//! This module contains the [TrieNode] type, which represents a node within a standard Merkle
//! Patricia Trie.

use crate::{
    TrieHinter, TrieNodeError, TrieProvider,
    errors::TrieNodeResult,
    util::{rlp_list_element_length, unpack_path_to_nibbles},
};
use alloc::{boxed::Box, string::ToString, vec, vec::Vec};
use alloy_primitives::{B256, Bytes, keccak256};
use alloy_rlp::{Buf, Decodable, EMPTY_STRING_CODE, Encodable, Header, length_of_length};
use alloy_trie::{EMPTY_ROOT_HASH, Nibbles};

/// The length of the branch list when RLP encoded
const BRANCH_LIST_LENGTH: usize = 17;

/// The length of a leaf or extension node's RLP encoded list
const LEAF_OR_EXTENSION_LIST_LENGTH: usize = 2;

/// The number of nibbles traversed in a branch node.
const BRANCH_NODE_NIBBLES: usize = 1;

/// Prefix for even-nibbled extension node paths.
const PREFIX_EXTENSION_EVEN: u8 = 0;

/// Prefix for odd-nibbled extension node paths.
const PREFIX_EXTENSION_ODD: u8 = 1;

/// Prefix for even-nibbled leaf node paths.
const PREFIX_LEAF_EVEN: u8 = 2;

/// Prefix for odd-nibbled leaf node paths.
const PREFIX_LEAF_ODD: u8 = 3;

/// Nibble bit width.
const NIBBLE_WIDTH: usize = 4;

/// A [TrieNode] is a node within a standard Ethereum Merkle Patricia Trie. In this implementation,
/// keys are expected to be fixed-size nibble sequences, and values are arbitrary byte sequences.
///
/// The [TrieNode] has several variants:
/// - [TrieNode::Empty] represents an empty node.
/// - [TrieNode::Blinded] represents a node that has been blinded by a commitment.
/// - [TrieNode::Leaf] represents a 2-item node with the encoding `rlp([encoded_path, value])`.
/// - [TrieNode::Extension] represents a 2-item pointer node with the encoding `rlp([encoded_path,
///   key])`.
/// - [TrieNode::Branch] represents a node that refers to up to 16 child nodes with the encoding
///   `rlp([ v0, ..., v15, value ])`.
///
/// In the Ethereum Merkle Patricia Trie, nodes longer than an encoded 32 byte string (33 total
/// bytes) are blinded with [keccak256] hashes. When a node is "opened", it is replaced with the
/// [TrieNode] that is decoded from to the preimage of the hash.
///
/// The [alloy_rlp::Encodable] and [alloy_rlp::Decodable] traits are implemented for [TrieNode],
/// allowing for RLP encoding and decoding of the types for storage and retrieval. The
/// implementation of these traits will implicitly blind nodes that are longer than 32 bytes in
/// length when encoding. When decoding, the implementation will leave blinded nodes in place.
///
/// ## SAFETY
/// As this implementation only supports uniform key sizes, the [TrieNode] data structure will fail
/// to behave correctly if confronted with keys of varying lengths. Namely, this is because it does
/// not support the `value` field in branch nodes, just like the Ethereum Merkle Patricia Trie.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrieNode {
    /// An empty [TrieNode] is represented as an [EMPTY_STRING_CODE] (0x80).
    Empty,
    /// A blinded node is a node that has been blinded by a [keccak256] commitment.
    Blinded {
        /// The commitment that blinds the node.
        commitment: B256,
    },
    /// A leaf node is a 2-item node with the encoding `rlp([encoded_path, value])`
    Leaf {
        /// The key of the leaf node
        prefix: Nibbles,
        /// The value of the leaf node
        value: Bytes,
    },
    /// An extension node is a 2-item pointer node with the encoding `rlp([encoded_path, key])`
    Extension {
        /// The path prefix of the extension
        prefix: Nibbles,
        /// The pointer to the child node
        node: Box<TrieNode>,
    },
    /// A branch node refers to up to 16 child nodes with the encoding
    /// `rlp([ v0, ..., v15, value ])`
    Branch {
        /// The 16 child nodes and value of the branch.
        stack: Vec<TrieNode>,
    },
}

impl TrieNode {
    /// Creates a new [TrieNode::Blinded] node.
    ///
    /// ## Takes
    /// - `commitment` - The commitment that blinds the node
    ///
    /// ## Returns
    /// - `Self` - The new blinded [TrieNode].
    pub const fn new_blinded(commitment: B256) -> Self {
        Self::Blinded { commitment }
    }

    /// Blinds the [TrieNode].. Alternatively, if the [TrieNode] is a [TrieNode::Blinded] node
    /// already, its commitment is returned directly.
    pub fn blind(&self) -> B256 {
        match self {
            Self::Blinded { commitment } => *commitment,
            Self::Empty => EMPTY_ROOT_HASH,
            _ => {
                let mut rlp_buf = Vec::with_capacity(self.length());
                self.encode(&mut rlp_buf);
                keccak256(rlp_buf)
            }
        }
    }

    /// Unblinds the [TrieNode] if it is a [TrieNode::Blinded] node.
    pub fn unblind<F: TrieProvider>(&mut self, fetcher: &F) -> TrieNodeResult<()> {
        if let Self::Blinded { commitment } = self {
            if *commitment == EMPTY_ROOT_HASH {
                // If the commitment is the empty root hash, the node is empty, and we don't need to
                // reach out to the fetcher.
                *self = Self::Empty;
            } else {
                *self = fetcher
                    .trie_node_by_hash(*commitment)
                    .map_err(|e| TrieNodeError::Provider(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Walks down the trie to a leaf value with the given key, if it exists. Preimages for blinded
    /// nodes along the path are fetched using the `fetcher` function, and persisted in the inner
    /// [TrieNode] elements.
    ///
    /// ## Takes
    /// - `self` - The root trie node
    /// - `path` - The nibbles representation of the path to the leaf node
    /// - `fetcher` - The preimage fetcher for intermediate blinded nodes
    ///
    /// ## Returns
    /// - `Err(_)` - Could not retrieve the node with the given key from the trie.
    /// - `Ok(None)` - The node with the given key does not exist in the trie.
    /// - `Ok(Some(_))` - The value of the node
    pub fn open<'a, F: TrieProvider>(
        &'a mut self,
        path: &Nibbles,
        fetcher: &F,
    ) -> TrieNodeResult<Option<&'a mut Bytes>> {
        match self {
            Self::Branch { stack } => {
                let branch_nibble = path.get(0).ok_or(TrieNodeError::PathTooShort)? as usize;
                stack
                    .get_mut(branch_nibble)
                    .map(|node| node.open(&path.slice(BRANCH_NODE_NIBBLES..), fetcher))
                    .unwrap_or(Ok(None))
            }
            Self::Leaf { prefix, value } => Ok((path == prefix).then_some(value)),
            Self::Extension { prefix, node } => {
                if path.slice(..prefix.len()) == *prefix {
                    // Follow extension branch
                    node.unblind(fetcher)?;
                    node.open(&path.slice(prefix.len()..), fetcher)
                } else {
                    Ok(None)
                }
            }
            Self::Blinded { .. } => {
                self.unblind(fetcher)?;
                self.open(path, fetcher)
            }
            Self::Empty => Ok(None),
        }
    }

    /// Inserts a [TrieNode] at the given path into the trie rooted at Self.
    ///
    /// ## Takes
    /// - `self` - The root trie node
    /// - `path` - The nibbles representation of the path to the leaf node
    /// - `node` - The node to insert at the given path
    /// - `fetcher` - The preimage fetcher for intermediate blinded nodes
    ///
    /// ## Returns
    /// - `Err(_)` - Could not insert the node at the given path in the trie.
    /// - `Ok(())` - The node was successfully inserted at the given path.
    pub fn insert<F: TrieProvider>(
        &mut self,
        path: &Nibbles,
        value: Bytes,
        fetcher: &F,
    ) -> TrieNodeResult<()> {
        match self {
            Self::Empty => {
                // If the trie node is null, insert the leaf node at the current path.
                *self = Self::Leaf { prefix: *path, value };
                Ok(())
            }
            Self::Leaf { prefix, value: leaf_value } => {
                let shared_extension_nibbles = path.common_prefix_length(prefix);

                // If all nibbles are shared, update the leaf node with the new value.
                if path == prefix {
                    *self = Self::Leaf { prefix: *prefix, value };
                    return Ok(());
                }

                // Create a branch node stack containing the leaf node and the new value.
                let mut stack = vec![Self::Empty; BRANCH_LIST_LENGTH];

                // Insert the shortened extension into the branch stack.
                let extension_nibble =
                    prefix.get(shared_extension_nibbles).ok_or(TrieNodeError::PathTooShort)?
                        as usize;
                stack[extension_nibble] = Self::Leaf {
                    prefix: prefix.slice(shared_extension_nibbles + BRANCH_NODE_NIBBLES..),
                    value: leaf_value.clone(),
                };

                // Insert the new value into the branch stack.
                let branch_nibble_new =
                    path.get(shared_extension_nibbles).ok_or(TrieNodeError::PathTooShort)? as usize;
                stack[branch_nibble_new] = Self::Leaf {
                    prefix: path.slice(shared_extension_nibbles + BRANCH_NODE_NIBBLES..),
                    value,
                };

                // Replace the leaf node with the branch if no nibbles are shared, else create an
                // extension.
                if shared_extension_nibbles == 0 {
                    *self = Self::Branch { stack };
                } else {
                    let raw_ext_nibbles = path.slice(..shared_extension_nibbles);
                    *self = Self::Extension {
                        prefix: raw_ext_nibbles,
                        node: Box::new(Self::Branch { stack }),
                    };
                }
                Ok(())
            }
            Self::Extension { prefix, node } => {
                let shared_extension_nibbles = path.common_prefix_length(prefix);
                if shared_extension_nibbles == prefix.len() {
                    node.insert(&path.slice(shared_extension_nibbles..), value, fetcher)?;
                    return Ok(());
                }

                // Create a branch node stack containing the leaf node and the new value.
                let mut stack = vec![Self::Empty; BRANCH_LIST_LENGTH];

                // Insert the shortened extension into the branch stack.
                let extension_nibble =
                    prefix.get(shared_extension_nibbles).ok_or(TrieNodeError::PathTooShort)?
                        as usize;
                let new_prefix = prefix.slice(shared_extension_nibbles + BRANCH_NODE_NIBBLES..);
                stack[extension_nibble] = if new_prefix.is_empty() {
                    // In the case that the extension node no longer has a prefix, insert the node
                    // verbatim into the branch.
                    node.as_ref().clone()
                } else {
                    Self::Extension { prefix: new_prefix, node: node.clone() }
                };

                // Insert the new value into the branch stack.
                let branch_nibble_new =
                    path.get(shared_extension_nibbles).ok_or(TrieNodeError::PathTooShort)? as usize;
                stack[branch_nibble_new] = Self::Leaf {
                    prefix: path.slice(shared_extension_nibbles + BRANCH_NODE_NIBBLES..),
                    value,
                };

                // Replace the extension node with the branch if no nibbles are shared, else create
                // an extension.
                if shared_extension_nibbles == 0 {
                    *self = Self::Branch { stack };
                } else {
                    let extension = path.slice(..shared_extension_nibbles);
                    *self = Self::Extension {
                        prefix: extension,
                        node: Box::new(Self::Branch { stack }),
                    };
                }
                Ok(())
            }
            Self::Branch { stack } => {
                // Follow the branch node to the next node in the path.
                let branch_nibble = path.get(0).ok_or(TrieNodeError::PathTooShort)? as usize;
                stack[branch_nibble].insert(&path.slice(BRANCH_NODE_NIBBLES..), value, fetcher)
            }
            Self::Blinded { .. } => {
                // If a blinded node is approached, reveal the node and continue the insertion
                // recursion.
                self.unblind(fetcher)?;
                self.insert(path, value, fetcher)
            }
        }
    }

    /// Deletes a node in the trie at the given path.
    ///
    /// ## Takes
    /// - `self` - The root trie node
    /// - `path` - The nibbles representation of the path to the leaf node
    ///
    /// ## Returns
    /// - `Err(_)` - Could not delete the node at the given path in the trie.
    /// - `Ok(())` - The node was successfully deleted at the given path.
    pub fn delete<F: TrieProvider, H: TrieHinter>(
        &mut self,
        path: &Nibbles,
        fetcher: &F,
        hinter: &H,
    ) -> TrieNodeResult<()> {
        match self {
            Self::Empty => Err(TrieNodeError::KeyNotFound),
            Self::Leaf { prefix, .. } => {
                if path == prefix {
                    *self = Self::Empty;
                    Ok(())
                } else {
                    Err(TrieNodeError::KeyNotFound)
                }
            }
            Self::Extension { prefix, node } => {
                let shared_nibbles = path.common_prefix_length(prefix);
                if shared_nibbles < prefix.len() {
                    return Err(TrieNodeError::KeyNotFound);
                } else if shared_nibbles == path.len() {
                    *self = Self::Empty;
                    return Ok(());
                }

                node.delete(&path.slice(prefix.len()..), fetcher, hinter)?;

                // Simplify extension if possible after the deletion
                self.collapse_if_possible(fetcher, hinter)
            }
            Self::Branch { stack } => {
                let branch_nibble = path.get(0).ok_or(TrieNodeError::PathTooShort)? as usize;
                stack[branch_nibble].delete(&path.slice(BRANCH_NODE_NIBBLES..), fetcher, hinter)?;

                // Simplify the branch if possible after the deletion
                self.collapse_if_possible(fetcher, hinter)
            }
            Self::Blinded { .. } => {
                self.unblind(fetcher)?;
                self.delete(path, fetcher, hinter)
            }
        }
    }

    /// If applicable, collapses `self` into a more compact form.
    ///
    /// ## Takes
    /// - `self` - The root trie node
    ///
    /// ## Returns
    /// - `Ok(())` - The node was successfully collapsed
    /// - `Err(_)` - Could not collapse the node
    fn collapse_if_possible<F: TrieProvider, H: TrieHinter>(
        &mut self,
        fetcher: &F,
        hinter: &H,
    ) -> TrieNodeResult<()> {
        match self {
            Self::Extension { prefix, node } => match node.as_mut() {
                Self::Extension { prefix: child_prefix, node: child_node } => {
                    // Double extensions are collapsed into a single extension.
                    let new_prefix = Nibbles::from_nibbles_unchecked(
                        [prefix.to_vec(), child_prefix.to_vec()].concat(),
                    );
                    *self = Self::Extension { prefix: new_prefix, node: child_node.clone() };
                }
                Self::Leaf { prefix: child_prefix, value: child_value } => {
                    // If the child node is a leaf, convert the extension into a leaf with the full
                    // path.
                    let new_prefix = Nibbles::from_nibbles_unchecked(
                        [prefix.to_vec(), child_prefix.to_vec()].concat(),
                    );
                    *self = Self::Leaf { prefix: new_prefix, value: child_value.clone() };
                }
                Self::Empty => {
                    // If the child node is empty, convert the extension into an empty node.
                    *self = Self::Empty;
                }
                _ => {
                    // If the child is a (blinded?) branch then no need for collapse
                    // because deletion did not collapse the (blinded?) branch
                }
            },
            Self::Branch { stack } => {
                // Count non-empty children
                let mut non_empty_children = stack
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, node)| !matches!(node, Self::Empty))
                    .collect::<Vec<_>>();

                if non_empty_children.len() == 1 {
                    let (index, non_empty_node) = &mut non_empty_children[0];

                    // If only one non-empty child and no value, convert to extension or leaf
                    match non_empty_node {
                        Self::Leaf { prefix, value } => {
                            let new_prefix = Nibbles::from_nibbles_unchecked(
                                [&[*index as u8], prefix.to_vec().as_slice()].concat(),
                            );
                            *self = Self::Leaf { prefix: new_prefix, value: value.clone() };
                        }
                        Self::Extension { prefix, node } => {
                            let new_prefix = Nibbles::from_nibbles_unchecked(
                                [&[*index as u8], prefix.to_vec().as_slice()].concat(),
                            );
                            *self = Self::Extension { prefix: new_prefix, node: node.clone() };
                        }
                        Self::Branch { .. } => {
                            *self = Self::Extension {
                                prefix: Nibbles::from_nibbles_unchecked([*index as u8]),
                                node: Box::new(non_empty_node.clone()),
                            };
                        }
                        Self::Blinded { commitment } => {
                            // In this special case, we need to send a hint to fetch the preimage of
                            // the blinded node, since it is outside of the paths that have been
                            // traversed so far.
                            hinter
                                .hint_trie_node(*commitment)
                                .map_err(|e| TrieNodeError::Provider(e.to_string()))?;

                            non_empty_node.unblind(fetcher)?;
                            self.collapse_if_possible(fetcher, hinter)?;
                        }
                        _ => {}
                    };
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Attempts to convert a `path` and `value` into a [TrieNode], if they correspond to a
    /// [TrieNode::Leaf] or [TrieNode::Extension].
    ///
    /// **Note:** This function assumes that the passed reader has already consumed the RLP header
    /// of the [TrieNode::Leaf] or [TrieNode::Extension] node.
    fn try_decode_leaf_or_extension_payload(buf: &mut &[u8]) -> TrieNodeResult<Self> {
        // Decode the path and value of the leaf or extension node.
        let path = Bytes::decode(buf).map_err(TrieNodeError::RLPError)?;
        let first_nibble = path[0] >> NIBBLE_WIDTH;
        let first = match first_nibble {
            PREFIX_EXTENSION_ODD | PREFIX_LEAF_ODD => Some(path[0] & 0x0F),
            PREFIX_EXTENSION_EVEN | PREFIX_LEAF_EVEN => None,
            _ => return Err(TrieNodeError::InvalidNodeType),
        };

        // Check the high-order nibble of the path to determine the type of node.
        match first_nibble {
            PREFIX_EXTENSION_EVEN | PREFIX_EXTENSION_ODD => {
                // Extension node
                let extension_node_value = Self::decode(buf).map_err(TrieNodeError::RLPError)?;
                Ok(Self::Extension {
                    prefix: unpack_path_to_nibbles(first, path[1..].as_ref()),
                    node: Box::new(extension_node_value),
                })
            }
            PREFIX_LEAF_EVEN | PREFIX_LEAF_ODD => {
                // Leaf node
                let value = Bytes::decode(buf).map_err(TrieNodeError::RLPError)?;
                Ok(Self::Leaf { prefix: unpack_path_to_nibbles(first, path[1..].as_ref()), value })
            }
            _ => Err(TrieNodeError::InvalidNodeType),
        }
    }

    /// Returns the RLP payload length of the [TrieNode].
    pub(crate) fn payload_length(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Blinded { commitment } => commitment.len(),
            Self::Leaf { prefix, value } => {
                let mut encoded_key_len = prefix.len() / 2 + 1;
                if encoded_key_len != 1 {
                    encoded_key_len += length_of_length(encoded_key_len);
                }
                encoded_key_len + value.length()
            }
            Self::Extension { prefix, node } => {
                let mut encoded_key_len = prefix.len() / 2 + 1;
                if encoded_key_len != 1 {
                    encoded_key_len += length_of_length(encoded_key_len);
                }
                encoded_key_len + node.blinded_length()
            }
            Self::Branch { stack } => {
                // In branch nodes, if an element is longer than an encoded 32 byte string, it is
                // blinded. Assuming we have an open trie node, we must re-hash the
                // elements that are longer than an encoded 32 byte string
                // in length.
                stack.iter().fold(0, |mut acc, node| {
                    acc += node.blinded_length();
                    acc
                })
            }
        }
    }

    /// Returns the encoded length of the trie node, blinding it if it is longer than an encoded
    /// [B256] string in length.
    ///
    /// ## Returns
    /// - `usize` - The encoded length of the value, blinded if the raw encoded length is longer
    ///   than a [B256].
    fn blinded_length(&self) -> usize {
        let encoded_len = self.length();
        if encoded_len >= B256::ZERO.len() { B256::ZERO.length() } else { encoded_len }
    }
}

impl Encodable for TrieNode {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        let payload_length = self.payload_length();
        match self {
            Self::Empty => out.put_u8(EMPTY_STRING_CODE),
            Self::Blinded { commitment } => commitment.encode(out),
            Self::Leaf { prefix, value } => {
                // Encode the leaf node's header and key-value pair.
                Header { list: true, payload_length }.encode(out);
                alloy_trie::nodes::encode_path_leaf(prefix, true).as_slice().encode(out);
                value.encode(out);
            }
            Self::Extension { prefix, node } => {
                // Encode the extension node's header, prefix, and pointer node.
                Header { list: true, payload_length }.encode(out);
                alloy_trie::nodes::encode_path_leaf(prefix, false).as_slice().encode(out);
                if node.length() >= B256::ZERO.len() {
                    let hash = node.blind();
                    hash.encode(out);
                } else {
                    node.encode(out);
                }
            }
            Self::Branch { stack } => {
                // In branch nodes, if an element is longer than 32 bytes in length, it is blinded.
                // Assuming we have an open trie node, we must re-hash the elements
                // that are longer than 32 bytes in length.
                Header { list: true, payload_length }.encode(out);
                stack.iter().for_each(|node| {
                    if node.length() >= B256::ZERO.len() {
                        let hash = node.blind();
                        hash.encode(out);
                    } else {
                        node.encode(out);
                    }
                });
            }
        }
    }

    fn length(&self) -> usize {
        match self {
            Self::Empty => 1,
            Self::Blinded { commitment } => commitment.length(),
            Self::Leaf { .. } => {
                let payload_length = self.payload_length();
                Header { list: true, payload_length }.length() + payload_length
            }
            Self::Extension { .. } => {
                let payload_length = self.payload_length();
                Header { list: true, payload_length }.length() + payload_length
            }
            Self::Branch { .. } => {
                let payload_length = self.payload_length();
                Header { list: true, payload_length }.length() + payload_length
            }
        }
    }
}

impl Decodable for TrieNode {
    /// Attempts to decode the [TrieNode].
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        // Peek at the header to determine the type of Trie node we're currently decoding.
        let header = Header::decode(&mut (**buf).as_ref())?;

        if header.list {
            // Peek at the RLP stream to determine the number of elements in the list.
            let list_length = rlp_list_element_length(&mut (**buf).as_ref())?;

            match list_length {
                BRANCH_LIST_LENGTH => {
                    let list = Vec::<Self>::decode(buf)?;
                    Ok(Self::Branch { stack: list })
                }
                LEAF_OR_EXTENSION_LIST_LENGTH => {
                    // Advance the buffer to the start of the list payload.
                    buf.advance(header.length());
                    // Decode the leaf or extension node's raw payload.
                    Self::try_decode_leaf_or_extension_payload(buf)
                        .map_err(|_| alloy_rlp::Error::UnexpectedList)
                }
                _ => Err(alloy_rlp::Error::UnexpectedLength),
            }
        } else {
            match header.payload_length {
                0 => {
                    buf.advance(header.length());
                    Ok(Self::Empty)
                }
                32 => {
                    let commitment = B256::decode(buf)?;
                    Ok(Self::new_blinded(commitment))
                }
                _ => Err(alloy_rlp::Error::UnexpectedLength),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        NoopTrieHinter, NoopTrieProvider, TrieNode, ordered_trie_with_encoder,
        test_util::TrieNodeProvider,
    };
    use alloc::{collections::BTreeMap, vec, vec::Vec};
    use alloy_primitives::{b256, bytes, hex, keccak256};
    use alloy_rlp::{Decodable, EMPTY_STRING_CODE, Encodable};
    use alloy_trie::{HashBuilder, Nibbles};
    use rand::prelude::IteratorRandom;

    #[test]
    fn test_empty_blinded() {
        let trie_node = TrieNode::Empty;
        assert_eq!(trie_node.blind(), EMPTY_ROOT_HASH);
    }

    #[test]
    fn test_decode_branch() {
        const BRANCH_RLP: [u8; 83] = hex!(
            "f851a0eb08a66a94882454bec899d3e82952dcc918ba4b35a09a84acd98019aef4345080808080808080a05d87a81d9bbf5aee61a6bfeab3a5643347e2c751b36789d988a5b6b163d496518080808080808080"
        );
        let expected = TrieNode::Branch {
            stack: vec![
                TrieNode::new_blinded(b256!(
                    "eb08a66a94882454bec899d3e82952dcc918ba4b35a09a84acd98019aef43450"
                )),
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::new_blinded(b256!(
                    "5d87a81d9bbf5aee61a6bfeab3a5643347e2c751b36789d988a5b6b163d49651"
                )),
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
                TrieNode::Empty,
            ],
        };

        let mut rlp_buf = Vec::with_capacity(expected.length());
        expected.encode(&mut rlp_buf);
        assert_eq!(rlp_buf.len(), BRANCH_RLP.len());
        assert_eq!(expected.length(), BRANCH_RLP.len());

        assert_eq!(expected, TrieNode::decode(&mut BRANCH_RLP.as_slice()).unwrap());
        assert_eq!(rlp_buf.as_slice(), &BRANCH_RLP[..]);
    }

    #[test]
    fn test_encode_decode_extension_open_short() {
        const EXTENSION_RLP: [u8; 19] = hex!("d28300646fcd308b8a74657374207468726565");

        let opened = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles([0x00]),
            value: bytes!("8a74657374207468726565"),
        };
        let expected =
            TrieNode::Extension { prefix: Nibbles::unpack(bytes!("646f")), node: Box::new(opened) };

        let mut rlp_buf = Vec::with_capacity(expected.length());
        expected.encode(&mut rlp_buf);

        assert_eq!(expected, TrieNode::decode(&mut EXTENSION_RLP.as_slice()).unwrap());
    }

    #[test]
    fn test_encode_decode_extension_blinded_long() {
        const EXTENSION_RLP: [u8; 38] =
            hex!("e58300646fa0f3fe8b3c5b21d3e52860f1e4a5825a6100bb341069c1e88f4ebf6bd98de0c190");
        let mut rlp_buf = Vec::new();

        let opened =
            TrieNode::Leaf { prefix: Nibbles::from_nibbles([0x00]), value: [0xFF; 64].into() };
        opened.encode(&mut rlp_buf);
        let blinded = TrieNode::new_blinded(keccak256(&rlp_buf));

        rlp_buf.clear();
        let opened_extension =
            TrieNode::Extension { prefix: Nibbles::unpack(bytes!("646f")), node: Box::new(opened) };
        opened_extension.encode(&mut rlp_buf);

        let expected = TrieNode::Extension {
            prefix: Nibbles::unpack(bytes!("646f")),
            node: Box::new(blinded),
        };
        assert_eq!(expected, TrieNode::decode(&mut EXTENSION_RLP.as_slice()).unwrap());
    }

    #[test]
    fn test_decode_leaf() {
        const LEAF_RLP: [u8; 11] = hex!("ca8320646f8576657262FF");
        let expected =
            TrieNode::Leaf { prefix: Nibbles::unpack(bytes!("646f")), value: bytes!("76657262FF") };
        assert_eq!(expected, TrieNode::decode(&mut LEAF_RLP.as_slice()).unwrap());
    }

    #[test]
    fn test_retrieve_from_trie_simple() {
        const VALUES: [&str; 5] = ["yeah", "dog", ", ", "laminar", "flow"];

        let mut trie = ordered_trie_with_encoder(&VALUES, |v, buf| {
            let mut encoded_value = Vec::with_capacity(v.length());
            v.encode(&mut encoded_value);
            TrieNode::new_blinded(keccak256(encoded_value)).encode(buf);
        });
        let root = trie.root();

        let preimages = trie.take_proof_nodes().into_inner().into_iter().fold(
            BTreeMap::default(),
            |mut acc, (_, value)| {
                acc.insert(keccak256(value.as_ref()), value);
                acc
            },
        );
        let fetcher = TrieNodeProvider::new(preimages);

        let mut root_node = fetcher.trie_node_by_hash(root).unwrap();
        for (i, value) in VALUES.iter().enumerate() {
            let path_nibbles = Nibbles::unpack([if i == 0 { EMPTY_STRING_CODE } else { i as u8 }]);
            let v = root_node.open(&path_nibbles, &fetcher).unwrap().unwrap();

            let mut encoded_value = Vec::with_capacity(value.length());
            value.encode(&mut encoded_value);
            let mut encoded_node = Vec::new();
            TrieNode::new_blinded(keccak256(&encoded_value)).encode(&mut encoded_node);

            assert_eq!(v, encoded_node.as_slice());
        }

        let commitment = root_node.blind();
        assert_eq!(commitment, root);
    }

    #[test]
    fn test_insert_static() {
        let mut node = TrieNode::Empty;
        let noop_fetcher = NoopTrieProvider;
        node.insert(&Nibbles::unpack(hex!("012345")), bytes!("01"), &noop_fetcher).unwrap();
        node.insert(&Nibbles::unpack(hex!("012346")), bytes!("02"), &noop_fetcher).unwrap();

        let expected = TrieNode::Extension {
            prefix: Nibbles::from_nibbles([0, 1, 2, 3, 4]),
            node: Box::new(TrieNode::Branch {
                stack: vec![
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Leaf { prefix: Nibbles::default(), value: bytes!("01") },
                    TrieNode::Leaf { prefix: Nibbles::default(), value: bytes!("02") },
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                    TrieNode::Empty,
                ],
            }),
        };

        assert_eq!(node, expected);
    }

    proptest::proptest! {
        /// Differential test for inserting an arbitrary number of keys into an empty `TrieNode` / `HashBuilder`.
        #[test]
        fn diff_hash_builder_insert(mut keys in proptest::collection::vec(proptest::prelude::any::<[u8; 32]>(), 1..4096)) {
            // Ensure the keys are sorted; `HashBuilder` expects sorted keys.`
            keys.sort();

            let mut hb = HashBuilder::default();
            let mut node = TrieNode::Empty;

            for key in keys {
                hb.add_leaf(Nibbles::unpack(key), key.as_ref());
                node.insert(&Nibbles::unpack(key), key.into(), &NoopTrieProvider).unwrap();
            }

            assert_eq!(node.blind(), hb.root());
        }

        /// Differential test for deleting an arbitrary number of keys from a `TrieNode` / `HashBuilder`.
        #[test]
        fn diff_hash_builder_delete(mut keys in proptest::collection::vec(proptest::prelude::any::<[u8; 32]>(), 1..4096)) {
            // Ensure the keys are sorted; `HashBuilder` expects sorted keys.`
            keys.sort();

            let mut hb = HashBuilder::default();
            let mut node = TrieNode::Empty;

            let mut rng = rand::rng();
            let deleted_keys =
            keys.clone().into_iter().choose_multiple(&mut rng, 5.min(keys.len()));

            // Insert the keys into the `HashBuilder` and `TrieNode`.
            for key in keys {
                // Don't add any keys that are to be deleted from the trie node to the `HashBuilder`.
                if !deleted_keys.contains(&key) {
                    hb.add_leaf(Nibbles::unpack(key), key.as_ref());
                }
                node.insert(&Nibbles::unpack(key), key.into(), &NoopTrieProvider).unwrap();
            }

            // Delete the keys that were randomly selected from the trie node.
            for deleted_key in deleted_keys {
                node.delete(&Nibbles::unpack(deleted_key), &NoopTrieProvider, &NoopTrieHinter)
                    .unwrap();
            }

            // Blind manually, since the single node remaining may be a leaf or empty node, and always must be blinded.
            let mut rlp_buf = Vec::with_capacity(node.length());
            node.encode(&mut rlp_buf);
            let trie_root = keccak256(rlp_buf);

            assert_eq!(trie_root, hb.root());
        }
    }

    // Additional tests for uncovered paths

    #[test]
    fn test_delete_from_empty() {
        let mut node = TrieNode::Empty;
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03]);

        let result = node.delete(&path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrieNodeError::KeyNotFound);
    }

    #[test]
    fn test_delete_from_leaf_not_found() {
        let mut node = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("76616c7565"), // "value"
        };
        let wrong_path = Nibbles::from_nibbles_unchecked([0x01, 0x03]);

        let result = node.delete(&wrong_path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrieNodeError::KeyNotFound);
    }

    #[test]
    fn test_delete_from_leaf_success() {
        let mut node = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("76616c7565"), // "value"
        };
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02]);

        let result = node.delete(&path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());
        assert_eq!(node, TrieNode::Empty);
    }

    #[test]
    fn test_delete_extension_key_not_found() {
        let leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x03, 0x04]),
            value: bytes!("76616c7565"), // "value"
        };
        let mut node = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(leaf),
        };

        // Path doesn't match prefix
        let wrong_path = Nibbles::from_nibbles_unchecked([0x01, 0x05]);
        let result = node.delete(&wrong_path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrieNodeError::KeyNotFound);
    }

    #[test]
    fn test_delete_extension_exact_match() {
        let leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x03]),
            value: bytes!("76616c7565"), // "value"
        };
        let mut node = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(leaf),
        };

        // Exact match on extension prefix
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02]);
        let result = node.delete(&path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());
        assert_eq!(node, TrieNode::Empty);
    }

    #[test]
    fn test_collapse_extension_to_leaf() {
        // Extension pointing to a leaf should collapse into a single leaf
        let leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x03, 0x04]),
            value: bytes!("76616c7565"), // "value"
        };
        let mut extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(leaf),
        };

        let result = extension.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should be collapsed to a leaf with combined prefix
        match extension {
            TrieNode::Leaf { prefix, value } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03, 0x04]));
                assert_eq!(value, bytes!("76616c7565")); // "value"
            }
            _ => panic!("Expected Leaf node after collapse"),
        }
    }

    #[test]
    fn test_collapse_extension_to_extension() {
        // Extension pointing to extension should collapse into single extension
        let inner_leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x05, 0x06]),
            value: bytes!("76616c7565"), // "value"
        };
        let inner_extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x03, 0x04]),
            node: Box::new(inner_leaf),
        };
        let mut outer_extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(inner_extension),
        };

        let result = outer_extension.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should be collapsed to extension with combined prefix
        match outer_extension {
            TrieNode::Extension { prefix, node } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03, 0x04]));
                match node.as_ref() {
                    TrieNode::Leaf { .. } => {} // Expected
                    _ => panic!("Extension should point to leaf"),
                }
            }
            _ => panic!("Expected Extension node after collapse"),
        }
    }

    #[test]
    fn test_collapse_extension_to_empty() {
        // Extension pointing to empty should become empty
        let mut extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(TrieNode::Empty),
        };

        let result = extension.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());
        assert_eq!(extension, TrieNode::Empty);
    }

    #[test]
    fn test_collapse_branch_with_single_child_leaf() {
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        stack[5] = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x03, 0x04]),
            value: bytes!("76616c7565"), // "value"
        };

        let mut branch = TrieNode::Branch { stack };
        let result = branch.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should collapse to a leaf with branch index prepended
        match branch {
            TrieNode::Leaf { prefix, value } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x05, 0x03, 0x04]));
                assert_eq!(value, bytes!("76616c7565")); // "value"
            }
            _ => panic!("Expected Leaf after branch collapse"),
        }
    }

    #[test]
    fn test_collapse_branch_with_single_child_extension() {
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        stack[3] = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x07, 0x08]),
            node: Box::new(TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x09]),
                value: bytes!("76616c"), // "val"
            }),
        };

        let mut branch = TrieNode::Branch { stack };
        let result = branch.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should collapse to extension with branch index prepended
        match branch {
            TrieNode::Extension { prefix, .. } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x03, 0x07, 0x08]));
            }
            _ => panic!("Expected Extension after branch collapse"),
        }
    }

    #[test]
    fn test_collapse_branch_with_single_child_branch() {
        let mut inner_stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        inner_stack[2] = TrieNode::Leaf {
            prefix: Nibbles::default(),
            value: bytes!("76616c"), // "val"
        };

        let mut outer_stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        outer_stack[7] = TrieNode::Branch { stack: inner_stack };

        let mut branch = TrieNode::Branch { stack: outer_stack };
        let result = branch.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should collapse to extension pointing to branch
        match branch {
            TrieNode::Extension { prefix, node } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x07]));
                match node.as_ref() {
                    TrieNode::Branch { .. } => {} // Expected
                    _ => panic!("Should point to branch"),
                }
            }
            _ => panic!("Expected Extension after branch collapse"),
        }
    }

    #[test]
    fn test_open_path_too_short() {
        let mut branch = TrieNode::Branch {
            stack: vec![TrieNode::Empty; BRANCH_LIST_LENGTH],
        };

        let empty_path = Nibbles::default();
        let result = branch.open(&empty_path, &NoopTrieProvider);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrieNodeError::PathTooShort);
    }

    #[test]
    fn test_open_leaf_wrong_path() {
        let mut leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("76616c7565"), // "value"
        };

        let wrong_path = Nibbles::from_nibbles_unchecked([0x01, 0x03]);
        let result = leaf.open(&wrong_path, &NoopTrieProvider);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_open_leaf_correct_path() {
        let mut leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("74657374"), // "test"
        };

        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02]);
        let result = leaf.open(&path, &NoopTrieProvider);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(&mut bytes!("74657374"))); // "test"
    }

    #[test]
    fn test_open_extension_wrong_path() {
        let mut extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x03]),
                value: bytes!("76616c"), // "val"
            }),
        };

        let wrong_path = Nibbles::from_nibbles_unchecked([0x01, 0x05]);
        let result = extension.open(&wrong_path, &NoopTrieProvider);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_insert_leaf_update_value() {
        let mut node = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("6f6c64"), // "old"
        };

        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02]);
        let result = node.insert(&path, bytes!("6e6577"), &NoopTrieProvider); // "new"
        assert!(result.is_ok());

        match node {
            TrieNode::Leaf { prefix, value } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x01, 0x02]));
                assert_eq!(value, bytes!("6e6577")); // "new"
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_insert_creates_branch_from_leaf() {
        let mut node = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("666972"), // "fir"
        };

        // Insert with different second nibble - should create branch
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x03]);
        let result = node.insert(&path, bytes!("736563"), &NoopTrieProvider); // "sec"
        assert!(result.is_ok());

        // Should now be an extension pointing to a branch
        match node {
            TrieNode::Extension { prefix, node } => {
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x01]));
                match node.as_ref() {
                    TrieNode::Branch { stack } => {
                        // Should have two entries
                        assert!(matches!(stack[2], TrieNode::Leaf { .. }));
                        assert!(matches!(stack[3], TrieNode::Leaf { .. }));
                    }
                    _ => panic!("Expected branch node"),
                }
            }
            _ => panic!("Expected extension node"),
        }
    }

    #[test]
    fn test_encode_decode_empty() {
        let node = TrieNode::Empty;
        let mut buf = Vec::new();
        node.encode(&mut buf);

        let decoded = TrieNode::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, TrieNode::Empty);
    }

    #[test]
    fn test_blind_empty_returns_empty_root() {
        let node = TrieNode::Empty;
        assert_eq!(node.blind(), EMPTY_ROOT_HASH);
    }

    #[test]
    fn test_delete_path_too_short_branch() {
        let mut branch = TrieNode::Branch {
            stack: vec![TrieNode::Empty; BRANCH_LIST_LENGTH],
        };

        let empty_path = Nibbles::default();
        let result = branch.delete(&empty_path, &NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TrieNodeError::PathTooShort);
    }

    #[test]
    fn test_encode_extension_with_large_child() {
        // Create a large branch that will be blinded when encoded in extension
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        for i in 0..16 {
            stack[i] = TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x0a + i as u8, 0x0b]),
                value: bytes!("6c617267655f76616c7565"), // "large_value"
            };
        }

        let extension = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01]),
            node: Box::new(TrieNode::Branch { stack }),
        };

        let mut buf = Vec::new();
        extension.encode(&mut buf);

        // Should encode successfully with blinded child
        assert!(!buf.is_empty());
        assert!(buf.len() < 1000); // Much smaller due to blinding
    }

    #[test]
    fn test_encode_branch_with_blinded_children() {
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];

        // Add some large nodes that will be blinded (>= 32 bytes when encoded)
        // Create a large value (64 bytes) to ensure blinding
        let large_value = bytes!("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");

        for i in 0..5 {
            let mut inner_stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
            inner_stack[0] = TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x0a, 0x0b]),
                value: large_value.clone(),
            };
            stack[i] = TrieNode::Branch { stack: inner_stack };
        }

        let branch = TrieNode::Branch { stack };
        let mut buf = Vec::new();
        branch.encode(&mut buf);

        // Decode it back
        let decoded = TrieNode::decode(&mut buf.as_slice()).unwrap();
        match decoded {
            TrieNode::Branch { stack: dec_stack } => {
                // Some children should be blinded
                let blinded_count = dec_stack.iter().filter(|n| matches!(n, TrieNode::Blinded { .. })).count();
                assert!(blinded_count > 0);
            }
            _ => panic!("Expected branch node"),
        }
    }

    #[test]
    fn test_insert_extension_no_shared_nibbles() {
        // Extension with prefix [0x01, 0x02]
        let mut node = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            node: Box::new(TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x03]),
                value: bytes!("666972737476616c"), // "firstval"
            }),
        };

        // Insert with completely different prefix [0x05, 0x06]
        let path = Nibbles::from_nibbles_unchecked([0x05, 0x06, 0x07]);
        let result = node.insert(&path, bytes!("7365636f6e6476616c"), &NoopTrieProvider); // "secondval"
        assert!(result.is_ok());

        // Should create a branch at root level
        match node {
            TrieNode::Branch { stack } => {
                // Should have entries at positions 1 and 5
                assert!(matches!(stack[1], TrieNode::Extension { .. } | TrieNode::Leaf { .. }));
                assert!(matches!(stack[5], TrieNode::Extension { .. } | TrieNode::Leaf { .. }));
            }
            _ => panic!("Expected branch node after insert with no shared nibbles"),
        }
    }

    #[test]
    fn test_insert_extension_partial_match() {
        // Extension with longer prefix
        let mut node = TrieNode::Extension {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03]),
            node: Box::new(TrieNode::Leaf {
                prefix: Nibbles::from_nibbles_unchecked([0x04]),
                value: bytes!("76616c"), // "val"
            }),
        };

        // Insert that matches only first 2 nibbles
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x05]);
        let result = node.insert(&path, bytes!("6e657776616c"), &NoopTrieProvider); // "newval"
        assert!(result.is_ok());

        // Should split the extension
        match node {
            TrieNode::Extension { prefix, node } => {
                // Common prefix should be [0x01, 0x02]
                assert_eq!(prefix, Nibbles::from_nibbles_unchecked([0x01, 0x02]));
                // Should point to a branch
                match node.as_ref() {
                    TrieNode::Branch { .. } => {}, // Expected
                    _ => panic!("Should point to branch"),
                }
            }
            _ => panic!("Expected extension node"),
        }
    }

    #[test]
    fn test_decode_invalid_prefix() {
        // Create RLP with invalid prefix nibble (not 0, 1, 2, or 3)
        let mut buf = Vec::new();
        // Manually create invalid RLP
        Header { list: true, payload_length: 10 }.encode(&mut buf);
        bytes!("48").encode(&mut buf); // Invalid prefix (0x4 in high nibble)
        bytes!("010203").encode(&mut buf);
        bytes!("76616c7565").encode(&mut buf); // "value"

        let result = TrieNode::decode(&mut buf.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_unexpected_list_length() {
        // Create RLP list with 5 elements (invalid, should be 2 or 17)
        let mut buf = Vec::new();
        let five_empty = vec![TrieNode::Empty; 5];
        five_empty.encode(&mut buf);

        let result = TrieNode::decode(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            alloy_rlp::Error::UnexpectedLength => {}, // Expected
            _ => panic!("Expected UnexpectedLength error"),
        }
    }

    #[test]
    fn test_open_branch_empty_child() {
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        stack[5] = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x02]),
            value: bytes!("76616c"), // "val"
        };

        let mut branch = TrieNode::Branch { stack };

        // Try to open path that leads to empty child
        let path = Nibbles::from_nibbles_unchecked([0x03, 0x01]);
        let result = branch.open(&path, &NoopTrieProvider);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_insert_to_empty_creates_leaf() {
        let mut node = TrieNode::Empty;
        let path = Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03]);
        let result = node.insert(&path, bytes!("74657374"), &NoopTrieProvider); // "test"

        assert!(result.is_ok());
        match node {
            TrieNode::Leaf { prefix, value } => {
                assert_eq!(prefix, path);
                assert_eq!(value, bytes!("74657374")); // "test"
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_branch_collapse_with_multiple_children_no_collapse() {
        let mut stack = vec![TrieNode::Empty; BRANCH_LIST_LENGTH];
        stack[2] = TrieNode::Leaf {
            prefix: Nibbles::default(),
            value: bytes!("76616c31"), // "val1"
        };
        stack[5] = TrieNode::Leaf {
            prefix: Nibbles::default(),
            value: bytes!("76616c32"), // "val2"
        };

        let mut branch = TrieNode::Branch { stack: stack.clone() };
        let result = branch.collapse_if_possible(&NoopTrieProvider, &NoopTrieHinter);
        assert!(result.is_ok());

        // Should NOT collapse - still has multiple children
        match branch {
            TrieNode::Branch { .. } => {}, // Expected - no collapse
            _ => panic!("Should remain a branch with multiple children"),
        }
    }

    #[test]
    fn test_payload_length_variations() {
        // Test empty
        assert_eq!(TrieNode::Empty.payload_length(), 0);

        // Test blinded
        let blinded = TrieNode::Blinded {
            commitment: b256!("1234567890123456789012345678901234567890123456789012345678901234"),
        };
        assert_eq!(blinded.payload_length(), 32);

        // Test leaf with short value
        let short_leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01]),
            value: bytes!("01"),
        };
        assert!(short_leaf.payload_length() > 0);

        // Test leaf with long value
        let long_leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03, 0x04]),
            value: bytes!("0102030405060708090a0b0c0d0e0f10"),
        };
        assert!(long_leaf.payload_length() > short_leaf.payload_length());
    }

    #[test]
    fn test_length_method() {
        // Empty is always 1 byte
        assert_eq!(TrieNode::Empty.length(), 1);

        // Leaf length
        let leaf = TrieNode::Leaf {
            prefix: Nibbles::from_nibbles_unchecked([0x01, 0x02]),
            value: bytes!("76616c7565"), // "value"
        };
        let len = leaf.length();
        assert!(len > 1);

        // Encode and verify length matches
        let mut buf = Vec::new();
        leaf.encode(&mut buf);
        assert_eq!(buf.len(), len);
    }
}
