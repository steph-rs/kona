//! Utilities for `kona-mpt`

use alloc::vec::Vec;
use alloy_rlp::{Buf, BufMut, Encodable, Header};
use alloy_trie::{HashBuilder, Nibbles, proof::ProofRetainer};

/// Compute a trie root of the collection of items with a custom encoder.
pub fn ordered_trie_with_encoder<T, F>(items: &[T], mut encode: F) -> HashBuilder
where
    F: FnMut(&T, &mut dyn BufMut),
{
    let mut index_buffer = Vec::new();
    let mut value_buffer = Vec::new();
    let items_len = items.len();

    // Store preimages for all intermediates
    let path_nibbles = (0..items_len)
        .map(|i| {
            let index = adjust_index_for_rlp(i, items_len);
            index_buffer.clear();
            index.encode(&mut index_buffer);
            Nibbles::unpack(&index_buffer)
        })
        .collect::<Vec<_>>();

    let mut hb = HashBuilder::default().with_proof_retainer(ProofRetainer::new(path_nibbles));
    for i in 0..items_len {
        let index = adjust_index_for_rlp(i, items_len);

        index_buffer.clear();
        index.encode(&mut index_buffer);

        value_buffer.clear();
        encode(&items[index], &mut value_buffer);

        hb.add_leaf(Nibbles::unpack(&index_buffer), &value_buffer);
    }

    hb
}

/// Adjust the index of an item for rlp encoding.
pub(crate) const fn adjust_index_for_rlp(i: usize, len: usize) -> usize {
    if i > 0x7f {
        i
    } else if i == 0x7f || i + 1 == len {
        0
    } else {
        i + 1
    }
}

/// Walks through a RLP list's elements and returns the total number of elements in the list.
/// Returns [alloy_rlp::Error::UnexpectedString] if the RLP stream is not a list.
///
/// ## Takes
/// - `buf` - The RLP stream to walk through
///
/// ## Returns
/// - `Ok(usize)` - The total number of elements in the list
/// - `Err(_)` - The RLP stream is not a list
pub(crate) fn rlp_list_element_length(buf: &mut &[u8]) -> alloy_rlp::Result<usize> {
    let header = Header::decode(buf)?;
    if !header.list {
        return Err(alloy_rlp::Error::UnexpectedString);
    }
    let len_after_consume = buf.len() - header.payload_length;

    let mut list_element_length = 0;
    while buf.len() > len_after_consume {
        let header = Header::decode(buf)?;
        buf.advance(header.payload_length);
        list_element_length += 1;
    }
    Ok(list_element_length)
}

/// Unpack node path to nibbles.
///
/// ## Takes
/// - `first` - first nibble of the path if it is odd. Must be <= 0x0F, or will create invalid
///   nibbles.
/// - `rest` - rest of the nibbles packed
///
/// ## Returns
/// - `Nibbles` - unpacked nibbles
pub(crate) fn unpack_path_to_nibbles(first: Option<u8>, rest: &[u8]) -> Nibbles {
    let rest = Nibbles::unpack(rest);
    Nibbles::from_iter_unchecked(first.into_iter().chain(rest.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_rlp::Encodable;

    #[test]
    fn test_adjust_index_for_rlp_small() {
        assert_eq!(adjust_index_for_rlp(0, 100), 1);
        assert_eq!(adjust_index_for_rlp(5, 100), 6);
        assert_eq!(adjust_index_for_rlp(0x7e, 100), 0x7f);
    }

    #[test]
    fn test_adjust_index_for_rlp_boundary() {
        // At 0x7f boundary
        assert_eq!(adjust_index_for_rlp(0x7f, 200), 0);
        // Past 0x7f
        assert_eq!(adjust_index_for_rlp(0x80, 200), 0x80);
        assert_eq!(adjust_index_for_rlp(0x81, 200), 0x81);
    }

    #[test]
    fn test_adjust_index_for_rlp_last_element() {
        // Last element (i + 1 == len)
        assert_eq!(adjust_index_for_rlp(99, 100), 0);
        assert_eq!(adjust_index_for_rlp(9, 10), 0);
    }

    #[test]
    fn test_rlp_list_element_length_error_on_string() {
        // Create a RLP-encoded string (not a list)
        let mut buf = Vec::new();
        "hello".encode(&mut buf);
        let mut slice = &buf[..];

        // Should return UnexpectedString error
        let result = rlp_list_element_length(&mut slice);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), alloy_rlp::Error::UnexpectedString);
    }

    #[test]
    fn test_unpack_path_to_nibbles_with_first() {
        let rest = vec![0xab, 0xcd];
        let nibbles = unpack_path_to_nibbles(Some(0x05), &rest);

        // Should have first nibble + unpacked rest
        assert_eq!(nibbles.len(), 5); // 1 + 4 (2 bytes = 4 nibbles)
        assert_eq!(nibbles.get(0), Some(0x05));
    }

    #[test]
    fn test_unpack_path_to_nibbles_without_first() {
        let rest = vec![0xab, 0xcd];
        let nibbles = unpack_path_to_nibbles(None, &rest);

        // Should just unpack rest
        assert_eq!(nibbles.len(), 4); // 2 bytes = 4 nibbles
    }
}
