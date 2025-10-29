//! Trait implementations for `kona-mpt` traits that are effectively a no-op.
//! Provides trait implementations for downstream users who do not require hinting.

use crate::{TrieHinter, TrieNode, TrieProvider};
use alloc::string::String;
use alloy_primitives::{Address, B256, U256};
use core::fmt::Debug;

/// The default, no-op implementation of the [TrieProvider] trait, used for testing.
#[derive(Debug, Clone, Copy)]
pub struct NoopTrieProvider;

impl TrieProvider for NoopTrieProvider {
    type Error = String;

    fn trie_node_by_hash(&self, _key: B256) -> Result<TrieNode, Self::Error> {
        Ok(TrieNode::Empty)
    }
}

/// The default, no-op implementation of the [TrieHinter] trait, used for testing.
#[derive(Debug, Clone, Copy)]
pub struct NoopTrieHinter;

impl TrieHinter for NoopTrieHinter {
    type Error = String;

    fn hint_trie_node(&self, _hash: B256) -> Result<(), Self::Error> {
        Ok(())
    }

    fn hint_account_proof(&self, _address: Address, _block_number: u64) -> Result<(), Self::Error> {
        Ok(())
    }

    fn hint_storage_proof(
        &self,
        _address: Address,
        _slot: U256,
        _block_number: u64,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn hint_execution_witness(
        &self,
        _parent_hash: B256,
        _op_payload_attributes: &op_alloy_rpc_types_engine::OpPayloadAttributes,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{address, b256, uint};

    #[test]
    fn test_noop_trie_provider() {
        let provider = NoopTrieProvider;
        let hash = b256!("0x1234567890123456789012345678901234567890123456789012345678901234");

        // Should always return Empty node
        let result = provider.trie_node_by_hash(hash);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), TrieNode::Empty);
    }

    #[test]
    fn test_noop_trie_provider_debug() {
        let provider = NoopTrieProvider;
        let debug_str = format!("{:?}", provider);
        assert_eq!(debug_str, "NoopTrieProvider");
    }

    #[test]
    fn test_noop_trie_provider_clone() {
        let provider = NoopTrieProvider;
        let cloned = provider.clone();

        // Both should behave the same
        let hash = b256!("0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd");
        assert_eq!(
            provider.trie_node_by_hash(hash).unwrap(),
            cloned.trie_node_by_hash(hash).unwrap()
        );
    }

    #[test]
    fn test_noop_trie_hinter() {
        let hinter = NoopTrieHinter;
        let hash = b256!("0x1234567890123456789012345678901234567890123456789012345678901234");

        // Should always succeed
        assert!(hinter.hint_trie_node(hash).is_ok());
    }

    #[test]
    fn test_noop_trie_hinter_account_proof() {
        let hinter = NoopTrieHinter;
        let address = address!("0x1234567890123456789012345678901234567890");
        let block_number = 12345u64;

        // Should always succeed
        assert!(hinter.hint_account_proof(address, block_number).is_ok());
    }

    #[test]
    fn test_noop_trie_hinter_storage_proof() {
        let hinter = NoopTrieHinter;
        let address = address!("0x1234567890123456789012345678901234567890");
        let slot = uint!(1_U256);
        let block_number = 12345u64;

        // Should always succeed
        assert!(hinter.hint_storage_proof(address, slot, block_number).is_ok());
    }

    #[test]
    fn test_noop_trie_hinter_execution_witness() {
        let hinter = NoopTrieHinter;
        let parent_hash = b256!("0x1234567890123456789012345678901234567890123456789012345678901234");
        let attrs = op_alloy_rpc_types_engine::OpPayloadAttributes::default();

        // Should always succeed
        assert!(hinter.hint_execution_witness(parent_hash, &attrs).is_ok());
    }

    #[test]
    fn test_noop_trie_hinter_debug() {
        let hinter = NoopTrieHinter;
        let debug_str = format!("{:?}", hinter);
        assert_eq!(debug_str, "NoopTrieHinter");
    }

    #[test]
    fn test_noop_trie_hinter_clone() {
        let hinter = NoopTrieHinter;
        let cloned = hinter.clone();

        // Both should behave the same
        let hash = b256!("0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd");
        assert!(hinter.hint_trie_node(hash).is_ok());
        assert!(cloned.hint_trie_node(hash).is_ok());
    }
}
