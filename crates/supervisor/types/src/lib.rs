//! Core types shared across supervisor components.
//!
//! This crate defines the fundamental data structures used within the
//! Optimism supervisor.

pub mod head;
pub use head::SuperHead;

mod log;
pub use log::Log;

mod message;
pub use message::ExecutingMessage;

mod receipt;
pub use receipt::Receipts;

mod types;
pub use types::{BlockSeal, L2BlockRef, OutputV0, SubscriptionEvent};
