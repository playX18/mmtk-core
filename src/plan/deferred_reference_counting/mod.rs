//! Plan: deferred reference counting.

mod gc_work;
mod global;
pub mod mutator;

pub use self::global::DeferredReferenceCounting;
pub use self::global::DRC_CONSTRAINTS;
