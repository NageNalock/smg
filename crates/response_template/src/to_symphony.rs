//! What a symphony table built from a response template needs from this crate.
//!
//! symphony reads a model's output with a table ([`symphony::Format`]), and a table names the
//! reader that types its calls' values ([`symphony::Format::reads_values_with`]). [`TRANSFORMERS`]
//! types them as transformers does, with this crate's own port, so that such a table and the
//! template's own reader ([`parse_response`](crate::parse_response)) type a value with one
//! implementation.

mod values;

pub use self::values::{Transformers, TRANSFORMERS};
