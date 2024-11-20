#![warn(clippy::imprecise_flops, clippy::suboptimal_flops)]

pub mod engine;
mod search;

pub use search::{allocate_tt, is_repetition_draw, Search, SearchParams, TtEntry};
