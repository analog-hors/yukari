use std::simd::{cmp::SimdOrd, i16x64, i32x64, num::SimdInt};

use crate::{Colour, Piece, Square};

const HIDDEN_SIZE: usize = 512;
const OUTPUT_BUCKETS: usize = 8;
const DIVISOR: usize = 32_usize.div_ceil(OUTPUT_BUCKETS);
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

/// This is the quantised format that bullet outputs.
#[repr(C)]
struct RawNetwork {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `OUTPUT_BUCKETS x (2 * HIDDEN_SIZE)` matrix.
    output_weights: [[i16; OUTPUT_BUCKETS]; 2 * HIDDEN_SIZE],
    /// Scalar output biases.
    output_bias: [i16; OUTPUT_BUCKETS],
}

impl RawNetwork {
    #[allow(clippy::large_stack_frames)]
    const fn into_network(self) -> Network {
        let mut output_weights = [[self.feature_bias; 2]; OUTPUT_BUCKETS];

        // const for requires const Iterator which requires const traits...
        let mut bucket = 0;
        while bucket < OUTPUT_BUCKETS {
            let mut colour = 0;
            while colour < 2 {
                let mut i = 0;
                while i < HIDDEN_SIZE {
                    output_weights[bucket][colour].vals[i] = self.output_weights[HIDDEN_SIZE * colour + i][bucket];
                    i += 1;
                }
                colour += 1;
            }
            bucket += 1;
        }

        Network {
            feature_weights: self.feature_weights,
            feature_bias: self.feature_bias,
            output_weights,
            output_bias: self.output_bias,
        }
    }
}

static NNUE: Network = unsafe {
    std::mem::transmute::<[u8; std::mem::size_of::<RawNetwork>()], RawNetwork>(*include_bytes!("../../../yukari_38cb8ff4.bin"))
        .into_network()
};
const _RAW_AND_PROCESSED_NETWORKS_ARE_SAME_SIZE: () = assert!(std::mem::size_of::<RawNetwork>() == std::mem::size_of::<Network>());

/// This is the quantised format that yukari uses.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Row-Major `OUTPUT_BUCKETS x (2 * HIDDEN_SIZE)` matrix.
    output_weights: [[Accumulator; 2]; OUTPUT_BUCKETS],
    /// Scalar output biases.
    output_bias: [i16; OUTPUT_BUCKETS],
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator, output_bucket: usize) -> i32 {
        // Initialise output with bias.
        let mut output = i32x64::splat(0);
        let min = i16x64::splat(0);
        let max = i16x64::splat(QA);

        // Side-To-Move Accumulator -> Output.
        for (input, weight) in us.vals.array_chunks::<64>().zip(self.output_weights[output_bucket][0].vals.array_chunks::<64>()) {
            // Squared Clipped `ReLU` - Activation Function.
            // Note that this takes the i16s in the accumulator to i32s.
            let input = i16x64::from_array(*input).simd_clamp(min, max);
            let weight = input * i16x64::from_array(*weight);
            output += input.cast::<i32>() * weight.cast::<i32>();
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (input, weight) in them.vals.array_chunks::<64>().zip(self.output_weights[output_bucket][1].vals.array_chunks::<64>()) {
            let input = i16x64::from_array(*input).simd_clamp(min, max);
            let weight = input * i16x64::from_array(*weight);
            output += input.cast::<i32>() * weight.cast::<i32>();
        }

        let mut output = (output.reduce_sum() / i32::from(QA)) + i32::from(self.output_bias[output_bucket]);

        // Apply eval scale.
        output *= SCALE;

        // Remove quantisation.
        output / (i32::from(QA) * i32::from(QB))
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub const fn new(net: &Network) -> Self {
        net.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d;
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eval {
    white: Accumulator,
    black: Accumulator,
}

impl Eval {
    pub fn new() -> Self {
        Self { white: Accumulator::new(&NNUE), black: Accumulator::new(&NNUE) }
    }

    pub fn get(&self, piece_count: usize, colour: Colour) -> i32 {
        let output_bucket = (piece_count - 2) / DIVISOR;
        if colour == Colour::White {
            NNUE.evaluate(&self.white, &self.black, output_bucket)
        } else {
            NNUE.evaluate(&self.black, &self.white, output_bucket)
        }
    }

    pub fn add_piece(&mut self, piece: Piece, square: Square, colour: Colour) {
        if colour == Colour::White {
            self.white.add_feature(64 * (piece as usize) + square.into_inner() as usize, &NNUE);
            self.black.add_feature(64 * (6 + piece as usize) + square.flip().into_inner() as usize, &NNUE);
        } else {
            self.black.add_feature(64 * (piece as usize) + square.flip().into_inner() as usize, &NNUE);
            self.white.add_feature(64 * (6 + piece as usize) + square.into_inner() as usize, &NNUE);
        }
    }

    pub fn remove_piece(&mut self, piece: Piece, square: Square, colour: Colour) {
        if colour == Colour::White {
            self.white.remove_feature(64 * (piece as usize) + square.into_inner() as usize, &NNUE);
            self.black.remove_feature(64 * (6 + piece as usize) + square.flip().into_inner() as usize, &NNUE);
        } else {
            self.black.remove_feature(64 * (piece as usize) + square.flip().into_inner() as usize, &NNUE);
            self.white.remove_feature(64 * (6 + piece as usize) + square.into_inner() as usize, &NNUE);
        }
    }

    pub fn move_piece(&mut self, piece: Piece, from_square: Square, to_square: Square, colour: Colour) {
        if colour == Colour::White {
            self.white.remove_feature(64 * (piece as usize) + from_square.into_inner() as usize, &NNUE);
            self.black.remove_feature(64 * (6 + piece as usize) + from_square.flip().into_inner() as usize, &NNUE);
            self.white.add_feature(64 * (piece as usize) + to_square.into_inner() as usize, &NNUE);
            self.black.add_feature(64 * (6 + piece as usize) + to_square.flip().into_inner() as usize, &NNUE);
        } else {
            self.black.remove_feature(64 * (piece as usize) + from_square.flip().into_inner() as usize, &NNUE);
            self.white.remove_feature(64 * (6 + piece as usize) + from_square.into_inner() as usize, &NNUE);
            self.black.add_feature(64 * (piece as usize) + to_square.flip().into_inner() as usize, &NNUE);
            self.white.add_feature(64 * (6 + piece as usize) + to_square.into_inner() as usize, &NNUE);
        }
    }
}
