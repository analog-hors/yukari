use std::simd::{cmp::SimdOrd, i16x64, i32x64, num::SimdInt};

use crate::{Colour, Piece, Square};

const HIDDEN_SIZE: usize = 128;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

static NNUE: Network = unsafe { std::mem::transmute(*include_bytes!("../../../yukari_86582212.bin")) };

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`
    /// matrix, we use it like this to make the
    /// code nicer in `Network::evaluate`.
    output_weights: [Accumulator; 2],
    /// Scalar output bias.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        // Initialise output with bias.
        let mut output = i32x64::splat(0); 
        let min = i16x64::splat(0);
        let max = i16x64::splat(QA);

        // Side-To-Move Accumulator -> Output.
        for (input, weight) in us.vals.array_chunks::<64>().zip(self.output_weights[0].vals.array_chunks::<64>()) {
            // Squared Clipped `ReLU` - Activation Function.
            // Note that this takes the i16s in the accumulator to i32s.
            let input = i16x64::from_array(*input).simd_clamp(min, max);
            let weight = input * i16x64::from_array(*weight);
            output += input.cast::<i32>() * weight.cast::<i32>();
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (input, weight) in them.vals.array_chunks::<64>().zip(self.output_weights[1].vals.array_chunks::<64>()) {
            let input = i16x64::from_array(*input).simd_clamp(min, max);
            let weight = input * i16x64::from_array(*weight);
            output += input.cast::<i32>() * weight.cast::<i32>();
        }

        let mut output = (output.reduce_sum() / i32::from(QA)) + i32::from(self.output_bias);

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

    pub fn get(&self, colour: Colour) -> i32 {
        if colour == Colour::White {
            NNUE.evaluate(&self.white, &self.black)
        } else {
            NNUE.evaluate(&self.black, &self.white)
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
