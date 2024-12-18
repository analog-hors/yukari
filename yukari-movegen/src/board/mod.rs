use std::{
    convert::{TryFrom, TryInto},
    ffi::CString,
    fmt::{Display, Write},
};

use colored::Colorize;
use tinyvec::ArrayVec;
pub use zobrist::Zobrist;

use crate::{
    chessmove::{Move, MoveType},
    colour::Colour,
    piece::Piece,
    square::{File, Rank, Square, Square16x8},
};

mod bitlist;
mod data;
mod eval;
mod index;
mod piecelist;
mod piecemask;
mod pins;
mod zobrist;

use bitlist::Bitlist;
use data::BoardData;
pub use index::PieceIndex;

/// A chess position.
#[derive(Clone)]
pub struct Board {
    /// The chess board representation.
    data: data::BoardData,
    /// The side to move.
    side: Colour,
    /// Castling rights, if any.
    castle: (bool, bool, bool, bool),
    /// En-passant square, if any.
    ep: Option<Square>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Board {
    #[allow(clippy::missing_inline_in_public_items)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0_u8..64_u8 {
            let j = i ^ 56_u8;

            let square_colour = |s: &str| if (j & 1) ^ ((j >> 3) & 1) == 0 { s.on_green() } else { s.on_white() };

            if let (Some(piece), Some(colour)) = (
                self.data.piece_from_square(j.try_into().expect("square somehow out of bounds")),
                self.data.colour_from_square(j.try_into().expect("square somehow out of bounds")),
            ) {
                let c = match piece {
                    Piece::Pawn => '♙',
                    Piece::Knight => '♘',
                    Piece::Bishop => '♗',
                    Piece::Rook => '♖',
                    Piece::Queen => '♕',
                    Piece::King => '♔',
                };

                let c = if colour == Colour::White { c.to_string().bright_white() } else { c.to_string().black() };

                write!(f, "{}", square_colour(&format!("{c} ")))?;
            } else {
                write!(f, "{}", square_colour("  "))?;
            }

            if j & 7 == 7 {
                writeln!(f)?;
            }
        }
        if self.side == Colour::White {
            writeln!(f, "White to move.")?;
        } else {
            writeln!(f, "Black to move.")?;
        }
        if self.castle.0 {
            write!(f, "K")?;
        }
        if self.castle.1 {
            write!(f, "Q")?;
        }
        if self.castle.2 {
            write!(f, "k")?;
        }
        if self.castle.3 {
            write!(f, "q")?;
        }
        writeln!(f)?;
        if let Some(ep) = self.ep {
            writeln!(f, "{ep}")?;
        } else {
            writeln!(f, "-")?;
        }

        writeln!(f, "{:016x}", self.hash())?;

        Ok(())
    }
}

impl Board {
    /// Create a new empty board.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self { side: Colour::White, castle: (false, false, false, false), ep: None, data: BoardData::new() }
    }

    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    /// Check if this board is illegal by seeing if the enemy king is attacked by friendly pieces.
    /// If it is, it implies the move the enemy made left them in check, which is illegal.
    #[must_use]
    #[inline]
    pub fn illegal(&self) -> bool {
        // A valid chessboard has a white king and a black king.
        if (self.kings() & Bitlist::white()).empty() {
            return true;
        }
        if (self.kings() & Bitlist::black()).empty() {
            return true;
        }
        // The opponent's king should not be in check.
        if !self.data.attacks_to(self.data.king_square(!self.side), self.side).empty() {
            return true;
        }
        false
    }

    /// Parse a position in Forsyth-Edwards Notation into a board.
    ///
    /// # Panics
    /// Panics if `fen` is not ASCII.
    #[must_use]
    pub fn from_fen(fen: &str) -> Option<Self> {
        let fen = CString::new(fen).expect("FEN is not ASCII");
        let fen = fen.as_bytes();
        Self::from_fen_bytes(fen)
    }

    /// Parse a position in Forsyth-Edwards Notation into a board.
    ///
    /// # Panics
    /// Panics when invalid FEN is input.
    #[must_use]
    pub fn from_fen_bytes(fen: &[u8]) -> Option<Self> {
        let mut b = Self::new();

        let mut idx = 0_usize;
        let mut c = fen[idx];

        for rank in (0..=7).rev() {
            let mut file = 0;
            while file <= 7 {
                if (b'1'..=b'8').contains(&c) {
                    let length = c - b'0';
                    let mut i = 0;
                    while i < length {
                        file += 1;
                        i += 1;
                    }
                } else {
                    let piece = match c.to_ascii_lowercase() {
                        b'k' => Piece::King,
                        b'q' => Piece::Queen,
                        b'r' => Piece::Rook,
                        b'b' => Piece::Bishop,
                        b'n' => Piece::Knight,
                        b'p' => Piece::Pawn,
                        _ => return None,
                    };

                    let colour = if c.is_ascii_uppercase() { Colour::White } else { Colour::Black };

                    let square = Square::from_rank_file(rank.try_into().unwrap(), file.try_into().unwrap());

                    b.data.add_piece(piece, colour, square, false);

                    file += 1;
                }
                idx += 1;
                c = fen[idx];
            }
            if rank > 0 {
                idx += 1;
                c = fen[idx];
            }
        }
        idx += 1;
        c = fen[idx];
        b.side = match c {
            b'w' => Colour::White,
            b'b' => Colour::Black,
            _ => return None,
        };
        idx += 2;
        c = fen[idx];
        b.castle = (false, false, false, false);
        if c == b'-' {
            idx += 1;
        } else {
            if c == b'K' {
                b.castle.0 = true;
                b.data.add_castling(0);
                idx += 1;
                c = fen[idx];
            }
            if c == b'Q' {
                b.castle.1 = true;
                b.data.add_castling(1);
                idx += 1;
                c = fen[idx];
            }
            if c == b'k' {
                b.castle.2 = true;
                b.data.add_castling(2);
                idx += 1;
                c = fen[idx];
            }
            if c == b'q' {
                b.castle.3 = true;
                b.data.add_castling(3);
                idx += 1;
            }
        }
        idx += 1;
        c = fen[idx];
        if c == b'-' {
            b.ep = None;
        } else {
            let file = File::try_from(c - b'a').unwrap();
            idx += 1;
            c = fen[idx];
            let rank = Rank::try_from(c - b'1').unwrap();
            b.ep = Some(Square::from_rank_file(rank, file));
        }

        b.data.rebuild_attacks();

        if b.illegal() {
            return None;
        }

        Some(b)
    }

    fn set_ep(&mut self, ep: Option<Square>) {
        self.data.set_ep(self.ep, ep);
        self.ep = ep;
    }

    /// Make a move on the board.
    ///
    /// # Panics
    /// Panics when Lofty hasn't implemented necessary code.
    #[inline]
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn make(&self, m: Move) -> Self {
        let mut b = self.clone();
        match m.kind {
            MoveType::Promotion | MoveType::Normal | MoveType::DoublePush => {}
            MoveType::Capture | MoveType::CapturePromotion => {
                let piece_index = b.data.piece_index(m.dest).unwrap_or_else(|| panic!("move {m} attempts to capture an empty square"));
                b.data.remove_piece(piece_index, true);
            }
            MoveType::Castle => {
                let (rook_from, rook_to) = if m.dest > m.from {
                    (m.dest.east().unwrap(), m.dest.west().unwrap())
                } else {
                    (m.dest.west().unwrap().west().unwrap(), m.dest.east().unwrap())
                };
                b.data.move_piece(rook_from, rook_to);
            }
            MoveType::EnPassant => {
                let target_square = b.ep.unwrap().relative_south(b.side).unwrap();
                let target_piece = b.data.piece_index(target_square).unwrap();
                b.data.remove_piece(target_piece, true);
            }
        }

        b.data.move_piece(m.from, m.dest);

        if matches!(m.kind, MoveType::Promotion | MoveType::CapturePromotion) {
            let piece_index = b.data.piece_index(m.dest).unwrap();
            b.data.remove_piece(piece_index, true);
            b.data.add_piece(m.prom.unwrap(), b.side, m.dest, true);
        }

        if matches!(m.kind, MoveType::DoublePush) {
            b.set_ep(m.from.relative_north(b.side));
        } else {
            b.set_ep(None);
        }

        let a1 = Square::from_rank_file(Rank::One, File::A);
        let a8 = Square::from_rank_file(Rank::Eight, File::A);
        let e1 = Square::from_rank_file(Rank::One, File::E);
        let e8 = Square::from_rank_file(Rank::Eight, File::E);
        let h1 = Square::from_rank_file(Rank::One, File::H);
        let h8 = Square::from_rank_file(Rank::Eight, File::H);

        if m.from == e1 {
            if b.castle.0 {
                b.castle.0 = false;
                b.data.remove_castling(0);
            }
            if b.castle.1 {
                b.castle.1 = false;
                b.data.remove_castling(1);
            }
        }

        if m.from == e8 {
            if b.castle.2 {
                b.castle.2 = false;
                b.data.remove_castling(2);
            }
            if b.castle.3 {
                b.castle.3 = false;
                b.data.remove_castling(3);
            }
        }

        if (m.from == h1 || m.dest == h1) && b.castle.0 {
            b.castle.0 = false;
            b.data.remove_castling(0);
        }

        if (m.from == a1 || m.dest == a1) && b.castle.1 {
            b.castle.1 = false;
            b.data.remove_castling(1);
        }

        if (m.from == h8 || m.dest == h8) && b.castle.2 {
            b.castle.2 = false;
            b.data.remove_castling(2);
        }

        if (m.from == a8 || m.dest == a8) && b.castle.3 {
            b.castle.3 = false;
            b.data.remove_castling(3);
        }

        b.side = !b.side;
        b.data.toggle_side();
        b
    }

    fn try_push_move(
        &self, v: &mut ArrayVec<[Move; 256]>, from: Square, dest: Square, kind: MoveType, promotion_piece: Option<Piece>,
        pininfo: &pins::PinInfo,
    ) {
        if let Some(dir) = pininfo.pins[self.data.piece_index(from).unwrap().into_inner() as usize] {
            let Some(move_dir) = from.direction(dest) else {
                // Pinned knight can't move.
                return;
            };
            // Pinned slider can only move along pin ray.
            if dir != move_dir && dir != move_dir.opposite() {
                return;
            }
        }
        v.push(Move::new(from, dest, kind, promotion_piece));
    }

    /// Generate en-passant pawn moves.
    fn generate_pawn_enpassant(&self, v: &mut ArrayVec<[Move; 256]>, pininfo: &pins::PinInfo) {
        let Some(ep) = self.ep else {
            return;
        };
        for capturer in self.data.attacks_to(ep, self.side).and(self.data.pawns()).and(!pininfo.enpassant_pinned) {
            let from = self.data.square_of_piece(capturer);
            self.try_push_move(v, from, ep, MoveType::EnPassant, None, pininfo);
        }
    }

    /// Generate pawn-specific quiet moves.
    fn generate_pawn_quiet(&self, v: &mut ArrayVec<[Move; 256]>, from: Square, pininfo: &pins::PinInfo) {
        let promotion_pieces = [Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop];
        let north = from.relative_north(self.side);
        let Some(dest) = north else {
            return;
        };
        // Pawn single pushes.
        if self.data.has_piece(dest) {
            return;
        }
        if Rank::from(dest).is_relative_eighth(self.side) {
            for piece in &promotion_pieces {
                self.try_push_move(v, from, dest, MoveType::Promotion, Some(*piece), pininfo);
            }
        } else {
            self.try_push_move(v, from, dest, MoveType::Normal, None, pininfo);
        }

        // Pawn double pushes.
        let Some(dest) = dest.relative_north(self.side) else {
            return;
        };
        if Rank::from(dest).is_relative_fourth(self.side) && !self.data.has_piece(dest) {
            self.try_push_move(v, from, dest, MoveType::DoublePush, None, pininfo);
        }
    }

    /// Generate moves when in check by a single piece.
    #[allow(clippy::too_many_lines)]
    fn generate_single_check(&self, v: &mut ArrayVec<[Move; 256]>) {
        let king_square = self.data.king_square(self.side);
        let king_square_16x8 = Square16x8::from_square(king_square);
        let attacker_bit = self.data.attacks_to(king_square, !self.side);
        let attacker_index = unsafe { attacker_bit.peek_nonzero() };
        let attacker_piece = self.data.piece_from_bit(attacker_index);
        let attacker_square = self.data.square_of_piece(attacker_index);
        let attacker_direction = attacker_square.direction(king_square);

        let pininfo = pins::PinInfo::discover(self);

        let add_pawn_block = |v: &mut ArrayVec<[Move; 256]>, from, dest, kind| {
            let promotion_pieces = [Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop];
            let Some(colour) = self.data.colour_from_square(from) else { return };
            if colour != self.side {
                return;
            }
            if !Rank::from(dest).is_relative_eighth(self.side) {
                self.try_push_move(v, from, dest, kind, None, &pininfo);
                return;
            }
            for piece in &promotion_pieces {
                self.try_push_move(v, from, dest, MoveType::Promotion, Some(*piece), &pininfo);
            }
        };

        let add_pawn_blocks = |v: &mut ArrayVec<[Move; 256]>, dest: Square| {
            let Some(from) = dest.relative_south(self.side) else { return };
            match self.data.piece_from_square(from) {
                Some(Piece::Pawn) => add_pawn_block(v, from, dest, MoveType::Normal),
                Some(_) => {}
                None => {
                    if Rank::from(dest).is_relative_fourth(self.side) {
                        let Some(from) = from.relative_south(self.side) else { return };
                        if self.data.piece_from_square(from) == Some(Piece::Pawn) {
                            add_pawn_block(v, from, dest, MoveType::DoublePush);
                        }
                    }
                }
            }
        };

        // Can we capture the attacker?
        for capturer in self.data.attacks_to(attacker_square, self.side) {
            let promotion_pieces = [Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop];
            let from = self.data.square_of_piece(capturer);
            if self.data.piece_from_bit(capturer) == Piece::King && !self.data.attacks_to(attacker_square, !self.side).empty() {
                continue;
            }
            if self.data.piece_from_bit(capturer) != Piece::Pawn || !Rank::from(attacker_square).is_relative_eighth(self.side) {
                self.try_push_move(v, from, attacker_square, MoveType::Capture, None, &pininfo);
                continue;
            }
            for piece in &promotion_pieces {
                self.try_push_move(v, from, attacker_square, MoveType::CapturePromotion, Some(*piece), &pininfo);
            }
        }

        // en-passant
        (|| {
            let Some(ep) = self.ep else { return };
            let Some(ep_south) = ep.relative_south(self.side) else { return };
            if ep_south != attacker_square || attacker_piece != Piece::Pawn {
                return;
            }
            for capturer in self.data.attacks_to(ep, self.side) & self.data.pawns() & !pininfo.enpassant_pinned {
                self.try_push_move(v, self.data.square_of_piece(capturer), ep, MoveType::EnPassant, None, &pininfo);
            }
        })();

        // Can we block the check?
        if let Piece::Bishop | Piece::Rook | Piece::Queen = attacker_piece {
            let direction = king_square.direction(attacker_square).unwrap();
            for dest in king_square_16x8.ray_attacks(direction) {
                if dest == attacker_square {
                    break;
                }

                // Piece moves.
                for attacker in self.data.attacks_to(dest, self.side).and(!self.data.pawns()).and(!self.data.kings()) {
                    self.try_push_move(v, self.data.square_of_piece(attacker), dest, MoveType::Normal, None, &pininfo);
                }

                // Pawn moves.
                add_pawn_blocks(v, dest);
            }
        }

        // Can we move the king?
        for square in king_square.king_attacks() {
            let kind = if self.data.has_piece(square) {
                if square == attacker_square || self.data.colour_from_square(square) == Some(self.side) {
                    // Own-piece captures are illegal, captures of the attacker are handled elsewhere.
                    continue;
                }
                MoveType::Capture
            } else {
                MoveType::Normal
            };

            if !self.data.attacks_to(square, !self.side).empty() {
                // Moving into check is illegal.
                continue;
            }
            if let Some(attacker_direction) = attacker_direction {
                // Slider attacks x-ray through the king to attack that square.
                if let Some(xray_square) = king_square.travel(attacker_direction) {
                    if matches!(attacker_piece, Piece::Bishop | Piece::Rook | Piece::Queen) && xray_square == square {
                        continue;
                    }
                }
            }

            v.push(Move::new(king_square, square, kind, None));
        }
    }

    fn generate_double_check(&self, v: &mut ArrayVec<[Move; 256]>) {
        let king_square = self.data.king_square(self.side);
        let mut attacker_bits = self.data.attacks_to(king_square, !self.side);
        let attacker1_index = attacker_bits.pop().unwrap();
        let attacker1_piece = self.data.piece_from_bit(attacker1_index);
        let attacker1_square = self.data.square_of_piece(attacker1_index);
        let attacker1_direction = attacker1_square.direction(king_square);
        let attacker2_index = attacker_bits.pop().unwrap();
        let attacker2_piece = self.data.piece_from_bit(attacker2_index);
        let attacker2_square = self.data.square_of_piece(attacker2_index);
        let attacker2_direction = attacker2_square.direction(king_square);

        // Can we move the king?
        for square in king_square.king_attacks() {
            let kind = if self.data.has_piece(square) {
                if self.data.colour_from_square(square) == Some(self.side) {
                    // Own-piece captures are illegal.
                    continue;
                }
                MoveType::Capture
            } else {
                MoveType::Normal
            };

            if !self.data.attacks_to(square, !self.side).empty() {
                // Moving into check is illegal.
                continue;
            }

            // Slider attacks x-ray through the king to attack that square.
            if let Some(attacker1_direction) = attacker1_direction {
                if let Some(xray_square) = king_square.travel(attacker1_direction) {
                    if matches!(attacker1_piece, Piece::Bishop | Piece::Rook | Piece::Queen) && xray_square == square {
                        continue;
                    }
                }
            }

            if let Some(attacker2_direction) = attacker2_direction {
                if let Some(xray_square) = king_square.travel(attacker2_direction) {
                    if matches!(attacker2_piece, Piece::Bishop | Piece::Rook | Piece::Queen) && xray_square == square {
                        continue;
                    }
                }
            }

            v.push(Move::new(king_square, square, kind, None));
        }
    }

    pub fn generate_captures(&self, v: &mut ArrayVec<[Move; 256]>) {
        let pininfo = pins::PinInfo::discover(self);

        let mut find_attackers = |dest: Square| {
            let promotion_pieces = [Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop];
            let attacks = self.data.attacks_to(dest, self.side);
            for capturer in attacks & self.data.pawns() {
                let from = self.data.square_of_piece(capturer);
                if Rank::from(dest).is_relative_eighth(self.side) {
                    for piece in &promotion_pieces {
                        self.try_push_move(v, from, dest, MoveType::CapturePromotion, Some(*piece), &pininfo);
                    }
                } else {
                    self.try_push_move(v, from, dest, MoveType::Capture, None, &pininfo);
                }
            }
            let capturers = (attacks & self.data.knights())
                .into_iter()
                .chain(attacks & self.data.bishops())
                .chain(attacks & self.data.rooks())
                .chain(attacks & self.data.queens());

            for capturer in capturers {
                let from = self.data.square_of_piece(capturer);
                self.try_push_move(v, from, dest, MoveType::Capture, None, &pininfo);
            }
            for capturer in attacks & self.data.kings() {
                let from = self.data.square_of_piece(capturer);
                if !self.data.attacks_to(dest, !self.side).empty() {
                    // Moving into check is illegal.
                    continue;
                }
                self.try_push_move(v, from, dest, MoveType::Capture, None, &pininfo);
            }
        };

        let victims = (self.data.pieces_of_colour(!self.side) & self.data.queens())
            .into_iter()
            .chain(self.data.pieces_of_colour(!self.side) & self.data.rooks())
            .chain(self.data.pieces_of_colour(!self.side) & self.data.bishops())
            .chain(self.data.pieces_of_colour(!self.side) & self.data.knights())
            .chain(self.data.pieces_of_colour(!self.side) & self.data.pawns());

        for victim in victims {
            find_attackers(self.square_of_piece(victim));
        }

        self.generate_pawn_enpassant(v, &pininfo);
    }

    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn generate_captures_incremental<F: FnMut(Move) -> bool>(&self, mut f: F) {
        let king_square = self.data.king_square(self.side);
        let checks = self.data.attacks_to(king_square, !self.side);

        // special case: being in check.
        if checks.count_ones() != 0 {
            let mut v = ArrayVec::new();
            v.set_len(0);
            if checks.count_ones() == 1 {
                self.generate_single_check(&mut v);
            } else if checks.count_ones() == 2 {
                self.generate_double_check(&mut v);
            }

            for m in v {
                if m.is_capture() && !f(m) {
                    break;
                }
            }
            return;
        }

        let pininfo = pins::PinInfo::discover(self);

        let mut minor_mask = Bitlist::new();
        let mut rook_mask = Bitlist::new();
        let mut queen_mask = Bitlist::new();

        let mut try_move = |from: Square, dest: Square, kind: MoveType, promotion_piece: Option<Piece>, pininfo: &pins::PinInfo| {
            if let Some(dir) = pininfo.pins[self.data.piece_index(from).unwrap().into_inner() as usize] {
                if let Some(move_dir) = from.direction(dest) {
                    // Pinned slider can only move along pin ray.
                    if dir == move_dir || dir == move_dir.opposite() {
                        return f(Move::new(from, dest, kind, promotion_piece));
                    }
                }
                // Pinned knight can't move.
                return true;
            }
            f(Move::new(from, dest, kind, promotion_piece))
        };

        let mut find_attackers =
            |dest: Square, victim_type: Piece, minor_mask: Bitlist, rook_mask: Bitlist, queen_mask: Bitlist| -> bool {
                let promotion_pieces = [Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop];
                let attacks = self.data.attacks_to(dest, self.side);
                for capturer in attacks & self.data.pawns() {
                    let from = self.data.square_of_piece(capturer);
                    if Rank::from(dest).is_relative_eighth(self.side) {
                        for piece in &promotion_pieces {
                            if !try_move(from, dest, MoveType::CapturePromotion, Some(*piece), &pininfo) {
                                return false;
                            }
                        }
                    } else if !try_move(from, dest, MoveType::Capture, None, &pininfo) {
                        return false;
                    }
                }
                for capturer in attacks & (self.data.knights() | self.data.bishops()) {
                    let from = self.data.square_of_piece(capturer);
                    if victim_type < Piece::Bishop && !(self.data.attacks_to(dest, !self.side) & minor_mask).empty() {
                        // This is a bad capture.
                        continue;
                    }
                    if !try_move(from, dest, MoveType::Capture, None, &pininfo) {
                        return false;
                    }
                }
                for capturer in attacks & self.data.rooks() {
                    let from = self.data.square_of_piece(capturer);
                    if victim_type < Piece::Rook && !(self.data.attacks_to(dest, !self.side) & rook_mask).empty() {
                        // This is a bad capture.
                        continue;
                    }
                    if !try_move(from, dest, MoveType::Capture, None, &pininfo) {
                        return false;
                    }
                }
                for capturer in attacks & self.data.queens() {
                    let from = self.data.square_of_piece(capturer);
                    if victim_type < Piece::Queen && !(self.data.attacks_to(dest, !self.side) & queen_mask).empty() {
                        // This is a bad capture.
                        continue;
                    }
                    if !try_move(from, dest, MoveType::Capture, None, &pininfo) {
                        return false;
                    }
                }
                for capturer in attacks & self.data.kings() {
                    let from = self.data.square_of_piece(capturer);
                    if !self.data.attacks_to(dest, !self.side).empty() {
                        // Moving into check is illegal.
                        continue;
                    }
                    if !try_move(from, dest, MoveType::Capture, None, &pininfo) {
                        return false;
                    }
                }
                true
            };

        minor_mask |= self.data.pieces_of_colour(!self.side) & self.data.pawns();
        rook_mask |= self.data.pieces_of_colour(!self.side) & self.data.pawns();
        queen_mask |= self.data.pieces_of_colour(!self.side) & self.data.pawns();

        for victim in self.data.pieces_of_colour(!self.side) & self.data.queens() {
            if !find_attackers(self.square_of_piece(victim), Piece::Queen, minor_mask, rook_mask, queen_mask) {
                return;
            }
        }

        queen_mask |= self.data.pieces_of_colour(!self.side) & (self.data.knights() | self.data.bishops());

        for victim in self.data.pieces_of_colour(!self.side) & self.data.rooks() {
            if !find_attackers(self.square_of_piece(victim), Piece::Rook, minor_mask, rook_mask, queen_mask) {
                return;
            }
        }

        queen_mask |= self.data.pieces_of_colour(!self.side) & self.data.rooks();

        for victim in self.data.pieces_of_colour(!self.side) & (self.data.knights() | self.data.bishops()) {
            if !find_attackers(self.square_of_piece(victim), Piece::Bishop, minor_mask, rook_mask, queen_mask) {
                return;
            }
        }

        rook_mask |= self.data.pieces_of_colour(!self.side) & (self.data.knights() | self.data.bishops());

        for victim in self.data.pieces_of_colour(!self.side) & self.data.pawns() {
            if !find_attackers(self.square_of_piece(victim), Piece::Pawn, minor_mask, rook_mask, queen_mask) {
                return;
            }
        }
    }

    /// Generate a vector of moves on the board.
    ///
    /// # Panics
    /// Panics when Lofty writes shitty code.
    #[allow(clippy::missing_inline_in_public_items)]
    pub fn generate(&self, v: &mut ArrayVec<[Move; 256]>) {
        // Unless something has gone very badly wrong we have to have a king.
        let king_square = self.data.king_square(self.side);
        let checks = self.data.attacks_to(king_square, !self.side);

        if checks.count_ones() == 1 {
            return self.generate_single_check(v);
        }
        if checks.count_ones() == 2 {
            return self.generate_double_check(v);
        }

        let pininfo = pins::PinInfo::discover(self);
        self.generate_captures(v);

        // Pawns.
        for pawn in self.data.pawns().and(Bitlist::mask_from_colour(self.side)) {
            let from = self.data.square_of_piece(pawn);
            self.generate_pawn_quiet(v, from, &pininfo);
        }

        // General quiet move loop; pawns and kings handled separately.
        for dest in 0_u8..64 {
            // Squares will always be in range, so this will never panic.
            let dest = unsafe { Square::from_u8_unchecked(dest) };

            // Ignore captures.
            if self.data.has_piece(dest) {
                continue;
            }

            // For every piece that attacks this square, find its location and add it to the move list.
            for attacker in self.data.attacks_to(dest, self.side).and(!self.data.pawns()) {
                // It's illegal for kings to move to attacked squares; prune those out.
                if self.data.piece_from_bit(attacker) == Piece::King && !self.data.attacks_to(dest, !self.side).empty() {
                    continue;
                }

                let from = self.data.square_of_piece(attacker);
                self.try_push_move(v, from, dest, MoveType::Normal, None, &pininfo);
            }
        }

        // Kingside castling.
        if (self.side == Colour::White && self.castle.0) || (self.side == Colour::Black && self.castle.2) {
            let east1 = king_square.east().unwrap();
            let east2 = east1.east().unwrap();
            if self.data.attacks_to(king_square, !self.side).empty()
                && !self.data.has_piece(east1)
                && self.data.attacks_to(east1, !self.side).empty()
                && !self.data.has_piece(east2)
                && self.data.attacks_to(east2, !self.side).empty()
            {
                self.try_push_move(v, king_square, east2, MoveType::Castle, None, &pininfo);
            }
        }

        // Queenside castling.
        if (self.side == Colour::White && self.castle.1) || (self.side == Colour::Black && self.castle.3) {
            let west1 = king_square.west().unwrap();
            let west2 = west1.west().unwrap();
            let west3 = west2.west().unwrap();
            if self.data.attacks_to(king_square, !self.side).empty()
                && !self.data.has_piece(west1)
                && self.data.attacks_to(west1, !self.side).empty()
                && !self.data.has_piece(west2)
                && self.data.attacks_to(west2, !self.side).empty()
                && !self.data.has_piece(west3)
            {
                self.try_push_move(v, king_square, west2, MoveType::Castle, None, &pininfo);
            }
        }
    }

    #[must_use]
    pub const fn kings(&self) -> Bitlist {
        self.data.kings()
    }

    /// Return a bitlist of all pieces.
    #[must_use]
    pub const fn pieces(&self) -> Bitlist {
        self.data.pieces()
    }

    /// Given a piece index, return its piece type.
    #[must_use]
    pub const fn piece_from_bit(&self, bit: PieceIndex) -> Piece {
        self.data.piece_from_bit(bit)
    }

    #[must_use]
    pub fn piece_from_square(&self, square: Square) -> Option<Piece> {
        self.data.piece_from_square(square)
    }

    #[must_use]
    pub fn square_of_piece(&self, bit: PieceIndex) -> Square {
        self.data.square_of_piece(bit)
    }

    #[must_use]
    pub const fn ep(&self) -> Option<Square> {
        self.ep
    }

    #[must_use]
    pub const fn side(&self) -> Colour {
        self.side
    }

    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.data.hash()
    }

    #[must_use]
    pub fn hash_pawns(&self) -> u64 {
        self.data.hash_pawns()
    }

    #[must_use]
    pub fn eval(&self, colour: Colour) -> i32 {
        self.data.eval(colour)
    }

    #[must_use]
    pub fn in_check(&self) -> bool {
        !self.data.attacks_to(self.data.king_square(self.side), !self.side).empty()
    }

    #[must_use]
    pub fn make_null(&self) -> Self {
        let mut board = self.clone();
        board.side = !board.side;
        board.set_ep(None);
        board.data.toggle_side();
        board
    }

    /// # Panics
    /// Panics when a nonsense move is encountered.
    #[must_use]
    pub fn to_san(&self, m: Move) -> String {
        let mut san = String::new();

        // Special case: castling
        if m.kind == MoveType::Castle {
            if m.dest > m.from {
                write!(san, "O-O").unwrap();
            } else {
                write!(san, "O-O-O").unwrap();
            }
            return san;
        }

        // Moving piece
        let piece = self.piece_from_square(m.from).unwrap_or_else(|| panic!("{m} has no origin piece on board\n{self}"));
        let piece_char = match piece {
            Piece::Pawn => "",
            Piece::Knight => "♘ ",
            Piece::Bishop => "♗ ",
            Piece::Rook => "♖ ",
            Piece::Queen => "♕ ",
            Piece::King => "♔ ",
        };
        write!(san, "{piece_char}").unwrap();

        // Disambiguation
        let mut moves = ArrayVec::new();
        self.generate(&mut moves);

        let mut ambiguities = Vec::new();
        for mv in moves {
            if mv.dest == m.dest && self.piece_from_square(mv.from) == self.piece_from_square(m.from) && mv.from != m.from {
                ambiguities.push(mv);
            }
        }

        let must_disambiguate = !ambiguities.is_empty();
        let mut piece_on_same_rank = false;
        let mut piece_on_same_file = false;

        let rank = Rank::from(m.from);
        let file = File::from(m.from);
        for ambiguity in ambiguities {
            let attacker_rank = Rank::from(ambiguity.from);
            let attacker_file = File::from(ambiguity.from);
            piece_on_same_rank |= attacker_rank == rank;
            piece_on_same_file |= attacker_file == file;
        }

        if piece != Piece::Pawn && must_disambiguate {
            if piece_on_same_rank || !piece_on_same_file {
                write!(san, "{file}").unwrap();
            }
            if piece_on_same_file {
                write!(san, "{rank}").unwrap();
            }
        }

        // Capture?
        if m.is_capture() {
            // Pawns always have their file.
            if piece == Piece::Pawn {
                write!(san, "{file}").unwrap();
            }
            write!(san, "x").unwrap();
        }

        let rank = Rank::from(m.dest);
        let file = File::from(m.dest);
        write!(san, "{file}{rank}").unwrap();

        // Promotion?
        if matches!(m.kind, MoveType::Promotion | MoveType::CapturePromotion) {
            let piece_char = match m.prom.unwrap() {
                Piece::Pawn => '♙',
                Piece::Knight => '♘',
                Piece::Bishop => '♗',
                Piece::Rook => '♖',
                Piece::Queen => '♕',
                Piece::King => '♔',
            };
            write!(san, "={piece_char}").unwrap();
        }

        // Check?
        let child = self.make(m);
        if child.in_check() {
            // Checkmate?
            let mut moves = ArrayVec::new();
            child.generate(&mut moves);
            if moves.is_empty() {
                write!(san, "#").unwrap();
            } else {
                write!(san, "+").unwrap();
            }
        }

        san
    }

    #[must_use]
    pub fn pv_to_san(&self, pv: &[Move]) -> String {
        let mut s = String::new();
        let mut board = self.clone();
        for &m in pv {
            let san = board.to_san(m);
            if board.side() == Colour::White {
                write!(s, "{} ", san.bright_white()).unwrap();
            } else {
                write!(s, "{} ", san.bright_black()).unwrap();
            }
            board = board.make(m);
        }
        s
    }
}

/* impl Drop for Board {
    fn drop(&mut self) {
        if ::std::thread::panicking() {
            println!("{}", self);
        }
    }
} */
