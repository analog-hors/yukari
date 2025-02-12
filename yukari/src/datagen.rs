use std::{io::Write, str::FromStr, time::{Duration, Instant}};

use rand::seq::IteratorRandom;
use tinyvec::ArrayVec;
use yukari_movegen::{Board, Colour, File, Move, Piece, Rank, Square};

use crate::{output, search};

#[derive(Clone, Copy)]
#[repr(u8)]
enum MarlinWdl {
    BlackWin = 0,
    Draw = 1,
    WhiteWin = 2,
}

#[repr(C)]
struct MarlinFormat {
    occupancy: u64,
    pieces: [u8; 16], // [u4; 32]
    stm_ep_square: u8,
    halfmove_clock: u8,
    fullmove_number: u16,
    eval: i16,
    wdl: MarlinWdl,
    _extra: u8,
}

impl From<Board> for MarlinFormat {
    fn from(board: Board) -> Self {
        let a1 = Square::from_rank_file(Rank::One, File::A);
        let a8 = Square::from_rank_file(Rank::Eight, File::A);
        let h1 = Square::from_rank_file(Rank::One, File::H);
        let h8 = Square::from_rank_file(Rank::Eight, File::H);

        let mut this = Self {
            occupancy: 0,
            pieces: [0; 16],
            stm_ep_square: 0,
            halfmove_clock: 0,
            fullmove_number: 0,
            eval: 0,
            wdl: MarlinWdl::Draw,
            _extra: 0
        };
        for sq in 0..64 {
            let square = unsafe { Square::from_u8_unchecked(sq) };
            let Some(piece) = board.data().piece_index(square) else { continue };

            let mut piece = if board.piece_from_bit(piece) == Piece::Rook {
                if (board.castle().0 && square == h1) || (board.castle().1 && square == a1) || (board.castle().2 && square == h8) || (board.castle().3 && square == a8) {
                    // "unmoved rook" to represent castling rights.
                    (6_u8) | ((piece.colour() as u8) << 3)
                } else {
                    (board.piece_from_bit(piece) as u8) | ((piece.colour() as u8) << 3)
                }
            } else {
                (board.piece_from_bit(piece) as u8) | ((piece.colour() as u8) << 3)
            };
            let piece_count = this.occupancy.count_ones() as usize;
            if piece_count % 2 == 1 {
                piece <<= 4;
            }
            this.pieces[piece_count / 2] |= piece;
            this.occupancy |= 1_u64 << sq;
        }

        this.stm_ep_square = ((board.side() as u8) << 7) | (board.ep().map_or(64, |s| s.into_inner()));

        this
    }
}

impl MarlinFormat {
    pub fn write(&self, f: &mut impl Write) {
        f.write_all(&self.occupancy.to_le_bytes()).unwrap();
        f.write_all(&self.pieces).unwrap();
        f.write_all(&[self.stm_ep_square, self.halfmove_clock]).unwrap();
        f.write_all(&self.fullmove_number.to_le_bytes()).unwrap();
        f.write_all(&self.eval.to_le_bytes()).unwrap();
        f.write_all(&[self.wdl as u8, self._extra]).unwrap();
    }
}

#[repr(transparent)]
struct ViriMove(u16);

impl From<Move> for ViriMove {
    fn from(m: Move) -> Self {
        let (from, dest) = if let yukari_movegen::MoveType::Castle = m.kind {
            // convert from yukari's "king two squares" castling to viridithas' "king takes rook" castling.
            let rank = Rank::from(m.dest);
            let file = File::from(m.dest);
            let from = m.from.into_inner() as u16;
            let dest = match (rank, file) {
                (Rank::One, File::G) => Square::from_rank_file(Rank::One, File::H).into_inner() as u16,
                (Rank::One, File::C) => Square::from_rank_file(Rank::One, File::A).into_inner() as u16,
                (Rank::Eight, File::G) => Square::from_rank_file(Rank::Eight, File::H).into_inner() as u16,
                (Rank::Eight, File::C) => Square::from_rank_file(Rank::Eight, File::A).into_inner() as u16,
                _ => panic!("unrecognised castling to-square"),
            };
            (from, dest)
        } else {
            (m.from.into_inner() as u16, m.dest.into_inner() as u16)
        };
        let prom = match m.prom {
            None => 0,
            Some(Piece::Knight) => 0,
            Some(Piece::Bishop) => 1,
            Some(Piece::Rook) => 2,
            Some(Piece::Queen) => 3,
            Some(_) => unreachable!("invalid promotion piece"),
        };
        let flags = match m.kind {
            yukari_movegen::MoveType::Normal => 0,
            yukari_movegen::MoveType::Capture => 0,
            yukari_movegen::MoveType::Castle => 2,
            yukari_movegen::MoveType::DoublePush => 0,
            yukari_movegen::MoveType::EnPassant => 1,
            yukari_movegen::MoveType::Promotion => 3,
            yukari_movegen::MoveType::CapturePromotion => 3,
        };

        Self(from | (dest << 6) | (prom << 12) | (flags << 14))
    }
}

struct ViriFormat {
    position: MarlinFormat,
    moves: Vec<(ViriMove, i16)>,
}

impl ViriFormat {
    pub fn new(board: Board) -> Self {
        Self {
            position: MarlinFormat::from(board),
            moves: Vec::new(),
        }
    }

    pub fn push(&mut self, m: Move, score: i16) {
        self.moves.push((ViriMove::from(m), score));
    }

    pub fn finish(mut self, result: MarlinWdl, f: &mut impl Write) {
        self.position.wdl = result;

        // marlinformat header
        self.position.write(f);

        // viriformat moves
        for (m, score) in self.moves {
            f.write_all(&m.0.to_le_bytes()).unwrap();
            f.write_all(&score.to_le_bytes()).unwrap();
        }
        
        // viriformat footer
        f.write_all(&[0, 0, 0, 0]).unwrap();
    }
}

pub struct DataGen<'a, T: Write> {
    f: &'a mut T,
    rng: rand::rngs::ThreadRng,
    params: search::SearchParams,
    tt: Vec<search::TtEntry>,
    history: [[i16; 64]; 64],
    corrhist: [[i32; 16384]; 2],
    positions: usize,
}

impl<'a, T: Write> DataGen<'a, T> {
    pub fn new(f: &'a mut T) -> DataGen<'a, T> {
        Self { 
            f,
            rng: rand::rng(),
            params: search::SearchParams::default(),
            tt: search::allocate_tt(16),
            history: [[0; 64]; 64],
            corrhist: [[0; 16384]; 2],
            positions: 0,
        }
    }

    #[must_use]
    fn find_move(&self, board: &Board, from: Square, dest: Square, prom: Option<Piece>) -> Option<Move> {
        let mut moves = ArrayVec::new();
        board.generate(&mut moves);
        moves.into_iter().find(|&m| m.from == from && m.dest == dest && m.prom == prom)
    }

    pub fn test1(&mut self) {
        let mut board = Board::from_fen("rnbqk1nr/pppp3p/3bp3/5pp1/4P3/P1P5/1P1PKPPP/RNBQ1BNR w kq - 0 5").unwrap();
        let moves = [
            "e2e1",
            "d8e7",
            "d2d4",
            "f5e4",
            "g2g3",
            "b7b6",
            "f1g2",
            "c8b7",
            "d1h5",
            "e7f7",
            "h5f7",
            "e8f7",
            "c1g5",
            "h7h6",
            "g5e3",
            "g8f6",
            "b1d2",
            "d6e7",
            "g1e2",
            "d7d6",
            "a1d1",
            "b8d7",
            "c3c4",
            "a8e8",
            "f2f3",
            "e4f3",
            "g2f3",
            "d6d5",
            "e3f4",
            "c7c5",
            "c4d5",
            "e6d5",
            "e1f2",
            "h6h5",
            "h1e1",
            "h5h4",
            "f2g2",
            "f6e4",
            "e2c3",
            "h4h3",
            "g2g1",
            "e4c3",
            "b2c3",
            "d7f6",
            "f3e2",
            "b7c8",
            "d2f3",
            "f6e4",
            "e2b5",
            "e4c3",
            "b5e8",
            "h8e8",
            "d1d3",
            "c3e4",
            "d4c5",
            "e7c5",
            "f4e3",
            "c8e6",
            "e1f1",
            "f7e7",
            "f3d4",
            "e8c8",
            "d4e6",
            "e7e6",
            "g1h1",
            "a7a5",
            "a3a4",
            "c5e3",
            "d3e3",
            "c8c2",
            "g3g4",
            "e6d6",
            "e3h3",
            "e4f2",
            "f1f2",
            "c2f2",
            "h3g3",
            "f2f4",
            "h2h4",
            "f4a4",
            "h4h5",
            "d6e7",
            "h1g2",
            "b6b5",
            "g4g5",
            "a4h4",
            "g5g6"
        ];

        let mut game = ViriFormat::new(board.clone());
        //println!("{}", board.clone());
        for m_str in moves {
            let chars = m_str.as_bytes();
            let from = Square::from_str(&m_str[..2]).unwrap();
            let dest = Square::from_str(&m_str[2..4]).unwrap();
            let prom = if chars.len() == 5 {
                match chars[4] {
                    b'n' => Some(Piece::Knight),
                    b'b' => Some(Piece::Bishop),
                    b'r' => Some(Piece::Rook),
                    b'q' => Some(Piece::Queen),
                    _ => None,
                }
            } else {
                None
            };

            let m = self.find_move(&board, from, dest, prom).unwrap_or_else(|| panic!("Attempted move {m_str} not found!?"));
            //println!("{} {}", m, board.to_san(m));
            game.push(m, 0);
            board = board.make(m);
        }

        game.finish(MarlinWdl::Draw, self.f);
    }

    pub fn play(&mut self, mut games: usize) -> usize {
        while games > 0 {
            if self.play_game() {
                games -= 1;
            }
        }
        self.positions
    }

    fn search(&mut self, board: Board, keystack: &mut Vec<u64>, node_limit: bool) -> Option<(Move, i16)> {
        let start = Instant::now();
        let stop_after = start + Duration::from_secs_f32(if node_limit { 0.25 } else { 2.0 });
        let mut s = search::Search::new(start, Some(stop_after), &self.tt, &mut self.history, &mut self.corrhist, &self.params);
        let mut pv = ArrayVec::new();
        let mut score = 0;
        let mut lower_bound = 50;
        let mut upper_bound = 50;
        for depth in 0..=63 {
            loop {
                pv.set_len(0);
                let lower_window = score - lower_bound;
                let upper_window = score + upper_bound;
                let mut output = output::NoOp;
                score = s.search_root(&board, depth, lower_window, upper_window, &mut output, &mut pv, keystack);
                if score <= lower_window {
                    lower_bound *= 2;
                    continue;
                }
                if score >= upper_window {
                    upper_bound *= 2;
                    continue;
                }
                break;
            }
            if node_limit && (s.nodes() + s.qnodes()) > 5_000 {
                break;
            }
            if !node_limit && depth == 10 {
                break;
            }
        }
        if pv.is_empty() {
            return None;
        }
        Some((pv[0], score as i16))
    }

    fn play_game(&mut self) -> bool {
        let mut yukari_board_stack = Vec::new();
        let mut cc_board_stack = Vec::new();
        let mut keystack = Vec::new();
        cc_board_stack.push(cozy_chess::Board::startpos());
        yukari_board_stack.push(Board::startpos());
        keystack.push(yukari_board_stack.last().unwrap().hash());

        // Opening: eight random moves.
        for _ in 0..8 {
            yukari_board_stack.push(yukari_board_stack.last().unwrap().clone());
            let yukari_board = yukari_board_stack.last_mut().unwrap();
            let mut moves = ArrayVec::new();
            yukari_board.generate(&mut moves);
            let Some(&m) = moves.iter().choose(&mut self.rng) else {
                // checkmate in the opening, maybe?
                return false;
            };
            //println!("{}", yukari_board.to_san(m));
            *yukari_board = yukari_board.make(m);
            keystack.push(yukari_board.hash());
            let m_str = format!("{m}");
            cc_board_stack.push(cc_board_stack.last().unwrap().clone());
            let cc_board = cc_board_stack.last_mut().unwrap();
            let Ok(cc_m) = cozy_chess::util::parse_uci_move(cc_board, &m_str) else {
                eprintln!("cozy-chess considers move {m} on board {cc_board} to be invalid!");
                return false;
            };
            let Ok(()) = cc_board.try_play(cc_m) else {
                eprintln!("cozy-chess considers move {m} on board {cc_board} to be illegal!");
                return false;
            };
        }

        // Check: the "opening" must not be excessively lopsided.
        let mut game = {
            let yukari_board = yukari_board_stack.last_mut().unwrap();
            let Some((_, score)) = self.search(yukari_board.clone(), &mut keystack, false) else {
                // checkmate???
                return false;
            };
            if score.abs() >= 1000 {
                return false;
            }
            ViriFormat::new(yukari_board.clone())
        };

        // Rollout: "soft 5k nodes" until game end.
        loop {
            assert_eq!(cc_board_stack.len(), yukari_board_stack.len());
            assert_eq!(keystack.len(), yukari_board_stack.len());

            let cc_board = cc_board_stack.last().unwrap();
            let yukari_board = yukari_board_stack.last().unwrap();
            match cc_board.status() {
                cozy_chess::GameStatus::Ongoing => {
                    // insufficient material check.
                    if yukari_board.insufficient_material() {
                        println!("1/2-1/2 {{Draw by insufficient material}}");
                        game.finish(MarlinWdl::Draw, self.f);
                        return true;
                    }

                    // rep-draw check.
                    let cc_board = cc_board.clone();
                    let mut reps = 0;
                    for board in &cc_board_stack {
                        if board.same_position(&cc_board) {
                            reps += 1;
                            if reps == 3 {
                                let yukari_reps = keystack.iter().filter(|key| **key == yukari_board.hash()).count();
                                if yukari_reps != 3 {
                                    eprintln!("cozy-chess thinks position is a repdraw, but yukari thinks there are {yukari_reps} repetitions");
                                    for board in &yukari_board_stack {
                                        eprintln!("{board}");
                                    }
                                }
                                println!("1/2-1/2 {{Draw by threefold repetition}}");
                                game.finish(MarlinWdl::Draw, self.f);
                                return true;
                            }
                        }
                    }
                },
                cozy_chess::GameStatus::Drawn => {
                    game.finish(MarlinWdl::Draw, self.f);
                    return true;
                },
                cozy_chess::GameStatus::Won => {
                    if yukari_board.side() == Colour::White {
                        println!("0-1 {{Black wins}}");
                        game.finish(MarlinWdl::BlackWin, self.f);
                    } else {
                        println!("1-0 {{White wins}}");
                        game.finish(MarlinWdl::WhiteWin, self.f);
                    }
                    return true;
                }
            }

            cc_board_stack.push(cc_board_stack.last().unwrap().clone());
            let cc_board = cc_board_stack.last_mut().unwrap();

            yukari_board_stack.push(yukari_board_stack.last().unwrap().clone());
            let yukari_board = yukari_board_stack.last_mut().unwrap();

            let Some((m, score)) = self.search(yukari_board.clone(), &mut keystack, true) else {
                eprintln!("search did not find a move on board {yukari_board}");
                return false;
            };
            let m_str = format!("{m}");
            //println!("{} = {score}", yukari_board.to_san(m));
            let Ok(cc_m) = cozy_chess::util::parse_uci_move(cc_board, &m_str) else {
                eprintln!("cozy-chess considers move {m} on board {cc_board} to be invalid!");
                return false;
            };
            let Ok(()) = cc_board.try_play(cc_m) else {
                eprintln!("cozy-chess considers move {m} on board {cc_board} to be illegal!");
                return false;
            };

            // TODO FOR NEXT DATAGEN RUN: 
            // these scores need to be absolute, rather than relative.
            // unfortunately, it's probably better to continue datagen with the fucked score than to start from scratch.

            game.push(m, score);
            *yukari_board = yukari_board.make(m);
            keystack.push(yukari_board.hash());
            self.positions += 1;
        }
    }
}