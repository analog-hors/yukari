use std::{
    cmp::Ordering,
    sync::atomic::AtomicU64,
    time::{Duration, Instant},
};

use tinyvec::ArrayVec;
use yukari_movegen::{Board, Move};

use crate::output;

const MATE_VALUE: i32 = 10_000;

#[derive(Clone)]
pub struct SearchParams {
    pub rfp_margin_base: i32,
    pub rfp_margin_mul: i32,
    pub razor_margin_mul: i32,
    pub lmr_base: f32,
    pub lmr_mul: f32,
    pub hist_bonus_base: i32,
    pub hist_bonus_mul: i32,
    pub hist_pen_base: i32,
    pub hist_pen_mul: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            rfp_margin_base: 0,
            rfp_margin_mul: 37,
            razor_margin_mul: 250,
            lmr_base: 1.0,
            lmr_mul: 0.5,
            hist_bonus_base: 250,
            hist_bonus_mul: 300,
            hist_pen_base: 250,
            hist_pen_mul: 300,
        }
    }
}

// TODO: when 50-move rule is implemented, this can be limited to searching from the last irreversible move.
#[must_use]
pub fn is_repetition_draw(keystack: &[u64], hash: u64) -> bool {
    keystack.iter().filter(|key| **key == hash).count() >= 3
}

#[derive(Clone, Default)]
#[repr(u8)]
enum TtFlags {
    #[default]
    Exact = 0,
    Upper = 1,
    Lower = 2,
}

#[derive(Default)]
#[repr(align(16))]
pub struct TtEntry {
    key: AtomicU64,
    data: AtomicU64,
}

#[derive(Default)]
struct TtData {
    flags: TtFlags,
    depth: u8,
    score: i16,
    m: Option<Move>,
}

const _TT_ENTRY_IS_16_BYTE: () = assert!(std::mem::size_of::<TtEntry>() == 16);
const _TT_DATA_IS_8_BYTE: () = assert!(std::mem::size_of::<TtData>() == 8);

pub fn allocate_tt(megabytes: usize) -> Vec<TtEntry> {
    let target_bytes = megabytes * 1024 * 1024;

    let mut size = 1_usize;
    loop {
        if size > target_bytes {
            break;
        }
        size *= 2;
    }
    size /= 2;
    size /= std::mem::size_of::<TtEntry>();

    let mut tt: Vec<TtEntry> = Vec::new();
    tt.resize_with(size, Default::default);
    println!("# Allocated {} bytes of hash", size * std::mem::size_of::<TtEntry>());
    tt
}

pub struct Search<'a> {
    nodes: u64,
    qnodes: u64,
    nullmove_attempts: u64,
    nullmove_success: u64,
    start: Instant,
    stop_after: Option<Instant>,
    history: [[i16; 64]; 64],
    tt: &'a [TtEntry],
    corrhist: &'a mut [[i32; 16384]; 2],
    params: &'a SearchParams,
}

impl<'a> Search<'a> {
    #[must_use]
    pub fn new(
        start: Instant, stop_after: Option<Instant>, tt: &'a [TtEntry], corrhist: &'a mut [[i32; 16384]; 2],
        params: &'a SearchParams,
    ) -> Self {
        Self {
            nodes: 0,
            qnodes: 0,
            nullmove_attempts: 0,
            nullmove_success: 0,
            start,
            stop_after,
            history: [[0; 64]; 64],
            tt,
            corrhist,
            params,
        }
    }

    fn update_corrhist(&mut self, board: &Board, depth: i32, diff: i32) {
        const CORRHIST_GRAIN: i32 = 256;
        const CORRHIST_WEIGHT_SCALE: i32 = 256;
        const CORRHIST_MAX: i32 = 256 * 32;
        let entry = &mut self.corrhist[board.side() as usize][board.hash_pawns() as usize & 16383];
        let diff = diff * CORRHIST_GRAIN;
        let weight = 16.min(depth + 1);

        *entry = ((*entry * (CORRHIST_WEIGHT_SCALE - weight) + diff * weight) / CORRHIST_WEIGHT_SCALE)
            .clamp(-CORRHIST_MAX, CORRHIST_MAX);
    }

    fn eval_with_corrhist(&self, board: &Board, eval: i32) -> i32 {
        const CORRHIST_GRAIN: i32 = 256;
        let entry = &self.corrhist[board.side() as usize][board.hash_pawns() as usize & 16383];
        (eval + entry / CORRHIST_GRAIN).clamp(-MATE_VALUE + 1, MATE_VALUE - 1)
    }

    fn update_history(&mut self, m: Move, bonus: i32) {
        const HISTORY_MAX: i32 = 16384;
        let bonus = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
        let history = &mut self.history[m.from.into_inner() as usize][m.dest.into_inner() as usize];
        let bonus = bonus - (*history as i32) * bonus.abs() / HISTORY_MAX;
        *history += bonus as i16;
    }

    fn quiesce(&mut self, board: &Board, mut alpha: i32, beta: i32, pv: &mut ArrayVec<[Move; 64]>, ply: i32) -> i32 {
        let mut best_score = self.eval_with_corrhist(board, board.eval(board.side()));

        pv.set_len(0);

        // Emergency bailout
        if ply == 63 {
            return best_score;
        }

        if best_score >= beta {
            return best_score;
        }
        alpha = alpha.max(best_score);

        board.generate_captures_incremental(|m| {
            self.qnodes += 1;

            let board = board.make(m);

            let mut child_pv = ArrayVec::new();
            let score = -self.quiesce(&board, -beta, -alpha, &mut child_pv, ply + 1);

            best_score = best_score.max(score);

            if score >= beta {
                return false;
            }

            if score > alpha {
                alpha = score;
                pv.set_len(0);
                pv.push(m);
                for m in child_pv {
                    pv.push(m);
                }
            }

            true
        });

        best_score
    }

    fn probe_tt(
        &self, board: &Board, depth: i32, ply: i32, lower_bound: i32, upper_bound: i32, m: &mut Option<Move>,
    ) -> Option<i32> {
        let entry = (board.hash() & ((self.tt.len() - 1) as u64)) as usize;
        let entry = &self.tt[entry];
        let entry_key = entry.key.load(std::sync::atomic::Ordering::Relaxed);
        let entry_data = entry.data.load(std::sync::atomic::Ordering::Relaxed);
        let entry: TtData = unsafe { std::mem::transmute(entry_data) };

        if entry_key ^ entry_data == board.hash() {
            if entry.depth as i32 >= depth {
                let mut score = entry.score as i32;
                if score >= MATE_VALUE - 500 {
                    score -= ply;
                }
                if score <= -MATE_VALUE + 500 {
                    score += ply;
                }
                match entry.flags {
                    TtFlags::Exact => return Some(score),
                    TtFlags::Upper => {
                        if score <= lower_bound {
                            return Some(score);
                        }
                    }
                    TtFlags::Lower => {
                        if score >= upper_bound {
                            return Some(score);
                        }
                    }
                }
            }
            *m = entry.m;
        }
        None
    }

    fn write_tt(&self, board: &Board, ply: i32, mut data: TtData) {
        let entry = (board.hash() & ((self.tt.len() - 1) as u64)) as usize;
        let entry = &self.tt[entry];
        if i32::from(data.score) >= MATE_VALUE - 500 {
            data.score += ply as i16;
        }
        if i32::from(data.score) <= -MATE_VALUE + 500 {
            data.score -= ply as i16;
        }
        let data = unsafe { std::mem::transmute::<TtData, u64>(data) };
        entry.key.store(board.hash() ^ data, std::sync::atomic::Ordering::Relaxed);
        entry.data.store(data, std::sync::atomic::Ordering::Relaxed);
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self, board: &Board, mut depth: i32, mut lower_bound: i32, upper_bound: i32, output: &mut dyn output::Output, pv: &mut ArrayVec<[Move; 64]>, ply: i32,
        keystack: &mut Vec<u64>,
    ) -> i32 {
        // Emergency bailout
        if ply == 63 {
            return self.eval_with_corrhist(board, board.eval(board.side()));
        }

        let mut root_reduction = 0;

        // Check extension
        if board.in_check() {
            depth += 1;
            root_reduction += -1;
        }

        if depth <= 0 {
            return self.quiesce(board, lower_bound, upper_bound, pv, ply);
        }

        pv.set_len(0);

        let mut tt_move = None;
        if let Some(score) = self.probe_tt(board, depth, ply, lower_bound, upper_bound, &mut tt_move) {
            if lower_bound == upper_bound - 1 {
                return score;
            }
        } else if lower_bound != upper_bound - 1 && tt_move.is_none() && depth >= 3 {
            // internal iterative reduction
            depth -= 1;
            root_reduction += 1;
        }

        let eval_int = self.eval_with_corrhist(board, board.eval(board.side()));

        let rfp_margin = self.params.rfp_margin_base + self.params.rfp_margin_mul * depth;
        if !board.in_check() && depth <= 3 && eval_int - rfp_margin >= upper_bound {
            return eval_int - rfp_margin;
        }

        let razor_margin = self.params.razor_margin_mul * depth;
        if !board.in_check() && depth <= 3 && lower_bound.abs() < 2000 && eval_int + razor_margin <= lower_bound {
            let score = self.quiesce(board, lower_bound, lower_bound + 1, pv, ply);
            if score <= lower_bound {
                return score;
            }
        }

        let reduction = if depth > 6 { 4 } else { 3 } + ((eval_int - upper_bound) / 200).max(0);

        if !board.in_check() && depth >= 2 && eval_int >= upper_bound {
            keystack.push(board.hash());
            let board = board.make_null();
            let mut child_pv = ArrayVec::new();
            let score =
                -self.search(&board, depth - 1 - reduction, -upper_bound, -upper_bound + 1, output, &mut child_pv, ply + 1, keystack);
            keystack.pop();

            self.nullmove_attempts += 1;

            if score >= upper_bound {
                self.nullmove_success += 1;
                return score;
            }
        }

        let moves: [Move; 256] = [Move::default(); 256];
        let mut moves = ArrayVec::from(moves);
        moves.set_len(0);
        board.generate(&mut moves);

        // Is this checkmate or stalemate?
        if moves.is_empty() {
            pv.set_len(0);
            if board.in_check() {
                return -MATE_VALUE + ply;
            }
            return 0;
        }

        // Is this a repetition draw?
        if is_repetition_draw(keystack, board.hash()) {
            return 0;
        }

        moves.sort_by(|a, b| {
            if let Some(tt_move) = tt_move {
                if *a == tt_move {
                    return Ordering::Less;
                }
                if *b == tt_move {
                    return Ordering::Greater;
                }
            }

            match (a.is_capture(), b.is_capture()) {
                (false, false) => {
                    self.history[b.from.into_inner() as usize][b.dest.into_inner() as usize]
                    .cmp(&self.history[a.from.into_inner() as usize][a.dest.into_inner() as usize])
                }
                (false, true) => Ordering::Greater,
                (true, false) => Ordering::Less,
                (true, true) => {
                    board.piece_from_square(b.dest)
                        .cmp(&board.piece_from_square(a.dest))
                        .then_with(|| board.piece_from_square(a.from).cmp(&board.piece_from_square(b.from)))
                }
            }
        });

        let mut best_move = None;
        let mut best_score = i32::MIN;
        let mut raised_lower_bound = false;

        for (i, m) in moves.into_iter().enumerate() {
            self.nodes += 1;

            let mut child_pv = ArrayVec::new();
            let child_board = board.make(m);
            let mut score = 0;

            if ply == 0 {
                let now = Instant::now();
                if now >= self.start + Duration::from_secs(2) {
                    output.new_move(board, depth + root_reduction, now.duration_since(self.start), self.nodes() + self.qnodes(), m);
                }
            }

            // Push the move to check for repetition draws
            keystack.push(board.hash());

            let mut reduction = 1;

            if lower_bound == upper_bound - 1 && depth >= 3 && i >= 4 && !board.in_check() && !m.is_capture() {
                let depth = (depth as f32).ln();
                let i = (i as f32).ln();
                reduction += (depth * i).mul_add(self.params.lmr_mul, self.params.lmr_base) as i32;
                // credit: adam
            }

            if i > 0 {
                score =
                    -self.search(&child_board, depth - reduction, -lower_bound - 1, -lower_bound, output, &mut child_pv, ply + 1, keystack);
            }
            if i > 0 && reduction > 1 && score > lower_bound {
                reduction = 1;
                score =
                    -self.search(&child_board, depth - reduction, -lower_bound - 1, -lower_bound, output, &mut child_pv, ply + 1, keystack);
            }
            if i == 0 || lower_bound != upper_bound - 1 && score > lower_bound {
                reduction = 1;
                score = -self.search(&child_board, depth - reduction, -upper_bound, -lower_bound, output, &mut child_pv, ply + 1, keystack);
            }

            keystack.pop();

            if score > best_score {
                best_move = Some(m);
                best_score = score;
            }

            if self.nodes.trailing_zeros() >= 10 {
                if let Some(time) = self.stop_after {
                    if Instant::now() >= time {
                        return best_score;
                    }
                }
            }

            if score >= upper_bound {
                let bonus = self.params.hist_bonus_mul * depth - self.params.hist_bonus_base;
                let penalty = self.params.hist_pen_mul * depth - self.params.hist_pen_base;
                if !m.is_capture() {
                    for m in moves.into_iter().take(i) {
                        if m.is_capture() {
                            continue;
                        }
                        self.update_history(m, -penalty);
                    }
                    self.update_history(m, bonus);
                }

                break;
            }

            if score > lower_bound {
                lower_bound = score;
                pv.set_len(0);
                pv.push(m);
                for m in child_pv {
                    pv.push(m);
                }
                raised_lower_bound = true;

                if ply == 0 {
                    let now = Instant::now();
                    if now >= self.start + Duration::from_secs(2) {
                        output.new_pv(board, depth + root_reduction, score, now.duration_since(self.start), self.nodes() + self.qnodes(), pv);
                    }
                }
            }
        }

        self.write_tt(
            board,
            ply,
            TtData {
                m: best_move,
                score: best_score as i16,
                flags: if best_score >= upper_bound { TtFlags::Lower } else if raised_lower_bound { TtFlags::Exact } else { TtFlags::Upper },
                depth: depth as u8,
            },
        );

        if !board.in_check() && !best_move.unwrap().is_capture() && (raised_lower_bound || (best_score >= upper_bound && best_score >= eval_int) || (best_score <= lower_bound && best_score <= eval_int)) {
            self.update_corrhist(board, depth, best_score - eval_int);
        }

        best_score
    }

    pub fn search_root(&mut self, board: &Board, depth: i32, lower_bound: i32, upper_bound: i32, output: &mut dyn output::Output, pv: &mut ArrayVec<[Move; 64]>, keystack: &mut Vec<u64>) -> i32 {
        self.search(board, depth, lower_bound, upper_bound, output, pv, 0, keystack)
    }

    #[must_use]
    pub const fn nodes(&self) -> u64 {
        self.nodes
    }

    #[must_use]
    pub const fn qnodes(&self) -> u64 {
        self.qnodes
    }

    #[must_use]
    pub fn nullmove_success(&self) -> f64 {
        100.0 * (self.nullmove_success as f64) / (self.nullmove_attempts as f64)
    }
}
