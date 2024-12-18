use std::time::Duration;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use tinyvec::ArrayVec;
use yukari_movegen::{Board, Move};

pub trait Output {
    fn new_pv(&mut self, board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move]);
    fn new_move(&mut self, board: &Board, depth: i32, time: Duration, nodes: u64, m: Move);
    fn complete(
        &mut self, board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move], success: bool, fail_high: bool,
    );
    fn abort(&mut self);
}

pub struct Human {
    progress: ProgressBar,
}

impl Human {
    pub fn start(board: &Board) -> Self {
        let mut moves = ArrayVec::new();
        board.generate(&mut moves);
        let progress = ProgressBar::new(moves.len() as u64);
        progress.set_style(ProgressStyle::with_template("[{bar:40.magenta/red}] {msg:30!}").unwrap().progress_chars("━╸ "));
        Self { progress }
    }
}

impl Output for Human {
    fn new_pv(&mut self, board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move]) {
        let nodes = if nodes > 1_000_000_000 { format!("{:>8}k", nodes / 1_000) } else { format!("{nodes:>9}") };

        self.progress.println(format!(
            "{depth:>2} {:>+7.2} {:>8.3} {nodes}\t{}",
            ((score as f32) / 100.0),
            time.as_secs_f32(),
            board.pv_to_san(pv)
        ));
    }

    fn new_move(&mut self, board: &Board, _depth: i32, _time: Duration, nodes: u64, m: Move) {
        self.progress.inc(1);
        self.progress.set_message(format!("{} ({} nodes)", board.to_san(m), nodes));
    }

    fn complete(
        &mut self, board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move], success: bool, fail_high: bool,
    ) {
        self.progress.finish_and_clear();
        let nodes = if nodes > 1_000_000_000 { format!("{:>8}k", nodes / 1_000) } else { format!("{nodes:>9}") };
        if success {
            println!(
                "{:>2} {:>+7.2} {:>8.3} {nodes}\t{}",
                depth.to_string().bold(),
                ((score as f32) / 100.0),
                time.as_secs_f32(),
                board.pv_to_san(pv)
            );
        } else if fail_high {
            println!(
                "{:>2} {:>+7.2} {:>8.3} {nodes}\t{}",
                depth.to_string().green(),
                ((score as f32) / 100.0),
                time.as_secs_f32(),
                board.pv_to_san(pv)
            );
        } else {
            println!(
                "{:>2} {:>+7.2} {:>8.3} {nodes}\t{}",
                depth.to_string().red(),
                ((score as f32) / 100.0),
                time.as_secs_f32(),
                board.pv_to_san(pv)
            );
        }
    }

    fn abort(&mut self) {
        self.progress.finish_and_clear();
    }
}

pub struct Xboard {
    movecount: usize,
    movesleft: usize,
}

impl Xboard {
    pub fn start(board: &Board) -> Self {
        let mut moves = ArrayVec::new();
        board.generate(&mut moves);
        Self { movecount: moves.len(), movesleft: moves.len() }
    }
}

impl Output for Xboard {
    fn new_pv(&mut self, _board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move]) {
        print!("{depth} {score} {} {nodes} ", time.as_millis() / 10);
        for m in pv {
            print!("{m} ");
        }
        println!();
    }

    fn new_move(&mut self, _board: &Board, depth: i32, time: Duration, nodes: u64, m: Move) {
        println!("stat01: {} {} {} {} {} {}", time.as_millis() / 10, nodes, depth, self.movesleft, self.movecount, m);
        self.movesleft -= 1;
    }

    fn complete(
        &mut self, board: &Board, depth: i32, score: i32, time: Duration, nodes: u64, pv: &[Move], success: bool, fail_high: bool,
    ) {
        if success {
            self.new_pv(board, depth, score, time, nodes, pv);
        } else if fail_high {
            self.new_pv(board, depth, score, time, nodes, pv);
            println!("++");
        } else {
            self.new_pv(board, depth, score, time, nodes, pv);
            println!("--");
        }
    }

    fn abort(&mut self) {
        /* no-op */
    }
}
