//! scientist — a local instrument for materials and chemistry investigations.
//! Architecture and every design decision: docs/design.md. Start reading at actions.rs.

mod actions;
mod cli;
mod config;
mod events;
mod learn;
mod planner;
mod repl;
mod rng;
mod search;
mod state;
mod store;
mod tools;
mod types;
mod worker;

use clap::Parser;

fn main() {
    let cli = cli::Cli::parse();
    let result = tools::Runtime::open().and_then(|mut rt| cli::dispatch(&mut rt, cli));
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
