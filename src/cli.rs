//! Commands. The same parser serves the shell (`scientist relax c3`) and chat (`/relax c3`),
//! so anything the planner can do you can do by hand, through the same validated path.

use crate::actions::{self, Action, Ui};
use crate::config::load_experiment_file;
use crate::events::{Actor, Event};
use crate::learn;
use crate::repl;
use crate::search::{SearchInput, Strategy};
use crate::state::{ExperimentState, JobState};
use crate::tools::Runtime;
use crate::types::*;
use anyhow::{anyhow, bail, Result};
use clap::{Parser, Subcommand};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "scientist", version, args_conflicts_with_subcommands = true,
    about = "A local instrument for materials and chemistry investigations.\n\
Run `scientist` in any directory and start typing: each directory has its own experiment.\n\
`scientist \"<request>\"` starts with that message. Subcommands below do single steps from the shell.")]
pub struct Cli {
    /// Experiment to act on (default: the one selected with `scientist use`)
    #[arg(short = 'e', long, global = true)]
    pub exp: Option<String>,
    /// Explain each computation as it happens, using its own numbers
    #[arg(long, global = true)]
    pub learn: bool,
    #[command(subcommand)]
    pub cmd: Option<Cmd>,
    /// First message for the interactive session (like `claude "…"`)
    pub prompt: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Create an experiment: describe it in plain words and the planner drafts it for your review
    /// (scientist new carbon nanotube strain vs diameter). A .toml file or --name flags also work.
    New {
        what: Vec<String>,
        /// create the planner's draft without asking
        #[arg(short = 'y', long)]
        yes: bool,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        objective: Option<String>,
        #[arg(long, value_delimiter = ',')]
        elements: Vec<String>,
        #[arg(long)]
        autonomy: Option<u8>,
    },
    /// Select the experiment that later commands act on
    Use { name: String },
    /// List experiments
    List,
    /// Overview: objective, budget, best candidates, searches, recent events
    Status,
    /// Talk to the planner; /commands run directly (type /help inside)
    Chat,
    /// One planner turn from the shell: scientist ask "what should we try first?"
    Ask {
        text: Vec<String>,
        /// auto-approve every valid action (still validated and within budget)
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Let the planner continue for up to N steps within the autonomy level
    Run {
        #[arg(long, default_value_t = 6)]
        steps: u32,
        /// auto-approve every valid action (still validated and within budget)
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// The event log
    Log {
        #[arg(long, default_value_t = 40)]
        last: usize,
    },
    /// Drill into anything: c3 (candidate), o7 (observation), j4 (job), s1 (search), h1, sha256:…
    Inspect { id: String },
    /// Why should anyone believe candidate X is interesting? The evidence chain.
    Evidence { candidate: String },
    /// Record a note (your scientific thinking is part of the history)
    Note {
        text: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        refs: Vec<String>,
    },
    /// Record an interpretation (kept separate from observations)
    Interpret {
        text: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        refs: Vec<String>,
    },
    /// Record a hypothesis
    Hypothesis {
        statement: String,
        #[arg(long, default_value = "")]
        rationale: String,
        #[arg(long, default_value = "")]
        falsified_if: String,
    },
    /// Build a structure from a prototype: diamond, zincblende, rocksalt, fcc, bcc, hcp, wurtzite, graphene, molecule, …
    Build {
        #[arg(long)]
        prototype: String,
        #[arg(long)]
        formula: String,
        /// Lattice constant in Å (default: ASE's reference value)
        #[arg(long)]
        a: Option<f64>,
        #[arg(long)]
        c: Option<f64>,
        /// e.g. 2x2x2
        #[arg(long)]
        supercell: Option<String>,
        /// nanotube chirality n,m, e.g. 5,5 (armchair) or 9,0 (zigzag)
        #[arg(long)]
        chirality: Option<String>,
        /// nanotube repeat units along the axis
        #[arg(long)]
        length: Option<u32>,
    },
    /// Change allowed elements and the maximum atoms per structure
    Constraints {
        #[arg(long, value_delimiter = ',')]
        elements: Option<Vec<String>>,
        #[arg(long)]
        max_sites: Option<u32>,
    },
    /// Import structures from a CIF / POSCAR / extxyz file
    Import { path: String },
    /// Search papers (OpenAlex + arXiv): scientist papers carbon nanotube strain energy --since 2010
    Papers {
        query: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: u32,
        #[arg(long)]
        since: Option<u32>,
    },
    /// Known phases from the Materials Project database: scientist known --system B-C-N
    Known {
        #[arg(long)]
        system: String,
        /// maximum energy above MP's hull, in meV/atom
        #[arg(long, default_value_t = 100.0)]
        max_hull: f64,
        #[arg(long, default_value_t = 30)]
        limit: u32,
    },
    /// Write a candidate's current structure to a file (cif, poscar, xyz)
    Export {
        candidate: String,
        #[arg(long, default_value = "cif")]
        format: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Generate candidates: --with mattergen --system B-C-N -n 4, or --with chemfm -n 200
    Generate {
        #[arg(long = "with")]
        generator: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(short = 'n', long, default_value_t = 4)]
        n: u32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        guidance: Option<f64>,
    },
    /// Relax candidates (geometry optimization) with --with mattersim|mace
    Relax {
        candidates: Vec<String>,
        #[arg(long = "with", default_value = "mattersim")]
        evaluator: String,
        #[arg(long)]
        fmax: Option<f64>,
        #[arg(long)]
        steps: Option<u32>,
        /// Keep the cell fixed; only atoms move
        #[arg(long)]
        fixed_cell: bool,
    },
    /// Energy, forces and stress without moving atoms
    Point {
        candidates: Vec<String>,
        #[arg(long = "with", default_value = "mattersim")]
        evaluator: String,
    },
    /// Change one thing: --substitute 3:N | --vacancy 2 | --strain 2 (percent) | --rattle 0.05 | --swap 0:3 | --supercell 2x2x1
    Mutate {
        candidate: String,
        #[arg(long)]
        substitute: Option<String>,
        #[arg(long)]
        vacancy: Option<usize>,
        #[arg(long, allow_hyphen_values = true)]
        strain: Option<f64>,
        #[arg(long)]
        rattle: Option<f64>,
        #[arg(long)]
        swap: Option<String>,
        #[arg(long)]
        supercell: Option<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Strain series (percent), cell fixed, atoms relaxed: energy vs strain
    Sweep {
        candidate: String,
        #[arg(long, allow_hyphen_values = true, default_value_t = -3.0)]
        from: f64,
        #[arg(long, allow_hyphen_values = true, default_value_t = 3.0)]
        to: f64,
        #[arg(long, default_value_t = 7)]
        points: u32,
        #[arg(long = "with", default_value = "mattersim")]
        evaluator: String,
    },
    /// A vs B: structure match and every recorded property side by side
    Compare { a: String, b: String },
    /// Import measured superconductors (SuperCon via 3DSC): scientist superconductors --elements Nb,Sn --min-tc 10
    Superconductors {
        #[arg(long, value_delimiter = ',')]
        elements: Vec<String>,
        #[arg(long)]
        min_tc: Option<f64>,
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// largest allowed composition difference between measurement and structure (3DSC totreldiff)
        #[arg(long)]
        max_mismatch: Option<f64>,
        /// include cuprates, iron-based and heavy-fermion superconductors (our Tc models do not cover them)
        #[arg(long)]
        all_classes: bool,
    },
    /// Re-check the Tc models against materials with measured Tc (benchmarks/<suite>.toml).
    /// Runs in its own new experiment. Suites: supercon (12, with baseline), supercon_unseen (276, never trained on).
    Benchmark {
        #[arg(long, default_value = "supercon")]
        suite: String,
    },
    /// Estimate superconducting Tc (two ML models + verdict): scientist tc c1 c2
    Tc { candidates: Vec<String> },
    /// Space group (crystals) or RDKit descriptors (molecules)
    Describe { candidates: Vec<String> },
    /// Formation energy and energy above this experiment's convex hull
    Hull {
        #[arg(long = "with", default_value = "mattersim")]
        evaluator: String,
    },
    /// Define, run and show searches
    #[command(subcommand)]
    Search(SearchCmd),
    /// Mark a candidate as promoted (needs evaluation evidence)
    Promote { candidate: String, reason: Vec<String> },
    /// Mark a candidate as rejected
    Reject { candidate: String, reason: Vec<String> },
    /// Change budget limits
    Budget {
        #[arg(long)]
        mattergen: Option<u64>,
        #[arg(long)]
        relaxations: Option<u64>,
        #[arg(long)]
        single_points: Option<u64>,
        #[arg(long)]
        chemfm: Option<u64>,
        #[arg(long)]
        planner_calls: Option<u64>,
    },
    /// Autonomy level 0 (manual) .. 4 (autonomous within budget)
    Level { level: u8 },
    /// Configured workers, their last known identity, and resident processes
    Workers,
    /// Concept notes: `scientist learn` lists them, `scientist learn relaxation` opens one
    Learn { concept: Option<String> },
    /// Write the lab notebook (markdown) for this experiment
    Notebook {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Branch this experiment: shares history and artifacts up to --at (default: now)
    Fork {
        name: String,
        #[arg(long)]
        at: Option<u64>,
    },
    /// End the investigation (recorded with your reason)
    Terminate { reason: Vec<String> },
}

#[derive(Subcommand, Debug)]
pub enum SearchCmd {
    /// e.g. define --strategy evolutionary --objective min:energy_per_atom --pool c1,c2
    Define {
        /// generative | evolutionary | random
        #[arg(long, default_value = "evolutionary")]
        strategy: String,
        /// for generative: mattergen | chemfm
        #[arg(long, default_value = "mattergen")]
        generator: String,
        /// direction:property, e.g. min:formation_energy_per_atom (repeat for several)
        #[arg(long = "objective", required = true)]
        objectives: Vec<String>,
        #[arg(long = "with", default_value = "mattersim")]
        evaluator: String,
        #[arg(long, default_value_t = 4)]
        per_round: u32,
        #[arg(long, default_value_t = 5)]
        rounds: u32,
        #[arg(long, default_value_t = 2)]
        patience: u32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_delimiter = ',')]
        elements: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        pool: Vec<String>,
        /// allow element substitution (changes composition)
        #[arg(long)]
        substitution: bool,
        /// maximum strain in percent
        #[arg(long, default_value_t = 3.0)]
        strain_max: f64,
        /// rattle standard deviation in Å
        #[arg(long, default_value_t = 0.05)]
        rattle: f64,
    },
    /// Run rounds of a defined search
    Run {
        id: String,
        #[arg(long, default_value_t = 1)]
        rounds: u32,
    },
    /// Show searches (or one search's rounds)
    Show { id: Option<String> },
}

pub fn parse<T: DeserializeOwned>(s: &str, what: &str) -> Result<T> {
    serde_json::from_value(Value::String(s.to_lowercase())).map_err(|_| anyhow!("unknown {what} {s:?}"))
}

/// Settings key mapping a working directory to its experiment (the directory is the project).
pub fn dir_key() -> String {
    let cwd = std::env::current_dir().ok().and_then(|p| std::fs::canonicalize(p).ok()).unwrap_or_default();
    format!("dir:{}", cwd.display())
}

/// The experiment for this directory, if one has been created or selected here.
pub fn dir_experiment(rt: &Runtime) -> Result<Option<String>> {
    Ok(rt.store.setting(&dir_key())?.filter(|e| rt.store.experiment_exists(e).unwrap_or(false)))
}

fn current(rt: &Runtime, cli_exp: &Option<String>) -> Result<String> {
    if let Some(e) = cli_exp {
        return Ok(e.clone());
    }
    if let Some(e) = dir_experiment(rt)? {
        return Ok(e);
    }
    rt.store.setting("current")?.ok_or_else(|| anyhow!("no experiment for this directory: run `scientist` and describe what you want to investigate"))
}

/// Select an experiment for this directory (and as the global fallback).
pub fn select(rt: &Runtime, name: &str) -> Result<()> {
    rt.store.set_setting("current", name)?;
    rt.store.set_setting(&dir_key(), name)?;
    Ok(())
}

/// Run an action typed by you: approval is implied, recorded as such, then executed.
pub fn manual(rt: &mut Runtime, exp: &str, action: Action, learn_mode: bool) -> Result<()> {
    rt.open_for_write(exp)?;
    let st = rt.state(exp)?;
    let errs = action.validate(&st, &rt.store);
    if !errs.is_empty() {
        bail!("not run: {}", errs.join("; "));
    }
    rt.append(exp, Actor::Human, vec![Event::ActionApproved { action: action.clone(), how: "manual command".into() }])?;
    let result = actions::execute(rt, exp, &action, Actor::Human, &Ui { learn: learn_mode });
    if rt.interrupted() {
        rt.append(exp, Actor::Human, vec![Event::Interrupted { during: action.describe() }])?;
    }
    let done = result?;
    println!("{}", done.summary);
    if let Some(t) = &done.text {
        print!("{t}");
    }
    if learn_mode {
        learn::render(rt, exp, &done)?;
    }
    match (done.candidates.first(), done.observations.first()) {
        (Some(c), _) => println!("drill down: scientist inspect {c}"),
        (None, Some(o)) => println!("drill down: scientist inspect {o}"),
        _ => {}
    }
    Ok(())
}

pub fn dispatch(rt: &mut Runtime, cli: Cli) -> Result<()> {
    let learn_mode = cli.learn;
    let Some(cmd) = cli.cmd else {
        let first = if cli.prompt.is_empty() { None } else { Some(cli.prompt.join(" ")) };
        return repl::session(rt, cli.exp, first, learn_mode);
    };
    let exp = || current(rt, &cli.exp);
    match cmd {
        Cmd::New { what, yes, name, objective, elements, autonomy } => {
            if what.len() == 1 && what[0].ends_with(".toml") {
                let ef = load_experiment_file(std::path::Path::new(&what[0]))?;
                let a = autonomy.unwrap_or(ef.planner.autonomy);
                create(rt, &ef.name, ef.objective.description.trim().to_string(), ef.constraints.elements, ef.constraints.max_sites, ef.budget, a, None)?;
            } else if let Some(name) = name {
                create(rt, &name, objective.unwrap_or_default(), elements, 40, Budget::default(), autonomy.unwrap_or(1), None)?;
            } else {
                let desc = if what.is_empty() { ask_description()? } else { Some(what.join(" ")) };
                if let Some(desc) = desc {
                    create_from_description(rt, &desc, yes)?;
                }
            }
        }
        Cmd::Use { name } => {
            if !rt.store.experiment_exists(&name)? {
                bail!("no experiment {name}");
            }
            select(rt, &name)?;
            println!("selected {name} (also for this directory)");
        }
        Cmd::List => {
            let cur = rt.store.setting("current")?;
            for e in rt.store.experiments()? {
                println!(
                    "{} {:<24} {:>5} events  created {}{}",
                    if Some(&e.name) == cur.as_ref() { "*" } else { " " },
                    e.name,
                    e.events,
                    clock(e.created_ms),
                    e.parent.map(|p| format!("  (fork of {p})")).unwrap_or_default()
                );
            }
        }
        Cmd::Status => status(rt, &exp()?)?,
        Cmd::Chat => return repl::session(rt, cli.exp, None, learn_mode),
        Cmd::Ask { text, mut yes } => {
            let e = exp()?;
            rt.open_for_write(&e)?;
            repl::agent_turn(rt, &e, Some(text.join(" ")), learn_mode, &mut yes)?;
        }
        Cmd::Run { steps, mut yes } => {
            let e = exp()?;
            rt.open_for_write(&e)?;
            repl::agent_turn_n(rt, &e, Some("Continue the investigation toward the objective.".into()), learn_mode, steps, &mut yes)?;
        }
        Cmd::Log { last } => {
            let events = rt.store.load_events(&exp()?)?;
            for r in events.iter().skip(events.len().saturating_sub(last)) {
                println!("{:>6} {} {:<8} {}", r.seq, clock(r.ts_ms), r.actor.as_str(), r.event.summary());
            }
        }
        Cmd::Inspect { id } => inspect(rt, &exp()?, &id)?,
        Cmd::Evidence { candidate } => evidence(rt, &exp()?, &candidate)?,
        Cmd::Note { text, refs } => manual(rt, &exp()?, Action::Note { text: text.join(" "), refs }, false)?,
        Cmd::Interpret { text, refs } => manual(rt, &exp()?, Action::Interpret { text: text.join(" "), refs }, false)?,
        Cmd::Hypothesis { statement, rationale, falsified_if } => manual(rt, &exp()?, Action::ProposeHypothesis { statement, rationale, falsified_if }, false)?,
        Cmd::Build { prototype, formula, a, c, supercell, chirality, length } => {
            let sc = match supercell {
                Some(s) => Some(triple(&s)?),
                None => None,
            };
            let ch = match chirality {
                Some(s) => {
                    let v: Vec<u32> = s.split(',').map(|p| p.trim().parse()).collect::<std::result::Result<_, _>>()?;
                    Some(<[u32; 2]>::try_from(v).map_err(|_| anyhow!("--chirality n,m e.g. 5,5"))?)
                }
                None => None,
            };
            manual(rt, &exp()?, Action::Build { prototype, formula, a, c, supercell: sc, chirality: ch, length }, learn_mode)?
        }
        Cmd::Constraints { elements, max_sites } => {
            let e = exp()?;
            rt.open_for_write(&e)?;
            let st = rt.state(&e)?;
            let (elements, max_sites) = (elements.unwrap_or(st.elements), max_sites.unwrap_or(st.max_sites));
            rt.append(&e, Actor::Human, vec![Event::ConstraintsSet { elements: elements.clone(), max_sites }])?;
            println!("constraints: elements {elements:?}, max {max_sites} sites per structure");
        }
        Cmd::Import { path } => manual(rt, &exp()?, Action::Import { path }, learn_mode)?,
        Cmd::Papers { query, limit, since } => manual(rt, &exp()?, Action::SearchLiterature { query: query.join(" "), limit: Some(limit), from_year: since }, false)?,
        Cmd::Known { system, max_hull, limit } => manual(rt, &exp()?, Action::LookupKnown { chemical_system: system, max_e_above_hull: Some(max_hull / 1000.0), limit: Some(limit) }, false)?,
        Cmd::Export { candidate, format, out } => {
            let e = exp()?;
            let st = rt.state(&e)?;
            let h = st.current_structure(&candidate, None).ok_or_else(|| anyhow!("no candidate {candidate}"))?;
            let s = rt.store.structure(&h)?;
            let mut ids = st.ids.clone();
            rt.open_for_write(&e)?;
            let job = rt.run_job(&e, Actor::Human, &mut ids, "structure", "export", serde_json::json!({"format": format}), vec![serde_json::to_value(&s)?], vec![h], true, &mut |_| {})?;
            let text = job.items.first().and_then(|i| i.output["text"].as_str()).ok_or_else(|| anyhow!("export failed"))?;
            let path = out.unwrap_or_else(|| PathBuf::from(format!("{candidate}.{format}")));
            std::fs::write(&path, text)?;
            println!("wrote {}", path.display());
        }
        Cmd::Generate { generator, system, n, seed, guidance } => {
            manual(rt, &exp()?, Action::Generate { generator: parse(&generator, "generator")?, chemical_system: system, n, seed, guidance }, learn_mode)?
        }
        Cmd::Relax { candidates, evaluator, fmax, steps, fixed_cell } => manual(
            rt,
            &exp()?,
            Action::Relax { candidates, evaluator: parse(&evaluator, "evaluator")?, fmax, steps, relax_cell: Some(!fixed_cell) },
            learn_mode,
        )?,
        Cmd::Point { candidates, evaluator } => manual(rt, &exp()?, Action::SinglePoint { candidates, evaluator: parse(&evaluator, "evaluator")? }, learn_mode)?,
        Cmd::Mutate { candidate, substitute, vacancy, strain, rattle, swap, supercell, seed } => {
            let m = if let Some(s) = substitute {
                let (site, el) = s.split_once(':').ok_or_else(|| anyhow!("--substitute SITE:ELEMENT, e.g. 3:N"))?;
                Mutation::Substitute { site: site.parse()?, element: el.to_string() }
            } else if let Some(v) = vacancy {
                Mutation::Vacancy { site: v }
            } else if let Some(p) = strain {
                Mutation::Strain { strain: p / 100.0 }
            } else if let Some(r) = rattle {
                Mutation::Rattle { stdev: r, seed }
            } else if let Some(s) = swap {
                let (i, j) = s.split_once(':').ok_or_else(|| anyhow!("--swap I:J"))?;
                Mutation::Swap { i: i.parse()?, j: j.parse()? }
            } else if let Some(s) = supercell {
                Mutation::Supercell { scale: triple(&s)? }
            } else {
                bail!("give one of --substitute --vacancy --strain --rattle --swap --supercell");
            };
            manual(rt, &exp()?, Action::Mutate { candidate, mutation: m }, learn_mode)?
        }
        Cmd::Sweep { candidate, from, to, points, evaluator } => manual(
            rt,
            &exp()?,
            Action::Sweep { candidate, strain_from: from / 100.0, strain_to: to / 100.0, points, evaluator: parse(&evaluator, "evaluator")? },
            learn_mode,
        )?,
        Cmd::Compare { a, b } => manual(rt, &exp()?, Action::Compare { a, b }, false)?,
        Cmd::Superconductors { elements, min_tc, limit, max_mismatch, all_classes } => manual(rt, &exp()?, Action::ImportSuperconductors { elements, min_tc, max_mismatch, include_unconventional: Some(all_classes), limit: Some(limit) }, false)?,
        Cmd::Benchmark { suite } => benchmark(rt, &suite)?,
        Cmd::Tc { candidates } => manual(rt, &exp()?, Action::EstimateTc { candidates }, false)?,
        Cmd::Describe { candidates } => manual(rt, &exp()?, Action::Describe { candidates }, false)?,
        Cmd::Hull { evaluator } => manual(rt, &exp()?, Action::ComputeHull { evaluator: parse(&evaluator, "evaluator")? }, false)?,
        Cmd::Search(SearchCmd::Define { strategy, generator, objectives, evaluator, per_round, rounds, patience, seed, elements, pool, substitution, strain_max, rattle }) => {
            let strategy = match strategy.as_str() {
                "generative" => Strategy::Generative { generator: parse(&generator, "generator")? },
                "evolutionary" => Strategy::Evolutionary { substitution, strain_max: strain_max / 100.0, rattle_stdev: rattle },
                "random" => Strategy::Random { substitution, strain_max: strain_max / 100.0, rattle_stdev: rattle },
                s => bail!("unknown strategy {s:?} (generative | evolutionary | random)"),
            };
            let objectives = objectives
                .iter()
                .map(|o| {
                    let (d, p) = o.split_once(':').ok_or_else(|| anyhow!("objective as direction:property, e.g. min:formation_energy_per_atom"))?;
                    Ok(Objective { direction: parse(d, "direction")?, property: parse(p, "property")? })
                })
                .collect::<Result<Vec<_>>>()?;
            let spec = SearchInput { strategy, objectives, evaluator: parse(&evaluator, "evaluator")?, per_round, max_rounds: rounds, patience, seed, elements, pool };
            manual(rt, &exp()?, Action::DefineSearch { spec }, false)?
        }
        Cmd::Search(SearchCmd::Run { id, rounds }) => manual(rt, &exp()?, Action::RunSearch { search: id, rounds }, learn_mode)?,
        Cmd::Search(SearchCmd::Show { id }) => search_show(rt, &exp()?, id)?,
        Cmd::Promote { candidate, reason } => manual(rt, &exp()?, Action::Promote { candidate, reason: reason.join(" ") }, false)?,
        Cmd::Reject { candidate, reason } => manual(rt, &exp()?, Action::Reject { candidate, reason: reason.join(" ") }, false)?,
        Cmd::Terminate { reason } => manual(rt, &exp()?, Action::Terminate { reason: reason.join(" ") }, false)?,
        Cmd::Budget { mattergen, relaxations, single_points, chemfm, planner_calls } => {
            let e = exp()?;
            rt.open_for_write(&e)?;
            let mut b = rt.state(&e)?.budget;
            if let Some(v) = mattergen {
                b.mattergen_structures = v;
            }
            if let Some(v) = relaxations {
                b.relaxations = v;
            }
            if let Some(v) = single_points {
                b.single_points = v;
            }
            if let Some(v) = chemfm {
                b.chemfm_molecules = v;
            }
            if let Some(v) = planner_calls {
                b.planner_calls = v;
            }
            rt.append(&e, Actor::Human, vec![Event::BudgetSet { budget: b.clone() }])?;
            println!("budget: {}", serde_json::to_string(&b)?);
        }
        Cmd::Level { level } => {
            if level > 4 {
                bail!("levels are 0..4");
            }
            let e = exp()?;
            rt.open_for_write(&e)?;
            rt.append(&e, Actor::Human, vec![Event::AutonomySet { level }])?;
            println!("autonomy level {level}: {}", level_text(level));
        }
        Cmd::Workers => workers(rt)?,
        Cmd::Learn { concept } => match concept {
            None => {
                println!("concepts (scientist learn <name>):");
                for c in learn::concepts(&rt.cfg) {
                    println!("  {c}");
                }
            }
            Some(c) => match learn::concept(&rt.cfg, &c) {
                Some(t) => print!("{t}"),
                None => bail!("no concept note {c:?}; available: {}", learn::concepts(&rt.cfg).join(", ")),
            },
        },
        Cmd::Notebook { out } => {
            let e = exp()?;
            let events = rt.store.load_events(&e)?;
            let st = ExperimentState::fold(events.clone());
            let path = out.unwrap_or_else(|| rt.cfg.data_dir().join("notebooks").join(format!("{e}.md")));
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p)?;
            }
            std::fs::write(&path, learn::notebook(&st, &events))?;
            println!("notebook written to {}", path.display());
        }
        Cmd::Fork { name, at } => {
            let e = exp()?;
            let st = rt.state(&e)?;
            let at = at.unwrap_or(st.last_seq);
            rt.store.create_experiment(&name, Some((&e, at)))?;
            rt.append(&name, Actor::Human, vec![
                Event::ExperimentCreated { name: name.clone(), objective: st.objective.clone(), elements: st.elements.clone(), max_sites: st.max_sites, budget: st.budget.clone(), autonomy: st.autonomy },
                Event::ExperimentForked { parent: e.clone(), at_seq: at },
            ])?;
            rt.store.set_setting("current", &name)?;
            println!("forked {e} at event {at} into {name} (now selected); parent artifacts are shared, not copied");
        }
    }
    Ok(())
}

pub fn level_text(level: u8) -> &'static str {
    match level {
        0 => "manual: planner off; you run every command",
        1 => "assisted: planner proposes, you approve every tool run",
        2 => "search-assisted: you approve new searches, generation and big evaluations; search rounds run alone",
        3 => "planner-assisted: you approve generation, new searches and judgments only",
        _ => "autonomous inside budget and stop rules; judgments still recorded",
    }
}

fn triple(s: &str) -> Result<[u32; 3]> {
    let v: Vec<u32> = s.split(['x', ',']).map(|p| p.parse()).collect::<std::result::Result<_, _>>()?;
    v.try_into().map_err(|_| anyhow!("expected AxBxC, e.g. 2x2x1"))
}

pub fn clock(ts_ms: u64) -> String {
    let s = (ts_ms / 1000) % 86400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

fn fmtv(v: Option<f64>, scale: f64, prec: usize) -> String {
    v.map(|x| format!("{:.*}", prec, x * scale)).unwrap_or_else(|| "—".into())
}

// ---------------------------------------------------------------- views (read-only)

fn status(rt: &Runtime, exp: &str) -> Result<()> {
    let events = rt.store.load_events(exp)?;
    let recent: Vec<String> = events[events.len().saturating_sub(6)..]
        .iter()
        .map(|r| format!("  {:>6} {} {:<8} {}", r.seq, clock(r.ts_ms), r.actor.as_str(), r.event.summary()))
        .collect();
    let st = ExperimentState::fold(events);
    let (b, u) = (&st.budget, &st.usage);
    println!("EXPERIMENT {}  [{:?}]  autonomy {} ({})", st.name, st.status, st.autonomy, level_text(st.autonomy));
    if let Some((p, at)) = &st.forked_from {
        println!("  forked from {p} at event {at}");
    }
    println!("OBJECTIVE  {}", st.objective);
    println!("ELEMENTS   {:?}, max {} sites", st.elements, st.max_sites);
    println!(
        "BUDGET     used/limit: mattergen {}/{}  relaxations {}/{}  single points {}/{}  chemfm {}/{}  planner calls {}/{}",
        u.mattergen_structures, b.mattergen_structures, u.relaxations, b.relaxations, u.single_points, b.single_points, u.chemfm_molecules, b.chemfm_molecules, u.planner_calls, b.planner_calls
    );
    let count = |s: CandidateStatus| st.candidates.iter().filter(|c| c.status == s).count();
    println!(
        "CANDIDATES {} total: {} proposed, {} promoted, {} rejected ({} crystals, {} molecules)",
        st.candidates.len(), count(CandidateStatus::Proposed), count(CandidateStatus::Promoted), count(CandidateStatus::Rejected),
        st.candidates.iter().filter(|c| c.kind == ArtifactKind::Structure).count(),
        st.candidates.iter().filter(|c| c.kind == ArtifactKind::Molecule).count()
    );
    let ms = Some(Evaluator::Mattersim);
    let mut xt: Vec<_> = st.candidates.iter().filter(|c| c.kind == ArtifactKind::Structure && c.status != CandidateStatus::Rejected).collect();
    xt.sort_by(|a, b| {
        let k = |id: &str| st.latest(id, PropertyKind::EAboveHull, ms).map(|v| v.0).unwrap_or(f64::INFINITY);
        k(&a.id).partial_cmp(&k(&b.id)).unwrap()
    });
    if !xt.is_empty() {
        println!("  id      formula      status     E/atom[ms]  E/atom[mace]  E_form[ms]  E_hull[ms] meV  origin");
        for c in xt.iter().take(12) {
            println!(
                "  {:<7} {:<12} {:<10} {:>10}  {:>12}  {:>10}  {:>13}  {}",
                c.id, c.formula, format!("{:?}", c.status),
                fmtv(st.latest(&c.id, PropertyKind::EnergyPerAtom, ms).map(|v| v.0), 1.0, 4),
                fmtv(st.latest(&c.id, PropertyKind::EnergyPerAtom, Some(Evaluator::Mace)).map(|v| v.0), 1.0, 4),
                fmtv(st.latest(&c.id, PropertyKind::FormationEnergyPerAtom, ms).map(|v| v.0), 1.0, 4),
                fmtv(st.latest(&c.id, PropertyKind::EAboveHull, ms).map(|v| v.0), 1000.0, 1),
                crate::events::origin_summary(&c.origin)
            );
        }
        if xt.len() > 12 {
            println!("  … {} more", xt.len() - 12);
        }
    }
    for (id, s) in &st.searches {
        println!("SEARCH {id}  rounds {}  front {:?}  {}", s.rounds_done, s.front, s.stopped.as_deref().unwrap_or("active"));
    }
    for h in &st.hypotheses {
        println!("HYPOTHESIS {} [{:?}] {}", h.id, h.status, h.statement);
    }
    let jobs = st.jobs.values().filter(|j| j.state == JobState::Completed).count();
    let cache = st.jobs.values().filter(|j| j.cache_hit).count();
    let gpu_s: f64 = st.jobs.values().filter(|j| !j.cache_hit).map(|j| j.wall_s).sum();
    println!("JOBS       {jobs} completed ({cache} cache hits), {:.0} s of worker compute, {} failed", gpu_s, st.jobs.values().filter(|j| j.failure.is_some()).count());
    println!("RECENT");
    for line in &recent {
        println!("{line}");
    }
    Ok(())
}

fn lattice_text(s: &Structure) -> String {
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let ang = |a: [f64; 3], b: [f64; 3]| ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (len(a) * len(b))).acos().to_degrees();
    let [a, b, c] = s.lattice;
    format!(
        "  cell lengths a={:.4} b={:.4} c={:.4} Å, angles α={:.2} β={:.2} γ={:.2}°, volume {:.3} Å³\n",
        len(a), len(b), len(c), ang(b, c), ang(a, c), ang(a, b), s.volume()
    )
}

fn inspect(rt: &Runtime, exp: &str, id: &str) -> Result<()> {
    let events = rt.store.load_events(exp)?;
    let st = ExperimentState::fold(events.clone());
    if id.starts_with("sha256:") {
        let bytes = rt.store.read_artifact(id)?;
        println!("artifact {id}\nfile {}\n{}", rt.store.artifact_file(id).display(), serde_json::to_string_pretty(&serde_json::from_slice::<Value>(&bytes)?)?);
        return Ok(());
    }
    if let Some(c) = st.candidate(id) {
        println!("CANDIDATE {} {} ({:?}) status {:?}{}", c.id, c.formula, c.kind, c.status, if c.status_reason.is_empty() { String::new() } else { format!(": {}", c.status_reason) });
        let mut chain = vec![];
        let mut cur = c.clone();
        loop {
            chain.push(format!("{} <- {}", cur.id, crate::events::origin_summary(&cur.origin)));
            match &cur.origin {
                Origin::Mutated { parent, .. } => match st.candidate(parent) {
                    Some(p) => cur = p.clone(),
                    None => break,
                },
                _ => break,
            }
        }
        println!("ORIGIN     {}", chain.join("  <-  "));
        if let Some(s) = &c.search {
            println!("SEARCH     {s}");
        }
        println!("ARTIFACT   {}  ({})", c.artifact, rt.store.artifact_file(&c.artifact).display());
        match c.kind {
            ArtifactKind::Structure => {
                let s = rt.store.structure(&c.artifact)?;
                print!("STRUCTURE (as created; site indices are what `mutate` uses)\n{}", lattice_text(&s));
                for (i, (sp, f)) in s.species.iter().zip(&s.frac_coords).enumerate() {
                    println!("  site {i:>3}  {sp:<2}  frac ({:.5}, {:.5}, {:.5})", f[0], f[1], f[2]);
                }
                if let Some(h) = st.current_structure(id, None).filter(|h| *h != c.artifact) {
                    let r = rt.store.structure(&h)?;
                    print!("CURRENT (latest relaxed) {}\n{}", short(&h), lattice_text(&r));
                }
            }
            _ => println!("SMILES     {}", rt.store.molecule(&c.artifact)?.smiles),
        }
        println!("OBSERVATIONS");
        for o in st.observations_of(id) {
            let props: Vec<String> = o.properties.iter().map(|p| format!("{:?} {:.4} {}", p.kind, p.value, p.unit)).collect();
            println!("  {} [{:?}] {}.{} model {}: {}", o.id, o.fidelity, o.method.worker, o.method.op, o.model().unwrap_or("-"), props.join(", "));
        }
        for n in st.notes.iter().filter(|n| n.refs.iter().any(|r| r == id)) {
            println!("  {} ({}): {}", if n.interpretation { "INTERPRETATION" } else { "NOTE" }, n.author.as_str(), n.text);
        }
        println!("next: scientist evidence {id} | scientist inspect <observation id>");
        return Ok(());
    }
    if let Some(o) = st.observations.iter().find(|o| o.id == id) {
        println!("OBSERVATION {} on {} [fidelity {:?}] from job {}", o.id, o.candidate, o.fidelity, o.job);
        println!("METHOD   worker {} op {} model {} checkpoint {} {} dtype {}", o.method.worker, o.method.op, o.model().unwrap_or("-"), o.method.checkpoint.as_deref().unwrap_or("-"), o.method.checkpoint_sha256.as_deref().unwrap_or("-"), o.method.dtype.as_deref().unwrap_or("-"));
        println!("VERSIONS {}", o.method.versions);
        println!("PARAMS   {}", o.method.params);
        println!("INPUT    {}", o.input);
        if let Some(out) = &o.output {
            println!("OUTPUT   {out}");
        }
        println!("PROPERTIES");
        for p in &o.properties {
            println!("  {:?} = {:.6} {}   — {}", p.kind, p.value, p.unit, p.kind.meaning());
        }
        let mut d = o.details.clone();
        if let Some(obj) = d.as_object_mut() {
            obj.remove("forces_eV_A");
        }
        println!("DETAILS  {}", serde_json::to_string_pretty(&d)?);
        if let Some(th) = o.details["trajectory"].as_str() {
            let t: Value = serde_json::from_slice(&rt.store.read_artifact(th)?)?;
            println!("TRAJECTORY ({th})\n  step   E/atom (eV)   max|F| (eV/Å)   V/atom (Å³)");
            for r in t["steps"].as_array().into_iter().flatten() {
                println!("  {:>4}  {:>12.6}  {:>13.5}  {:>11.4}", r["step"], r["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN), r["max_force_eV_A"].as_f64().unwrap_or(f64::NAN), r["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN));
            }
        }
        return Ok(());
    }
    if id.starts_with('j') {
        for r in events.iter().filter(|r| r.event.summary().starts_with(&format!("{id} ")) || matches!(&r.event, Event::ObservationRecorded { job, .. } if job == id)) {
            println!("{:>6} {} {}", r.seq, clock(r.ts_ms), r.event.summary());
            if let Event::JobQueued { params, inputs, cache_key, .. } = &r.event {
                println!("        params {params}\n        inputs {}\n        cache key {}", inputs.iter().map(|h| short(h).to_string()).collect::<Vec<_>>().join(", "), short(cache_key));
            }
            if let Event::JobFailed { failure, .. } = &r.event {
                println!("        {}", failure.detail.as_deref().unwrap_or(""));
            }
        }
        return Ok(());
    }
    if let Some(s) = st.searches.get(id) {
        println!("SEARCH {id}: {}", s.spec.describe());
        for r in &events {
            if let Event::SearchRoundCompleted { search, .. } | Event::SearchStopped { search, .. } = &r.event {
                if search == id {
                    println!("  {}", r.event.summary());
                }
            }
        }
        return Ok(());
    }
    if let Some(p) = st.papers.iter().find(|p| p.id == id) {
        let r: Value = serde_json::from_slice(&rt.store.read_artifact(&p.artifact)?)?;
        println!("PAPER {} [{} {}] retrieved record {}", p.id, p.source, p.source_id, short(&p.artifact));
        println!("TITLE    {}", p.title);
        println!("AUTHORS  {}", r["authors"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default());
        println!("WHERE    {} {}  cited {}×", p.venue.as_deref().unwrap_or("-"), p.year.map(|y| y.to_string()).unwrap_or_default(), p.cited_by.map(|c| c.to_string()).unwrap_or("?".into()));
        println!("LINK     {}", p.url);
        println!("ABSTRACT (external text, the authors' claims)\n  {}", r["abstract"].as_str().unwrap_or("(none)"));
        return Ok(());
    }
    if let Some(h) = st.hypotheses.iter().find(|h| h.id == id) {
        println!("HYPOTHESIS {} [{:?}] by {}\n  {}\n  rationale: {}\n  falsified if: {}\n  {}", h.id, h.status, h.author.as_str(), h.statement, h.rationale, h.falsified_if, h.reason);
        return Ok(());
    }
    bail!("nothing called {id} in {exp}")
}

fn evidence(rt: &Runtime, exp: &str, cid: &str) -> Result<()> {
    let st = rt.state(exp)?;
    let c = st.candidate(cid).ok_or_else(|| anyhow!("no candidate {cid}"))?;
    println!("EVIDENCE for {} {} — status {:?}{}", c.id, c.formula, c.status, if c.status_reason.is_empty() { String::new() } else { format!(" ({})", c.status_reason) });
    println!("1. WHERE IT CAME FROM: {}", crate::events::origin_summary(&c.origin));
    if let Origin::Generated { generator, params } = &c.origin {
        println!("   {generator:?} with {params} — a proposal from a generative model; by itself not evidence of anything.");
    }
    let obs = st.observations_of(cid);
    let best = obs.iter().map(|o| o.fidelity).max();
    println!("2. OBSERVATIONS ({} total, highest fidelity {:?}):", obs.len(), best);
    for o in &obs {
        let props: Vec<String> = o.properties.iter().map(|p| format!("{:?} {:.4} {}", p.kind, p.value, p.unit)).collect();
        let conv = o.details.get("converged").and_then(|v| v.as_bool()).map(|b| if b { " converged" } else { " NOT CONVERGED" }).unwrap_or("");
        println!("   {} [{:?}] {}.{} {}{}: {}", o.id, o.fidelity, o.method.worker, o.method.op, o.model().unwrap_or(""), conv, props.join(", "));
    }
    println!("3. MODEL AGREEMENT:");
    let mut any = false;
    for k in [PropertyKind::EnergyPerAtom, PropertyKind::FormationEnergyPerAtom, PropertyKind::EAboveHull] {
        if let (Some((a, _)), Some((b, _))) = (st.latest(cid, k, Some(Evaluator::Mattersim)), st.latest(cid, k, Some(Evaluator::Mace))) {
            any = true;
            let d = (a - b).abs() * 1000.0;
            println!("   {k:?}: MatterSim {a:.4} vs MACE {b:.4} {} (|Δ| {d:.1} meV/atom){}", k.unit(), if d > 30.0 { " — LARGE disagreement: ranking is uncertain" } else { "" });
        }
    }
    if !any {
        println!("   only one model has evaluated this candidate (cross-check with: scientist relax {cid} --with mace)");
    }
    let measured = st.observations_of(cid).into_iter().rev().find(|o| o.fidelity == Fidelity::Experiment && o.property(PropertyKind::Tc).is_some());
    let a = st.latest_model(cid, PropertyKind::Tc, "alex_supercon_Tc");
    let b = st.latest_model(cid, PropertyKind::Tc, "bete-net-cso");
    if measured.is_some() || a.is_some() || b.is_some() {
        println!("SUPERCONDUCTIVITY (Tc):");
        if let Some(m) = measured {
            println!("   MEASURED   {:.2} K  ({} measurement(s), range {}–{} K; {})", m.property(PropertyKind::Tc).unwrap_or(f64::NAN),
                m.details["n_measurements"], m.details["tc_min_K"], m.details["tc_max_K"], m.details["source"].as_str().unwrap_or(""));
        } else {
            println!("   MEASURED   none — no experiment has shown this material superconducts");
        }
        if let Some((v, o)) = a {
            println!("   ALIGNN     {v:.2} K  (ML surrogate of DFT; {})", o.id);
        }
        if let Some((v, o)) = b {
            println!("   BETE-NET   {v:.2} ± {:.2} K  (ML surrogate of DFT; λ {:.2}; {})", o.details["tc_std_K"].as_f64().unwrap_or(f64::NAN), o.property(PropertyKind::Lambda).unwrap_or(f64::NAN), o.id);
        }
        if let (Some(m), Some((v, _))) = (measured, a) {
            let mt = m.property(PropertyKind::Tc).unwrap_or(f64::NAN);
            println!("   model vs measurement: ALIGNN off by {:+.1} K — a calibration point for how far to trust the models on similar materials", v - mt);
        }
    }
    let notes: Vec<_> = st.notes.iter().filter(|n| n.refs.iter().any(|r| r == cid)).collect();
    if !notes.is_empty() {
        println!("4. NOTES AND INTERPRETATIONS (not observations):");
        for n in notes {
            println!("   [{}{}] {}", n.author.as_str(), if n.interpretation { ", interpretation" } else { "" }, n.text);
        }
    }
    println!("WHAT THIS CAN SUPPORT");
    match best {
        None | Some(Fidelity::Generated) => println!("   nothing yet beyond 'a model proposed it'."),
        Some(Fidelity::StructuralCheck) => println!("   geometrically plausible (distances, density); no energy evaluated yet."),
        Some(Fidelity::Descriptor) => println!("   computed molecular descriptors; no physical simulation."),
        Some(Fidelity::MlPotential) => println!(
            "   ML-potential predictions only. Good for ranking and filtering; NOT a demonstration of stability.\n   Missing: DFT (no tool installed), dynamical stability (phonons), comparison against known phases outside this experiment."
        ),
        Some(Fidelity::Dft) if obs.iter().any(|o| o.fidelity == Fidelity::Dft && o.method.worker == "mp") => println!(
            "   DFT values exist, but from the Materials Project DATABASE (not computed here): this is a known phase.\n   Its MP values are on MP's energy scale; compare with your models only after relaxing it with them."
        ),
        Some(Fidelity::Experiment) => println!("   MEASURED values exist (see the Experiment observations above): this is a known material.\n   Check the details: the measurement may be on a slightly different (e.g. doped) composition than this structure."),
        Some(f) => println!("   {f:?}-level evidence."),
    }
    Ok(())
}

fn search_show(rt: &Runtime, exp: &str, id: Option<String>) -> Result<()> {
    let st = rt.state(exp)?;
    match id {
        Some(id) => inspect(rt, exp, &id)?,
        None => {
            if st.searches.is_empty() {
                println!("no searches yet (scientist search define --help)");
            }
            for (id, s) in &st.searches {
                println!("{id}: {}\n    rounds {}  front {:?}  {}", s.spec.describe(), s.rounds_done, s.front, s.stopped.as_deref().unwrap_or("active"));
            }
        }
    }
    Ok(())
}

pub fn workers(rt: &Runtime) -> Result<()> {
    println!("CONFIGURED WORKERS (scientist.toml); GPU budget for resident workers {:.0} MB", rt.cfg.gpu.budget_mb);
    for (name, w) in &rt.cfg.workers {
        let ident = rt.store.worker_identity(name)?;
        let detail = match &ident {
            Some((_, h)) => format!(
                "model {} checkpoint {} {} load {}s gpu at load {} MB",
                h["model"].as_str().unwrap_or("-"), h["checkpoint"].as_str().unwrap_or("-"),
                h["checkpoint_sha256"].as_str().map(short).unwrap_or("-"), h["load_s"], h["gpu_mb"]
            ),
            None => "never started".into(),
        };
        println!("  {name:<10} {} timeout {}s  {}", if w.gpu { "GPU" } else { "CPU" }, w.timeout_s, detail);
    }
    let resident = rt.pool.status();
    if resident.is_empty() {
        println!("RESIDENT   none in this process (workers live as long as the scientist process that started them; `scientist chat` keeps them loaded)");
    } else {
        println!("RESIDENT   name       dev     pid  model        load s  peak GPU MB  requests  idle s  up s");
        for w in resident {
            println!("           {:<10} {:<3} {:>7}  {:<12} {:>6.1}  {:>11.0}  {:>8}  {:>6.0}  {:>4.0}", w.name, if w.gpu { "GPU" } else { "CPU" }, w.pid, w.model, w.load_s, w.peak_gpu_mb, w.requests, w.idle_s, w.uptime_s);
        }
    }
    let (n, bytes) = rt.store.cache_stats()?;
    println!("CACHE      {n} entries, {:.1} kB of stored results", bytes as f64 / 1024.0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create(rt: &mut Runtime, name: &str, objective: String, elements: Vec<String>, max_sites: u32, budget: Budget, autonomy: u8, from: Option<String>) -> Result<()> {
    rt.store.create_experiment(name, None)?;
    let mut events = vec![Event::ExperimentCreated { name: name.to_string(), objective, elements, max_sites, budget, autonomy }];
    if let Some(d) = from {
        // Your own words start the history: the planner and the notebook see the original goal.
        events.push(Event::UserMessage { text: format!("(starting this experiment) {d}") });
    }
    rt.append(name, Actor::Human, events)?;
    select(rt, name)?;
    println!("created experiment {name} for this directory");
    Ok(())
}

fn ask_description() -> Result<Option<String>> {
    let mut ed = rustyline::DefaultEditor::new()?;
    match ed.readline("  what do you want to investigate? > ") {
        Ok(s) if !s.trim().is_empty() => Ok(Some(s.trim().to_string())),
        _ => Ok(None),
    }
}

/// The planner drafts the experiment from your words; you review it (y / n / text = change it).
pub fn create_from_description(rt: &mut Runtime, desc: &str, yes: bool) -> Result<Option<String>> {
    let existing: Vec<String> = rt.store.experiments()?.into_iter().map(|e| e.name).collect();
    let mut previous: Option<crate::planner::Draft> = None;
    let mut change: Option<String> = None;
    let mut ed = rustyline::DefaultEditor::new()?;
    let literature = literature_for(rt, desc);
    loop {
        eprint!("  planner drafting the experiment… ");
        let t0 = std::time::Instant::now();
        let d = match crate::planner::draft_experiment(&rt.cfg.planner, desc, &literature, previous.as_ref(), change.as_deref(), &existing, rt.interrupt_flag()) {
            Ok(d) => d,
            Err(e) if e.is::<crate::planner::Interrupted>() => {
                eprintln!();
                println!("  ⎿ Interrupted · nothing created");
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        eprintln!("{:.1} s", t0.elapsed().as_secs_f64());
        let b = &d.budget;
        println!("\nDRAFT EXPERIMENT");
        println!("  name        {}", d.name);
        println!("  objective   {}", d.objective);
        println!("  elements    {:?}", d.elements);
        println!("  max sites   {} atoms per structure", d.max_sites);
        println!("  budget      mattergen {} · relaxations {} · single points {} · chemfm {} · planner calls {}", b.mattergen_structures, b.relaxations, b.single_points, b.chemfm_molecules, b.planner_calls);
        println!("  autonomy    {} ({})", d.autonomy, level_text(d.autonomy));
        println!("  why         {}", d.why);
        println!("\nBACKGROUND (from {} paper(s); the authors' claims, not observations)\n  {}", literature.len(), d.background);
        for f in &d.key_facts {
            let src = if f.source == "none" { "UNVERIFIED: no paper found supports this".to_string() } else {
                let i: usize = f.source[1..].parse().unwrap_or(1) - 1;
                format!("{} — {} ({})", f.source, literature[i]["title"].as_str().unwrap_or(""), literature[i]["year"])
            };
            println!("  • {}  [{src}]", f.fact);
        }
        for q in &d.open_questions {
            println!("  ? {q}");
        }
        if yes {
            create(rt, &d.name, d.objective.clone(), d.elements.clone(), d.max_sites, d.budget.clone(), d.autonomy, Some(desc.to_string()))?;
            record_context(rt, &d, &literature, desc)?;
            return Ok(Some(d.name));
        }
        let answer = match ed.readline("  create it? [y] yes  [n] no  other text = change it > ") {
            Ok(a) => a.trim().to_string(),
            Err(_) => {
                println!("  nothing created");
                return Ok(None);
            }
        };
        match answer.as_str() {
            "y" | "yes" | "Y" => {
                create(rt, &d.name, d.objective.clone(), d.elements.clone(), d.max_sites, d.budget.clone(), d.autonomy, Some(desc.to_string()))?;
                record_context(rt, &d, &literature, desc)?;
                return Ok(Some(d.name));
            }
            "n" | "no" | "N" | "" => {
                println!("  nothing created");
                return Ok(None);
            }
            other => {
                change = Some(other.to_string());
                previous = Some(d);
            }
        }
    }
}

/// Papers for the description, fetched before the experiment exists (the search worker is
/// called directly); they are recorded into the experiment once it is created.
fn literature_for(rt: &mut Runtime, desc: &str) -> Vec<Value> {
    eprint!("  researching your description… ");
    let t0 = std::time::Instant::now();
    if let Err(f) = rt.pool.ensure("search") {
        eprintln!("search unavailable ({}); drafting without literature", f.message);
        return vec![];
    }
    let params = serde_json::json!({"query": desc, "limit": 5, "sources": ["openalex", "arxiv"]});
    let out = rt.pool.request("search", "draft", "papers", &params, &[], &mut |_| {});
    let papers: Vec<Value> = out.items.into_iter().filter(|i| i.ok).map(|i| i.output).collect();
    eprintln!("{} paper(s) in {:.1} s", papers.len(), t0.elapsed().as_secs_f64());
    papers
}

/// Record the papers (p1…pN) and the planner's background note in the new experiment, and
/// write experiment.toml + context.md under ~/scientist-data/experiments/<name>/.
fn record_context(rt: &mut Runtime, d: &crate::planner::Draft, literature: &[Value], desc: &str) -> Result<()> {
    let mut events = vec![];
    let mut ids = vec![];
    for (i, p) in literature.iter().enumerate() {
        let bytes = serde_json::to_vec(p)?;
        let (hash, created) = rt.store.put_artifact(ArtifactKind::Record, &bytes)?;
        if created {
            events.push(Event::ArtifactCreated { hash: hash.clone(), kind: ArtifactKind::Record, bytes: bytes.len() as u64 });
        }
        let id = format!("p{}", i + 1);
        events.push(Event::PaperRecorded {
            id: id.clone(), source: p["source"].as_str().unwrap_or("?").into(), source_id: p["id"].as_str().unwrap_or("?").into(),
            title: p["title"].as_str().unwrap_or("").into(), year: p["year"].as_u64().map(|y| y as u32),
            venue: p["venue"].as_str().map(String::from), cited_by: p["cited_by"].as_u64(), url: p["url"].as_str().unwrap_or("").into(), artifact: hash,
        });
        ids.push(id);
    }
    rt.append(&d.name, Actor::Runtime, events)?;
    let facts: Vec<String> = d.key_facts.iter().map(|f| format!("{} [{}]", f.fact, if f.source == "none" { "unverified" } else { &f.source })).collect();
    let note = format!(
        "CONTEXT from literature (drafted with the experiment; authors' claims, not observations): {} Key facts: {} Open questions: {}",
        d.background, facts.join("; "), d.open_questions.join(" | ")
    );
    rt.append(&d.name, Actor::Planner, vec![Event::NoteAdded { text: note, refs: ids }])?;

    let dir = rt.cfg.data_dir().join("experiments").join(&d.name);
    std::fs::create_dir_all(&dir)?;
    let q = |s: &str| serde_json::to_string(s).unwrap();
    let b = &d.budget;
    let toml = format!(
        "# Drafted by the planner from: {}\n# Recreate with: scientist new {}\nname = {}\n\n[objective]\ndescription = {}\n\n[constraints]\nelements = {:?}\nmax_sites = {}\n\n[budget]\nmattergen_structures = {}\nchemfm_molecules = {}\nrelaxations = {}\nsingle_points = {}\nplanner_calls = {}\n\n[planner]\nautonomy = {}\n",
        desc.replace('\n', " "), dir.join("experiment.toml").display(), q(&d.name), q(&d.objective), d.elements, d.max_sites,
        b.mattergen_structures, b.chemfm_molecules, b.relaxations, b.single_points, b.planner_calls, d.autonomy
    );
    std::fs::write(dir.join("experiment.toml"), toml)?;
    let mut md = format!("# {}\n\n**Your description:** {desc}\n\n## Objective\n\n{}\n\n## Why these settings\n\n{}\n\n## Background (literature; the authors' claims)\n\n{}\n\n## Key facts\n\n", d.name, d.objective, d.why, d.background);
    for f in &d.key_facts {
        md.push_str(&format!("- {} — {}\n", f.fact, if f.source == "none" { "**unverified** (no paper found supports it)".into() } else { f.source.clone() }));
    }
    md.push_str("\n## Open questions\n\n");
    for q in &d.open_questions {
        md.push_str(&format!("- {q}\n"));
    }
    md.push_str("\n## References\n\n");
    for (i, p) in literature.iter().enumerate() {
        md.push_str(&format!("- **p{}** {} ({}, {}). {}\n", i + 1, p["title"].as_str().unwrap_or(""), p["year"], p["venue"].as_str().unwrap_or("?"), p["url"].as_str().unwrap_or("")));
    }
    std::fs::write(dir.join("context.md"), md)?;
    println!("wrote {} and context.md (papers recorded as p1..p{})", dir.join("experiment.toml").display(), literature.len());
    Ok(())
}

#[derive(serde::Deserialize)]
struct BenchFile {
    material: Vec<BenchMat>,
    baseline: Option<Baseline>,
}

#[derive(serde::Deserialize)]
struct BenchMat {
    name: String,
    mp_id: Option<String>,
    prototype: Option<String>,
    formula: Option<String>,
    a: Option<f64>,
    tc_measured_k: f64,
    #[serde(default)]
    blind_spot: bool,
    #[serde(default)]
    seen_alignn: String,
    #[serde(default)]
    seen_betenet: String,
}

#[derive(serde::Deserialize)]
struct Baseline {
    alignn_yes_no: u32,
    betenet_yes_no: u32,
    alignn_mae_k: f64,
    betenet_mae_k: f64,
    verdicts_consistent: u32,
}

/// Rebuild the benchmark set in a fresh experiment, estimate Tc, score against measurements.
fn benchmark(rt: &mut Runtime, suite: &str) -> Result<()> {
    let path = rt.cfg.root.join(format!("benchmarks/{suite}.toml"));
    let bf: BenchFile = toml::from_str(&std::fs::read_to_string(&path)?)?;
    let name = format!("benchmark-{}", crate::store::now_ms() / 1000);
    rt.store.create_experiment(&name, None)?;
    rt.append(&name, Actor::Runtime, vec![Event::ExperimentCreated {
        name: name.clone(), objective: format!("superconductivity benchmark from {}", path.display()),
        elements: vec![], max_sites: 64, budget: Budget::default(), autonomy: 0,
    }])?;
    rt.open_for_write(&name)?;
    let mut ids = rt.state(&name)?.ids.clone();
    let mp_ids: Vec<String> = bf.material.iter().filter_map(|m| m.mp_id.clone()).collect();
    let mp_job = rt.run_job(&name, Actor::Runtime, &mut ids, "mp", "structures", serde_json::json!({"ids": mp_ids}), vec![], vec![], true, &mut |_| {})?;
    let mut list = vec![];
    for m in &bf.material {
        let s: Structure = if let Some(id) = &m.mp_id {
            let i = mp_ids.iter().position(|x| x == id).unwrap();
            let item = mp_job.items.iter().find(|it| it.index == i && it.ok).ok_or_else(|| anyhow!("{id} not retrieved"))?;
            serde_json::from_value(item.output["structure"].clone())?
        } else {
            let params = serde_json::json!({"prototype": m.prototype, "formula": m.formula, "a": m.a});
            let job = rt.run_job(&name, Actor::Runtime, &mut ids, "structure", "build", params, vec![], vec![], true, &mut |_| {})?;
            serde_json::from_value(job.items.first().ok_or_else(|| anyhow!("build failed for {}", m.name))?.output["structure"].clone())?
        };
        let origin = match &m.mp_id {
            Some(id) => Origin::Database { source: "materials_project".into(), id: id.clone(), version: "benchmark".into(), url: String::new() },
            None => Origin::Built { prototype: m.prototype.clone().unwrap_or_default(), params: serde_json::json!({"a": m.a}) },
        };
        list.push((s, origin));
    }
    let hashes: Vec<String> = list.iter().map(|(s, _)| crate::types::sha256(&serde_json::to_vec(&s.canonical()).unwrap())).collect();
    actions::add_structures(rt, &name, Actor::Runtime, &mut ids, list, None)?;
    let st = rt.state(&name)?;
    let cids: Vec<String> = hashes.iter().map(|h| st.candidate_by_artifact(h).unwrap().to_string()).collect();
    let mut ids = st.ids.clone();
    actions::estimate_tc(rt, &name, &st, Actor::Runtime, &mut ids, &cids)?;
    let st = rt.state(&name)?;
    let cfg = rt.cfg.supercon.clone();
    // Per-material results.
    struct Row {
        name: String,
        measured: f64,
        a: f64,
        b: f64,
        bs: f64,
        verdict: &'static str,
        blind: bool,
        seen_a: String,
        seen_b: String,
    }
    let rows: Vec<Row> = bf.material.iter().zip(&cids).map(|(m, c)| {
        let a = st.latest_model(c, PropertyKind::Tc, "alex_supercon_Tc").map(|v| v.0).unwrap_or(f64::NAN);
        let (b, bs) = st.latest_model(c, PropertyKind::Tc, "bete-net-cso").map(|(v, o)| (v, o.details["tc_std_K"].as_f64().unwrap_or(0.0))).unwrap_or((f64::NAN, 0.0));
        Row { name: m.name.clone(), measured: m.tc_measured_k, a, b, bs, verdict: actions::tc_verdict(a, b, bs, &cfg), blind: m.blind_spot, seen_a: m.seen_alignn.clone(), seen_b: m.seen_betenet.clone() }
    }).collect();
    println!("SUPERCONDUCTIVITY BENCHMARK `{suite}` ({} materials, measured Tc) — experiment {name}", rows.len());
    let seen = |s: &str| if s == "no" || s.is_empty() { "" } else if s.starts_with("train") { "trained" } else if s.starts_with("test") { "test" } else { "?" };
    if rows.len() <= 24 {
        println!("  material  measured   ALIGNN   BETE-NET (±σ)    verdict                       consistent?");
        for r in &rows {
            let consistent = if r.measured > 0.0 { r.verdict != "unlikely" } else { r.verdict == "unlikely" };
            println!("  {:<8} {:>7.2} K {:>7.2} K {:>7.2} K (±{:>4.1})   {:<28}  {}  seen A:{:<7} B:{:<7}{}", r.name, r.measured, r.a, r.b, r.bs, r.verdict,
                if consistent { "✓" } else { "✗" }, seen(&r.seen_a), seen(&r.seen_b), if r.blind { "  (known blind spot)" } else { "" });
        }
    } else {
        let dir = rt.cfg.data_dir().join("benchmarks");
        std::fs::create_dir_all(&dir)?;
        let csv = dir.join(format!("{name}.csv"));
        let mut out = String::from("material,measured_K,alignn_K,betenet_K,betenet_std_K,verdict\n");
        for r in &rows {
            out.push_str(&format!("\"{}\",{:.3},{:.3},{:.3},{:.3},{}\n", r.name, r.measured, r.a, r.b, r.bs, r.verdict));
        }
        std::fs::write(&csv, out)?;
        println!("  full per-material table: {}", csv.display());
    }
    // Scores by measured-Tc band. "detected" = predicted Tc >= possible_tc_k (default 1 K).
    let med = |mut v: Vec<f64>| -> f64 { if v.is_empty() { return f64::NAN; } v.sort_by(|x, y| x.partial_cmp(y).unwrap()); v[v.len() / 2] };
    let spearman = |p: &[f64], q: &[f64]| -> f64 {
        let rank = |v: &[f64]| { let mut idx: Vec<usize> = (0..v.len()).collect(); idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap()); let mut r = vec![0.0; v.len()]; for (k, &i) in idx.iter().enumerate() { r[i] = k as f64; } r };
        let (rp, rq) = (rank(p), rank(q));
        let n = p.len() as f64;
        let (mp, mq) = (rp.iter().sum::<f64>() / n, rq.iter().sum::<f64>() / n);
        let cov: f64 = rp.iter().zip(&rq).map(|(x, y)| (x - mp) * (y - mq)).sum();
        cov / (rp.iter().map(|x| (x - mp).powi(2)).sum::<f64>() * rq.iter().map(|y| (y - mq).powi(2)).sum::<f64>()).sqrt()
    };
    println!("  band                 n   model     detected   MAE (K)  median error (K)   verdict L/p/u");
    let bands: [(&str, Box<dyn Fn(f64) -> bool>); 4] = [
        ("Tc >= 10 K", Box::new(|t| t >= 10.0)),
        ("5 <= Tc < 10 K", Box::new(|t| (5.0..10.0).contains(&t))),
        ("0 < Tc < 5 K", Box::new(|t| t > 0.0 && t < 5.0)),
        ("not superconducting", Box::new(|t| t == 0.0)),
    ];
    for (label, f) in &bands {
        let sel: Vec<&Row> = rows.iter().filter(|r| f(r.measured) && !r.blind).collect();
        if sel.is_empty() {
            continue;
        }
        let l = sel.iter().filter(|r| r.verdict.starts_with("LIKELY")).count();
        let p = sel.iter().filter(|r| r.verdict == "possible").count();
        let u = sel.len() - l - p;
        for (mname, pick) in [("ALIGNN", 0usize), ("BETE-NET", 1)] {
            let pred = |r: &Row| if pick == 0 { r.a } else { r.b };
            let det = sel.iter().filter(|r| pred(r) >= cfg.possible_tc_k).count();
            let mae = sel.iter().map(|r| (pred(r) - r.measured).abs()).sum::<f64>() / sel.len() as f64;
            let bias = med(sel.iter().map(|r| pred(r) - r.measured).collect());
            let det_txt = if *label == "not superconducting" { format!("{det} false alarms") } else { format!("{det}/{}", sel.len()) };
            println!("  {:<19} {:>3}   {:<8} {:>12}  {:>7.2}  {:>+16.2}   {}", if pick == 0 { *label } else { "" }, if pick == 0 { sel.len().to_string() } else { String::new() },
                mname, det_txt, mae, bias, if pick == 0 { format!("{l}/{p}/{u}") } else { String::new() });
        }
    }
    let sc: Vec<&Row> = rows.iter().filter(|r| r.measured > 0.0 && !r.blind).collect();
    if sc.len() >= 5 {
        let m: Vec<f64> = sc.iter().map(|r| r.measured).collect();
        println!("  ranking of superconductors by Tc (Spearman, 1 = perfect order): ALIGNN {:.2}, BETE-NET {:.2}  (n = {})",
            spearman(&sc.iter().map(|r| r.a).collect::<Vec<_>>(), &m), spearman(&sc.iter().map(|r| r.b).collect::<Vec<_>>(), &m), sc.len());
    }
    let mut verdict = "no baseline recorded for this suite: results reported, not judged".to_string();
    if let Some(b) = &bf.baseline {
        let ay = rows.iter().filter(|r| (r.a >= cfg.possible_tc_k) == (r.measured > 0.0)).count() as f64;
        let by = rows.iter().filter(|r| (r.b >= cfg.possible_tc_k) == (r.measured > 0.0)).count() as f64;
        let vc = rows.iter().filter(|r| if r.measured > 0.0 { r.verdict != "unlikely" } else { r.verdict == "unlikely" }).count() as f64;
        let nb: Vec<&Row> = rows.iter().filter(|r| !r.blind).collect();
        let amae = nb.iter().map(|r| (r.a - r.measured).abs()).sum::<f64>() / nb.len() as f64;
        let bmae = nb.iter().map(|r| (r.b - r.measured).abs()).sum::<f64>() / nb.len() as f64;
        let checks = [("ALIGNN yes/no", ay, b.alignn_yes_no as f64, true), ("BETE-NET yes/no", by, b.betenet_yes_no as f64, true),
            ("verdicts consistent", vc, b.verdicts_consistent as f64, true), ("ALIGNN MAE (K)", amae, b.alignn_mae_k, false), ("BETE-NET MAE (K)", bmae, b.betenet_mae_k, false)];
        let mut regressions = vec![];
        println!("  metric                  now     baseline");
        for (label, now, base, hb) in checks {
            let worse = if hb { now < base } else { now > base + 0.2 };
            println!("  {label:<22} {now:>6.2}   {base:>6.2}   {}", if worse { "WORSE" } else { "ok" });
            if worse {
                regressions.push(label);
            }
        }
        verdict = if regressions.is_empty() { "PASS".into() } else { format!("REGRESSION in {}", regressions.join(", ")) };
    }
    println!("{verdict}");
    rt.append(&name, Actor::Runtime, vec![Event::NoteAdded { text: format!("benchmark {suite}: {verdict}"), refs: vec![] }])?;
    Ok(())
}
