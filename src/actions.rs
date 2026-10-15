//! Actions: the single vocabulary shared by you (CLI and /commands) and the planner.
//!
//!   validate(action, state)   ids exist? budget left? elements allowed? evidence present?
//!   needs_approval(level)     the autonomy gate (design.md D)
//!   execute(action)           jobs -> artifacts -> candidates/observations -> events
//!
//! A planner action and the same command typed by you take the identical path; only the
//! `actor` column and the approval record differ.

use crate::events::{Actor, Event, ExperimentStatus, HypothesisStatus};
use crate::search::{self, SearchInput};
use crate::state::ExperimentState;
use crate::store::Store;
use crate::tools::{failure_text, JobResult, Runtime};
use crate::types::*;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::Write;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Talk to you: answer a question, explain, teach. Runs nothing.
    Respond { message: String },
    /// Ask you to decide something. Runs nothing.
    AskHuman { question: String },
    ProposeHypothesis { statement: String, rationale: String, falsified_if: String },
    UpdateHypothesis { id: String, status: HypothesisStatus, reason: String },
    Build {
        prototype: String,
        formula: String,
        a: Option<f64>,
        c: Option<f64>,
        supercell: Option<[u32; 3]>,
        /// nanotube (n, m); `a` is then the C-C bond length (default 1.42 Å)
        #[serde(default)]
        chirality: Option<[u32; 2]>,
        /// nanotube repeat units along the axis
        #[serde(default)]
        length: Option<u32>,
    },
    Import { path: String },
    /// Papers from OpenAlex (published) and arXiv (preprints). External text: claims to check.
    SearchLiterature { query: String, limit: Option<u32>, from_year: Option<u32> },
    /// Measured superconductors (SuperCon via 3DSC) as candidates, with measured Tc as evidence.
    ImportSuperconductors { elements: Vec<String>, min_tc: Option<f64>, max_mismatch: Option<f64>, include_unconventional: Option<bool>, limit: Option<u32> },
    /// Known phases from the Materials Project database for a chemical system (and its subsystems).
    LookupKnown { chemical_system: String, max_e_above_hull: Option<f64>, limit: Option<u32> },
    Generate { generator: Generator, chemical_system: Option<String>, n: u32, seed: Option<u64>, guidance: Option<f64> },
    Relax { candidates: Vec<String>, evaluator: Evaluator, fmax: Option<f64>, steps: Option<u32>, relax_cell: Option<bool> },
    SinglePoint { candidates: Vec<String>, evaluator: Evaluator },
    Mutate { candidate: String, mutation: Mutation },
    Sweep { candidate: String, strain_from: f64, strain_to: f64, points: u32, evaluator: Evaluator },
    Compare { a: String, b: String },
    Describe { candidates: Vec<String> },
    /// ML-estimated superconducting Tc from two models (ALIGNN Alexandria + BETE-NET), with a verdict.
    EstimateTc { candidates: Vec<String> },
    ComputeHull { evaluator: Evaluator },
    DefineSearch { spec: SearchInput },
    RunSearch { search: String, rounds: u32 },
    Promote { candidate: String, reason: String },
    Reject { candidate: String, reason: String },
    Note { text: String, refs: Vec<String> },
    Interpret { text: String, refs: Vec<String> },
    Terminate { reason: String },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cost {
    Free,
    Cheap,
    Expensive,
    Judgment,
}

pub struct Ui {
    pub learn: bool,
}

#[derive(Default)]
pub struct Done {
    pub summary: String,
    pub text: Option<String>,
    pub candidates: Vec<String>,
    pub observations: Vec<String>,
}

impl Action {
    pub fn describe(&self) -> String {
        use Action::*;
        match self {
            Respond { .. } => "respond".into(),
            AskHuman { question } => format!("ask: {question}"),
            ProposeHypothesis { statement, .. } => format!("propose hypothesis: {statement}"),
            UpdateHypothesis { id, status, .. } => format!("mark {id} {status:?}"),
            Build { prototype, formula, a, chirality, .. } => format!(
                "build {formula} {prototype}{}{}",
                chirality.map(|[n, m]| format!(" ({n},{m})")).unwrap_or_default(),
                a.map(|a| format!(" a={a}")).unwrap_or_default()
            ),
            Import { path } => format!("import {path}"),
            SearchLiterature { query, .. } => format!("search papers: {query}"),
            ImportSuperconductors { elements, min_tc, .. } => format!("import measured superconductors{}{}", if elements.is_empty() { String::new() } else { format!(" in {}", elements.join("-")) }, min_tc.map(|t| format!(" with Tc ≥ {t} K")).unwrap_or_default()),
            LookupKnown { chemical_system, max_e_above_hull, .. } => format!("look up known phases of {chemical_system} in Materials Project (≤{:.0} meV/atom above its hull)", max_e_above_hull.unwrap_or(0.1) * 1000.0),
            Generate { generator, chemical_system, n, .. } => format!("generate {n} with {generator:?}{}", chemical_system.as_ref().map(|s| format!(" in {s}")).unwrap_or_default()),
            Relax { candidates, evaluator, .. } => format!("relax {} with {evaluator:?}", list(candidates)),
            SinglePoint { candidates, evaluator } => format!("single point {} with {evaluator:?}", list(candidates)),
            Mutate { candidate, mutation } => format!("mutate {candidate}: {}", mutation.describe()),
            Sweep { candidate, strain_from, strain_to, points, evaluator } => format!("strain sweep {candidate} {:+.1}%..{:+.1}% ({points} points) with {evaluator:?}", strain_from * 100.0, strain_to * 100.0),
            Compare { a, b } => format!("compare {a} vs {b}"),
            Describe { candidates } => format!("describe {}", list(candidates)),
            EstimateTc { candidates } => format!("estimate superconducting Tc of {}", list(candidates)),
            ComputeHull { evaluator } => format!("compute convex hull with {evaluator:?} energies"),
            DefineSearch { spec } => format!("define search: {}", spec.describe()),
            RunSearch { search, rounds } => format!("run search {search} for {rounds} round(s)"),
            Promote { candidate, reason } => format!("promote {candidate}: {reason}"),
            Reject { candidate, reason } => format!("reject {candidate}: {reason}"),
            Note { text, .. } => format!("note: {text}"),
            Interpret { text, .. } => format!("interpretation: {text}"),
            Terminate { reason } => format!("terminate investigation: {reason}"),
        }
    }

    pub fn cost(&self) -> Cost {
        use Action::*;
        match self {
            // Lookups only read external sources and record what came back: the planner may
            // research on its own, the way a person checks a reference before answering.
            Respond { .. } | AskHuman { .. } | ProposeHypothesis { .. } | Compare { .. } | Note { .. } | Interpret { .. }
            | SearchLiterature { .. } | LookupKnown { .. } | ImportSuperconductors { .. } => Cost::Free,
            Build { .. } | Import { .. } | Mutate { .. } | SinglePoint { .. } | Describe { .. } | ComputeHull { .. } | EstimateTc { .. } => Cost::Cheap,
            Relax { candidates, .. } if candidates.len() <= 5 => Cost::Cheap,
            Relax { .. } | Generate { .. } | Sweep { .. } | DefineSearch { .. } | RunSearch { .. } => Cost::Expensive,
            Promote { .. } | Reject { .. } | Terminate { .. } | UpdateHypothesis { .. } => Cost::Judgment,
        }
    }

    /// Autonomy gate (design.md D). Scientific judgments always need you below level 4.
    pub fn needs_approval(&self, level: u8) -> bool {
        let c = self.cost();
        match level {
            0 | 1 => c != Cost::Free,
            2 => c == Cost::Judgment || (c == Cost::Expensive && !matches!(self, Action::RunSearch { .. })),
            3 => c == Cost::Judgment || matches!(self, Action::Generate { .. } | Action::DefineSearch { .. }),
            _ => false,
        }
    }

    pub fn validate(&self, st: &ExperimentState, store: &Store) -> Vec<String> {
        use Action::*;
        let mut errs = vec![];
        if st.status == ExperimentStatus::Terminated && !matches!(self, Respond { .. } | AskHuman { .. } | Note { .. } | Interpret { .. } | Compare { .. }) {
            errs.push(format!("experiment is terminated ({}); fork it to continue", st.status_reason));
            return errs;
        }
        let rem = st.remaining();
        let need = |errs: &mut Vec<String>, id: &str, kind: Option<ArtifactKind>| match st.candidate(id) {
            None => errs.push(format!("no candidate {id} (existing: {})", existing(st))),
            Some(c) if kind.is_some_and(|k| k != c.kind) => errs.push(format!("{id} is a {:?}, this needs a {:?}", c.kind, kind.unwrap())),
            _ => {}
        };
        let allowed = |e: &str| st.elements.is_empty() || st.elements.iter().any(|x| x == e);
        match self {
            Build { formula, prototype, chirality, .. } => {
                if prototype == "nanotube" && chirality.is_none() {
                    errs.push("a nanotube needs chirality (n, m), e.g. [5,5] armchair or [9,0] zigzag".into());
                }
                if prototype != "nanotube" && chirality.is_some() {
                    errs.push(format!("chirality only applies to the nanotube prototype, not {prototype}; set it to null"));
                }
                for e in elements_in(formula) {
                    if !allowed(&e) && prototype != "molecule" {
                        errs.push(format!("{e} is outside the experiment's elements {:?}", st.elements));
                    }
                }
            }
            SearchLiterature { query, limit, .. } => {
                if query.trim().len() < 3 {
                    errs.push("give a search query".into());
                }
                if limit.unwrap_or(8) > 25 {
                    errs.push("limit at most 25 per source".into());
                }
            }
            LookupKnown { chemical_system, limit, .. } => {
                for e in chemical_system.split('-') {
                    if !allowed(e) {
                        errs.push(format!("{e} is outside the experiment's elements {:?}", st.elements));
                    }
                }
                if limit.unwrap_or(30) > 200 {
                    errs.push("limit at most 200 per lookup".into());
                }
            }
            Generate { generator: Generator::Mattergen, chemical_system, n, .. } => {
                match chemical_system {
                    None => errs.push("mattergen needs chemical_system, e.g. \"B-C-N\"".into()),
                    Some(s) => {
                        for e in s.split('-') {
                            if !allowed(e) {
                                errs.push(format!("{e} is outside the experiment's elements {:?}", st.elements));
                            }
                        }
                    }
                }
                if *n as u64 > rem.mattergen_structures {
                    errs.push(format!("budget: {n} structures requested, {} mattergen structures left", rem.mattergen_structures));
                }
            }
            Generate { generator: Generator::Chemfm, n, .. } => {
                if *n as u64 > rem.chemfm_molecules {
                    errs.push(format!("budget: {n} molecules requested, {} chemfm molecules left", rem.chemfm_molecules));
                }
            }
            Relax { candidates, .. } | SinglePoint { candidates, .. } => {
                if candidates.is_empty() {
                    errs.push("no candidates given".into());
                }
                for c in candidates {
                    need(&mut errs, c, Some(ArtifactKind::Structure));
                }
                let left = if matches!(self, Relax { .. }) { rem.relaxations } else { rem.single_points };
                if candidates.len() as u64 > left {
                    errs.push(format!("budget: {} requested, {left} left", candidates.len()));
                }
            }
            Mutate { candidate, mutation } => {
                need(&mut errs, candidate, Some(ArtifactKind::Structure));
                if let (Some(h), true) = (st.current_structure(candidate, None), errs.is_empty()) {
                    if let Ok(s) = store.structure(&h) {
                        let n = s.species.len();
                        match mutation {
                            Mutation::Substitute { site, element } => {
                                if *site >= n {
                                    errs.push(format!("{candidate} has sites 0..{}", n - 1));
                                }
                                if !allowed(element) {
                                    errs.push(format!("{element} is outside the experiment's elements {:?}", st.elements));
                                }
                            }
                            Mutation::Vacancy { site } if *site >= n => errs.push(format!("{candidate} has sites 0..{}", n - 1)),
                            Mutation::Swap { i, j } if *i >= n || *j >= n => errs.push(format!("{candidate} has sites 0..{}", n - 1)),
                            Mutation::Strain { strain } if strain.abs() > 0.25 => errs.push("strain beyond ±25% is not physical for these models".into()),
                            _ => {}
                        }
                    }
                }
            }
            Sweep { candidate, strain_from, strain_to, points, .. } => {
                need(&mut errs, candidate, Some(ArtifactKind::Structure));
                if !(2..=25).contains(points) {
                    errs.push("points must be 2..25".into());
                }
                if strain_from.abs() > 0.2 || strain_to.abs() > 0.2 {
                    errs.push("keep strain within ±20%".into());
                }
                if *points as u64 > rem.relaxations {
                    errs.push(format!("budget: {points} relaxations needed, {} left", rem.relaxations));
                }
            }
            Compare { a, b } => {
                need(&mut errs, a, None);
                need(&mut errs, b, None);
            }
            Describe { candidates } => candidates.iter().for_each(|c| need(&mut errs, c, None)),
            EstimateTc { candidates } => {
                if candidates.is_empty() {
                    errs.push("no candidates given".into());
                }
                candidates.iter().for_each(|c| need(&mut errs, c, Some(ArtifactKind::Structure)));
            }
            RunSearch { search, .. } => match st.searches.get(search) {
                None => errs.push(format!("no search {search} (existing: {:?})", st.searches.keys().collect::<Vec<_>>())),
                Some(s) if s.stopped.is_some() => errs.push(format!("search {search} already stopped: {}", s.stopped.as_ref().unwrap())),
                _ => {}
            },
            DefineSearch { spec } => errs.extend(spec.validate(st)),
            Promote { candidate, .. } => {
                need(&mut errs, candidate, None);
                let evidence = st.observations_of(candidate).iter().any(|o| o.fidelity >= Fidelity::Descriptor);
                if st.candidate(candidate).is_some() && !evidence {
                    errs.push(format!("{candidate} has no evaluation yet; generation is not evidence (relax or describe it first)"));
                }
            }
            Reject { candidate, .. } => need(&mut errs, candidate, None),
            UpdateHypothesis { id, .. } => {
                if !st.hypotheses.iter().any(|h| &h.id == id) {
                    errs.push(format!("no hypothesis {id}"));
                }
            }
            _ => {}
        }
        errs
    }
}

fn list(v: &[String]) -> String {
    if v.len() <= 6 { v.join(", ") } else { format!("{}, … ({} total)", v[..5].join(", "), v.len()) }
}

fn existing(st: &ExperimentState) -> String {
    let ids: Vec<&str> = st.candidates.iter().map(|c| c.id.as_str()).collect();
    if ids.is_empty() { "none yet".into() } else { list(&ids.iter().map(|s| s.to_string()).collect::<Vec<_>>()) }
}

/// Element symbols in a formula like "BN", "C", "MgO2".
pub fn elements_in(formula: &str) -> Vec<String> {
    let mut out = vec![];
    let cs: Vec<char> = formula.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_ascii_uppercase() {
            let mut e = cs[i].to_string();
            if i + 1 < cs.len() && cs[i + 1].is_ascii_lowercase() {
                e.push(cs[i + 1]);
                i += 1;
            }
            out.push(e);
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------- progress display

pub fn progress_printer(learn: bool, labels: Vec<String>) -> impl FnMut(&Value) {
    let mut last_item = usize::MAX;
    move |v: &Value| {
        if let Some(step) = v["step"].as_u64() {
            let item = v["item"].as_u64().unwrap_or(0) as usize;
            let label = labels.get(item).cloned().unwrap_or_else(|| format!("#{item}"));
            let line = format!(
                "  {label:>6} step {step:>3}  E/atom {:>10.5} eV  max|F| {:>8.4} eV/Å  V/atom {:>7.3} Å³",
                v["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN),
                v["max_force_eV_A"].as_f64().unwrap_or(f64::NAN),
                v["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN)
            );
            if learn {
                if item != last_item {
                    eprintln!("  -- optimizer steps for {label} (each line = one force evaluation on the GPU) --");
                    last_item = item;
                }
                eprintln!("{line}");
            } else {
                eprint!("\r{line}");
                let _ = std::io::stderr().flush();
            }
        } else if let Some(f) = v["fraction"].as_f64() {
            eprint!("\r  progress {:>5.1}%", f * 100.0);
            let _ = std::io::stderr().flush();
        }
    }
}

fn end_progress_line() {
    eprint!("\r{}\r", " ".repeat(110));
}

// ---------------------------------------------------------------- execution

pub fn execute(rt: &mut Runtime, exp: &str, action: &Action, actor: Actor, ui: &Ui) -> Result<Done> {
    let st = rt.state(exp)?;
    let errs = action.validate(&st, &rt.store);
    if !errs.is_empty() {
        bail!("{}", errs.join("; "));
    }
    let result = run(rt, exp, &st, action, actor, ui);
    match &result {
        Ok(d) => rt.append(exp, actor, vec![Event::ActionCompleted { action: action.clone(), summary: d.summary.clone() }])?,
        Err(e) => rt.append(exp, actor, vec![Event::ActionFailed { action: action.clone(), error: e.to_string() }])?,
    }
    result
}

fn run(rt: &mut Runtime, exp: &str, st: &ExperimentState, action: &Action, actor: Actor, ui: &Ui) -> Result<Done> {
    use Action::*;
    let mut ids = st.ids.clone();
    match action {
        Respond { message } => Ok(Done { summary: message.clone(), ..Default::default() }),
        AskHuman { question } => Ok(Done { summary: question.clone(), ..Default::default() }),
        ProposeHypothesis { statement, rationale, falsified_if } => {
            let id = ids.hypothesis();
            rt.append(exp, actor, vec![Event::HypothesisProposed { id: id.clone(), statement: statement.clone(), rationale: rationale.clone(), falsified_if: falsified_if.clone() }])?;
            Ok(Done { summary: format!("hypothesis {id} recorded: {statement}"), ..Default::default() })
        }
        UpdateHypothesis { id, status, reason } => {
            rt.append(exp, actor, vec![Event::HypothesisUpdated { id: id.clone(), status: *status, reason: reason.clone() }])?;
            Ok(Done { summary: format!("{id} -> {status:?}"), ..Default::default() })
        }
        Note { text, refs } => {
            rt.append(exp, actor, vec![Event::NoteAdded { text: text.clone(), refs: refs.clone() }])?;
            Ok(Done { summary: "note recorded".into(), ..Default::default() })
        }
        Interpret { text, refs } => {
            rt.append(exp, actor, vec![Event::InterpretationAdded { text: text.clone(), refs: refs.clone() }])?;
            Ok(Done { summary: "interpretation recorded (kept separate from observations)".into(), ..Default::default() })
        }
        Build { prototype, formula, a, c, supercell, chirality, length } => {
            let params = json!({"prototype": prototype, "formula": formula, "a": a, "c": c, "supercell": supercell, "chirality": chirality, "length": length});
            let job = rt.run_job(exp, actor, &mut ids, "structure", "build", params.clone(), vec![], vec![], true, &mut |_| {})?;
            check(&job)?;
            let s: Structure = serde_json::from_value(job.items[0].output["structure"].clone())?;
            let added = add_structures(rt, exp, actor, &mut ids, vec![(s, Origin::Built { prototype: prototype.clone(), params })], None)?;
            Ok(added.done(format!("built {formula} ({prototype})")))
        }
        Import { path } => {
            let abs = std::fs::canonicalize(path)?.display().to_string();
            let job = rt.run_job(exp, actor, &mut ids, "structure", "import", json!({"path": abs}), vec![], vec![], false, &mut |_| {})?;
            check(&job)?;
            let list = job
                .ok_items()
                .map(|i| Ok((serde_json::from_value(i.output["structure"].clone())?, Origin::Imported { path: abs.clone() })))
                .collect::<Result<Vec<_>>>()?;
            let added = add_structures(rt, exp, actor, &mut ids, list, None)?;
            Ok(added.done(format!("imported {abs}")))
        }
        ImportSuperconductors { elements, min_tc, max_mismatch, include_unconventional, limit } => import_superconductors(rt, exp, st, actor, &mut ids, elements, *min_tc, *max_mismatch, include_unconventional.unwrap_or(false), limit.unwrap_or(20)),
        SearchLiterature { query, limit, from_year } => search_literature(rt, exp, st, actor, &mut ids, query, limit.unwrap_or(8), *from_year),
        LookupKnown { chemical_system, max_e_above_hull, limit } => lookup_known(rt, exp, actor, &mut ids, chemical_system, max_e_above_hull.unwrap_or(0.1), limit.unwrap_or(30)),
        Generate { generator: Generator::Mattergen, chemical_system, n, seed, guidance } => {
            let added = generate_mattergen(rt, exp, st, actor, &mut ids, chemical_system.as_deref().unwrap_or(""), *n, *seed, *guidance, None)?;
            Ok(added.done(format!("MatterGen generated in {}", chemical_system.as_deref().unwrap_or("?"))))
        }
        Generate { generator: Generator::Chemfm, n, seed, .. } => {
            let (added, text) = generate_chemfm(rt, exp, st, actor, &mut ids, *n, *seed, None)?;
            let mut d = added.done("ChemFM generated molecules".into());
            d.text = Some(text);
            Ok(d)
        }
        Relax { candidates, evaluator, fmax, steps, relax_cell } => {
            let params = json!({"fmax": fmax.unwrap_or(0.02), "steps": steps.unwrap_or(300), "relax_cell": relax_cell.unwrap_or(true), "optimizer": "FIRE"});
            relax(rt, exp, actor, &mut ids, candidates, *evaluator, params, ui)
        }
        SinglePoint { candidates, evaluator } => single_point(rt, exp, st, actor, &mut ids, candidates, *evaluator, ui),
        Mutate { candidate, mutation } => {
            let added = mutate(rt, exp, st, actor, &mut ids, candidate, mutation.clone(), None)?;
            Ok(added.done(format!("{candidate}: {}", mutation.describe())))
        }
        Sweep { candidate, strain_from, strain_to, points, evaluator } => sweep(rt, exp, st, actor, &mut ids, candidate, *strain_from, *strain_to, *points, *evaluator, ui),
        Compare { a, b } => compare(rt, exp, st, actor, &mut ids, a, b),
        Describe { candidates } => describe(rt, exp, st, actor, &mut ids, candidates),
        EstimateTc { candidates } => estimate_tc(rt, exp, st, actor, &mut ids, candidates),
        ComputeHull { evaluator } => compute_hull(rt, exp, st, actor, &mut ids, *evaluator),
        DefineSearch { spec } => search::define(rt, exp, st, actor, &mut ids, spec),
        RunSearch { search, rounds } => search::run(rt, exp, actor, search, *rounds, ui),
        Promote { candidate, reason } | Reject { candidate, reason } => {
            let status = if matches!(action, Promote { .. }) { CandidateStatus::Promoted } else { CandidateStatus::Rejected };
            rt.append(exp, actor, vec![Event::CandidateStatusChanged { candidate: candidate.clone(), status, reason: reason.clone() }])?;
            Ok(Done { summary: format!("{candidate} -> {status:?}: {reason}"), ..Default::default() })
        }
        Terminate { reason } => {
            rt.append(exp, actor, vec![Event::StatusSet { status: ExperimentStatus::Terminated, reason: reason.clone() }])?;
            Ok(Done { summary: format!("investigation terminated: {reason}"), ..Default::default() })
        }
    }
}

fn check(job: &JobResult) -> Result<()> {
    match &job.failure {
        Some(f) => bail!("{} {}", job.job, failure_text(f)),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------- candidates

#[derive(Default)]
pub struct Added {
    pub added: Vec<String>,
    pub rejected: Vec<(String, String)>,
    pub duplicates: Vec<String>,
    pub observations: Vec<String>,
}

impl Added {
    pub fn done(self, what: String) -> Done {
        let mut s = format!("{what}: {} new candidate(s) {}", self.added.len(), list(&self.added));
        if !self.rejected.is_empty() {
            let r: Vec<String> = self.rejected.iter().map(|(c, why)| format!("{c} ({why})")).collect();
            s.push_str(&format!("; failed structural check: {}", r.join("; ")));
        }
        if !self.duplicates.is_empty() {
            s.push_str(&format!("; {} identical to existing candidates {}", self.duplicates.len(), list(&self.duplicates)));
        }
        Done { summary: s, candidates: self.added, observations: self.observations, text: None }
    }
}

/// Store structures, add them as candidates, and run the cheap structural check
/// (distances, elements, size, symmetry) as the first rung of the fidelity ladder.
pub fn add_structures(rt: &mut Runtime, exp: &str, actor: Actor, ids: &mut crate::state::Ids, list: Vec<(Structure, Origin)>, search: Option<String>) -> Result<Added> {
    let st = rt.state(exp)?;
    let mut out = Added::default();
    let mut events = vec![];
    let mut new: Vec<(String, Hash, Structure)> = vec![];
    let mut seen: BTreeSet<Hash> = BTreeSet::new();
    for (s, origin) in list {
        let (hash, canon, created) = rt.store.put_structure(&s)?;
        if created {
            events.push(Event::ArtifactCreated { hash: hash.clone(), kind: ArtifactKind::Structure, bytes: rt.store.artifact_bytes_len(&hash).unwrap_or(0) });
        }
        if let Some(existing) = st.candidate_by_artifact(&hash) {
            let c = st.candidate(existing).unwrap();
            // Rejected only by the structural check? Constraints may have changed: check again.
            if c.status == CandidateStatus::Rejected && c.status_reason.starts_with("failed structural check") && seen.insert(hash.clone()) {
                new.push((existing.to_string(), hash, canon));
            } else {
                out.duplicates.push(existing.to_string());
            }
            continue;
        }
        if !seen.insert(hash.clone()) {
            continue;
        }
        let id = ids.candidate();
        events.push(Event::CandidateAdded {
            id: id.clone(),
            artifact: hash.clone(),
            kind: ArtifactKind::Structure,
            formula: canon.formula(),
            origin,
            job: None,
            search: search.clone(),
        });
        new.push((id, hash, canon));
    }
    rt.append(exp, actor, events)?;
    if new.is_empty() {
        return Ok(out);
    }
    let params = json!({"min_distance": 0.5, "allowed_elements": st.elements, "max_sites": st.max_sites});
    let job = rt.run_job(
        exp,
        actor,
        ids,
        "structure",
        "validate",
        params,
        new.iter().map(|n| serde_json::to_value(&n.2).unwrap()).collect(),
        new.iter().map(|n| n.1.clone()).collect(),
        true,
        &mut |_| {},
    )?;
    let mut events = vec![];
    for item in job.ok_items() {
        let (cid, hash, _) = &new[item.index];
        let m = &item.output["metrics"];
        let oid = ids.observation();
        events.push(Event::ObservationRecorded {
            id: oid.clone(),
            candidate: cid.clone(),
            job: job.job.clone(),
            method: job.method.clone(),
            fidelity: Fidelity::StructuralCheck,
            input: hash.clone(),
            output: None,
            properties: vec![
                Property::new(PropertyKind::VolumePerAtom, m["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN)),
                Property::new(PropertyKind::Density, m["density_g_cm3"].as_f64().unwrap_or(f64::NAN)),
            ],
            details: item.output.clone(),
        });
        out.observations.push(oid);
        let was_rejected = st.candidate(cid).is_some_and(|c| c.status == CandidateStatus::Rejected);
        if item.output["valid"] == false {
            let why = item.output["reasons"].as_array().map(|r| r.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
            if !was_rejected {
                events.push(Event::CandidateStatusChanged { candidate: cid.clone(), status: CandidateStatus::Rejected, reason: format!("failed structural check: {why}") });
            }
            out.rejected.push((cid.clone(), why));
        } else {
            if was_rejected {
                events.push(Event::CandidateStatusChanged { candidate: cid.clone(), status: CandidateStatus::Proposed, reason: "passed the structural check under the current constraints".into() });
            }
            out.added.push(cid.clone());
        }
    }
    rt.append(exp, actor, events)?;
    check(&job)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_mattergen(
    rt: &mut Runtime,
    exp: &str,
    st: &ExperimentState,
    actor: Actor,
    ids: &mut crate::state::Ids,
    system: &str,
    n: u32,
    seed: Option<u64>,
    guidance: Option<f64>,
    search: Option<String>,
) -> Result<Added> {
    // A seed derived from the log position makes every call reproducible yet distinct.
    let seed = seed.unwrap_or(st.last_seq);
    let params = json!({"chemical_system": system, "n": n, "batch_size": n.min(16), "guidance": guidance.unwrap_or(2.0), "seed": seed});
    eprintln!("  MatterGen: {n} structure(s) in {system}; 1000 diffusion steps per batch (measured ~29 s/structure at batch 64, slower at small batches)");
    let job = rt.run_job(exp, actor, ids, "mattergen", "generate", params.clone(), vec![], vec![], true, &mut progress_printer(false, vec![]))?;
    end_progress_line();
    check(&job)?;
    let list = job
        .ok_items()
        .map(|i| Ok((serde_json::from_value(i.output["structure"].clone())?, Origin::Generated { generator: Generator::Mattergen, params: params.clone() })))
        .collect::<Result<Vec<_>>>()?;
    add_structures(rt, exp, actor, ids, list, search)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_chemfm(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, n: u32, seed: Option<u64>, search: Option<String>) -> Result<(Added, String)> {
    let seed = seed.unwrap_or(st.last_seq);
    let params = json!({"n": n, "seed": seed, "temperature": 1.0, "top_k": 0, "max_new_tokens": 128, "batch_size": n.min(256)});
    let job = rt.run_job(exp, actor, ids, "chem", "chemfm_generate", params.clone(), vec![], vec![], true, &mut progress_printer(false, vec![]))?;
    end_progress_line();
    check(&job)?;
    let toolkit = format!("RDKit {}", job.hello["rdkit"].as_str().unwrap_or("?"));
    let mut out = Added::default();
    let mut events = vec![];
    let mut seen = BTreeSet::new();
    let (mut invalid, mut reasons) = (0, std::collections::BTreeMap::<String, u32>::new());
    for item in job.ok_items() {
        let o = &item.output;
        if o["valid"] != true {
            invalid += 1;
            *reasons.entry(o["reason"].as_str().unwrap_or("?").to_string()).or_default() += 1;
            continue;
        }
        let m = Molecule { smiles: o["smiles"].as_str().unwrap_or("").to_string(), toolkit: toolkit.clone() };
        let (hash, created) = rt.store.put_artifact(ArtifactKind::Molecule, &serde_json::to_vec(&m)?)?;
        if created {
            events.push(Event::ArtifactCreated { hash: hash.clone(), kind: ArtifactKind::Molecule, bytes: rt.store.artifact_bytes_len(&hash).unwrap_or(0) });
        }
        if let Some(e) = st.candidate_by_artifact(&hash) {
            out.duplicates.push(e.to_string());
            continue;
        }
        if !seen.insert(hash.clone()) {
            out.duplicates.push("(same batch)".into());
            continue;
        }
        let cid = ids.candidate();
        events.push(Event::CandidateAdded {
            id: cid.clone(),
            artifact: hash,
            kind: ArtifactKind::Molecule,
            formula: o["formula"].as_str().unwrap_or("?").to_string(),
            origin: Origin::Generated { generator: Generator::Chemfm, params: params.clone() },
            job: Some(job.job.clone()),
            search: search.clone(),
        });
        let outside: Vec<String> = o["elements"]
            .as_array()
            .map(|a| a.iter().filter_map(|e| e.as_str()).filter(|e| *e != "H" && !st.elements.is_empty() && !st.elements.iter().any(|x| x == e)).map(String::from).collect())
            .unwrap_or_default();
        if outside.is_empty() {
            out.added.push(cid);
        } else {
            events.push(Event::CandidateStatusChanged { candidate: cid.clone(), status: CandidateStatus::Rejected, reason: format!("contains {outside:?}, outside the experiment's elements") });
            out.rejected.push((cid, format!("elements {outside:?}")));
        }
    }
    rt.append(exp, actor, events)?;
    let total = job.items.len();
    let mut text = format!(
        "ChemFM sampled {total} strings: {} valid ({:.0}%), {invalid} invalid, {} duplicates, {} rejected for elements.\n",
        total - invalid,
        100.0 * (total - invalid) as f64 / total.max(1) as f64,
        out.duplicates.len(),
        out.rejected.len()
    );
    for (r, k) in &reasons {
        text.push_str(&format!("  invalid because: {r} ({k})\n"));
    }
    Ok((out, text))
}

// ---------------------------------------------------------------- evaluation

#[allow(clippy::too_many_arguments)]
pub fn relax(rt: &mut Runtime, exp: &str, actor: Actor, ids: &mut crate::state::Ids, cids: &[String], ev: Evaluator, params: Value, ui: &Ui) -> Result<Done> {
    let st = rt.state(exp)?;
    let hashes: Vec<Hash> = cids.iter().map(|c| st.candidate(c).unwrap().artifact.clone()).collect();
    let inputs = hashes.iter().map(|h| Ok(serde_json::to_value(rt.store.structure(h)?)?)).collect::<Result<Vec<_>>>()?;
    let job = rt.run_job(exp, actor, ids, ev.worker(), "relax", params, inputs, hashes.clone(), true, &mut progress_printer(ui.learn, cids.to_vec()))?;
    end_progress_line();
    let (events, obs, lines) = record_relaxations(rt, exp, ids, &job, cids, &hashes)?;
    rt.append(exp, actor, events)?;
    let mut summary = format!("relaxed with {ev:?} ({}{:.1} s): {}", if job.cache_hit { "cache hit, " } else { "" }, job.wall_s, lines.join("; "));
    if let Some(f) = &job.failure {
        summary.push_str(&format!("; JOB FAILED after {} item(s): {}", job.items.len(), failure_text(f)));
    }
    Ok(Done { summary, text: None, candidates: vec![], observations: obs })
}

/// Turn relax items into relaxed-structure artifacts, trajectory artifacts and observations.
pub fn record_relaxations(rt: &mut Runtime, exp: &str, ids: &mut crate::state::Ids, job: &JobResult, cids: &[String], hashes: &[Hash]) -> Result<(Vec<Event>, Vec<String>, Vec<String>)> {
    let mut events = vec![];
    let mut obs = vec![];
    let mut lines = vec![];
    let st = rt.state(exp)?;
    for item in &job.items {
        let cid = &cids[item.index];
        // A cache hit for a computation this candidate already has: point to it, don't duplicate.
        if job.cache_hit {
            if let Some(prev) = st.observations_of(cid).into_iter().find(|o| o.method.op == "relax" && o.input == hashes[item.index] && o.method.model == job.method.model && o.method.params == job.method.params) {
                lines.push(format!("{cid}: already computed as {} (cache hit, nothing new recorded)", prev.id));
                obs.push(prev.id.clone());
                continue;
            }
        }
        let oid = ids.observation();
        if !item.ok {
            events.push(Event::ObservationRecorded {
                id: oid.clone(),
                candidate: cid.clone(),
                job: job.job.clone(),
                method: job.method.clone(),
                fidelity: Fidelity::MlPotential,
                input: hashes[item.index].clone(),
                output: None,
                properties: vec![],
                details: json!({"failed": item.error}),
            });
            lines.push(format!("{cid} FAILED ({})", item.error["message"].as_str().unwrap_or("?")));
            obs.push(oid);
            continue;
        }
        let o = &item.output;
        let relaxed: Structure = serde_json::from_value(o["structure"].clone())?;
        let (rhash, _, created) = rt.store.put_structure(&relaxed)?;
        if created {
            events.push(Event::ArtifactCreated { hash: rhash.clone(), kind: ArtifactKind::Structure, bytes: rt.store.artifact_bytes_len(&rhash).unwrap_or(0) });
        }
        let traj = serde_json::to_vec(&json!({"candidate": cid, "job": job.job, "steps": o["trajectory"]}))?;
        let (thash, tcreated) = rt.store.put_artifact(ArtifactKind::Trajectory, &traj)?;
        if tcreated {
            events.push(Event::ArtifactCreated { hash: thash.clone(), kind: ArtifactKind::Trajectory, bytes: traj.len() as u64 });
        }
        let f = &o["final"];
        let e0 = o["initial"]["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN);
        let e1 = f["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN);
        let v0 = o["initial"]["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN);
        let v1 = f["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN);
        events.push(Event::ObservationRecorded {
            id: oid.clone(),
            candidate: cid.clone(),
            job: job.job.clone(),
            method: job.method.clone(),
            fidelity: Fidelity::MlPotential,
            input: hashes[item.index].clone(),
            output: Some(rhash),
            properties: vec![
                Property::new(PropertyKind::EnergyPerAtom, e1),
                Property::new(PropertyKind::MaxForce, f["max_force_eV_A"].as_f64().unwrap_or(f64::NAN)),
                Property::new(PropertyKind::Pressure, f["pressure_GPa"].as_f64().unwrap_or(f64::NAN)),
                Property::new(PropertyKind::VolumePerAtom, v1),
            ],
            details: json!({
                "converged": o["converged"], "steps": o["steps"], "fmax_target_eV_A": o["fmax_target_eV_A"],
                "relax_cell": o["relax_cell"], "optimizer": o["optimizer"],
                "initial_energy_per_atom_eV": e0, "energy_change_meV_per_atom": (e1 - e0) * 1000.0,
                "volume_change_pct": (v1 / v0 - 1.0) * 100.0, "max_displacement_A": o["max_displacement_A"],
                "initial_max_force_eV_A": o["initial"]["max_force_eV_A"], "compute_s": o["compute_s"],
                "trajectory": thash, "cache_hit": job.cache_hit,
            }),
        });
        lines.push(format!(
            "{cid}: E/atom {e1:.4} eV (Δ{:+.0} meV/atom), {} in {} steps, relaxed {}",
            (e1 - e0) * 1000.0,
            if o["converged"] == true { "converged" } else { "NOT converged" },
            o["steps"],
            relaxed.cell_text()
        ));
        obs.push(oid);
    }
    Ok((events, obs, lines))
}

#[allow(clippy::too_many_arguments)]
fn single_point(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, cids: &[String], ev: Evaluator, _ui: &Ui) -> Result<Done> {
    let hashes: Vec<Hash> = cids.iter().map(|c| st.current_structure(c, None).unwrap()).collect();
    let inputs = hashes.iter().map(|h| Ok(serde_json::to_value(rt.store.structure(h)?)?)).collect::<Result<Vec<_>>>()?;
    let job = rt.run_job(exp, actor, ids, ev.worker(), "single_point", json!({}), inputs, hashes.clone(), true, &mut |_| {})?;
    let mut events = vec![];
    let mut obs = vec![];
    let mut lines = vec![];
    for item in job.ok_items() {
        let o = &item.output;
        let cid = &cids[item.index];
        let oid = ids.observation();
        let e = o["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN);
        events.push(Event::ObservationRecorded {
            id: oid.clone(),
            candidate: cid.clone(),
            job: job.job.clone(),
            method: job.method.clone(),
            fidelity: Fidelity::MlPotential,
            input: hashes[item.index].clone(),
            output: None,
            properties: vec![
                Property::new(PropertyKind::EnergyPerAtom, e),
                Property::new(PropertyKind::MaxForce, o["max_force_eV_A"].as_f64().unwrap_or(f64::NAN)),
                Property::new(PropertyKind::Pressure, o["pressure_GPa"].as_f64().unwrap_or(f64::NAN)),
                Property::new(PropertyKind::VolumePerAtom, o["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN)),
            ],
            details: json!({"stress_GPa": o["stress_GPa"], "forces_eV_A": o["forces_eV_A"], "compute_s": o["compute_s"], "cache_hit": job.cache_hit}),
        });
        lines.push(format!("{cid}: E/atom {e:.4} eV, max|F| {:.3} eV/Å, P {:.2} GPa", o["max_force_eV_A"].as_f64().unwrap_or(f64::NAN), o["pressure_GPa"].as_f64().unwrap_or(f64::NAN)));
        obs.push(oid);
    }
    rt.append(exp, actor, events)?;
    check(&job)?;
    Ok(Done { summary: format!("single point with {ev:?}: {}", lines.join("; ")), observations: obs, ..Default::default() })
}

#[allow(clippy::too_many_arguments)]
pub fn mutate(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, cid: &str, m: Mutation, search: Option<String>) -> Result<Added> {
    let h = st.current_structure(cid, None).unwrap();
    let s = rt.store.structure(&h)?;
    let params = serde_json::to_value(&m)?;
    let job = rt.run_job(exp, actor, ids, "structure", "mutate", params, vec![serde_json::to_value(&s)?], vec![h], true, &mut |_| {})?;
    check(&job)?;
    let child: Structure = serde_json::from_value(job.items[0].output["structure"].clone())?;
    add_structures(rt, exp, actor, ids, vec![(child, Origin::Mutated { parent: cid.to_string(), mutation: m })], search)
}

#[allow(clippy::too_many_arguments)]
fn sweep(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, cid: &str, from: f64, to: f64, points: u32, ev: Evaluator, ui: &Ui) -> Result<Done> {
    let mut members: Vec<(f64, String)> = vec![];
    for k in 0..points {
        let strain = from + (to - from) * k as f64 / (points - 1) as f64;
        let added = mutate(rt, exp, st, actor, ids, cid, Mutation::Strain { strain: (strain * 1e6).round() / 1e6 }, None)?;
        if let Some(c) = added.added.first().or(added.duplicates.first()) {
            members.push((strain, c.clone()));
        }
    }
    // Cell held fixed (that is the strain); atoms may still move inside it.
    let params = json!({"fmax": 0.02, "steps": 300, "relax_cell": false, "optimizer": "FIRE"});
    let cids: Vec<String> = members.iter().map(|m| m.1.clone()).collect();
    let done = relax(rt, exp, actor, ids, &cids, ev, params, ui)?;
    let st = rt.state(exp)?;
    let rows: Vec<(f64, String, f64, f64)> = members
        .iter()
        .filter_map(|(s, c)| {
            let (e, o) = st.latest(c, PropertyKind::EnergyPerAtom, Some(ev))?;
            Some((*s, c.clone(), e, o.property(PropertyKind::Pressure).unwrap_or(f64::NAN)))
        })
        .collect();
    let emin = rows.iter().map(|r| r.2).fold(f64::INFINITY, f64::min);
    let mut text = format!("strain sweep of {cid} with {ev:?} (cell fixed, atoms relaxed)\n  strain    cand    E/atom (eV)   E-Emin (meV/atom)   pressure (GPa)\n");
    for (s, c, e, p) in &rows {
        text.push_str(&format!("  {:+6.2}%  {c:>6}  {e:>12.5}  {:>18.2}  {p:>15.2}\n", s * 100.0, (e - emin) * 1000.0));
    }
    text.push_str("OBSERVATION: the lowest-energy strain is where the model's preferred lattice size lies;\n  pressure crosses zero near the same point. Compare the curve shape between models.\n");
    Ok(Done { summary: format!("strain sweep {cid}: {}", done.summary), text: Some(text), candidates: cids, observations: done.observations })
}

fn compare(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, a: &str, b: &str) -> Result<Done> {
    let (ca, cb) = (st.candidate(a).unwrap(), st.candidate(b).unwrap());
    let mut text = format!("{a} {} ({:?}, {})  vs  {b} {} ({:?}, {})\n", ca.formula, ca.status, crate::events::origin_summary(&ca.origin), cb.formula, cb.status, crate::events::origin_summary(&cb.origin));
    if ca.kind == ArtifactKind::Structure && cb.kind == ArtifactKind::Structure {
        let (ha, hb) = (st.current_structure(a, None).unwrap(), st.current_structure(b, None).unwrap());
        let inputs = vec![serde_json::to_value(rt.store.structure(&ha)?)?, serde_json::to_value(rt.store.structure(&hb)?)?];
        let job = rt.run_job(exp, actor, ids, "structure", "compare", json!({}), inputs, vec![ha, hb], true, &mut |_| {})?;
        check(&job)?;
        let o = &job.items[0].output;
        text.push_str(&format!(
            "  same crystal (pymatgen StructureMatcher on current structures): {}{}\n",
            o["same_structure"],
            o["rms_displacement"].as_f64().map(|r| format!(", normalized RMS displacement {r:.4}")).unwrap_or_default()
        ));
        for (k, label) in [("a", a), ("b", b)] {
            let m = &o[k];
            text.push_str(&format!(
                "  {label}: {} sites, min distance {} Å, V/atom {} Å³, density {} g/cm³\n",
                m["num_sites"], m["min_distance_A"], m["volume_per_atom_A3"], m["density_g_cm3"]
            ));
        }
    }
    let kinds = [PropertyKind::EnergyPerAtom, PropertyKind::FormationEnergyPerAtom, PropertyKind::EAboveHull, PropertyKind::Pressure, PropertyKind::VolumePerAtom, PropertyKind::Logp, PropertyKind::Qed];
    for ev in [Some(Evaluator::Mattersim), Some(Evaluator::Mace), None] {
        for k in kinds {
            let (va, vb) = (st.latest(a, k, ev), st.latest(b, k, ev));
            if va.is_none() && vb.is_none() {
                continue;
            }
            let f = |v: Option<(f64, &crate::state::Observation)>| v.map(|(x, o)| format!("{x:.4} ({})", o.id)).unwrap_or_else(|| "—".into());
            text.push_str(&format!("  {:<26} {:>22} {:>22}  {}\n", format!("{k:?} [{}]", ev.map(|e| e.worker()).unwrap_or("any")), f(va), f(vb), k.unit()));
        }
    }
    text.push_str("OBSERVATION only: differences above are model outputs; energies per atom compare only at equal composition.\n");
    Ok(Done { summary: format!("compared {a} and {b}"), text: Some(text), ..Default::default() })
}

pub fn describe(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, cids: &[String]) -> Result<Done> {
    let (mols, xtals): (Vec<&String>, Vec<&String>) = cids.iter().partition(|c| st.candidate(c).unwrap().kind == ArtifactKind::Molecule);
    let mut events = vec![];
    let mut obs = vec![];
    let mut text = String::new();
    if !xtals.is_empty() {
        let hashes: Vec<Hash> = xtals.iter().map(|c| st.current_structure(c, None).unwrap()).collect();
        let inputs = hashes.iter().map(|h| Ok(serde_json::to_value(rt.store.structure(h)?)?)).collect::<Result<Vec<_>>>()?;
        let params = json!({"min_distance": 0.5, "allowed_elements": st.elements, "max_sites": st.max_sites});
        let job = rt.run_job(exp, actor, ids, "structure", "validate", params, inputs, hashes.clone(), true, &mut |_| {})?;
        check(&job)?;
        for item in job.ok_items() {
            let m = &item.output["metrics"];
            let cell = rt.store.structure(&hashes[item.index])?.cell_text();
            text.push_str(&format!(
                "  {} {}: {cell}; space group {} (#{}), {}, {} sites, min distance {} Å, density {} g/cm³\n",
                xtals[item.index], m["formula"].as_str().unwrap_or("?"), m["spacegroup"].as_str().unwrap_or("?"), m["spacegroup_number"], m["crystal_system"].as_str().unwrap_or("?"), m["num_sites"], m["min_distance_A"], m["density_g_cm3"]
            ));
        }
    }
    if !mols.is_empty() {
        let hashes: Vec<Hash> = mols.iter().map(|c| st.candidate(c).unwrap().artifact.clone()).collect();
        let smiles = hashes.iter().map(|h| Ok(Value::String(rt.store.molecule(h)?.smiles))).collect::<Result<Vec<_>>>()?;
        let job = rt.run_job(exp, actor, ids, "chem", "rdkit_descriptors", json!({}), smiles, hashes.clone(), true, &mut |_| {})?;
        check(&job)?;
        for item in job.ok_items() {
            let o = &item.output;
            let g = |k: &str| o[k].as_f64().unwrap_or(f64::NAN);
            let oid = ids.observation();
            events.push(Event::ObservationRecorded {
                id: oid.clone(),
                candidate: mols[item.index].clone(),
                job: job.job.clone(),
                method: job.method.clone(),
                fidelity: Fidelity::Descriptor,
                input: hashes[item.index].clone(),
                output: None,
                properties: vec![
                    Property::new(PropertyKind::MolWt, g("mol_wt_g_mol")),
                    Property::new(PropertyKind::Logp, g("logp")),
                    Property::new(PropertyKind::Tpsa, g("tpsa_A2")),
                    Property::new(PropertyKind::Qed, g("qed")),
                    Property::new(PropertyKind::FractionSp3, g("fraction_sp3")),
                    Property::new(PropertyKind::AromaticRings, g("aromatic_rings")),
                ],
                details: o.clone(),
            });
            obs.push(oid);
            text.push_str(&format!(
                "  {} {} {}: MolWt {:.1} g/mol, LogP {:.2}, TPSA {:.1} Å², rings {} (aromatic {}), sp3 fraction {:.2}, QED {:.2}\n",
                mols[item.index], o["formula"].as_str().unwrap_or("?"), o["smiles"].as_str().unwrap_or(""), g("mol_wt_g_mol"), g("logp"), g("tpsa_A2"), o["rings"], o["aromatic_rings"], g("fraction_sp3"), g("qed")
            ));
        }
    }
    rt.append(exp, actor, events)?;
    // The text goes into the summary too: the summary is what the planner sees next step.
    Ok(Done { summary: format!("described {}:\n{}", list(cids), text.trim_end()), text: None, observations: obs, ..Default::default() })
}

/// Formation energies and energy above the convex hull of THIS experiment's relaxed
/// structures, from one evaluator's energies (pymatgen PhaseDiagram in the structure worker).
pub fn compute_hull(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, ev: Evaluator) -> Result<Done> {
    let mut entries = vec![];
    let mut used: Vec<(String, Hash)> = vec![];
    for c in st.candidates.iter().filter(|c| c.kind == ArtifactKind::Structure && c.status != CandidateStatus::Rejected) {
        let relaxed = st.observations_of(&c.id).into_iter().rev().find(|o| o.method.op == "relax" && o.model() == Some(ev.worker()) && o.output.is_some());
        if let Some(o) = relaxed {
            let s = rt.store.structure(o.output.as_ref().unwrap())?;
            let e = o.property(PropertyKind::EnergyPerAtom).unwrap_or(f64::NAN);
            entries.push(json!({"id": c.id, "composition": s.composition(), "energy_eV": e * s.species.len() as f64}));
            used.push((c.id.clone(), o.output.clone().unwrap()));
        }
    }
    if entries.is_empty() {
        bail!("no structures relaxed with {ev:?} yet; relax some first");
    }
    let mut have: BTreeSet<String> = BTreeSet::new();
    let mut elemental: BTreeSet<String> = BTreeSet::new();
    for e in &entries {
        let comp = e["composition"].as_object().unwrap();
        have.extend(comp.keys().cloned());
        if comp.len() == 1 {
            elemental.extend(comp.keys().cloned());
        }
    }
    let missing: Vec<&String> = have.difference(&elemental).collect();
    if !missing.is_empty() {
        bail!(
            "formation energy needs an elemental reference phase for every element, relaxed with {ev:?}. Missing: {missing:?}.\n\
             A reference is the pure element in a stable form, e.g. `scientist build --prototype diamond --formula C` then relax it.\n\
             Some elements (B: alpha-boron, N: N2 molecular crystal) have complex ground states; a simpler stand-in gives a\n\
             different reference energy, which shifts every formation energy containing that element. Record which you used."
        );
    }
    let job = rt.run_job(exp, actor, ids, "structure", "hull", json!({"entries": entries}), vec![], used.iter().map(|u| u.1.clone()).collect(), true, &mut |_| {})?;
    check(&job)?;
    let mut method = job.method.clone();
    method.model = Some(ev.worker().to_string()); // energies came from this evaluator
    let mut events = vec![];
    let mut obs = vec![];
    let mut rows = vec![];
    for item in job.ok_items() {
        let (cid, h) = &used[item.index];
        let (ef, eh) = (item.output["formation_energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN), item.output["e_above_hull_eV"].as_f64().unwrap_or(f64::NAN));
        let oid = ids.observation();
        events.push(Event::ObservationRecorded {
            id: oid.clone(),
            candidate: cid.clone(),
            job: job.job.clone(),
            method: method.clone(),
            fidelity: Fidelity::MlPotential,
            input: h.clone(),
            output: None,
            properties: vec![Property::new(PropertyKind::FormationEnergyPerAtom, ef), Property::new(PropertyKind::EAboveHull, eh)],
            details: json!({"entries": entries.len(), "elements": have, "note": "internal hull: only phases relaxed in this experiment"}),
        });
        obs.push(oid);
        rows.push((cid.clone(), st.candidate(cid).unwrap().formula.clone(), ef, eh));
    }
    rt.append(exp, actor, events)?;
    rows.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
    let mut text = format!("convex hull over {} relaxed structure(s), {ev:?} energies, elements {have:?}\n  cand    formula      E_form (eV/atom)   E_above_hull (meV/atom)\n", rows.len());
    for (c, f, ef, eh) in &rows {
        text.push_str(&format!("  {c:>5}  {f:<10}  {ef:>16.4}  {:>22.1}\n", eh * 1000.0));
    }
    text.push_str("OBSERVATION: 0 meV/atom = on this experiment's hull. It is NOT the Materials Project hull: phases\n  absent from this experiment cannot compete, so 'on the hull' here can still be unstable in reality.\n");
    Ok(Done { summary: format!("hull computed for {} structures with {ev:?}", rows.len()), text: Some(text), observations: obs, ..Default::default() })
}

/// Known phases from Materials Project: their structures become candidates (so the same
/// model can relax them and they can compete on this experiment's hull), and the database's
/// own DFT values are recorded as a separate, clearly labelled observation.
fn lookup_known(rt: &mut Runtime, exp: &str, actor: Actor, ids: &mut crate::state::Ids, system: &str, max_hull: f64, limit: u32) -> Result<Done> {
    let params = json!({"chemical_system": system, "include_subsystems": true, "max_e_above_hull": max_hull, "limit": limit});
    let job = rt.run_job(exp, actor, ids, "mp", "search", params, vec![], vec![], true, &mut |_| {})?;
    check(&job)?;
    let mut list = vec![];
    for item in job.ok_items() {
        let o = &item.output;
        let s: Structure = serde_json::from_value(o["structure"].clone())?;
        let origin = Origin::Database {
            source: "materials_project".into(),
            id: o["material_id"].as_str().unwrap_or("?").into(),
            version: o["database_version"].as_str().unwrap_or("?").into(),
            url: o["url"].as_str().unwrap_or("").into(),
        };
        list.push((s, origin));
    }
    let added = add_structures(rt, exp, actor, ids, list, None)?;
    // Database values, attached to whichever candidate now holds each structure.
    let st = rt.state(exp)?;
    let mut events = vec![];
    let mut rows = vec![];
    for item in job.ok_items() {
        let o = &item.output;
        let s: Structure = serde_json::from_value(o["structure"].clone())?;
        let hash = sha256(&serde_json::to_vec(&s.canonical())?);
        let Some(cid) = st.candidate_by_artifact(&hash).map(String::from) else { continue };
        let g = |k: &str| o[k].as_f64().unwrap_or(f64::NAN);
        let mut method = job.method.clone();
        method.model = Some("materials_project".into());
        events.push(Event::ObservationRecorded {
            id: ids.observation(),
            candidate: cid.clone(),
            job: job.job.clone(),
            method,
            fidelity: Fidelity::Dft,
            input: hash,
            output: None,
            properties: vec![
                Property::new(PropertyKind::FormationEnergyPerAtom, g("formation_energy_per_atom_eV")),
                Property::new(PropertyKind::EAboveHull, g("e_above_hull_eV")),
                Property::new(PropertyKind::BandGap, g("band_gap_eV")),
            ],
            details: json!({"source": "Materials Project database, NOT computed here", "material_id": o["material_id"],
                "database_version": o["database_version"], "url": o["url"], "theoretical": o["theoretical"],
                "spacegroup": o["spacegroup"], "mp_energy_per_atom_eV": o["energy_per_atom_eV"],
                "note": "MP energies are on MP's own DFT scale; compare with MatterSim/MACE only after relaxing this structure with them"}),
        });
        rows.push(format!("  {cid:>5}  {:<11} {:<10} {:>6.1} meV  gap {:>5.2} eV  {}  {}",
            o["formula"].as_str().unwrap_or("?"), o["spacegroup"].as_str().unwrap_or("?"), g("e_above_hull_eV") * 1000.0,
            g("band_gap_eV"), o["material_id"].as_str().unwrap_or("?"),
            if o["theoretical"] == true { "theoretical" } else { "experimentally reported (ICSD)" }));
    }
    rt.append(exp, actor, events)?;
    let text = format!(
        "Materials Project, {system} and subsystems, ≤{:.0} meV/atom above MP's hull ({} found, limit {limit}):\n  cand   formula     spacegroup  E_hull(MP)   band gap(MP)  id\n{}\n\
         SOURCE: database DFT values (fidelity Dft, not computed here). To put these phases on THIS experiment's\n  hull, relax them with the same evaluator as your candidates, then run `scientist hull`.\n",
        max_hull * 1000.0, job.items.len(), rows.join("\n"));
    let mut d = added.done(format!("looked up {system} in Materials Project"));
    d.text = Some(text);
    Ok(d)
}

/// Literature search. Each paper gets an id (p1, p2, …) and its full record is stored as an
/// artifact. The summary the planner reads is labelled as external claims.
#[allow(clippy::too_many_arguments)]
fn search_literature(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, query: &str, limit: u32, from_year: Option<u32>) -> Result<Done> {
    let params = json!({"query": query, "limit": limit, "from_year": from_year, "sources": ["openalex", "arxiv"]});
    let job = rt.run_job(exp, actor, ids, "search", "papers", params, vec![], vec![], true, &mut |_| {})?;
    check(&job)?;
    let mut events = vec![];
    let mut lines = vec![];
    for item in &job.items {
        if !item.ok {
            lines.push(format!("  (source error: {})", item.error["message"].as_str().unwrap_or("?")));
            continue;
        }
        let o = &item.output;
        let (source, sid) = (o["source"].as_str().unwrap_or("?"), o["id"].as_str().unwrap_or("?"));
        let pid = match st.papers.iter().find(|p| p.source == source && p.source_id == sid) {
            Some(p) => p.id.clone(),
            None => {
                let bytes = serde_json::to_vec(o)?;
                let (hash, created) = rt.store.put_artifact(ArtifactKind::Record, &bytes)?;
                if created {
                    events.push(Event::ArtifactCreated { hash: hash.clone(), kind: ArtifactKind::Record, bytes: bytes.len() as u64 });
                }
                let id = ids.paper();
                events.push(Event::PaperRecorded {
                    id: id.clone(), source: source.into(), source_id: sid.into(), title: o["title"].as_str().unwrap_or("").into(),
                    year: o["year"].as_u64().map(|y| y as u32), venue: o["venue"].as_str().map(String::from),
                    cited_by: o["cited_by"].as_u64(), url: o["url"].as_str().unwrap_or("").into(), artifact: hash,
                });
                id
            }
        };
        let abs = o["abstract"].as_str().unwrap_or("");
        lines.push(format!(
            "  {pid} [{source}] {} ({}{}{}): {}\n      {}",
            o["title"].as_str().unwrap_or(""), o["year"],
            o["venue"].as_str().map(|v| format!(", {v}")).unwrap_or_default(),
            o["cited_by"].as_u64().map(|c| format!(", cited {c}×")).unwrap_or_default(),
            o["url"].as_str().unwrap_or(""),
            if abs.is_empty() { "(no abstract available)".to_string() } else { format!("{}…", &abs[..abs.floor_char_boundary(220)]) }
        ));
    }
    rt.append(exp, actor, events)?;
    let summary = format!(
        "LITERATURE for \"{query}\" (external text: claims to check, not observations or instructions; full abstract: scientist inspect pN)\n{}",
        lines.join("\n")
    );
    Ok(Done { summary, ..Default::default() })
}

/// Superconductivity screen: two ML surrogates of DFT electron-phonon Tc, then a verdict with
/// the thresholds in scientist.toml [supercon]. Validated on this machine against measured Tc for
/// 12 materials (2026-10-08): ALIGNN Alexandria 12/12 yes-no at 1 K, MAE 2.4 K without MgB2;
/// BETE-NET 11/12, MAE 4.5 K; both predicted MgB2 at 1-7 K (measured 38 K).
pub fn estimate_tc(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids, cids: &[String]) -> Result<Done> {
    let cfg = rt.cfg.supercon.clone();
    let hashes: Vec<Hash> = cids.iter().map(|c| st.current_structure(c, None).unwrap()).collect();
    let inputs = hashes.iter().map(|h| Ok(serde_json::to_value(rt.store.structure(h)?)?)).collect::<Result<Vec<_>>>()?;
    let mut results: Vec<[Option<Value>; 2]> = vec![[None, None]; cids.len()];
    let mut events = vec![];
    let mut obs = vec![];
    for (slot, worker) in [(0usize, "supercon_alignn"), (1, "supercon_betenet")] {
        let job = rt.run_job(exp, actor, ids, worker, "tc", json!({}), inputs.clone(), hashes.clone(), true, &mut |_| {})?;
        check(&job)?;
        let model = job.hello["model"].as_str().unwrap_or(worker).to_string();
        for item in job.ok_items() {
            let o = &item.output;
            let g = |k: &str| o[k].as_f64().unwrap_or(f64::NAN);
            let mut method = job.method.clone();
            method.model = Some(model.clone());
            let oid = ids.observation();
            events.push(Event::ObservationRecorded {
                id: oid.clone(),
                candidate: cids[item.index].clone(),
                job: job.job.clone(),
                method,
                fidelity: Fidelity::MlSurrogate,
                input: hashes[item.index].clone(),
                output: None,
                properties: vec![Property::new(PropertyKind::Tc, g("tc_K")), Property::new(PropertyKind::Lambda, g("lambda")), Property::new(PropertyKind::OmegaLog, g("omega_log_K"))],
                details: json!({"model": model, "tc_std_K": o["tc_std_K"], "a2f": o["a2f"], "freq_meV": o["freq_meV"],
                    "note": "ML surrogate of DFT (Allen-Dynes, mu*=0.09); not a measurement"}),
            });
            obs.push(oid);
            results[item.index][slot] = Some(o.clone());
        }
    }
    rt.append(exp, actor, events)?;
    let mut text = String::from("  cand    formula      ALIGNN Tc   BETE-NET Tc (±σ)   λ(A/B)       verdict\n");
    let mut lines = vec![];
    for (j, cid) in cids.iter().enumerate() {
        let a = results[j][0].as_ref().and_then(|o| o["tc_K"].as_f64()).unwrap_or(f64::NAN);
        let b = results[j][1].as_ref().and_then(|o| o["tc_K"].as_f64()).unwrap_or(f64::NAN);
        let bs = results[j][1].as_ref().and_then(|o| o["tc_std_K"].as_f64()).unwrap_or(f64::NAN);
        let la = results[j][0].as_ref().and_then(|o| o["lambda"].as_f64()).unwrap_or(f64::NAN);
        let lb = results[j][1].as_ref().and_then(|o| o["lambda"].as_f64()).unwrap_or(f64::NAN);
        let verdict = tc_verdict(a, b, bs, &cfg);
        let mut flags = vec![];
        if a.max(b) >= cfg.record_tc_k {
            flags.push(format!("ABOVE THE {:.0} K RECORD (MgB2): extrapolation, treat as a hypothesis for DFT", cfg.record_tc_k));
        } else if a.max(b) > cfg.training_max_tc_k {
            flags.push(format!("above training range (~{:.0} K): extrapolation", cfg.training_max_tc_k));
        }
        if (a - b).abs() > 5.0 {
            flags.push("models disagree by >5 K".into());
        }
        let formula = st.candidate(cid).map(|c| c.formula.clone()).unwrap_or_default();
        text.push_str(&format!("  {cid:>5}  {formula:<11} {a:>8.2} K   {b:>6.2} K (±{bs:>4.1})   {la:.2}/{lb:.2}    {verdict}{}\n",
            if flags.is_empty() { String::new() } else { format!(" — {}", flags.join("; ")) }));
        lines.push(format!("{cid} {formula}: ALIGNN {a:.1} K, BETE-NET {b:.1}±{bs:.1} K, λ {la:.2}/{lb:.2} → {verdict}{}", if flags.is_empty() { String::new() } else { format!(" [{}]", flags.join("; ")) }));
    }
    text.push_str("FIDELITY: ML surrogates of DFT electron-phonon Tc. Only a measurement (zero resistance AND\n  magnetic-field expulsion) shows superconductivity. Known blind spot: multi-gap superconductors (MgB2 predicted 1-7 K, measured 38 K).\n");
    text.push_str("RELIABILITY: On 216 superconductors neither model trained on (scientist benchmark --suite supercon_unseen, 2026-10-08): ranking by Tc is near random (Spearman 0.13); high-Tc materials are underestimated by ~10 K; 'possible' covers 45/60 measured non-superconductors, so it carries little information; LIKELY was right 9/9 but caught only 9/216.\n  Use LIKELY as a high-precision filter; do not rank candidates by predicted Tc.\n");
    Ok(Done { summary: format!("Tc estimates: {}", lines.join("; ")), text: Some(text), observations: obs, ..Default::default() })
}

/// Known superconductors from 3DSC: undoped parent structure from Materials Project + the median
/// measured Tc of SuperCon entries whose composition is within `max_mismatch` of that parent.
#[allow(clippy::too_many_arguments)]
fn import_superconductors(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut crate::state::Ids,
    elements: &[String], min_tc: Option<f64>, max_mismatch: Option<f64>, unconventional: bool, limit: u32) -> Result<Done> {
    let elements = if elements.is_empty() { st.elements.clone() } else { elements.to_vec() };
    let csv = crate::config::expand(&rt.cfg.supercon.data_3dsc);
    let params = json!({"csv": csv, "elements": elements, "min_tc": min_tc.unwrap_or(0.0), "max_mismatch": max_mismatch.unwrap_or(0.05),
                        "include_unconventional": unconventional, "limit": limit});
    let job = rt.run_job(exp, actor, ids, "mp", "known_superconductors", params, vec![], vec![], true, &mut |_| {})?;
    check(&job)?;
    let list = job.ok_items().map(|i| {
        let o = &i.output;
        let mid = o["material_id"].as_str().unwrap_or("?").to_string();
        Ok((serde_json::from_value(o["structure"].clone())?, Origin::Database {
            source: "3DSC (SuperCon + Materials Project)".into(), id: mid.clone(), version: "3DSC_MP".into(),
            url: format!("https://next-gen.materialsproject.org/materials/{mid}"),
        }))
    }).collect::<Result<Vec<_>>>()?;
    let added = add_structures(rt, exp, actor, ids, list, None)?;
    let st2 = rt.state(exp)?;
    let mut events = vec![];
    let mut rows = vec![];
    for item in job.ok_items() {
        let o = &item.output;
        let s: Structure = serde_json::from_value(o["structure"].clone())?;
        let hash = sha256(&serde_json::to_vec(&s.canonical())?);
        let Some(cid) = st2.candidate_by_artifact(&hash).map(String::from) else { continue };
        let mut method = job.method.clone();
        method.model = Some("measured (SuperCon via 3DSC)".into());
        events.push(Event::ObservationRecorded {
            id: ids.observation(), candidate: cid.clone(), job: job.job.clone(), method, fidelity: Fidelity::Experiment,
            input: hash, output: None, properties: vec![Property::new(PropertyKind::Tc, o["tc_median_K"].as_f64().unwrap_or(f64::NAN))],
            details: json!({"tc_min_K": o["tc_min_K"], "tc_max_K": o["tc_max_K"], "n_measurements": o["n_measurements"],
                "measured_formulas": o["measured_formulas"], "composition_mismatch_max": o["composition_mismatch_max"],
                "sc_class": o["sc_class"], "mp_e_above_hull_eV": o["e_above_hull_eV"], "source": o["source"],
                "note": "Tc measured on the listed compositions. 3DSC matched them to this Materials Project structure by COMPOSITION only: the measured sample may be a different polymorph (e.g. NbN: Tc belongs to cubic rock-salt, 3DSC pairs it with hexagonal mp-2634). Check the space group before trusting the pairing."}),
        });
        rows.push(format!("  {cid:>5}  {:<12} Tc {:>6.2} K (median of {}, range {:.1}–{:.1})  {}  {}",
            o["parent_formula"].as_str().unwrap_or("?"), o["tc_median_K"].as_f64().unwrap_or(0.0), o["n_measurements"],
            o["tc_min_K"].as_f64().unwrap_or(0.0), o["tc_max_K"].as_f64().unwrap_or(0.0), o["material_id"].as_str().unwrap_or(""),
            o["sc_class"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default()));
    }
    rt.append(exp, actor, events)?;
    let mut d = added.done(format!("imported {} measured superconductor(s)", rows.len()));
    d.text = Some(format!("Known superconductors (MEASURED Tc, fidelity Experiment; structure = undoped Materials Project parent):\n{}\nCAUTION: measurements are matched to structures by composition only; the measured sample can be a different\n  polymorph than the structure shown (check `scientist describe <id>` for the space group).\n", rows.join("\n")));
    Ok(d)
}

/// The one rule for labelling ML Tc estimates (thresholds from scientist.toml [supercon]).
/// a = ALIGNN Tc, b = BETE-NET mean Tc, bs = BETE-NET ensemble spread, all in K.
pub fn tc_verdict(a: f64, b: f64, bs: f64, cfg: &crate::config::SuperconCfg) -> &'static str {
    if a >= cfg.likely_tc_k && b >= cfg.likely_tc_k && b - bs >= cfg.possible_tc_k {
        "LIKELY (both models agree)"
    } else if a >= cfg.possible_tc_k || b >= cfg.possible_tc_k {
        "possible"
    } else {
        "unlikely"
    }
}
