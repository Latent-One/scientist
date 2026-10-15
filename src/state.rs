//! S_t = fold(events). The only way state is computed (design.md A5).
//!
//! Nothing here is persisted: every command reloads the experiment's events from SQLite
//! and folds them in order. A fork sees its parent's events up to the fork point first.

use crate::events::{Actor, Event, EventRecord, ExperimentStatus, FailureKind, Failure, HypothesisStatus};
use crate::search::SearchSpec;
use crate::types::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: String,
    pub artifact: Hash,
    pub kind: ArtifactKind,
    pub formula: String,
    pub origin: Origin,
    pub search: Option<String>,
    pub status: CandidateStatus,
    pub status_reason: String,
    pub observations: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub id: String,
    pub candidate: String,
    pub job: String,
    pub method: Method,
    pub fidelity: Fidelity,
    pub input: Hash,
    pub output: Option<Hash>,
    pub properties: Vec<Property>,
    pub details: Value,
}

impl Observation {
    pub fn property(&self, kind: PropertyKind) -> Option<f64> {
        self.properties.iter().find(|p| p.kind == kind).map(|p| p.value)
    }
    pub fn model(&self) -> Option<&str> {
        self.method.model.as_deref()
    }
}

#[derive(Clone, Debug)]
pub struct Hypothesis {
    pub id: String,
    pub statement: String,
    pub rationale: String,
    pub falsified_if: String,
    pub status: HypothesisStatus,
    pub reason: String,
    pub author: Actor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    UnknownAfterCrash,
}

#[derive(Clone, Debug)]
pub struct JobInfo {
    pub id: String,
    pub worker: String,
    pub op: String,
    pub state: JobState,
    pub wall_s: f64,
    pub items: u32,
    pub cache_hit: bool,
    pub failure: Option<Failure>,
}

#[derive(Clone, Debug)]
pub struct SearchState {
    pub spec: SearchSpec,
    pub rounds_done: u32,
    pub front: Vec<String>,
    pub rounds_without_improvement: u32,
    pub stopped: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Paper {
    pub id: String,
    pub source: String,
    pub source_id: String,
    pub title: String,
    pub year: Option<u32>,
    pub venue: Option<String>,
    pub cited_by: Option<u64>,
    pub url: String,
    pub artifact: Hash,
}

#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub mattergen_structures: u64,
    pub chemfm_molecules: u64,
    pub relaxations: u64,
    pub single_points: u64,
    pub planner_calls: u64,
}

#[derive(Clone, Debug)]
pub struct Note {
    pub seq: u64,
    pub author: Actor,
    pub text: String,
    pub refs: Vec<String>,
    pub interpretation: bool,
}

/// Next free id numbers. Commands copy this, hand out ids, and append events; the next
/// fold sees the new events. Only one process writes an experiment at a time.
#[derive(Clone, Debug, Default)]
pub struct Ids {
    pub candidate: u64,
    pub observation: u64,
    pub job: u64,
    pub hypothesis: u64,
    pub search: u64,
    pub paper: u64,
}

impl Ids {
    pub fn candidate(&mut self) -> String {
        self.candidate += 1;
        format!("c{}", self.candidate)
    }
    pub fn observation(&mut self) -> String {
        self.observation += 1;
        format!("o{}", self.observation)
    }
    pub fn job(&mut self) -> String {
        self.job += 1;
        format!("j{}", self.job)
    }
    pub fn hypothesis(&mut self) -> String {
        self.hypothesis += 1;
        format!("h{}", self.hypothesis)
    }
    pub fn search(&mut self) -> String {
        self.search += 1;
        format!("s{}", self.search)
    }
    pub fn paper(&mut self) -> String {
        self.paper += 1;
        format!("p{}", self.paper)
    }
}

fn num(id: &str) -> u64 {
    id.trim_start_matches(|c: char| c.is_ascii_alphabetic()).parse().unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct ExperimentState {
    pub name: String,
    pub objective: String,
    pub elements: Vec<String>,
    pub max_sites: u32,
    pub budget: Budget,
    pub autonomy: u8,
    pub status: ExperimentStatus,
    pub status_reason: String,
    pub candidates: Vec<Candidate>,
    index: HashMap<String, usize>,
    by_artifact: HashMap<Hash, String>,
    pub observations: Vec<Observation>,
    pub hypotheses: Vec<Hypothesis>,
    pub notes: Vec<Note>,
    pub references: BTreeMap<String, (String, Evaluator)>,
    pub searches: BTreeMap<String, SearchState>,
    pub papers: Vec<Paper>,
    pub jobs: BTreeMap<u64, JobInfo>,
    pub usage: Usage,
    pub ids: Ids,
    pub last_seq: u64,
    pub forked_from: Option<(String, u64)>,
}

impl ExperimentState {
    pub fn fold(records: &[EventRecord]) -> ExperimentState {
        let mut s = ExperimentState {
            name: String::new(),
            objective: String::new(),
            elements: vec![],
            max_sites: 40,
            budget: Budget::default(),
            autonomy: 1,
            status: ExperimentStatus::Active,
            status_reason: String::new(),
            candidates: vec![],
            index: HashMap::new(),
            by_artifact: HashMap::new(),
            observations: vec![],
            hypotheses: vec![],
            notes: vec![],
            references: BTreeMap::new(),
            searches: BTreeMap::new(),
            papers: vec![],
            jobs: BTreeMap::new(),
            usage: Usage::default(),
            ids: Ids::default(),
            last_seq: 0,
            forked_from: None,
        };
        for r in records {
            s.apply(r);
        }
        s
    }

    fn apply(&mut self, r: &EventRecord) {
        self.last_seq = r.seq;
        match &r.event {
            Event::ExperimentCreated { name, objective, elements, max_sites, budget, autonomy } => {
                // A fork's own ExperimentCreated renames the state; everything else carries over.
                self.name = name.clone();
                self.objective = objective.clone();
                self.elements = elements.clone();
                self.max_sites = *max_sites;
                self.budget = budget.clone();
                self.autonomy = *autonomy;
            }
            Event::ExperimentForked { parent, at_seq } => self.forked_from = Some((parent.clone(), *at_seq)),
            Event::ObjectiveSet { text } => self.objective = text.clone(),
            Event::ConstraintsSet { elements, max_sites } => {
                self.elements = elements.clone();
                self.max_sites = *max_sites;
            }
            Event::BudgetSet { budget } => self.budget = budget.clone(),
            Event::AutonomySet { level } => self.autonomy = *level,
            Event::StatusSet { status, reason } => {
                self.status = *status;
                self.status_reason = reason.clone();
            }
            Event::NoteAdded { text, refs } => {
                self.notes.push(Note { seq: r.seq, author: r.actor, text: text.clone(), refs: refs.clone(), interpretation: false })
            }
            Event::InterpretationAdded { text, refs } => {
                self.notes.push(Note { seq: r.seq, author: r.actor, text: text.clone(), refs: refs.clone(), interpretation: true })
            }
            Event::HypothesisProposed { id, statement, rationale, falsified_if } => {
                self.ids.hypothesis = self.ids.hypothesis.max(num(id));
                self.hypotheses.push(Hypothesis {
                    id: id.clone(),
                    statement: statement.clone(),
                    rationale: rationale.clone(),
                    falsified_if: falsified_if.clone(),
                    status: HypothesisStatus::Open,
                    reason: String::new(),
                    author: r.actor,
                });
            }
            Event::HypothesisUpdated { id, status, reason } => {
                if let Some(h) = self.hypotheses.iter_mut().find(|h| &h.id == id) {
                    h.status = *status;
                    h.reason = reason.clone();
                }
            }
            Event::PlannerProposed { .. } => self.usage.planner_calls += 1,
            Event::JobQueued { job, worker, op, .. } => {
                let n = num(job);
                self.ids.job = self.ids.job.max(n);
                self.jobs.insert(n, JobInfo {
                    id: job.clone(),
                    worker: worker.clone(),
                    op: op.clone(),
                    state: JobState::Queued,
                    wall_s: 0.0,
                    items: 0,
                    cache_hit: false,
                    failure: None,
                });
            }
            Event::JobStarted { job, .. } => {
                if let Some(j) = self.jobs.get_mut(&num(job)) {
                    j.state = JobState::Running;
                }
            }
            Event::JobCompleted { job, wall_s, items, cache_hit, .. } => {
                if let Some(j) = self.jobs.get_mut(&num(job)) {
                    j.state = JobState::Completed;
                    j.wall_s = *wall_s;
                    j.items = *items;
                    j.cache_hit = *cache_hit;
                    if !cache_hit {
                        // Budget is consumed by compute, not by cache reuse.
                        let n = *items as u64;
                        match (j.worker.as_str(), j.op.as_str()) {
                            ("mattergen", "generate") => self.usage.mattergen_structures += n,
                            ("chem", "chemfm_generate") => self.usage.chemfm_molecules += n,
                            (_, "relax") => self.usage.relaxations += n,
                            (_, "single_point") => self.usage.single_points += n,
                            _ => {}
                        }
                    }
                }
            }
            Event::JobFailed { job, failure } => {
                if let Some(j) = self.jobs.get_mut(&num(job)) {
                    j.state = if failure.kind == FailureKind::Cancelled { JobState::Cancelled } else { JobState::Failed };
                    j.failure = Some(failure.clone());
                }
            }
            Event::JobUnknownAfterCrash { job } => {
                if let Some(j) = self.jobs.get_mut(&num(job)) {
                    j.state = JobState::UnknownAfterCrash;
                }
            }
            Event::CandidateAdded { id, artifact, kind, formula, origin, search, .. } => {
                self.ids.candidate = self.ids.candidate.max(num(id));
                self.index.insert(id.clone(), self.candidates.len());
                self.by_artifact.entry(artifact.clone()).or_insert_with(|| id.clone());
                self.candidates.push(Candidate {
                    id: id.clone(),
                    artifact: artifact.clone(),
                    kind: *kind,
                    formula: formula.clone(),
                    origin: origin.clone(),
                    search: search.clone(),
                    status: CandidateStatus::Proposed,
                    status_reason: String::new(),
                    observations: vec![],
                });
            }
            Event::ObservationRecorded { id, candidate, job, method, fidelity, input, output, properties, details } => {
                self.ids.observation = self.ids.observation.max(num(id));
                let idx = self.observations.len();
                self.observations.push(Observation {
                    id: id.clone(),
                    candidate: candidate.clone(),
                    job: job.clone(),
                    method: method.clone(),
                    fidelity: *fidelity,
                    input: input.clone(),
                    output: output.clone(),
                    properties: properties.clone(),
                    details: details.clone(),
                });
                if let Some(&ci) = self.index.get(candidate) {
                    self.candidates[ci].observations.push(idx);
                }
            }
            Event::CandidateStatusChanged { candidate, status, reason } => {
                if let Some(&ci) = self.index.get(candidate) {
                    self.candidates[ci].status = *status;
                    self.candidates[ci].status_reason = reason.clone();
                }
            }
            Event::ReferenceSet { element, candidate, evaluator } => {
                self.references.insert(element.clone(), (candidate.clone(), *evaluator));
            }
            Event::SearchDefined { spec } => {
                self.ids.search = self.ids.search.max(num(&spec.id));
                self.searches.insert(spec.id.clone(), SearchState {
                    spec: spec.clone(),
                    rounds_done: 0,
                    front: vec![],
                    rounds_without_improvement: 0,
                    stopped: None,
                });
            }
            Event::SearchRoundCompleted { search, round, front, improved, .. } => {
                if let Some(s) = self.searches.get_mut(search) {
                    s.rounds_done = *round;
                    s.front = front.clone();
                    s.rounds_without_improvement = if *improved { 0 } else { s.rounds_without_improvement + 1 };
                }
            }
            Event::SearchStopped { search, reason } => {
                if let Some(s) = self.searches.get_mut(search) {
                    s.stopped = Some(reason.clone());
                }
            }
            Event::PaperRecorded { id, source, source_id, title, year, venue, cited_by, url, artifact } => {
                self.ids.paper = self.ids.paper.max(num(id));
                self.papers.push(Paper { id: id.clone(), source: source.clone(), source_id: source_id.clone(), title: title.clone(), year: *year, venue: venue.clone(), cited_by: *cited_by, url: url.clone(), artifact: artifact.clone() });
            }
            Event::Interrupted { .. } => {}
            Event::UserMessage { .. }
            | Event::ActionApproved { .. }
            | Event::ActionRejected { .. }
            | Event::ActionCompleted { .. }
            | Event::ActionFailed { .. }
            | Event::ArtifactCreated { .. } => {}
        }
    }

    pub fn candidate(&self, id: &str) -> Option<&Candidate> {
        self.index.get(id).map(|&i| &self.candidates[i])
    }

    pub fn candidate_by_artifact(&self, hash: &str) -> Option<&str> {
        self.by_artifact.get(hash).map(|s| s.as_str())
    }

    pub fn observations_of(&self, cid: &str) -> Vec<&Observation> {
        self.candidate(cid).map(|c| c.observations.iter().map(|&i| &self.observations[i]).collect()).unwrap_or_default()
    }

    /// The structure to use for further work on a candidate: the most recent relaxed
    /// output if one exists (optionally from a given evaluator), else the original.
    pub fn current_structure(&self, cid: &str, evaluator: Option<Evaluator>) -> Option<Hash> {
        let c = self.candidate(cid)?;
        let relaxed = self
            .observations_of(cid)
            .into_iter()
            .rev()
            .find(|o| o.method.op == "relax" && o.output.is_some() && evaluator.is_none_or(|e| o.model() == Some(e.worker())));
        Some(relaxed.and_then(|o| o.output.clone()).unwrap_or_else(|| c.artifact.clone()))
    }

    /// Latest value of a property for a candidate, optionally from one evaluator model.
    pub fn latest(&self, cid: &str, kind: PropertyKind, evaluator: Option<Evaluator>) -> Option<(f64, &Observation)> {
        self.observations_of(cid)
            .into_iter()
            .rev()
            .filter(|o| evaluator.is_none_or(|e| o.model() == Some(e.worker())))
            .find_map(|o| o.property(kind).map(|v| (v, o)))
    }

    /// Latest value of a property recorded by one named model (e.g. "alex_supercon_Tc").
    pub fn latest_model(&self, cid: &str, kind: PropertyKind, model: &str) -> Option<(f64, &Observation)> {
        self.observations_of(cid).into_iter().rev().filter(|o| o.model() == Some(model)).find_map(|o| o.property(kind).map(|v| (v, o)))
    }

    pub fn remaining(&self) -> Usage {
        let (b, u) = (&self.budget, &self.usage);
        Usage {
            mattergen_structures: b.mattergen_structures.saturating_sub(u.mattergen_structures),
            chemfm_molecules: b.chemfm_molecules.saturating_sub(u.chemfm_molecules),
            relaxations: b.relaxations.saturating_sub(u.relaxations),
            single_points: b.single_points.saturating_sub(u.single_points),
            planner_calls: b.planner_calls.saturating_sub(u.planner_calls),
        }
    }
}
