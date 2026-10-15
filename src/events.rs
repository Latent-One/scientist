//! The append-only event log (design.md, sections A5 and C).
//!
//! Every change to an experiment is one of these variants, serialized as JSON into the
//! `events` table. Nothing is ever updated or deleted. `state.rs` rebuilds the current
//! state by folding over them in sequence order.
//!
//! Observation vs interpretation is a type distinction: `ObservationRecorded` is only
//! written by the runtime from tool output; `InterpretationAdded`, `HypothesisProposed`
//! and `NoteAdded` always carry a human or planner author (the `actor` column).

use crate::actions::Action;
use crate::search::SearchSpec;
use crate::types::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Actor {
    Human,
    Planner,
    Search,
    Runtime,
}

impl Actor {
    pub fn as_str(self) -> &'static str {
        match self {
            Actor::Human => "human",
            Actor::Planner => "planner",
            Actor::Search => "search",
            Actor::Runtime => "runtime",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Active,
    Paused,
    Terminated,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    Open,
    Supported,
    Weakened,
    Rejected,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    ToolError,
    Oom,
    Timeout,
    WorkerCrashed,
    Cancelled,
    InvalidOutput,
    SpawnFailed,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    ExperimentCreated { name: String, objective: String, elements: Vec<String>, max_sites: u32, budget: Budget, autonomy: u8 },
    ObjectiveSet { text: String },
    ConstraintsSet { elements: Vec<String>, max_sites: u32 },
    BudgetSet { budget: Budget },
    AutonomySet { level: u8 },
    StatusSet { status: ExperimentStatus, reason: String },
    UserMessage { text: String },
    NoteAdded { text: String, refs: Vec<String> },
    HypothesisProposed { id: String, statement: String, rationale: String, falsified_if: String },
    HypothesisUpdated { id: String, status: HypothesisStatus, reason: String },
    InterpretationAdded { text: String, refs: Vec<String> },
    PlannerProposed { output: Value, model: String, prompt_tokens: u64, completion_tokens: u64, latency_ms: u64, valid: bool, errors: Vec<String> },
    ActionApproved { action: Action, how: String },
    ActionRejected { action: Value, reason: String },
    ActionCompleted { action: Action, summary: String },
    ActionFailed { action: Action, error: String },
    JobQueued { job: String, worker: String, op: String, params: Value, inputs: Vec<Hash>, cache_key: String },
    JobStarted { job: String, pid: u32, cold_start: bool, load_s: Option<f64> },
    JobCompleted { job: String, wall_s: f64, items: u32, gpu_mb: f64, cache_hit: bool, details: Value },
    JobFailed { job: String, failure: Failure },
    JobUnknownAfterCrash { job: String },
    ArtifactCreated { hash: Hash, kind: ArtifactKind, bytes: u64 },
    CandidateAdded { id: String, artifact: Hash, kind: ArtifactKind, formula: String, origin: Origin, job: Option<String>, search: Option<String> },
    ObservationRecorded { id: String, candidate: String, job: String, method: Method, fidelity: Fidelity, input: Hash, output: Option<Hash>, properties: Vec<Property>, details: Value },
    CandidateStatusChanged { candidate: String, status: CandidateStatus, reason: String },
    ReferenceSet { element: String, candidate: String, evaluator: Evaluator },
    SearchDefined { spec: SearchSpec },
    SearchRoundCompleted { search: String, round: u32, proposed: Vec<String>, rejected: Vec<String>, duplicates: u32, evaluated: Vec<String>, front: Vec<String>, improved: bool },
    SearchStopped { search: String, reason: String },
    ExperimentForked { parent: String, at_seq: u64 },
    /// You pressed Ctrl-C while the agent was working.
    Interrupted { during: String },
    /// A paper found by literature search; full record (abstract etc.) in `artifact`.
    PaperRecorded { id: String, source: String, source_id: String, title: String, year: Option<u32>, venue: Option<String>, cited_by: Option<u64>, url: String, artifact: Hash },
}

#[derive(Clone, Debug)]
pub struct EventRecord {
    pub seq: u64,
    pub ts_ms: u64,
    pub actor: Actor,
    pub event: Event,
}

impl Event {
    pub fn kind(&self) -> &'static str {
        use Event::*;
        match self {
            ExperimentCreated { .. } => "experiment_created",
            ObjectiveSet { .. } => "objective_set",
            ConstraintsSet { .. } => "constraints_set",
            BudgetSet { .. } => "budget_set",
            AutonomySet { .. } => "autonomy_set",
            StatusSet { .. } => "status_set",
            UserMessage { .. } => "user_message",
            NoteAdded { .. } => "note_added",
            HypothesisProposed { .. } => "hypothesis_proposed",
            HypothesisUpdated { .. } => "hypothesis_updated",
            InterpretationAdded { .. } => "interpretation_added",
            PlannerProposed { .. } => "planner_proposed",
            ActionApproved { .. } => "action_approved",
            ActionRejected { .. } => "action_rejected",
            ActionCompleted { .. } => "action_completed",
            ActionFailed { .. } => "action_failed",
            JobQueued { .. } => "job_queued",
            JobStarted { .. } => "job_started",
            JobCompleted { .. } => "job_completed",
            JobFailed { .. } => "job_failed",
            JobUnknownAfterCrash { .. } => "job_unknown_after_crash",
            ArtifactCreated { .. } => "artifact_created",
            CandidateAdded { .. } => "candidate_added",
            ObservationRecorded { .. } => "observation_recorded",
            CandidateStatusChanged { .. } => "candidate_status_changed",
            ReferenceSet { .. } => "reference_set",
            SearchDefined { .. } => "search_defined",
            SearchRoundCompleted { .. } => "search_round_completed",
            SearchStopped { .. } => "search_stopped",
            ExperimentForked { .. } => "experiment_forked",
            PaperRecorded { .. } => "paper_recorded",
            Interrupted { .. } => "interrupted",
        }
    }

    /// One line for `scientist log`, the notebook, and the planner's recent-events window.
    pub fn summary(&self) -> String {
        use Event::*;
        match self {
            ExperimentCreated { name, objective, elements, .. } => format!("experiment {name} created; elements {elements:?}; objective: {objective}"),
            ObjectiveSet { text } => format!("objective: {text}"),
            ConstraintsSet { elements, max_sites } => format!("constraints: elements {elements:?}, max {max_sites} sites"),
            BudgetSet { budget } => format!("budget set: {}", serde_json::to_string(budget).unwrap_or_default()),
            AutonomySet { level } => format!("autonomy level {level}"),
            StatusSet { status, reason } => format!("status {status:?}: {reason}"),
            UserMessage { text } => format!("you said: {text}"),
            NoteAdded { text, refs } => format!("note {refs:?}: {text}"),
            HypothesisProposed { id, statement, .. } => format!("hypothesis {id}: {statement}"),
            HypothesisUpdated { id, status, reason } => format!("hypothesis {id} -> {status:?}: {reason}"),
            InterpretationAdded { text, refs } => format!("interpretation {refs:?}: {text}"),
            PlannerProposed { output, valid, .. } => format!(
                "planner proposed {} ({})",
                output.pointer("/action/type").and_then(|v| v.as_str()).unwrap_or("?"),
                if *valid { "valid" } else { "invalid" }
            ),
            ActionApproved { action, how } => format!("approved [{how}]: {}", action.describe()),
            ActionRejected { reason, .. } => format!("action rejected: {reason}"),
            ActionCompleted { summary, .. } => format!("done: {summary}"),
            ActionFailed { action, error } => format!("failed: {} — {error}", action.describe()),
            JobQueued { job, worker, op, inputs, .. } => format!("{job} queued: {worker}.{op} on {} input(s)", inputs.len()),
            JobStarted { job, pid, cold_start, load_s } => format!(
                "{job} started on pid {pid}{}",
                if *cold_start { format!(" (cold start, model load {:.1} s)", load_s.unwrap_or(0.0)) } else { " (model already resident)".into() }
            ),
            JobCompleted { job, wall_s, items, gpu_mb, cache_hit, .. } => {
                if *cache_hit { format!("{job} cache hit: {items} item(s) reused, no compute") } else { format!("{job} completed: {items} item(s) in {wall_s:.1} s, GPU peak {gpu_mb:.0} MB") }
            }
            JobFailed { job, failure } => format!("{job} failed ({:?}): {}", failure.kind, failure.message),
            JobUnknownAfterCrash { job } => format!("{job} was running when the runtime stopped; outcome unknown"),
            ArtifactCreated { hash, kind, bytes } => format!("artifact {} ({kind:?}, {bytes} bytes)", short(hash)),
            CandidateAdded { id, formula, origin, search, .. } => format!(
                "{id} added: {formula} from {}{}",
                origin_summary(origin),
                search.as_ref().map(|s| format!(" (search {s})")).unwrap_or_default()
            ),
            ObservationRecorded { id, candidate, method, properties, details, .. } => {
                let props: Vec<String> = properties.iter().map(|p| format!("{:?}={:.4} {}", p.kind, p.value, p.unit)).collect();
                let conv = details.get("converged").and_then(|c| c.as_bool()).map(|c| if c { " converged" } else { " NOT converged" }).unwrap_or("");
                format!("{id} on {candidate}: {}.{}{conv} {}", method.worker, method.op, props.join(", "))
            }
            CandidateStatusChanged { candidate, status, reason } => format!("{candidate} -> {status:?}: {reason}"),
            ReferenceSet { element, candidate, evaluator } => format!("{element} reference = {candidate} ({evaluator:?})"),
            SearchDefined { spec } => format!("search {} defined: {}", spec.id, spec.describe()),
            SearchRoundCompleted { search, round, proposed, rejected, duplicates, evaluated, front, improved } => format!(
                "search {search} round {round}: {} proposed, {} rejected, {duplicates} duplicate, {} evaluated; front {front:?}{}",
                proposed.len(), rejected.len(), evaluated.len(), if *improved { " (improved)" } else { "" }
            ),
            SearchStopped { search, reason } => format!("search {search} stopped: {reason}"),
            ExperimentForked { parent, at_seq } => format!("forked from {parent} at event {at_seq}"),
            Interrupted { during } => format!("interrupted by you (Ctrl-C) during {during}"),
            PaperRecorded { id, title, year, source, .. } => format!("{id} found ({source}, {}): {title}", year.map(|y| y.to_string()).unwrap_or("?".into())),
        }
    }
}

pub fn origin_summary(o: &Origin) -> String {
    match o {
        Origin::Built { prototype, params } => {
            let mut s = format!("prototype {prototype}");
            if let Some([n, m]) = params["chirality"].as_array().map(|a| [a.first().and_then(|x| x.as_u64()), a.get(1).and_then(|x| x.as_u64())]) {
                s.push_str(&format!(" ({},{})", n.unwrap_or(0), m.unwrap_or(0)));
            }
            if let Some(a) = params["a"].as_f64() {
                s.push_str(&format!(" a={a}"));
            }
            s
        }
        Origin::Imported { path } => format!("file {path}"),
        Origin::Generated { generator, .. } => format!("{generator:?} generation"),
        Origin::Mutated { parent, mutation } => format!("{parent} by {}", mutation.describe()),
        Origin::Database { source, id, .. } => format!("{source} {id}"),
    }
}
