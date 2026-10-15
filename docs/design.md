# scientist — design (V0)

A local computational instrument for materials and chemistry investigations on one
DGX Spark. The human is the investigator. A local LLM planner and numerical searchers
propose; a deterministic Rust runtime validates, executes, records, and explains.

This document answers section 49 of the plan (A–H) plus the human-operator addendum.
Every decision lists: problem, options, choice, reason, tradeoff, revisit-if.
Numbers marked (measured) come from `~/scientist-spike/results/` on this machine.

---

## 0. What you are looking at

```
 you ── terminal ──► scientist (one Rust process, CPU)
                        │  owns: experiment history, artifacts, budgets, GPU queue,
                        │        approvals, caching, provenance
                        │
          ┌─────────────┼──────────────────────────────┬──────────────────┐
          │ HTTP (localhost:8000)                       │ pipes (JSON lines)│
          ▼                                             ▼                   ▼
  vLLM server (Docker)                        Python worker processes, one per model
  Qwen3.6-35B-A3B "planner"                   potential:mattersim  (env-mattersim)
  proposes typed actions + explanations       potential:mace       (env-mace)
                                              mattergen            (env-mattergen)
                                              chem: chemfm + rdkit (env-chem)
                                              structure: pymatgen/ASE, CPU only
          │                                             │
          └──────────────── nothing persists here ──────┘
                        all state lives in ~/scientist-data/
                          scientist.db   (SQLite: append-only events, jobs, cache)
                          artifacts/     (content-addressed files: structures, molecules,
                                          trajectories), named by SHA-256 of their bytes
```

The loop the plan describes, made concrete:

```
S_t  = fold(events of experiment)            -- state is rebuilt from history
A_t  = policy(S_t)                           -- policy = you (a command), the planner,
                                                or a searcher (MatterGen, evolution, sweep)
       validate(A_t, S_t)                    -- ids exist? budget left? elements allowed?
       approve(A_t, autonomy level)          -- you say yes/no/edit, unless pre-authorized
O_t  = execute(A_t)                          -- worker job(s), cache checked first
S_t+1 = S_t + events(A_t, O_t)               -- persisted before anyone relies on them
```

Manual commands and planner actions are the **same Action type** and go through the
same validate → approve → execute → record path. Level 0 (manual) is not a separate
system; it is the same system with the planner switched off.

---

## A. Architecture decisions

### A1. One Rust binary, one crate, modules instead of many crates
- Problem: plan proposes ~8 crates. Each boundary costs build config and indirection.
- Options: workspace of 8 crates; one crate with modules.
- Choice: one crate `scientist`, ~12 modules (`src/*.rs`).
- Reason: one engineer must read it; no module is reused by another binary yet.
- Tradeoff: no compile-time enforcement of layer boundaries; enforced by convention.
- Revisit if: a second binary (daemon, replay tool) needs a subset, or build time hurts.

### A2. Synchronous runtime, no async executor
- Problem: plan asks to choose an async runtime.
- Options: tokio; std threads + blocking I/O.
- Choice: blocking I/O; one reader thread per worker pipe; the main loop is sequential.
- Reason: there is one GPU and it is the bottleneck. Running two GPU jobs at once made
  both slower (measured: planner decode fell from ~30 to 16–20 tok/s while MatterGen
  ran). A sequential executor makes that contention impossible by construction and
  every execution trace is a straight line you can read.
- Tradeoff: CPU work (validation, RDKit) does not overlap GPU work in V0.
- Revisit if: profiles show the GPU idle while CPU-side validation runs.

### A3. Planner stays an HTTP server (vLLM), not embedded
- Measured earlier: HTTP hop overhead 3–28 ms per call vs ~5 s per planner decision
  (measured median 4.9 s, reasoning off). Embedding saves <1%.
- vLLM gives prefix caching, structured outputs (schema-constrained decoding), and
  continuous batching; re-implementing these is out of scope (plan §34: no custom
  inference engine).
- Revisit if: a model is needed that vLLM cannot serve.

### A4. One Python process per model family, each in its own environment
- Problem: MatterGen needs numpy<2 and old pymatgen; MatterSim 1.2.5 needs numpy>=2
  (measured install conflict). Packages also fight over torch extensions.
- Choice: each worker is launched with its own interpreter
  (`~/scientist-spike/env-*/bin/python`), configured in `scientist.toml`.
- Reason: crash isolation (an OOM kills one worker, not the runtime), dependency
  isolation, and the model stays loaded between requests (residency).
- Revisit if: two families converge on one environment.

### A5. Experiment state = fold over an append-only event log
- History is never edited. State (candidates, properties, budgets, hypotheses) is
  rebuilt by replaying events in order. No derived tables to keep in sync.
- Experiments are small (thousands of events); a full replay takes milliseconds.
- Revisit if: replay time becomes visible (> ~100 ms), then add a snapshot table.

### A6. The searcher sits under the planner
- Plan §19: the LLM decides *what* to investigate; numerical searchers decide *where
  in the space* to sample. The planner emits `DefineSearch {space, strategy,
  objectives, budget}` and `RunSearch {rounds}`; the Rust search module then proposes
  candidates round by round without calling the LLM.
- Searchers implemented (`src/search.rs`):
  - `generative` — a **model is the searcher**: MatterGen samples crystals from a
    chemical system; ChemFM samples molecules.
  - `evolutionary` — Pareto-ranked parents, mutations (substitute, strain, rattle,
    swap) applied by the CPU structure worker, seeded and recorded.
  - `sweep` — one controlled variable (strain or substitution sites) for A-vs-B work.
  - `random` — baseline for comparing strategies (plan §48).
- The planner can also act as a searcher directly by issuing `Mutate` actions; this
  allows the LLM-only vs algorithm vs combined comparison of plan §48.

---

## B. Dependencies

| Need | Choice | Why not the alternative |
|---|---|---|
| CLI | `clap` (derive) | standard; generates help text you can read |
| Serialization | `serde`, `serde_json` | worker protocol and events are JSON (debuggable by eye) |
| Storage | `rusqlite` (bundled SQLite) | access pattern is small transactional appends + point lookups (OLTP). DuckDB is columnar analytics; it can attach the SQLite file later for analysis |
| Hashing | `sha2` | SHA-256 content addresses |
| HTTP to vLLM | `ureq` (blocking) | fits A2; no async stack |
| Config | `toml` | experiment files are TOML (plan §37) |
| REPL line editing | `rustyline` | history + editing in `scientist chat` |
| Ctrl-C | `ctrlc` | "stop after current job" |
| Random numbers | own SplitMix64 (20 lines) | a library's algorithm can change between versions and silently break replay; ours cannot |

Python side (already installed and measured on GB10, sm_121, torch 2.14.1+cu130):
MatterSim 1.2.5, MACE 0.3.16 (MACE-MP-0 medium), MatterGen 1.0.3 (pins overridden,
torch_scatter/sparse/cluster built from source), ChemFM-1B + RDKit 2026.03.6,
pymatgen + ASE (structure worker, inside env-mattersim).

DGX Spark constraints that shaped the design:
- CPU and GPU share 128 GB. The planner holds 64.69 GiB of weights + 13.96 GiB KV
  cache (measured, `gpu-memory-utilization 0.65`). Workers are small (measured peak
  CUDA allocation: MatterSim 0.2 GB, MACE 0.45 GB, ChemFM bf16 2–3.3 GB). MatterGen is
  the slow stage: ~29 s per structure at batch 64, 1000 diffusion steps (measured).
- Therefore: keep all workers resident when they fit the budget in `scientist.toml`;
  evict least-recently-used GPU workers when they do not.

---

## C. Data model (`src/types.rs`, `src/events.rs`)

```rust
// identity
type Hash = String;                      // "sha256:…" of canonical bytes
struct ArtifactRef { hash: Hash, kind: ArtifactKind }   // Structure | Molecule | Trajectory | Raw

// scientific objects
struct Structure {                       // periodic crystal, canonical form
    lattice: [[f64; 3]; 3],              // Å, rows are cell vectors
    species: Vec<String>,                // element symbol per site
    frac_coords: Vec<[f64; 3]>,          // fractional, wrapped to [0,1)
}                                        // sites sorted (species, coords) before hashing
struct Molecule { smiles: String, toolkit: String }    // canonical SMILES + RDKit version

struct Property {
    kind: PropertyKind,                  // EnergyPerAtom, FormationEnergyPerAtom, MaxForce,
                                         // Pressure, VolumePerAtom, Density, MolWt, LogP, …
    value: f64, unit: Unit,              // eV/atom, eV/Å, GPa, Å³/atom, g/cm³, g/mol, …
    uncertainty: Option<f64>,
    method: Method,                      // tool, model, checkpoint, versions, params hash
    fidelity: Fidelity,                  // Generated < StructuralCheck < MLPotential < DFT < Experiment
    structure: Hash,                     // the exact structure the number belongs to
    job: JobId,
}

struct Candidate {                       // one material/molecule idea, id c1, c2, …
    id, artifact: ArtifactRef, formula, origin: Origin,
    // Origin: Generated{tool,params} | Mutated{parent,mutation} | Built{prototype}
    //       | Imported{path} | Search{search,round}
    status: Proposed | Promoted{reason,by} | Rejected{reason,by},
    observations: Vec<ObservationId>,   // relaxations and single points, each with Properties
}

struct Budget { mattergen_structures, chemfm_molecules, relaxations, single_points,
                planner_calls }          // limits; consumption is derived from events

struct Job { id, tool, op, request, state: Queued|Running|Completed|Failed|Cancelled|
             UnknownAfterCrash, worker_pid, timings }
```

Events (append-only, `kind` + typed JSON payload, actor = human | planner | search | runtime):
`ExperimentCreated, ObjectiveSet, BudgetSet, AutonomySet, NoteAdded, HypothesisProposed,
HypothesisUpdated, PlannerProposed, ActionApproved, ActionRejected, JobQueued, JobStarted,
JobCompleted, JobFailed, JobUnknownAfterCrash, CandidateAdded, ObservationRecorded,
CandidateStatusChanged, ReferenceSet, SearchDefined, SearchRoundCompleted, SearchStopped,
InterpretationAdded, ExperimentForked`.

Observation vs interpretation (addendum §13) is a type distinction:
`ObservationRecorded` is only ever written by the runtime from tool output;
`InterpretationAdded` / `HypothesisProposed` carry an author (you or the planner).

---

## D. State machines

Experiment: `Active ⇄ Paused → Terminated` (Terminated via budget exhausted, stop rule,
or human). Forking creates a new experiment whose state = parent events up to the fork
point + its own events; parent artifacts are shared, not copied.

Job:
```
Queued ──► Running ──► Completed
   │          ├──────► Failed{failure}
   │          └──────► Cancelled           (Ctrl-C twice: worker killed)
   └──► Cancelled
Running (no terminal event, process restarted) ──► UnknownAfterCrash   (at startup)
```

Action (per planner proposal or command):
```
Proposed ─validate─► Invalid{errors}  (returned to the planner as a structured failure)
    └──────────────► Valid ─gate─► Rejected{by human} | Approved ─► Executed ─► Observed
```

Approval gate by autonomy level (addendum §3):

| level | planner | needs your approval |
|---|---|---|
| 0 manual | off | — (you type commands) |
| 1 assisted | proposes | every action that runs a tool |
| 2 search-assisted | proposes | defining a search, generation; search rounds and evaluations of an approved search run alone |
| 3 planner-assisted | proposes | only expensive actions (generation, new searches) |
| 4 autonomous | acts | nothing inside budget and stop rules |

Stopping (plan §40) is decided by the runtime, not the LLM: budget exhausted, search
patience (no Pareto improvement for N rounds), human stop, or planner `Terminate`
(which you confirm at levels ≤ 3).

---

## E. Worker protocol (`workers/common.py`, `src/worker.rs`)

Transport: the runtime spawns the worker with piped stdin/stdout/stderr. One JSON object
per line. stderr goes to `~/scientist-data/logs/<worker>.log`. At startup the worker moves
file descriptor 1 aside and points fd 1 at stderr, so library prints (tqdm, loguru) can
never corrupt the protocol stream.

```
worker → runtime   {"type":"hello","worker":"potential","model":"mattersim",
                    "versions":{...},"checkpoint":"…","load_s":2.4,"gpu_mb":203,"ops":[...]}
runtime → worker   {"type":"request","id":"j17","op":"relax",
                    "params":{"fmax":0.02,"steps":200},"inputs":[{structure},...]}
worker → runtime   {"type":"progress","id":"j17","item":0,"step":3,"energy":…,"fmax":…}
worker → runtime   {"type":"item","id":"j17","index":0,"ok":true,"output":{…}}   (per input)
worker → runtime   {"type":"done","id":"j17","wall_s":…,"gpu_mb":…}
                   {"type":"done","id":"j17","error":{"kind":"oom","message":…}}
```

Why per-item messages: a batch of 200 relaxations interrupted at item 137 keeps 137
recorded observations (plan §23). Why JSON lines: you can run any worker by hand and
type requests into it. Overhead is negligible next to model compute. Large data
(trajectories) still travel inline in V0 because they are small (kB); a staging-file
path is the planned upgrade when they are not.

Execution trace of `scientist generate --with mattergen --system B-C-N -n 8`:
1. CLI parses arguments into `Action::Generate{generator:MatterGen, system, n:8}`.
2. Runtime folds events → state; validates: elements allowed, budget ≥ 8 left.
3. Gate: you ran it yourself, so approved; `ActionApproved{by: human}` appended.
4. Cache key = sha256(tool, model version, checkpoint, params incl. seed); a hit would
   reuse the earlier job's outputs and skip to 11.
5. `JobQueued{j}` appended and committed to SQLite **before** anything runs.
6. Scheduler: is a `mattergen` worker resident? If not, check GPU budget, evict LRU
   GPU workers if needed, spawn `env-mattergen/bin/python workers/mattergen_worker.py`,
   wait for `hello` (model load time recorded). `JobStarted{j, pid}` appended.
7. Runtime writes one request line to the worker's stdin pipe.
8. Worker (Python, GPU): diffusion sampling, 1000 steps; progress lines stream back.
9. Worker converts each pymatgen Structure to the canonical JSON form, writes one
   `item` line per structure, then `done`.
10. Runtime canonicalizes again (sort sites, wrap coords, round), hashes, writes each
    artifact file (idempotent: same bytes → same path).
11. One SQLite transaction appends `ArtifactCreated…`, `CandidateAdded{c…}`,
    `JobCompleted{j, wall, gpu_mb}`. Only now does the state contain the candidates.
12. Terminal prints the new candidate ids, timings (queue, load, compute), cache status.

What crosses process boundaries: request/response text over pipes (copies through the
kernel pipe buffer), never GPU tensors. GPU memory is consumed only inside workers and
vLLM. What is persisted before failure: the queued job (5); after: artifacts + events (11).

---

## F. Failure model

| Failure | Detected by | Recorded as | Recovery |
|---|---|---|---|
| Python exception in an op | worker catches | `JobFailed{ToolError{kind,message}}` | worker stays up |
| CUDA out of memory | worker catches `OutOfMemoryError` | `JobFailed{Oom}` | worker restarted; you can reduce batch |
| Invalid / unphysical structure | structure worker validation | candidate `Rejected{reason}` + observation | data, not crash |
| Relaxation not converged | potential worker | observation `converged=false` | data; you decide |
| Worker crash / exit | EOF on stdout pipe | `JobFailed{WorkerCrashed{exit,stderr tail}}` | respawn on next use |
| Worker hang | per-op timeout in runtime | `JobFailed{Timeout}` | kill + respawn |
| Ctrl-C once / twice | signal handler | stop after current job / `Cancelled` | completed items kept |
| Runtime crash or reboot | `JobStarted` without terminal event at startup | `JobUnknownAfterCrash` | rerun is safe (cache + idempotent artifacts) |
| Planner unreachable | HTTP error | message to you | continue manually (level 0) |
| Planner proposes invalid action | validation | `ActionRejected{errors}` sent back to planner (≤2 retries) | — |
| Budget exhausted | validation | refusal with remaining budget | you raise budget (`BudgetSet`) |

---

## G. Repository layout

```
scientist/
  Cargo.toml
  scientist.toml           machine config: data dir, worker interpreters, planner URL, GPU budget
  src/
    main.rs                CLI commands → Actions
    config.rs              scientist.toml + experiment.toml
    types.rs               scientific objects (C)
    events.rs              event enum (C)
    store.rs               SQLite + content-addressed artifacts
    state.rs               fold(events) → ExperimentState
    worker.rs              spawn/hello/request/progress/timeout/evict
    tools.rs               tool registry, caching, job lifecycle
    actions.rs             Action enum, validation, approval gate, execution
    search.rs              searchers + Pareto front
    planner.rs             context builder, action schema, vLLM call
    learn.rs               --learn renderers, concept library lookup, notebook
    repl.rs                `scientist chat`
    rng.rs                 SplitMix64
  workers/
    common.py              protocol loop shared by all workers
    potential_worker.py    MatterSim or MACE: single_point, relax
    structure_worker.py    build, import, validate, mutate, compare, symmetry, export
    mattergen_worker.py    generate
    chem_worker.py         chemfm_generate, rdkit_validate, rdkit_descriptors
  concepts/*.md            short explanations used by --learn and /learn
  experiments/*.toml       example experiment definitions
  docs/design.md           this file
```

---

## H. Build sequence (each step ends in something you can run)

1. Store + events + state fold; `scientist new / list / status / note / inspect`.
2. Worker protocol + structure worker; `build`, `import`, `inspect`, `export`.
3. Potential workers; `relax`, `point` (single point) with `--learn`; caching; evidence.
4. Mutations + comparison; `mutate`, `compare`, `sweep`.
5. MatterGen worker; `generate`; GPU residency + eviction; `workers` view.
6. Chem worker; `molecules generate / describe`.
7. Searchers; `search define / run / show`, Pareto fronts, stop rules.
8. Planner; `chat` (the agent loop with approval gate) and `run` (autonomy ≥ 2).
9. Notebook, forking, crash reconciliation, references and formation energy.
