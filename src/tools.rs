//! The runtime: owns the store, the worker pool, and the job lifecycle.
//!
//! run_job is the one place where computation happens. Its order is the design's
//! execution trace (design.md, section E):
//!   1. job id handed out
//!   2. cache key = sha256(worker, op, params, input hashes, worker identity)
//!   3. JobQueued committed to SQLite   <- persisted before anything runs
//!   4. cache hit?  -> JobCompleted{cache_hit} and return the stored items
//!   5. worker started if not resident (model load timed) -> JobStarted
//!   6. request sent over the pipe; progress streamed to the terminal
//!   7. JobCompleted (and cache entry) or JobFailed
//! The caller then turns items into artifacts, candidates and observations.

use crate::config::Config;
use crate::events::{Actor, Event, Failure, FailureKind};
use crate::state::{ExperimentState, Ids, JobState};
use crate::store::Store;
use crate::types::{canonical_json, sha256, Hash, Method};
use crate::worker::{ItemOut, WorkerPool};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub struct Runtime {
    pub cfg: Config,
    pub store: Store,
    pub pool: WorkerPool,
    pub interrupt: Arc<AtomicU8>,
    locks: Vec<PathBuf>,
}

pub struct JobResult {
    pub job: String,
    pub items: Vec<ItemOut>,
    pub failure: Option<Failure>,
    pub cache_hit: bool,
    pub wall_s: f64,
    pub method: Method,
    pub hello: Value,
}

impl JobResult {
    pub fn ok_items(&self) -> impl Iterator<Item = &ItemOut> {
        self.items.iter().filter(|i| i.ok)
    }
}

/// Fields of a worker's hello that identify the computation (not pid, timings, memory).
fn identity(hello: &Value) -> Value {
    json!({
        "worker": hello["worker"], "model": hello["model"], "checkpoint": hello["checkpoint"],
        "checkpoint_sha256": hello["checkpoint_sha256"], "dtype": hello["dtype"],
        "versions": hello["versions"], "rdkit": hello["rdkit"],
    })
}

pub fn method_from(hello: &Value, worker: &str, op: &str, params: &Value) -> Method {
    let s = |k: &str| hello[k].as_str().map(String::from);
    Method {
        worker: worker.to_string(),
        op: op.to_string(),
        model: s("model"),
        checkpoint: s("checkpoint"),
        checkpoint_sha256: s("checkpoint_sha256"),
        dtype: s("dtype"),
        versions: hello["versions"].clone(),
        params: params.clone(),
    }
}

impl Runtime {
    pub fn open() -> Result<Runtime> {
        let cfg = Config::load()?;
        let store = Store::open(&cfg.data_dir())?;
        let interrupt = Arc::new(AtomicU8::new(0));
        let flag = interrupt.clone();
        // Ctrl-C while working interrupts at once (planner stream dropped, running worker
        // stopped, finished items kept). At the chat prompt rustyline handles Ctrl-C instead.
        let _ = ctrlc::set_handler(move || {
            if flag.fetch_add(1, Ordering::SeqCst) == 0 {
                eprintln!("\n  ⎿ Interrupted");
            }
        });
        let pool = WorkerPool::new(cfg.clone(), interrupt.clone());
        Ok(Runtime { cfg, store, pool, interrupt, locks: vec![] })
    }

    pub fn interrupt_flag(&self) -> &AtomicU8 {
        &self.interrupt
    }

    pub fn interrupted(&self) -> bool {
        self.interrupt.load(Ordering::SeqCst) > 0
    }

    pub fn clear_interrupt(&self) {
        self.interrupt.store(0, Ordering::SeqCst);
    }

    pub fn state(&self, exp: &str) -> Result<ExperimentState> {
        Ok(ExperimentState::fold(self.store.load_events(exp)?))
    }

    pub fn append(&mut self, exp: &str, actor: Actor, events: Vec<Event>) -> Result<()> {
        if !events.is_empty() {
            self.store.append(exp, actor, &events)?;
        }
        Ok(())
    }

    /// Take the write lock for an experiment and reconcile jobs left unfinished by a crash.
    /// Only one scientist process may write an experiment; readers (status, inspect) never lock.
    pub fn open_for_write(&mut self, exp: &str) -> Result<()> {
        let dir = self.cfg.data_dir().join("locks");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{exp}.lock"));
        if let Ok(pid) = std::fs::read_to_string(&path) {
            let pid = pid.trim();
            if pid != std::process::id().to_string() && std::path::Path::new(&format!("/proc/{pid}")).exists() {
                bail!("experiment {exp} is open in another scientist process (pid {pid}); finish or close it first");
            }
        }
        std::fs::write(&path, std::process::id().to_string())?;
        self.locks.push(path);
        // Crash reconciliation (design.md F): a job that started but never finished.
        let st = self.state(exp)?;
        let orphans: Vec<Event> = st
            .jobs
            .values()
            .filter(|j| matches!(j.state, JobState::Queued | JobState::Running))
            .map(|j| Event::JobUnknownAfterCrash { job: j.id.clone() })
            .collect();
        if !orphans.is_empty() {
            eprintln!("  reconciled {} job(s) left unfinished by an earlier crash", orphans.len());
            self.append(exp, Actor::Runtime, orphans)?;
        }
        Ok(())
    }

    /// Give up the write lock on an experiment (chat switching to another one).
    pub fn release(&mut self, exp: &str) {
        let path = self.cfg.data_dir().join("locks").join(format!("{exp}.lock"));
        if std::fs::read_to_string(&path).is_ok_and(|p| p.trim() == std::process::id().to_string()) {
            let _ = std::fs::remove_file(&path);
        }
        self.locks.retain(|l| *l != path);
    }

    fn known_identity(&mut self, worker: &str) -> Result<String> {
        if let Some((fp, _)) = self.store.worker_identity(worker)? {
            return Ok(fp);
        }
        // Never seen this worker: start it once to learn what it is.
        match self.pool.ensure(worker) {
            Ok(s) => {
                let fp = sha256(canonical_json(&identity(&s.hello)).as_bytes());
                self.store.set_worker_identity(worker, &fp, &s.hello)?;
                Ok(fp)
            }
            Err(f) => bail!("could not start {worker}: {}\n{}", f.message, f.detail.unwrap_or_default()),
        }
    }

    /// Run one op on one worker as a recorded job. See module docs for the exact order.
    #[allow(clippy::too_many_arguments)]
    pub fn run_job(
        &mut self,
        exp: &str,
        actor: Actor,
        ids: &mut Ids,
        worker: &str,
        op: &str,
        params: Value,
        inputs: Vec<Value>,
        input_hashes: Vec<Hash>,
        use_cache: bool,
        progress: &mut dyn FnMut(&Value),
    ) -> Result<JobResult> {
        let job = ids.job();
        let fp = self.known_identity(worker)?;
        let key = sha256(canonical_json(&json!({"worker": worker, "op": op, "params": params, "inputs": input_hashes, "identity": fp})).as_bytes());
        self.append(exp, actor, vec![Event::JobQueued {
            job: job.clone(),
            worker: worker.into(),
            op: op.into(),
            params: params.clone(),
            inputs: input_hashes.clone(),
            cache_key: key.clone(),
        }])?;

        if use_cache {
            if let Some(hit) = self.store.cache_get(&key)? {
                let items: Vec<ItemOut> = hit
                    .items
                    .iter()
                    .map(|v| ItemOut { index: v["index"].as_u64().unwrap_or(0) as usize, ok: v["ok"].as_bool().unwrap_or(false), output: v["output"].clone(), error: v["error"].clone() })
                    .collect();
                self.append(exp, actor, vec![Event::JobCompleted {
                    job: job.clone(),
                    wall_s: 0.0,
                    items: items.len() as u32,
                    gpu_mb: 0.0,
                    cache_hit: true,
                    details: json!({"reused_job": hit.job, "from_experiment": hit.experiment}),
                }])?;
                eprintln!("  {job}: cache hit — reusing {} result(s) from {} in experiment {} (no GPU time)", items.len(), hit.job, hit.experiment);
                let method = method_from(&hit.hello, worker, op, &params);
                return Ok(JobResult { job, items, failure: None, cache_hit: true, wall_s: 0.0, method, hello: hit.hello });
            }
        }

        let started = match self.pool.ensure(worker) {
            Ok(s) => s,
            Err(f) => {
                self.append(exp, actor, vec![Event::JobFailed { job: job.clone(), failure: f.clone() }])?;
                return Ok(JobResult { job, items: vec![], failure: Some(f), cache_hit: false, wall_s: 0.0, method: Method::default(), hello: Value::Null });
            }
        };
        for e in self.pool.evictions.drain(..) {
            eprintln!("  {e}");
        }
        if started.cold {
            let fp_now = sha256(canonical_json(&identity(&started.hello)).as_bytes());
            if fp_now != fp {
                self.store.set_worker_identity(worker, &fp_now, &started.hello)?;
            }
            eprintln!("  {job}: started {worker} (pid {}), model load {:.1} s", started.pid, started.load_s.unwrap_or(0.0));
        }
        self.append(exp, actor, vec![Event::JobStarted { job: job.clone(), pid: started.pid, cold_start: started.cold, load_s: started.load_s }])?;

        let mut out = self.pool.request(worker, &job, op, &params, &inputs, progress);
        out.items.sort_by_key(|i| i.index);
        let method = method_from(&started.hello, worker, op, &params);
        match &out.failure {
            None => {
                let cached: Vec<Value> = out.items.iter().map(|i| json!({"index": i.index, "ok": i.ok, "output": i.output, "error": i.error})).collect();
                let fp_now = sha256(canonical_json(&identity(&started.hello)).as_bytes());
                let key_now = sha256(canonical_json(&json!({"worker": worker, "op": op, "params": params, "inputs": input_hashes, "identity": fp_now})).as_bytes());
                if use_cache {
                    self.store.cache_put(&key_now, worker, op, &job, exp, &cached, &started.hello)?;
                }
                self.append(exp, actor, vec![Event::JobCompleted {
                    job: job.clone(),
                    wall_s: out.wall_s,
                    items: out.items.len() as u32,
                    gpu_mb: out.gpu_mb,
                    cache_hit: false,
                    details: json!({"cold_start": started.cold, "model_load_s": started.load_s}),
                }])?;
            }
            Some(f) => {
                self.append(exp, actor, vec![Event::JobFailed { job: job.clone(), failure: f.clone() }])?;
            }
        }
        Ok(JobResult { job, items: out.items, failure: out.failure, cache_hit: false, wall_s: out.wall_s, method, hello: started.hello })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.pool.stop_all();
        for l in &self.locks {
            let _ = std::fs::remove_file(l);
        }
    }
}

pub fn failure_text(f: &Failure) -> String {
    let mut s = format!("{:?}: {}", f.kind, f.message);
    if f.kind == FailureKind::WorkerCrashed || f.kind == FailureKind::SpawnFailed {
        if let Some(d) = &f.detail {
            let tail: Vec<&str> = d.lines().rev().take(6).collect();
            s.push_str("\n  last worker log lines:\n    ");
            s.push_str(&tail.into_iter().rev().collect::<Vec<_>>().join("\n    "));
        }
    }
    s
}
