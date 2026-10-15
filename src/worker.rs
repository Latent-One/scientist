//! Runtime side of the worker protocol (design.md, section E).
//!
//! Each worker is a child process: `<python> <script> <args>` with stdin/stdout piped and
//! stderr appended to ~/scientist-data/logs/<name>.log. A reader thread turns stdout lines
//! into messages on a channel; the main thread waits on that channel with a timeout, so
//! a hung worker cannot hang the runtime.
//!
//! Workers are started in their own process group. Pressing Ctrl-C in the terminal sends
//! SIGINT to the foreground group (the runtime), not to the workers; the runtime then
//! stops the running request at once (worker killed, finished items kept), like Claude Code.
//!
//! Residency: a started worker stays alive for the life of this process (all of
//! `scientist chat`, one `search run`, or one command). Before starting a GPU worker the
//! pool checks the GPU budget and stops the least-recently-used GPU worker if needed.

use crate::config::Config;
use crate::events::{Failure, FailureKind};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

enum Line {
    Msg(Value),
    Bad(String),
    Eof,
}

struct Running {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Line>,
    hello: Value,
    started: Instant,
    last_used: Instant,
    requests: u64,
    peak_gpu_mb: f64,
    load_s: f64,
}

pub struct ItemOut {
    pub index: usize,
    pub ok: bool,
    pub output: Value,
    pub error: Value,
}

pub struct Outcome {
    pub items: Vec<ItemOut>,
    pub wall_s: f64,
    pub gpu_mb: f64,
    pub failure: Option<Failure>,
}

pub struct Started {
    pub cold: bool,
    pub load_s: Option<f64>,
    pub pid: u32,
    pub hello: Value,
}

pub struct WorkerStatus {
    pub name: String,
    pub pid: u32,
    pub model: String,
    pub load_s: f64,
    pub peak_gpu_mb: f64,
    pub requests: u64,
    pub idle_s: f64,
    pub uptime_s: f64,
    pub gpu: bool,
}

pub struct WorkerPool {
    cfg: Config,
    running: BTreeMap<String, Running>,
    interrupt: Arc<AtomicU8>,
    log_dir: PathBuf,
    pub evictions: Vec<String>,
}

fn fail(kind: FailureKind, message: impl Into<String>, detail: Option<String>) -> Failure {
    Failure { kind, message: message.into(), detail }
}

impl WorkerPool {
    pub fn new(cfg: Config, interrupt: Arc<AtomicU8>) -> WorkerPool {
        let log_dir = cfg.data_dir().join("logs");
        WorkerPool { cfg, running: BTreeMap::new(), interrupt, log_dir, evictions: vec![] }
    }

    fn log_path(&self, name: &str) -> PathBuf {
        self.log_dir.join(format!("{name}.log"))
    }

    fn stderr_tail(&self, name: &str) -> String {
        let mut f = match std::fs::File::open(self.log_path(name)) {
            Ok(f) => f,
            Err(_) => return String::new(),
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        let _ = f.seek(SeekFrom::Start(len.saturating_sub(2000)));
        let mut s = String::new();
        let _ = f.read_to_string(&mut s);
        s
    }

    fn gpu_in_use(&self, except: &str) -> f64 {
        self.running
            .iter()
            .filter(|(n, _)| n.as_str() != except && self.cfg.workers.get(n.as_str()).is_some_and(|w| w.gpu))
            .map(|(n, r)| r.peak_gpu_mb.max(self.cfg.workers[n.as_str()].est_gpu_mb))
            .sum()
    }

    /// Make sure the named worker is running; start it (after evicting others) if not.
    pub fn ensure(&mut self, name: &str) -> Result<Started, Failure> {
        if let Some(r) = self.running.get_mut(name) {
            if matches!(r.child.try_wait(), Ok(None)) {
                return Ok(Started { cold: false, load_s: None, pid: r.child.id(), hello: r.hello.clone() });
            }
            self.running.remove(name);
        }
        let wcfg = self
            .cfg
            .workers
            .get(name)
            .cloned()
            .ok_or_else(|| fail(FailureKind::SpawnFailed, format!("no worker named {name:?} in scientist.toml"), None))?;
        if wcfg.gpu {
            while self.gpu_in_use(name) + wcfg.est_gpu_mb > self.cfg.gpu.budget_mb {
                let lru = self
                    .running
                    .iter()
                    .filter(|(n, _)| self.cfg.workers.get(n.as_str()).is_some_and(|w| w.gpu))
                    .min_by_key(|(_, r)| r.last_used)
                    .map(|(n, _)| n.clone());
                match lru {
                    Some(n) => {
                        self.evictions.push(format!("stopped {n} to free GPU memory for {name}"));
                        self.stop(&n);
                    }
                    None => break,
                }
            }
        }
        let t0 = Instant::now();
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path(name))
            .map_err(|e| fail(FailureKind::SpawnFailed, format!("opening log: {e}"), None))?;
        let python = crate::config::expand(&wcfg.python);
        let mut child = Command::new(&python)
            .arg(self.cfg.script(&wcfg))
            .args(&wcfg.args)
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .env("HF_XET_CACHE", crate::config::expand("~/.cache/hf-xet-user"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .process_group(0)
            .spawn()
            .map_err(|e| fail(FailureKind::SpawnFailed, format!("starting {}: {e}", python.display()), None))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let msg = match line {
                    Ok(l) => match serde_json::from_str::<Value>(&l) {
                        Ok(v) => Line::Msg(v),
                        Err(_) => Line::Bad(l),
                    },
                    Err(_) => break,
                };
                if tx.send(msg).is_err() {
                    return;
                }
            }
            let _ = tx.send(Line::Eof);
        });
        let pid = child.id();
        let mut r = Running {
            child,
            stdin,
            rx,
            hello: Value::Null,
            started: t0,
            last_used: Instant::now(),
            requests: 0,
            peak_gpu_mb: 0.0,
            load_s: 0.0,
        };
        // Wait for hello: the worker has imported its libraries and loaded its model.
        let deadline = Instant::now() + Duration::from_secs(900);
        loop {
            if self.interrupt.load(Ordering::SeqCst) >= 1 || Instant::now() > deadline {
                let _ = r.child.kill();
                let kind = if Instant::now() > deadline { FailureKind::Timeout } else { FailureKind::Cancelled };
                return Err(fail(kind, format!("{name} did not finish starting"), Some(self.stderr_tail(name))));
            }
            match r.rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Line::Msg(v)) if v["type"] == "hello" => {
                    r.load_s = v["load_s"].as_f64().unwrap_or(t0.elapsed().as_secs_f64());
                    r.peak_gpu_mb = v["gpu_mb"].as_f64().unwrap_or(0.0);
                    r.hello = v;
                    break;
                }
                Ok(Line::Msg(_)) | Ok(Line::Bad(_)) | Err(RecvTimeoutError::Timeout) => continue,
                Ok(Line::Eof) | Err(RecvTimeoutError::Disconnected) => {
                    let _ = r.child.wait();
                    return Err(fail(FailureKind::WorkerCrashed, format!("{name} exited during startup"), Some(self.stderr_tail(name))));
                }
            }
        }
        let started = Started { cold: true, load_s: Some(r.load_s), pid, hello: r.hello.clone() };
        self.running.insert(name.to_string(), r);
        Ok(started)
    }

    /// Send one request and collect its items. `progress` sees every progress message.
    pub fn request(&mut self, name: &str, id: &str, op: &str, params: &Value, inputs: &[Value], progress: &mut dyn FnMut(&Value)) -> Outcome {
        let timeout = Duration::from_secs(self.cfg.workers.get(name).map(|w| w.timeout_s).unwrap_or(3600));
        let t0 = Instant::now();
        let mut out = Outcome { items: vec![], wall_s: 0.0, gpu_mb: 0.0, failure: None };
        let line = json!({"type": "request", "id": id, "op": op, "params": params, "inputs": inputs}).to_string();
        let Some(r) = self.running.get_mut(name) else {
            out.failure = Some(fail(FailureKind::WorkerCrashed, format!("{name} is not running"), None));
            return out;
        };
        r.requests += 1;
        if let Err(e) = writeln!(r.stdin, "{line}").and_then(|_| r.stdin.flush()) {
            out.failure = Some(fail(FailureKind::WorkerCrashed, format!("writing to {name}: {e}"), None));
            self.running.remove(name);
            return out;
        }
        loop {
            if self.interrupt.load(Ordering::SeqCst) >= 1 {
                self.stop(name);
                out.failure = Some(fail(FailureKind::Cancelled, "interrupted by Ctrl-C (worker stopped); completed items are kept", None));
                break;
            }
            if t0.elapsed() > timeout {
                self.stop(name);
                out.failure = Some(fail(FailureKind::Timeout, format!("{name}.{op} exceeded {} s", timeout.as_secs()), None));
                break;
            }
            let r = self.running.get_mut(name).unwrap();
            match r.rx.recv_timeout(Duration::from_millis(200)) {
                Err(RecvTimeoutError::Timeout) => continue,
                Ok(Line::Bad(l)) => eprintln!("  [{name}] non-protocol output ignored: {}", &l[..l.len().min(120)]),
                Ok(Line::Msg(v)) => match v["type"].as_str() {
                    Some("progress") => progress(&v),
                    Some("item") => out.items.push(ItemOut {
                        index: v["index"].as_u64().unwrap_or(0) as usize,
                        ok: v["ok"].as_bool().unwrap_or(false),
                        output: v["output"].clone(),
                        error: v["error"].clone(),
                    }),
                    Some("done") => {
                        out.wall_s = v["wall_s"].as_f64().unwrap_or(t0.elapsed().as_secs_f64());
                        out.gpu_mb = v["gpu_mb"].as_f64().unwrap_or(0.0);
                        r.peak_gpu_mb = r.peak_gpu_mb.max(out.gpu_mb);
                        r.last_used = Instant::now();
                        if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
                            let kind = if err["kind"] == "oom" { FailureKind::Oom } else { FailureKind::ToolError };
                            out.failure = Some(fail(kind, err["message"].as_str().unwrap_or("?"), err["traceback"].as_str().map(String::from)));
                        }
                        break;
                    }
                    _ => {}
                },
                Ok(Line::Eof) | Err(RecvTimeoutError::Disconnected) => {
                    let code = r.child.wait().ok().and_then(|s| s.code());
                    self.running.remove(name);
                    out.failure = Some(fail(
                        FailureKind::WorkerCrashed,
                        format!("{name} exited during {op} (exit code {code:?})"),
                        Some(self.stderr_tail(name)),
                    ));
                    break;
                }
            }
        }
        if out.wall_s == 0.0 {
            out.wall_s = t0.elapsed().as_secs_f64();
        }
        out
    }

    pub fn stop(&mut self, name: &str) {
        if let Some(mut r) = self.running.remove(name) {
            let _ = writeln!(r.stdin, "{}", json!({"type": "shutdown"}));
            drop(r.stdin);
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline {
                if let Ok(Some(_)) = r.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = r.child.kill();
            let _ = r.child.wait();
        }
    }

    pub fn stop_all(&mut self) {
        let names: Vec<String> = self.running.keys().cloned().collect();
        for n in names {
            self.stop(&n);
        }
    }

    pub fn status(&self) -> Vec<WorkerStatus> {
        self.running
            .iter()
            .map(|(n, r)| WorkerStatus {
                name: n.clone(),
                pid: r.child.id(),
                model: r.hello["model"].as_str().unwrap_or("-").to_string(),
                load_s: r.load_s,
                peak_gpu_mb: r.peak_gpu_mb,
                requests: r.requests,
                idle_s: r.last_used.elapsed().as_secs_f64(),
                uptime_s: r.started.elapsed().as_secs_f64(),
                gpu: self.cfg.workers.get(n).is_some_and(|w| w.gpu),
            })
            .collect()
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.stop_all();
    }
}
