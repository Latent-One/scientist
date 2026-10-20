//! Persistence: one SQLite file + a content-addressed artifact directory.
//!
//! ~/scientist-data/
//!   scientist.db                 events (append-only, enforced by triggers), artifact
//!                                index, cache, last-seen worker identities, settings
//!   artifacts/ab/abcdef….json    immutable files named by the SHA-256 of their bytes
//!   logs/<worker>.log            worker stderr
//!
//! Ordering rule (design.md F): artifact files are written first (idempotent: same bytes,
//! same path), then the events that reference them are committed in one transaction.
//! A crash between the two leaves an unreferenced file, never a dangling reference.

use crate::events::{Actor, Event, EventRecord};
use crate::types::{sha256, ArtifactKind, Hash, Molecule, Structure};
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Row (seq, ts_ms, actor, payload) -> EventRecord, decoded straight from SQLite's row buffer:
/// no intermediate String per column.
fn decode_row(r: &rusqlite::Row) -> Result<EventRecord> {
    let seq = r.get::<_, i64>(0)? as u64;
    let event: Event = serde_json::from_slice(r.get_ref(3)?.as_bytes()?).with_context(|| format!("decoding event {seq}"))?;
    let actor = Actor::parse(r.get_ref(2)?.as_str()?).ok_or_else(|| anyhow!("event {seq}: unknown actor"))?;
    Ok(EventRecord { seq, ts_ms: r.get::<_, i64>(1)? as u64, actor, event })
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
CREATE TABLE IF NOT EXISTS experiments (
    name TEXT PRIMARY KEY, created_ms INTEGER NOT NULL, parent TEXT, fork_seq INTEGER);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, experiment TEXT NOT NULL, ts_ms INTEGER NOT NULL,
    actor TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_by_experiment ON events(experiment, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TABLE IF NOT EXISTS artifacts (
    hash TEXT PRIMARY KEY, kind TEXT NOT NULL, bytes INTEGER NOT NULL, created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, worker TEXT NOT NULL, op TEXT NOT NULL, job TEXT NOT NULL,
    experiment TEXT NOT NULL, items TEXT NOT NULL, hello TEXT NOT NULL, created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS worker_identity (
    worker TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, hello TEXT NOT NULL, updated_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
    experiment TEXT PRIMARY KEY, seq INTEGER NOT NULL, fingerprint TEXT NOT NULL, payload BLOB NOT NULL);
"#;

pub struct Store {
    conn: Connection,
    root: PathBuf,
}

pub struct CacheEntry {
    pub job: String,
    pub experiment: String,
    pub items: Vec<Value>,
    pub hello: Value,
}

pub struct ExperimentRow {
    pub name: String,
    pub created_ms: u64,
    pub parent: Option<String>,
    pub events: u64,
}

impl Store {
    pub fn open(root: &Path) -> Result<Store> {
        std::fs::create_dir_all(root.join("artifacts"))?;
        std::fs::create_dir_all(root.join("logs"))?;
        let conn = Connection::open(root.join("scientist.db"))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Store { conn, root: root.to_path_buf() })
    }

    // ---------- experiments and events ----------

    pub fn create_experiment(&self, name: &str, parent: Option<(&str, u64)>) -> Result<()> {
        if self.experiment_exists(name)? {
            bail!("experiment {name:?} already exists");
        }
        self.conn.execute(
            "INSERT INTO experiments (name, created_ms, parent, fork_seq) VALUES (?1, ?2, ?3, ?4)",
            params![name, now_ms() as i64, parent.map(|p| p.0), parent.map(|p| p.1 as i64)],
        )?;
        Ok(())
    }

    pub fn experiment_exists(&self, name: &str) -> Result<bool> {
        Ok(self.conn.query_row("SELECT 1 FROM experiments WHERE name = ?1", [name], |_| Ok(())).optional()?.is_some())
    }

    pub fn experiments(&self) -> Result<Vec<ExperimentRow>> {
        let mut st = self.conn.prepare(
            "SELECT e.name, e.created_ms, e.parent, (SELECT COUNT(*) FROM events v WHERE v.experiment = e.name)
             FROM experiments e ORDER BY e.created_ms",
        )?;
        let rows = st.query_map([], |r| {
            Ok(ExperimentRow { name: r.get(0)?, created_ms: r.get::<_, i64>(1)? as u64, parent: r.get(2)?, events: r.get::<_, i64>(3)? as u64 })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Append events in one transaction. Returns their sequence numbers.
    pub fn append(&mut self, experiment: &str, actor: Actor, events: &[Event]) -> Result<Vec<u64>> {
        let tx = self.conn.transaction()?;
        let mut seqs = Vec::with_capacity(events.len());
        for e in events {
            tx.execute(
                "INSERT INTO events (experiment, ts_ms, actor, kind, payload) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![experiment, now_ms() as i64, actor.as_str(), e.kind(), serde_json::to_string(e)?],
            )?;
            seqs.push(tx.last_insert_rowid() as u64);
        }
        tx.commit()?;
        Ok(seqs)
    }

    /// Events visible to an experiment: for a fork, the parent's events up to the fork
    /// point (recursively), then its own. Sequence numbers are global, so order is total.
    pub fn load_events(&self, experiment: &str) -> Result<Vec<EventRecord>> {
        let (parent, fork_seq): (Option<String>, Option<i64>) = self
            .conn
            .query_row("SELECT parent, fork_seq FROM experiments WHERE name = ?1", [experiment], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?
            .ok_or_else(|| anyhow!("no experiment named {experiment:?} (see `scientist list`)"))?;
        let mut out = match (parent, fork_seq) {
            (Some(p), Some(at)) => self.load_events(&p)?.into_iter().filter(|e| e.seq <= at as u64).collect(),
            _ => vec![],
        };
        let mut st = self.conn.prepare_cached("SELECT seq, ts_ms, actor, payload FROM events WHERE experiment = ?1 ORDER BY seq")?;
        let mut rows = st.query([experiment])?;
        while let Some(r) = rows.next()? {
            out.push(decode_row(r)?);
        }
        Ok(out)
    }

    /// Events of this experiment itself with seq > `after`. Valid for resuming from a snapshot
    /// taken at `after`: a fork's inherited parent events all precede its own first event.
    pub fn load_events_after(&self, experiment: &str, after: u64) -> Result<Vec<EventRecord>> {
        let mut st = self.conn.prepare_cached("SELECT seq, ts_ms, actor, payload FROM events WHERE experiment = ?1 AND seq > ?2 ORDER BY seq")?;
        let mut rows = st.query(params![experiment, after as i64])?;
        let mut out = vec![];
        while let Some(r) = rows.next()? {
            out.push(decode_row(r)?);
        }
        Ok(out)
    }

    /// The last `n` events of this experiment, oldest first.
    pub fn recent_events(&self, experiment: &str, n: usize) -> Result<Vec<EventRecord>> {
        let mut st = self.conn.prepare_cached("SELECT seq, ts_ms, actor, payload FROM events WHERE experiment = ?1 ORDER BY seq DESC LIMIT ?2")?;
        let mut rows = st.query(params![experiment, n as i64])?;
        let mut out = vec![];
        while let Some(r) = rows.next()? {
            out.push(decode_row(r)?);
        }
        out.reverse();
        Ok(out)
    }

    // ---------- state snapshots (a cache: deleting them only costs a full fold) ----------

    pub fn snapshot(&self, experiment: &str, fingerprint: u64) -> Result<Option<(u64, Vec<u8>)>> {
        Ok(self
            .conn
            .prepare_cached("SELECT seq, payload FROM snapshots WHERE experiment = ?1 AND fingerprint = ?2")?
            .query_row(params![experiment, format!("{fingerprint:016x}")], |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, Vec<u8>>(1)?)))
            .optional()?)
    }

    pub fn put_snapshot(&self, experiment: &str, seq: u64, fingerprint: u64, payload: &[u8]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (experiment, seq, fingerprint, payload) VALUES (?1, ?2, ?3, ?4)",
            params![experiment, seq as i64, format!("{fingerprint:016x}"), payload],
        )?;
        Ok(())
    }

    // ---------- artifacts ----------

    fn artifact_path(&self, hash: &str) -> PathBuf {
        let hex = hash.strip_prefix("sha256:").unwrap_or(hash);
        self.root.join("artifacts").join(&hex[..2]).join(format!("{hex}.json"))
    }

    /// Store bytes under their hash. Returns (hash, newly_created).
    pub fn put_artifact(&self, kind: ArtifactKind, bytes: &[u8]) -> Result<(Hash, bool)> {
        let hash = sha256(bytes);
        let path = self.artifact_path(&hash);
        let created = !path.exists();
        if created {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(&tmp, &path)?; // atomic on one filesystem: readers never see half a file
        }
        self.conn.execute(
            "INSERT OR IGNORE INTO artifacts (hash, kind, bytes, created_ms) VALUES (?1, ?2, ?3, ?4)",
            params![hash, serde_json::to_value(kind)?.as_str().unwrap_or("?"), bytes.len() as i64, now_ms() as i64],
        )?;
        Ok((hash, created))
    }

    pub fn put_structure(&self, s: &Structure) -> Result<(Hash, Structure, bool)> {
        let c = s.canonical();
        let (h, created) = self.put_artifact(ArtifactKind::Structure, &serde_json::to_vec(&c)?)?;
        Ok((h, c, created))
    }

    pub fn read_artifact(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.artifact_path(hash);
        let bytes = std::fs::read(&path).with_context(|| format!("reading artifact {}", path.display()))?;
        if sha256(&bytes) != hash {
            bail!("artifact {hash} is corrupted: content hash does not match its name");
        }
        Ok(bytes)
    }

    pub fn structure(&self, hash: &str) -> Result<Structure> {
        Ok(serde_json::from_slice(&self.read_artifact(hash)?)?)
    }

    pub fn molecule(&self, hash: &str) -> Result<Molecule> {
        Ok(serde_json::from_slice(&self.read_artifact(hash)?)?)
    }

    pub fn artifact_bytes_len(&self, hash: &str) -> Option<u64> {
        self.conn.query_row("SELECT bytes FROM artifacts WHERE hash = ?1", [hash], |r| r.get::<_, i64>(0)).optional().ok().flatten().map(|b| b as u64)
    }

    pub fn artifact_file(&self, hash: &str) -> PathBuf {
        self.artifact_path(hash)
    }

    // ---------- cache (shared across experiments) ----------

    pub fn cache_get(&self, key: &str) -> Result<Option<CacheEntry>> {
        let mut st = self.conn.prepare_cached("SELECT job, experiment, items, hello FROM cache WHERE key = ?1")?;
        let mut rows = st.query([key])?;
        let Some(r) = rows.next()? else { return Ok(None) };
        // items can be large (whole structures): decode from the row buffer, no String copy.
        Ok(Some(CacheEntry {
            job: r.get(0)?,
            experiment: r.get(1)?,
            items: serde_json::from_slice(r.get_ref(2)?.as_bytes()?)?,
            hello: serde_json::from_slice(r.get_ref(3)?.as_bytes()?)?,
        }))
    }

    pub fn cache_put<T: serde::Serialize>(&self, key: &str, worker: &str, op: &str, job: &str, experiment: &str, items: &[T], hello: &Value) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, worker, op, job, experiment, items, hello, created_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![key, worker, op, job, experiment, serde_json::to_string(items)?, serde_json::to_string(hello)?, now_ms() as i64],
        )?;
        Ok(())
    }

    pub fn cache_stats(&self) -> Result<(u64, u64)> {
        Ok(self.conn.query_row("SELECT COUNT(*), COALESCE(SUM(LENGTH(items)),0) FROM cache", [], |r| {
            Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64))
        })?)
    }

    // ---------- worker identities (for cache keys without starting a worker) ----------

    pub fn worker_identity(&self, worker: &str) -> Result<Option<(String, Value)>> {
        let row = self
            .conn
            .query_row("SELECT fingerprint, hello FROM worker_identity WHERE worker = ?1", [worker], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .optional()?;
        Ok(match row {
            Some((f, h)) => Some((f, serde_json::from_str(&h)?)),
            None => None,
        })
    }

    pub fn set_worker_identity(&self, worker: &str, fingerprint: &str, hello: &Value) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO worker_identity (worker, fingerprint, hello, updated_ms) VALUES (?1, ?2, ?3, ?4)",
            params![worker, fingerprint, serde_json::to_string(hello)?, now_ms() as i64],
        )?;
        Ok(())
    }

    // ---------- settings ----------

    pub fn setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)", params![key, value])?;
        Ok(())
    }
}
