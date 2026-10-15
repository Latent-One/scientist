//! Machine configuration (`scientist.toml`) and experiment definitions (`*.toml`).
//!
//! The machine file says where things are: data directory, each worker's Python
//! interpreter and script, the planner URL, the GPU memory budget for workers.
//! An experiment file says what you want: objective, allowed elements, budget,
//! autonomy level (plan section 37). It never says in which order to run tools.

use crate::types::Budget;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub data_dir: String,
    pub planner: PlannerCfg,
    pub gpu: GpuCfg,
    pub workers: BTreeMap<String, WorkerCfg>,
    #[serde(default)]
    pub supercon: SuperconCfg,
    #[serde(skip)]
    pub root: PathBuf,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PlannerCfg {
    pub url: String,
    pub model: String,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_recent")]
    pub recent_events: usize,
    #[serde(default = "default_max_steps")]
    pub max_steps_per_turn: u32,
}

fn default_max_tokens() -> u32 {
    2048
}
fn default_recent() -> usize {
    25
}
fn default_max_steps() -> u32 {
    6
}

#[derive(Deserialize, Debug, Clone)]
#[serde(default)]
pub struct SuperconCfg {
    pub likely_tc_k: f64,
    pub possible_tc_k: f64,
    pub record_tc_k: f64,
    pub training_max_tc_k: f64,
    /// 3DSC table: SuperCon measurements matched to Materials Project structures (CC BY 4.0).
    pub data_3dsc: String,
}

impl Default for SuperconCfg {
    fn default() -> Self {
        SuperconCfg { likely_tc_k: 5.0, possible_tc_k: 1.0, record_tc_k: 39.0, training_max_tc_k: 33.0, data_3dsc: "~/scientist-data/datasets/3DSC_MP.csv".into() }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct GpuCfg {
    /// Unified memory (MB) the runtime lets resident GPU workers use together. The
    /// planner's vLLM server is outside this budget; it reserved its share at startup.
    pub budget_mb: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct WorkerCfg {
    pub python: String,
    pub script: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub gpu: bool,
    /// Expected peak GPU memory before the first measurement replaces it.
    #[serde(default)]
    pub est_gpu_mb: f64,
    pub timeout_s: u64,
}

pub fn expand(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

impl Config {
    /// $SCIENTIST_HOME/scientist.toml, defaulting to ~/scientist/scientist.toml.
    pub fn load() -> Result<Config> {
        let root = std::env::var("SCIENTIST_HOME").map(PathBuf::from).unwrap_or_else(|_| expand("~/scientist"));
        let path = root.join("scientist.toml");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {} (first run: copy scientist.example.toml to scientist.toml and edit its paths)", path.display()))?;
        let mut cfg: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.root = root;
        Ok(cfg)
    }

    pub fn data_dir(&self) -> PathBuf {
        expand(&self.data_dir)
    }

    pub fn script(&self, w: &WorkerCfg) -> PathBuf {
        let p = expand(&w.script);
        if p.is_absolute() { p } else { self.root.join(p) }
    }
}

#[derive(Deserialize, Debug)]
pub struct ExperimentFile {
    pub name: String,
    pub objective: ObjectiveSec,
    #[serde(default)]
    pub constraints: ConstraintsSec,
    #[serde(default)]
    pub budget: Budget,
    #[serde(default)]
    pub planner: PlannerSec,
}

#[derive(Deserialize, Debug)]
pub struct ObjectiveSec {
    pub description: String,
}

#[derive(Deserialize, Debug)]
pub struct ConstraintsSec {
    #[serde(default)]
    pub elements: Vec<String>,
    #[serde(default = "default_max_sites")]
    pub max_sites: u32,
}

impl Default for ConstraintsSec {
    fn default() -> Self {
        ConstraintsSec { elements: vec![], max_sites: default_max_sites() }
    }
}

fn default_max_sites() -> u32 {
    40
}

#[derive(Deserialize, Debug)]
pub struct PlannerSec {
    #[serde(default = "default_autonomy")]
    pub autonomy: u8,
}

impl Default for PlannerSec {
    fn default() -> Self {
        PlannerSec { autonomy: default_autonomy() }
    }
}

fn default_autonomy() -> u8 {
    1
}

pub fn load_experiment_file(path: &Path) -> Result<ExperimentFile> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}
