//! Learning features (addendum sections 5, 8, 14, 15):
//!   --learn          explains the computation that actually ran, with its own numbers
//!   concepts/*.md    short first-principles notes, opened by `scientist learn <concept>`
//!   notebook         a lab notebook rendered from the event log

use crate::actions::Done;
use crate::config::Config;
use crate::events::{Event, EventRecord};
use crate::state::{ExperimentState, Observation};
use crate::store::{now_ms, Store};
use crate::tools::Runtime;
use crate::types::*;
use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;

pub fn slug(name: &str) -> String {
    name.trim().to_lowercase().replace([' ', '_'], "-")
}

fn concept_path(cfg: &Config, name: &str) -> PathBuf {
    cfg.root.join("concepts").join(format!("{}.md", slug(name)))
}

pub fn concept(cfg: &Config, name: &str) -> Option<String> {
    std::fs::read_to_string(concept_path(cfg, name)).ok()
}

pub fn has_concept(cfg: &Config, name: &str) -> bool {
    concept_path(cfg, name).exists()
}

pub fn concepts(cfg: &Config) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(cfg.root.join("concepts"))
        .map(|d| d.filter_map(|e| e.ok()).filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".md")).map(String::from)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Explain each observation the action produced, using its recorded numbers.
pub fn render(rt: &Runtime, exp: &str, done: &Done) -> Result<()> {
    let st = rt.state(exp)?;
    for oid in &done.observations {
        if let Some(o) = st.observations.iter().find(|o| &o.id == oid) {
            if o.method.op == "relax" && !o.properties.is_empty() {
                println!("{}", relax_lesson(&rt.cfg, &rt.store, &st, o)?);
            }
        }
    }
    Ok(())
}

fn relax_lesson(cfg: &Config, store: &Store, st: &ExperimentState, o: &Observation) -> Result<String> {
    let d = &o.details;
    let c = st.candidate(&o.candidate).unwrap();
    let input = store.structure(&o.input)?;
    let f = |k: &str| d[k].as_f64().unwrap_or(f64::NAN);
    let e1 = o.property(PropertyKind::EnergyPerAtom).unwrap_or(f64::NAN);
    let fmax = o.property(PropertyKind::MaxForce).unwrap_or(f64::NAN);
    let p = o.property(PropertyKind::Pressure).unwrap_or(f64::NAN);
    let converged = d["converged"] == true;
    let mut s = String::new();
    s.push_str(&format!("\n=== LEARN: what {} just did to {} ({}) ===\n", o.id, c.id, c.formula));
    s.push_str(&format!(
        "TASK         geometry optimization (\"relaxation\"): move atoms{} downhill in energy until every force is small\n",
        if d["relax_cell"] == true { " and the cell shape" } else { " (cell held fixed)" }
    ));
    s.push_str(&format!(
        "INPUT        {} atoms {:?} in a periodic cell of {:.2} Å³ (artifact {})\n",
        input.species.len(), input.composition(), input.volume(), short(&o.input)
    ));
    s.push_str(&format!(
        "MODEL        {} — a machine-learned interatomic potential (neural network trained on DFT data);\n             checkpoint {} {}; dtype {}. Fidelity: ML potential, not quantum mechanics.\n",
        o.model().unwrap_or("?"), o.method.checkpoint.as_deref().unwrap_or("?"),
        o.method.checkpoint_sha256.as_deref().map(short).unwrap_or("-"), o.method.dtype.as_deref().unwrap_or("?")
    ));
    s.push_str(&format!(
        "OPTIMIZER    {} (ASE), stop when max |force| < {} eV/Å or after the step limit\n",
        d["optimizer"].as_str().unwrap_or("FIRE"), d["fmax_target_eV_A"]
    ));
    if let Some(th) = d["trajectory"].as_str() {
        if let Ok(bytes) = store.read_artifact(th) {
            let t: Value = serde_json::from_slice(&bytes)?;
            if let Some(steps) = t["steps"].as_array() {
                s.push_str("ITERATIONS   step   E/atom (eV)   max|F| (eV/Å)   V/atom (Å³)\n");
                let n = steps.len();
                for (i, r) in steps.iter().enumerate() {
                    if i < 4 || i + 3 >= n || i % (n / 6).max(1) == 0 {
                        s.push_str(&format!(
                            "             {:>4}  {:>12.5}  {:>13.4}  {:>11.3}\n",
                            r["step"], r["energy_per_atom_eV"].as_f64().unwrap_or(f64::NAN),
                            r["max_force_eV_A"].as_f64().unwrap_or(f64::NAN), r["volume_per_atom_A3"].as_f64().unwrap_or(f64::NAN)
                        ));
                    } else if i == 4 {
                        s.push_str("             …\n");
                    }
                }
            }
        }
    }
    s.push_str(&format!(
        "CONVERGENCE  {} after {} steps; final max|F| {:.4} eV/Å (target {}), residual pressure {:.2} GPa\n",
        if converged { "converged" } else { "NOT converged" }, d["steps"], fmax, d["fmax_target_eV_A"], p
    ));
    s.push_str(&format!(
        "CHANGED      energy {:+.1} meV/atom ({:.5} -> {e1:.5} eV/atom); volume {:+.2}%; largest atom move {:.3} Å\n",
        f("energy_change_meV_per_atom"), f("initial_energy_per_atom_eV"), f("volume_change_pct"), f("max_displacement_A")
    ));
    s.push_str("INTERPRETATION (rule-based reading of the numbers above, not a finding)\n");
    if !converged {
        s.push_str("  - Not converged: this energy is not at a minimum; do not rank it against converged ones.\n");
    }
    if f("energy_change_meV_per_atom").abs() < 5.0 && f("max_displacement_A") < 0.05 {
        s.push_str("  - Almost nothing moved: the input was already at (or very near) a minimum of this model.\n");
    }
    if f("volume_change_pct").abs() > 5.0 {
        s.push_str("  - The cell changed size a lot: the starting density was far from what this model prefers.\n");
    }
    if f("energy_change_meV_per_atom") < -100.0 {
        s.push_str("  - Large energy drop: the input was strained or distorted; the relaxed structure may differ in kind.\n");
    }
    if d["relax_cell"] == true && p.abs() > 1.0 {
        s.push_str("  - Residual pressure above 1 GPa: the cell is not fully relaxed (tighten fmax or allow more steps).\n");
    }
    s.push_str("  - A minimum found by relaxation is LOCAL: other arrangements of the same atoms may be lower.\n");
    s.push_str("  - This energy is only comparable to structures of the same composition, from the same model.\n");
    let names = ["relaxation", "potential-energy-surface", "forces", "ml-interatomic-potential", "periodic-boundary-conditions"];
    let have: Vec<&str> = names.iter().copied().filter(|n| has_concept(cfg, n)).collect();
    s.push_str(&format!("WHAT TO LEARN  scientist learn <one of: {}>\n", have.join(", ")));
    s.push_str(&format!("DRILL DOWN     scientist inspect {}   (method, versions, full trajectory, raw numbers)\n", o.id));
    Ok(s)
}

/// A lab notebook rendered from the event log (addendum section 15).
pub fn notebook(st: &ExperimentState, events: &[EventRecord]) -> String {
    let mut s = format!("# Experiment {}\n\n*Rendered from the event log at {} ms since epoch. Events: {}.*\n\n", st.name, now_ms(), events.len());
    s.push_str(&format!("## Question\n\n{}\n\nElements: {:?}. Autonomy level: {}.\n\n", st.objective, st.elements, st.autonomy));
    if !st.hypotheses.is_empty() {
        s.push_str("## Hypotheses\n\n");
        for h in &st.hypotheses {
            s.push_str(&format!("- **{}** [{:?}, by {}] {}\n  - rationale: {}\n  - falsified if: {}\n", h.id, h.status, h.author.as_str(), h.statement, h.rationale, h.falsified_if));
            if !h.reason.is_empty() {
                s.push_str(&format!("  - status reason: {}\n", h.reason));
            }
        }
        s.push('\n');
    }
    s.push_str("## Computations and observations\n\n");
    for r in events {
        let line = match &r.event {
            Event::UserMessage { text } => Some(format!("**You asked:** {text}")),
            Event::ActionCompleted { action, summary } => Some(format!("**{}** `{}` — {}", r.actor.as_str(), action.describe(), summary)),
            Event::ActionFailed { action, error } => Some(format!("**FAILED** `{}` — {}", action.describe(), error.lines().next().unwrap_or(""))),
            Event::SearchRoundCompleted { .. } | Event::SearchStopped { .. } | Event::StatusSet { .. } => Some(r.event.summary()),
            _ => None,
        };
        if let Some(l) = line {
            s.push_str(&format!("- [{}] {}\n", r.seq, l));
        }
    }
    s.push_str("\n## Candidates with evaluations\n\n| id | formula | status | E/atom (MatterSim) | E above hull (MatterSim, meV/atom) | E/atom (MACE) | origin |\n|---|---|---|---|---|---|---|\n");
    for c in st.candidates.iter().filter(|c| c.kind == ArtifactKind::Structure && st.latest(&c.id, PropertyKind::EnergyPerAtom, None).is_some()) {
        let g = |k, e| st.latest(&c.id, k, Some(e)).map(|v| format!("{:.4}", v.0)).unwrap_or("—".into());
        let hull = st.latest(&c.id, PropertyKind::EAboveHull, Some(Evaluator::Mattersim)).map(|v| format!("{:.1}", v.0 * 1000.0)).unwrap_or("—".into());
        s.push_str(&format!(
            "| {} | {} | {:?} | {} | {} | {} | {} |\n",
            c.id, c.formula, c.status, g(PropertyKind::EnergyPerAtom, Evaluator::Mattersim), hull, g(PropertyKind::EnergyPerAtom, Evaluator::Mace), crate::events::origin_summary(&c.origin)
        ));
    }
    let interp: Vec<_> = st.notes.iter().filter(|n| n.interpretation).collect();
    if !interp.is_empty() {
        s.push_str("\n## Interpretations (not observations)\n\n");
        for n in interp {
            s.push_str(&format!("- [{}] ({}) {} {:?}\n", n.seq, n.author.as_str(), n.text, n.refs));
        }
    }
    let notes: Vec<_> = st.notes.iter().filter(|n| !n.interpretation).collect();
    if !notes.is_empty() {
        s.push_str("\n## Notes\n\n");
        for n in notes {
            s.push_str(&format!("- [{}] ({}) {} {:?}\n", n.seq, n.author.as_str(), n.text, n.refs));
        }
    }
    let mut concepts: Vec<String> = vec![];
    for r in events {
        if let Event::PlannerProposed { output, .. } = &r.event {
            if let Some(a) = output.pointer("/explanation/new_concepts").and_then(|v| v.as_array()) {
                for c in a.iter().filter_map(|c| c.as_str()) {
                    if !concepts.iter().any(|x| x == c) {
                        concepts.push(c.to_string());
                    }
                }
            }
        }
    }
    if !concepts.is_empty() {
        s.push_str(&format!("\n## Concepts encountered\n\n{}\n", concepts.join(", ")));
    }
    let open: Vec<_> = st.hypotheses.iter().filter(|h| h.status == crate::events::HypothesisStatus::Open).collect();
    if !open.is_empty() {
        s.push_str("\n## Open questions\n\n");
        for h in open {
            s.push_str(&format!("- {}: {}\n", h.id, h.statement));
        }
    }
    s
}
