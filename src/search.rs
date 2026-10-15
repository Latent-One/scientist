//! Searchers: the numerical layer under the planner (plan section 19, design.md A6).
//!
//! The planner (or you) decides WHAT to investigate by defining a search: a space
//! (elements), a strategy, objectives, an evaluator and a budget. This module then decides
//! WHERE to sample, round by round, without calling the LLM:
//!
//!   generative    a model is the searcher: MatterGen samples crystals in a chemical
//!                 system; ChemFM samples molecules.
//!   evolutionary  parents chosen by Pareto rank (binary tournament), then mutated:
//!                 substitute / swap / strain / rattle. Seeded, so replayable.
//!   random        same mutations, parents chosen uniformly: the baseline that tells you
//!                 whether selection is doing any work (plan section 48).
//!
//! Each round: propose -> structural check -> evaluate -> (hull) -> Pareto front -> event.
//! Stop rules are enforced here, not by the LLM: max rounds, patience (rounds without a
//! new Pareto member), budget.

use crate::actions::{self, Added, Done, Ui};
use crate::events::{Actor, Event};
use crate::rng::Rng;
use crate::state::{ExperimentState, Ids};
use crate::tools::Runtime;
use crate::types::*;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Strategy {
    Generative { generator: Generator },
    Evolutionary { substitution: bool, strain_max: f64, rattle_stdev: f64 },
    Random { substitution: bool, strain_max: f64, rattle_stdev: f64 },
}

/// What you or the planner specify.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchInput {
    pub strategy: Strategy,
    pub objectives: Vec<Objective>,
    pub evaluator: Evaluator,
    pub per_round: u32,
    pub max_rounds: u32,
    pub patience: u32,
    pub seed: Option<u64>,
    /// Allowed elements for this search; empty = the experiment's elements.
    pub elements: Vec<String>,
    /// Starting population for evolutionary/random; empty = every evaluated candidate.
    pub pool: Vec<String>,
}

/// What is recorded: the input with every default resolved, plus an id.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchSpec {
    pub id: String,
    pub strategy: Strategy,
    pub objectives: Vec<Objective>,
    pub evaluator: Evaluator,
    pub per_round: u32,
    pub max_rounds: u32,
    pub patience: u32,
    pub seed: u64,
    pub elements: Vec<String>,
    pub pool: Vec<String>,
}

fn objectives_text(o: &[Objective]) -> String {
    o.iter().map(|o| format!("{:?} {:?}", o.direction, o.property)).collect::<Vec<_>>().join(", ")
}

fn strategy_text(s: &Strategy) -> String {
    match s {
        Strategy::Generative { generator } => format!("generative ({generator:?} samples candidates)"),
        Strategy::Evolutionary { substitution, strain_max, .. } => format!("evolutionary (Pareto-ranked parents; substitution {substitution}; strain ±{:.1}%)", strain_max * 100.0),
        Strategy::Random { substitution, strain_max, .. } => format!("random mutations (baseline; substitution {substitution}; strain ±{:.1}%)", strain_max * 100.0),
    }
}

impl SearchInput {
    pub fn describe(&self) -> String {
        format!("{}; objectives [{}]; evaluator {:?}; {} per round, ≤{} rounds", strategy_text(&self.strategy), objectives_text(&self.objectives), self.evaluator, self.per_round, self.max_rounds)
    }

    fn molecular(&self) -> bool {
        matches!(self.strategy, Strategy::Generative { generator: Generator::Chemfm })
    }

    pub fn validate(&self, st: &ExperimentState) -> Vec<String> {
        use PropertyKind::*;
        let mut errs = vec![];
        if self.objectives.is_empty() {
            errs.push("a search needs at least one objective".into());
        }
        if !(1..=64).contains(&self.per_round) || !(1..=50).contains(&self.max_rounds) || self.patience == 0 {
            errs.push("per_round must be 1..64, max_rounds 1..50, patience ≥ 1".into());
        }
        let mol_kinds = [MolWt, Logp, Tpsa, Qed, FractionSp3, AromaticRings];
        let xtal_kinds = [EnergyPerAtom, FormationEnergyPerAtom, EAboveHull, VolumePerAtom, Tc, Lambda];
        for o in &self.objectives {
            let ok = if self.molecular() { mol_kinds.contains(&o.property) } else { xtal_kinds.contains(&o.property) };
            if !ok {
                errs.push(format!("{:?} is not an objective for this kind of search (molecules: {mol_kinds:?}; crystals: {xtal_kinds:?})", o.property));
            }
        }
        let changes_composition = match &self.strategy {
            Strategy::Generative { generator: Generator::Mattergen } => true,
            Strategy::Evolutionary { substitution, .. } | Strategy::Random { substitution, .. } => *substitution,
            _ => false,
        };
        if changes_composition && self.objectives.iter().any(|o| o.property == EnergyPerAtom) {
            errs.push(
                "energy per atom cannot rank candidates of different compositions (a B atom and an N atom carry different \
                 reference energies). Use formation_energy_per_atom or e_above_hull (needs elemental references), or turn \
                 substitution off so composition stays fixed"
                    .into(),
            );
        }
        let elements = if self.elements.is_empty() { &st.elements } else { &self.elements };
        if matches!(self.strategy, Strategy::Generative { generator: Generator::Mattergen }) && elements.is_empty() {
            errs.push("a MatterGen search needs elements (set them in the search or the experiment)".into());
        }
        if matches!(self.strategy, Strategy::Evolutionary { .. } | Strategy::Random { .. }) {
            for c in &self.pool {
                match st.candidate(c) {
                    Some(c) if c.kind == ArtifactKind::Structure => {}
                    _ => errs.push(format!("pool member {c} is not a crystal candidate")),
                }
            }
            let any = st.candidates.iter().any(|c| c.kind == ArtifactKind::Structure && c.status != CandidateStatus::Rejected);
            if self.pool.is_empty() && !any {
                errs.push("evolutionary and random searches mutate existing crystals; build, import or generate some first".into());
            }
        }
        errs
    }
}

impl SearchSpec {
    pub fn describe(&self) -> String {
        format!("{}; objectives [{}]; evaluator {:?}; {} per round, ≤{} rounds, patience {}, seed {}", strategy_text(&self.strategy), objectives_text(&self.objectives), self.evaluator, self.per_round, self.max_rounds, self.patience, self.seed)
    }
}

pub fn define(rt: &mut Runtime, exp: &str, st: &ExperimentState, actor: Actor, ids: &mut Ids, input: &SearchInput) -> Result<Done> {
    let spec = SearchSpec {
        id: ids.search(),
        strategy: input.strategy.clone(),
        objectives: input.objectives.clone(),
        evaluator: input.evaluator,
        per_round: input.per_round,
        max_rounds: input.max_rounds,
        patience: input.patience,
        seed: input.seed.unwrap_or(st.last_seq),
        elements: if input.elements.is_empty() { st.elements.clone() } else { input.elements.clone() },
        pool: input.pool.clone(),
    };
    let mut summary = format!("search {} defined: {}", spec.id, spec.describe());
    if spec.objectives.iter().any(|o| matches!(o.property, PropertyKind::Tc | PropertyKind::Lambda)) {
        summary.push_str(". WARNING: predicted Tc ranks unseen superconductors near-randomly (Spearman 0.13 on 216, benchmark supercon_unseen); a Tc-maximizing search is close to undirected. Treat its front as candidates to check, not as ranked results.");
    }
    rt.append(exp, actor, vec![Event::SearchDefined { spec }])?;
    Ok(Done { summary, ..Default::default() })
}

// ---------------------------------------------------------------- Pareto

/// Values oriented so smaller is better on every axis.
fn oriented(v: &[f64], objs: &[Objective]) -> Vec<f64> {
    v.iter().zip(objs).map(|(x, o)| if o.direction == Direction::Max { -x } else { *x }).collect()
}

pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// Non-dominated sorting: rank 0 = the Pareto front, rank 1 = front after removing rank 0, …
pub fn pareto_ranks(points: &[Vec<f64>]) -> Vec<usize> {
    let n = points.len();
    let mut rank = vec![usize::MAX; n];
    let mut r = 0;
    let mut left: Vec<usize> = (0..n).collect();
    while !left.is_empty() {
        let front: Vec<usize> = left.iter().copied().filter(|&i| !left.iter().any(|&j| j != i && dominates(&points[j], &points[i]))).collect();
        for &i in &front {
            rank[i] = r;
        }
        left.retain(|i| !front.contains(i));
        r += 1;
    }
    rank
}

fn value(st: &ExperimentState, cid: &str, o: &Objective, ev: Evaluator) -> Option<f64> {
    use PropertyKind::*;
    if matches!(o.property, Tc | Lambda | OmegaLog) {
        // superconductivity objectives use the primary (best-validated) model
        return st.latest_model(cid, o.property, "alex_supercon_Tc").map(|v| v.0).filter(|v| v.is_finite());
    }
    let by_model = !matches!(o.property, MolWt | Logp | Tpsa | Qed | FractionSp3 | AromaticRings);
    st.latest(cid, o.property, if by_model { Some(ev) } else { None }).map(|v| v.0).filter(|v| v.is_finite())
}

/// Candidates that belong to this search's population and have every objective value.
fn population(st: &ExperimentState, spec: &SearchSpec) -> Vec<(String, Vec<f64>)> {
    st.candidates
        .iter()
        .filter(|c| c.status != CandidateStatus::Rejected)
        .filter(|c| c.search.as_deref() == Some(&spec.id) || spec.pool.contains(&c.id) || (spec.pool.is_empty() && c.search.is_none()))
        .filter_map(|c| {
            let v: Option<Vec<f64>> = spec.objectives.iter().map(|o| value(st, &c.id, o, spec.evaluator)).collect();
            v.map(|v| (c.id.clone(), v))
        })
        .collect()
}

/// A rattled or slightly strained crystal often relaxes back into the same minimum as its
/// parent; float noise then makes it look new. Treat same formula, energy within 1 meV/atom
/// and volume within 0.5% as the same minimum, keeping the earliest candidate.
fn distinct_minima(st: &ExperimentState, spec: &SearchSpec, ids: Vec<String>) -> Vec<String> {
    let key = |c: &str| {
        let e = st.latest(c, PropertyKind::EnergyPerAtom, Some(spec.evaluator)).map(|v| v.0);
        let v = st.latest(c, PropertyKind::VolumePerAtom, Some(spec.evaluator)).map(|v| v.0);
        (st.candidate(c).map(|x| x.formula.clone()).unwrap_or_default(), e, v)
    };
    let mut kept: Vec<String> = vec![];
    for c in ids {
        let (f, e, v) = key(&c);
        let same = kept.iter().any(|k| {
            let (f2, e2, v2) = key(k);
            f == f2
                && matches!((e, e2), (Some(a), Some(b)) if (a - b).abs() < 1e-3)
                && matches!((v, v2), (Some(a), Some(b)) if (a / b - 1.0).abs() < 5e-3)
        });
        if !same {
            kept.push(c);
        }
    }
    kept
}

// ---------------------------------------------------------------- rounds

fn choose_mutation(rng: &mut Rng, s: &Structure, elements: &[String], substitution: bool, strain_max: f64, rattle_stdev: f64) -> Mutation {
    let n = s.species.len();
    let distinct = s.composition().len();
    let mut menu: Vec<(&str, u32)> = vec![("strain", 2), ("rattle", 2)];
    if substitution && !elements.is_empty() {
        menu.push(("substitute", 3));
    }
    if distinct >= 2 {
        menu.push(("swap", 2));
    }
    let total: u32 = menu.iter().map(|m| m.1).sum();
    let mut pick = rng.below(total as usize) as u32;
    let kind = menu.iter().find(|m| if pick < m.1 { true } else { pick -= m.1; false }).unwrap().0;
    match kind {
        "substitute" => {
            let site = rng.below(n);
            let choices: Vec<&String> = elements.iter().filter(|e| **e != s.species[site]).collect();
            if choices.is_empty() {
                return Mutation::Strain { strain: (rng.uniform(-strain_max, strain_max) * 1e4).round() / 1e4 };
            }
            Mutation::Substitute { site, element: choices[rng.below(choices.len())].clone() }
        }
        "swap" => {
            let i = rng.below(n);
            let others: Vec<usize> = (0..n).filter(|&j| s.species[j] != s.species[i]).collect();
            Mutation::Swap { i, j: others[rng.below(others.len())] }
        }
        "rattle" => Mutation::Rattle { stdev: rattle_stdev, seed: rng.next_u64() % 1_000_000_000 },
        _ => Mutation::Strain { strain: (rng.uniform(-strain_max, strain_max) * 1e4).round() / 1e4 },
    }
}

pub fn run(rt: &mut Runtime, exp: &str, _actor: Actor, id: &str, rounds: u32, ui: &Ui) -> Result<Done> {
    let mut lines = vec![];
    for _ in 0..rounds {
        if rt.interrupted() {
            lines.push("stopped by Ctrl-C between rounds".to_string());
            break;
        }
        let st = rt.state(exp)?;
        let s = st.searches.get(id).unwrap().clone();
        if let Some(why) = &s.stopped {
            lines.push(format!("search already stopped: {why}"));
            break;
        }
        let round = s.rounds_done + 1;
        eprintln!("  search {id} round {round}/{}: {}", s.spec.max_rounds, strategy_text(&s.spec.strategy));
        let line = run_round(rt, exp, &st, &s.spec, round, ui)?;
        eprintln!("  {line}");
        lines.push(line);
        let st = rt.state(exp)?;
        let s2 = &st.searches[id];
        let rem = st.remaining();
        let budget_out = match s2.spec.strategy {
            Strategy::Generative { generator: Generator::Mattergen } => rem.mattergen_structures == 0 || rem.relaxations == 0,
            Strategy::Generative { generator: Generator::Chemfm } => rem.chemfm_molecules == 0,
            _ => rem.relaxations == 0,
        };
        let stop = if round >= s2.spec.max_rounds {
            Some(format!("reached max_rounds {}", s2.spec.max_rounds))
        } else if s2.rounds_without_improvement >= s2.spec.patience {
            Some(format!("no new Pareto-front member for {} round(s) (patience)", s2.spec.patience))
        } else if budget_out {
            Some("budget exhausted".to_string())
        } else {
            None
        };
        if let Some(reason) = stop {
            lines.push(format!("stopped: {reason}"));
            rt.append(exp, Actor::Search, vec![Event::SearchStopped { search: id.to_string(), reason }])?;
            break;
        }
    }
    let st = rt.state(exp)?;
    let s = &st.searches[id];
    let mut text = format!("search {id} front after {} round(s) ({}):\n", s.rounds_done, objectives_text(&s.spec.objectives));
    for c in &s.front {
        let cand = st.candidate(c).unwrap();
        let vals: Vec<String> = s.spec.objectives.iter().map(|o| value(&st, c, o, s.spec.evaluator).map(|v| format!("{v:.4}")).unwrap_or("—".into())).collect();
        text.push_str(&format!("  {c:>6} {:<12} [{}]  from {}\n", cand.formula, vals.join(", "), crate::events::origin_summary(&cand.origin)));
    }
    text.push_str("OBSERVATION: front members are not dominated on these objectives by anything evaluated here.\n  They are model predictions (energies: ML potential; Tc: ML surrogate of DFT), not verified materials.\n");
    Ok(Done { summary: format!("search {id}: {}", lines.join(" | ")), text: Some(text), candidates: s.front.clone(), ..Default::default() })
}

fn run_round(rt: &mut Runtime, exp: &str, st: &ExperimentState, spec: &SearchSpec, round: u32, ui: &Ui) -> Result<String> {
    let actor = Actor::Search;
    let mut ids = st.ids.clone();
    let rem = st.remaining();
    let round_seed = spec.seed ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let added: Added = match &spec.strategy {
        Strategy::Generative { generator: Generator::Mattergen } => {
            let n = spec.per_round.min(rem.mattergen_structures as u32);
            if n == 0 {
                bail!("MatterGen budget exhausted");
            }
            actions::generate_mattergen(rt, exp, st, actor, &mut ids, &spec.elements.join("-"), n, Some(round_seed), None, Some(spec.id.clone()))?
        }
        Strategy::Generative { generator: Generator::Chemfm } => {
            let n = spec.per_round.min(rem.chemfm_molecules as u32);
            if n == 0 {
                bail!("ChemFM budget exhausted");
            }
            actions::generate_chemfm(rt, exp, st, actor, &mut ids, n, Some(round_seed), Some(spec.id.clone()))?.0
        }
        Strategy::Evolutionary { substitution, strain_max, rattle_stdev } | Strategy::Random { substitution, strain_max, rattle_stdev } => {
            let pop = population(st, spec);
            if pop.is_empty() {
                bail!("no parent has all objective values from {:?} yet; evaluate the pool first (e.g. relax it{})", spec.evaluator, if spec.objectives.iter().any(|o| matches!(o.property, PropertyKind::FormationEnergyPerAtom | PropertyKind::EAboveHull)) { " and compute the hull" } else { "" });
            }
            let mut rng = Rng::new(round_seed);
            let points: Vec<Vec<f64>> = pop.iter().map(|p| oriented(&p.1, &spec.objectives)).collect();
            let ranks = pareto_ranks(&points);
            let evolutionary = matches!(spec.strategy, Strategy::Evolutionary { .. });
            let mut all = Added::default();
            for _ in 0..spec.per_round {
                let pick = if evolutionary {
                    let (a, b) = (rng.below(pop.len()), rng.below(pop.len()));
                    if ranks[a] < ranks[b] || (ranks[a] == ranks[b] && rng.below(2) == 0) { a } else { b }
                } else {
                    rng.below(pop.len())
                };
                let parent = &pop[pick].0;
                let h = st.current_structure(parent, Some(spec.evaluator)).unwrap();
                let s = rt.store.structure(&h)?;
                let m = choose_mutation(&mut rng, &s, &spec.elements, *substitution, *strain_max, *rattle_stdev);
                let a = actions::mutate(rt, exp, st, actor, &mut ids, parent, m, Some(spec.id.clone()))?;
                all.added.extend(a.added);
                all.rejected.extend(a.rejected);
                all.duplicates.extend(a.duplicates);
            }
            all
        }
    };
    // Evaluate what survived the structural check.
    let molecular = matches!(spec.strategy, Strategy::Generative { generator: Generator::Chemfm });
    let st_now = rt.state(exp)?;
    let mut evaluated = added.added.clone();
    if molecular {
        if !evaluated.is_empty() {
            actions::describe(rt, exp, &st_now, actor, &mut ids, &evaluated)?;
        }
    } else {
        evaluated.truncate(st_now.remaining().relaxations as usize);
        if !evaluated.is_empty() {
            let params = json!({"fmax": 0.02, "steps": 300, "relax_cell": true, "optimizer": "FIRE"});
            actions::relax(rt, exp, actor, &mut ids, &evaluated, spec.evaluator, params, ui)?;
        }
        if spec.objectives.iter().any(|o| matches!(o.property, PropertyKind::FormationEnergyPerAtom | PropertyKind::EAboveHull)) {
            let st_h = rt.state(exp)?;
            actions::compute_hull(rt, exp, &st_h, actor, &mut ids, spec.evaluator)?;
        }
        if !evaluated.is_empty() && spec.objectives.iter().any(|o| matches!(o.property, PropertyKind::Tc | PropertyKind::Lambda)) {
            let st_t = rt.state(exp)?;
            actions::estimate_tc(rt, exp, &st_t, actor, &mut ids, &evaluated)?;
        }
    }
    // Front over the whole population (pool + everything this search produced).
    let st_after = rt.state(exp)?;
    let pop = population(&st_after, spec);
    let points: Vec<Vec<f64>> = pop.iter().map(|p| oriented(&p.1, &spec.objectives)).collect();
    let ranks = pareto_ranks(&points);
    let front = distinct_minima(&st_after, spec, pop.iter().zip(&ranks).filter(|(_, r)| **r == 0).map(|(p, _)| p.0.clone()).collect());
    let improved = front.iter().any(|c| evaluated.contains(c));
    let rejected: Vec<String> = added.rejected.iter().map(|r| r.0.clone()).collect();
    let mut proposed = added.added.clone();
    proposed.extend(rejected.iter().cloned());
    let line = format!(
        "round {round}: {} proposed, {} failed structural check, {} duplicate(s), {} evaluated; front {:?}{}",
        proposed.len(),
        rejected.len(),
        added.duplicates.len(),
        evaluated.len(),
        front,
        if improved { " (improved)" } else { " (no new front member)" }
    );
    rt.append(exp, Actor::Search, vec![Event::SearchRoundCompleted {
        search: spec.id.clone(),
        round,
        proposed,
        rejected,
        duplicates: added.duplicates.len() as u32,
        evaluated,
        front,
        improved,
    }])?;
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pareto_ranks_two_objectives() {
        // minimize both: (1,3) and (3,1) are the front; (2,2) is also non-dominated; (3,3) dominated.
        let pts = vec![vec![1.0, 3.0], vec![3.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        assert_eq!(pareto_ranks(&pts), vec![0, 0, 0, 1]);
        assert!(dominates(&[1.0, 1.0], &[1.0, 2.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }
}
