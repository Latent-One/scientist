//! Scientific objects (design.md, section C).
//!
//! A number is never stored alone: properties live inside observations, which carry the
//! method (model, checkpoint hash, versions, parameters), the fidelity, the exact input
//! structure hash, and the job that produced them.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub type Hash = String;

/// All 118 element symbols, for validating anything that names an element.
pub const ELEMENTS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

pub fn sha256(bytes: &[u8]) -> Hash {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    // One allocation of the exact size; hex digits from a table instead of format! per byte.
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 15) as usize] as char);
    }
    out
}

/// JSON with object keys sorted at every level. Cache keys and identity fingerprints hash
/// this form, so they never depend on how a map happens to order its keys.
/// Written straight into one buffer: no sorted copy of the tree is built.
pub fn canonical_json(v: &serde_json::Value) -> String {
    fn write(v: &serde_json::Value, out: &mut Vec<u8>) {
        match v {
            serde_json::Value::Object(m) => {
                let mut entries: Vec<(&String, &serde_json::Value)> = m.iter().collect();
                entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
                out.push(b'{');
                for (i, (k, val)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    serde_json::to_writer(&mut *out, k).expect("string key");
                    out.push(b':');
                    write(val, out);
                }
                out.push(b'}');
            }
            serde_json::Value::Array(a) => {
                out.push(b'[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write(x, out);
                }
                out.push(b']');
            }
            scalar => serde_json::to_writer(&mut *out, scalar).expect("scalar"),
        }
    }
    let mut out = Vec::with_capacity(256);
    write(v, &mut out);
    // Only serde_json wrote string bytes, so the buffer is valid UTF-8.
    String::from_utf8(out).expect("utf-8")
}

pub fn short(hash: &str) -> &str {
    let h = hash.strip_prefix("sha256:").unwrap_or(hash);
    &h[..h.len().min(12)]
}

/// A periodic crystal: three cell vectors plus atoms in fractional coordinates.
/// Rows of `lattice` are the cell vectors a, b, c in ångström (Å, 1e-10 m).
/// A fractional coordinate (0.25, 0.25, 0.25) means 0.25·a + 0.25·b + 0.25·c.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Structure {
    pub lattice: [[f64; 3]; 3],
    pub species: Vec<String>,
    pub frac_coords: Vec<[f64; 3]>,
}

fn round_to(x: f64, scale: f64) -> f64 {
    (x * scale).round() / scale + 0.0 // "+ 0.0" turns -0.0 into 0.0 so bytes are stable
}

fn wrap(x: f64) -> f64 {
    let y = x - x.floor();
    if y >= 1.0 - 1e-9 { 0.0 } else { y }
}

impl Structure {
    /// Canonical form used for hashing: coordinates wrapped into [0,1), rounded
    /// (lattice to 1e-6 Å, fractions to 1e-7), sites sorted by (element, coordinates).
    /// Two runs that produce the same crystal up to float noise and atom order hash equal.
    /// Different cell choices for the same crystal still hash differently; the structure
    /// worker's `compare` op (pymatgen StructureMatcher) is the tool for that question.
    pub fn canonical(&self) -> Structure {
        let lattice = self.lattice.map(|row| row.map(|x| round_to(x, 1e6)));
        let mut sites: Vec<(String, [f64; 3])> = self
            .species
            .iter()
            .zip(&self.frac_coords)
            .map(|(s, f)| (s.clone(), f.map(|x| wrap(round_to(wrap(x), 1e7)))))
            .collect();
        sites.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        Structure {
            lattice,
            species: sites.iter().map(|s| s.0.clone()).collect(),
            frac_coords: sites.iter().map(|s| s.1).collect(),
        }
    }

    pub fn composition(&self) -> BTreeMap<String, u32> {
        let mut c = BTreeMap::new();
        for s in &self.species {
            *c.entry(s.clone()).or_insert(0) += 1;
        }
        c
    }

    /// Reduced formula with elements in alphabetical order, e.g. B2C4N2 -> "BC2N".
    pub fn formula(&self) -> String {
        formula_of(&self.composition())
    }

    /// Cell lengths (Å) and angles α=∠(b,c), β=∠(a,c), γ=∠(a,b) in degrees.
    pub fn cell_params(&self) -> ([f64; 3], [f64; 3]) {
        let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let ang = |a: [f64; 3], b: [f64; 3]| ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (len(a) * len(b))).acos().to_degrees();
        let [a, b, c] = self.lattice;
        ([len(a), len(b), len(c)], [ang(b, c), ang(a, c), ang(a, b)])
    }

    pub fn cell_text(&self) -> String {
        let ([a, b, c], [al, be, ga]) = self.cell_params();
        format!("cell a={a:.4} b={b:.4} c={c:.4} Å, α={al:.1} β={be:.1} γ={ga:.1}°")
    }

    pub fn volume(&self) -> f64 {
        let [a, b, c] = self.lattice;
        (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            .abs()
    }
}

pub fn formula_of(comp: &BTreeMap<String, u32>) -> String {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = comp.values().fold(0, |acc, &n| gcd(acc, n)).max(1);
    comp.iter()
        .map(|(e, &n)| if n / g == 1 { e.clone() } else { format!("{e}{}", n / g) })
        .collect()
}

/// A molecule as canonical SMILES (a line notation: atoms and bonds as text, no
/// coordinates). Canonical form depends on the RDKit version, which is recorded.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Molecule {
    pub smiles: String,
    pub toolkit: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Structure,
    Molecule,
    Trajectory,
    /// An external record (paper metadata + abstract) exactly as retrieved.
    Record,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PropertyKind {
    EnergyPerAtom,
    FormationEnergyPerAtom,
    EAboveHull,
    MaxForce,
    Pressure,
    VolumePerAtom,
    Density,
    MolWt,
    Logp,
    Tpsa,
    Qed,
    FractionSp3,
    AromaticRings,
    BandGap,
    /// Superconducting critical temperature (K), ML-estimated or measured.
    Tc,
    /// Electron-phonon coupling constant (dimensionless).
    Lambda,
    /// Logarithmic-average phonon frequency, in kelvin.
    OmegaLog,
}

impl PropertyKind {
    pub fn unit(self) -> &'static str {
        use PropertyKind::*;
        match self {
            EnergyPerAtom | FormationEnergyPerAtom | EAboveHull => "eV/atom",
            MaxForce => "eV/Å",
            Pressure => "GPa",
            VolumePerAtom => "Å³/atom",
            Density => "g/cm³",
            MolWt => "g/mol",
            Tpsa => "Å²",
            BandGap => "eV",
            Tc | OmegaLog => "K",
            Lambda => "",
            Logp | Qed | FractionSp3 | AromaticRings => "",
        }
    }

    /// One-line meaning, shown by `inspect`, `evidence`, and given to the planner.
    pub fn meaning(self) -> &'static str {
        use PropertyKind::*;
        match self {
            EnergyPerAtom => "model total energy divided by atom count; only differences between structures of the same composition are meaningful",
            FormationEnergyPerAtom => "energy released (negative) or needed (positive) to form the compound from its elements, per atom, using this experiment's elemental references",
            EAboveHull => "distance above the lowest-energy combination of phases known in THIS experiment; 0 = on the internal hull (not the Materials Project hull)",
            MaxForce => "largest force on any atom; near 0 means the atoms sit at a local energy minimum",
            Pressure => "pressure the model predicts the crystal exerts on its cell; positive = squeezed",
            VolumePerAtom => "cell volume divided by atom count",
            Density => "mass per volume",
            MolWt => "molecular weight",
            Logp => "Crippen estimate of octanol/water partition coefficient (log10); higher = more oil-loving",
            Tpsa => "topological polar surface area: surface from N and O atoms and their hydrogens",
            Qed => "quantitative estimate of drug-likeness in [0,1]; a pharma heuristic, not a materials metric",
            FractionSp3 => "fraction of carbon atoms with four single bonds (sp3)",
            AromaticRings => "number of aromatic rings",
            Tc => "superconducting critical temperature: below it resistance is exactly zero (only a measurement proves this; ML values estimate a DFT calculation)",
            Lambda => "electron-phonon coupling strength: how strongly electrons interact with lattice vibrations; Tc grows roughly exponentially with it",
            OmegaLog => "typical lattice-vibration frequency, expressed as a temperature; light atoms vibrate faster",
            BandGap => "energy gap between filled and empty electron states (database DFT value; common functionals underestimate gaps)",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Property {
    pub kind: PropertyKind,
    pub value: f64,
    pub unit: String,
}

impl Property {
    pub fn new(kind: PropertyKind, value: f64) -> Self {
        Property { kind, value, unit: kind.unit().to_string() }
    }
}

/// How much a number can be trusted, lowest to highest. Ordering is meaningful.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    /// A generative model proposed it. Not evidence of anything except "might exist".
    Generated,
    /// Deterministic geometry checks (distances, symmetry, density).
    StructuralCheck,
    /// Deterministic descriptors computed from a molecule graph (RDKit).
    Descriptor,
    /// ML model trained to imitate an expensive DFT-derived quantity (e.g. Tc from electron-phonon
    /// coupling). Two layers of error: model vs DFT, and DFT vs reality.
    MlSurrogate,
    /// Machine-learned interatomic potential trained on DFT data (MatterSim, MACE).
    MlPotential,
    /// Density functional theory. No DFT tool is installed in V0.
    Dft,
    /// Physical measurement.
    Experiment,
}

/// Everything needed to say which computation produced a number.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Method {
    pub worker: String,
    pub op: String,
    pub model: Option<String>,
    pub checkpoint: Option<String>,
    pub checkpoint_sha256: Option<String>,
    pub dtype: Option<String>,
    pub versions: serde_json::Value,
    pub params: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Evaluator {
    Mattersim,
    Mace,
}

impl Evaluator {
    pub fn worker(self) -> &'static str {
        match self {
            Evaluator::Mattersim => "mattersim",
            Evaluator::Mace => "mace",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Generator {
    Mattergen,
    Chemfm,
}

/// A change applied to a parent structure. Site indices refer to the canonical
/// (sorted) site order shown by `scientist inspect`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Mutation {
    Substitute { site: usize, element: String },
    Vacancy { site: usize },
    Strain { strain: f64 },
    Rattle { stdev: f64, seed: u64 },
    Swap { i: usize, j: usize },
    Supercell { scale: [u32; 3] },
}

impl Mutation {
    pub fn describe(&self) -> String {
        match self {
            Mutation::Substitute { site, element } => format!("replace site {site} with {element}"),
            Mutation::Vacancy { site } => format!("remove site {site}"),
            Mutation::Strain { strain } => format!("strain all cell vectors by {:+.2}%", strain * 100.0),
            Mutation::Rattle { stdev, seed } => format!("random displacements σ={stdev} Å (seed {seed})"),
            Mutation::Swap { i, j } => format!("swap sites {i} and {j}"),
            Mutation::Supercell { scale } => format!("supercell {}x{}x{}", scale[0], scale[1], scale[2]),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Origin {
    Built { prototype: String, params: serde_json::Value },
    Imported { path: String },
    Generated { generator: Generator, params: serde_json::Value },
    Mutated { parent: String, mutation: Mutation },
    /// A known phase retrieved from a database (structure + the database's own DFT values).
    Database { source: String, id: String, version: String, url: String },
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Proposed,
    Promoted,
    Rejected,
}

/// Limits. Consumption is derived from events (state.rs), never stored separately.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Budget {
    pub mattergen_structures: u64,
    pub chemfm_molecules: u64,
    pub relaxations: u64,
    pub single_points: u64,
    pub planner_calls: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Budget { mattergen_structures: 64, chemfm_molecules: 1000, relaxations: 200, single_points: 1000, planner_calls: 200 }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Min,
    Max,
}

/// One axis of a multi-objective search. Kept separate (no single score) so that
/// trade-offs stay visible on a Pareto front (plan section 20).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Objective {
    pub property: PropertyKind,
    pub direction: Direction,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The previous implementation (sorted copy of the tree, then to_string). Cache keys hash this
    /// form, so the fast writer must produce identical bytes or every cache entry would miss.
    fn canonical_json_reference(v: &serde_json::Value) -> String {
        fn sort(v: &serde_json::Value) -> serde_json::Value {
            match v {
                serde_json::Value::Object(m) => {
                    let mut keys: Vec<&String> = m.keys().collect();
                    keys.sort();
                    serde_json::Value::Object(keys.into_iter().map(|k| (k.clone(), sort(&m[k]))).collect())
                }
                serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(sort).collect()),
                other => other.clone(),
            }
        }
        sort(v).to_string()
    }

    #[test]
    fn canonical_json_matches_reference() {
        let cases = [
            serde_json::json!({"z": 1, "a": [3, {"y": "q\"\n\u{e9}", "b": null}], "m": {"k": 1.5e-9, "c": true, "b": -0.0}}),
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::json!("plain"),
            serde_json::json!({"worker": "mace", "op": "relax", "params": {"fmax": 0.01, "steps": 500}, "inputs": ["sha256:ab", "sha256:cd"], "identity": "sha256:ef"}),
        ];
        for c in &cases {
            assert_eq!(canonical_json(c), canonical_json_reference(c));
        }
        let s = sha256(b"abc");
        assert_eq!(s, "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn canonical_is_order_and_noise_invariant() {
        let a = Structure {
            lattice: [[0.0, 1.7835, 1.7835], [1.7835, 0.0, 1.7835], [1.7835, 1.7835, 0.0]],
            species: vec!["C".into(), "N".into()],
            frac_coords: vec![[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        };
        let b = Structure {
            lattice: a.lattice,
            species: vec!["N".into(), "C".into()],
            frac_coords: vec![[0.25 + 1e-10, 0.25, 1.25], [-1e-12, 1.0, 0.0]],
        };
        let (ca, cb) = (a.canonical(), b.canonical());
        assert_eq!(serde_json::to_vec(&ca).unwrap(), serde_json::to_vec(&cb).unwrap());
        assert_eq!(a.formula(), "CN");
        assert!((a.volume() - 11.3462).abs() < 1e-3);
    }

    #[test]
    fn reduced_formula() {
        let comp = BTreeMap::from([("B".to_string(), 2), ("C".to_string(), 4), ("N".to_string(), 2)]);
        assert_eq!(formula_of(&comp), "BC2N");
    }
}
