"""CPU structure toolkit worker (pymatgen + ASE). No GPU, no learned model.

Ops (inputs are canonical structure dicts unless noted):
  build       params: prototype, formula, a, c, vacuum, supercell     -> 1 structure
  import      params: path                                            -> structures in file
  validate    params: min_distance, allowed_elements, max_sites       -> checks + metrics
  mutate      params: one mutation (substitute|vacancy|strain|rattle|swap|supercell)
  compare     inputs: [a, b]                                          -> match + differences
  symmetry    params: symprec                                         -> space group, formula
  hull        params: entries [{id, composition, energy_eV}]          -> formation energy, e_above_hull
  export      params: format (cif|poscar|xyz)                         -> text
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, atoms_to_dict, dict_to_pmg, pmg_to_dict, versions  # noqa: E402

import numpy as np  # noqa: E402
from ase import build as ase_build  # noqa: E402
from ase.formula import Formula  # noqa: E402
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Composition  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

w = Worker("structure")

PROTOTYPES = ["sc", "fcc", "bcc", "hcp", "diamond", "zincblende", "rocksalt", "cesiumchloride",
              "fluorite", "wurtzite", "graphene", "nanotube", "molecule"]


@w.op("build")
def build(ctx, p, _inputs):
    proto, formula = p["prototype"], p["formula"]
    if proto == "graphene":  # 2D sheet: 2 atoms in a hexagonal cell, vacuum along c
        # ASE needs one symbol per site of the 2-atom cell: "C" -> "C2"; "BN" stays (h-BN).
        if len(Formula(formula)) == 1:
            formula = formula * 2
        atoms = ase_build.graphene(formula=formula, a=p.get("a") or 2.46, size=(1, 1, 1),
                                   vacuum=p.get("vacuum") or 7.5)
    elif proto == "nanotube":  # single-wall tube rolled from a graphene sheet, periodic along z
        n, m = p["chirality"]
        els = Formula(formula).count()
        if len(els) != 1:
            raise ValueError("ASE builds single-element nanotubes only (e.g. C); BN tubes are not supported")
        atoms = ase_build.nanotube(n, m, length=p.get("length") or 1, bond=p.get("a") or 1.42,
                                   symbol=next(iter(els)), vacuum=p.get("vacuum") or 7.5)
    elif proto == "molecule":  # isolated molecule in a periodic box
        atoms = ase_build.molecule(formula)
        atoms.center(vacuum=p.get("vacuum") or 7.5)
    elif proto in PROTOTYPES:
        kw = {k: p[k] for k in ("a", "c") if p.get(k)}
        atoms = ase_build.bulk(formula, proto, **kw)
    else:
        raise ValueError(f"unknown prototype {proto!r}; known: {PROTOTYPES}")
    atoms.pbc = True
    if p.get("supercell"):
        atoms = atoms.repeat(tuple(p["supercell"]))
    ctx.item(0, {"structure": atoms_to_dict(atoms)})


@w.op("import")
def import_file(ctx, p, _inputs):
    from ase.io import read
    for i, atoms in enumerate(read(p["path"], index=":")):
        atoms.pbc = True
        ctx.item(i, {"structure": atoms_to_dict(atoms)})


def metrics(s):
    n = len(s)
    dm = s.distance_matrix + np.eye(n) * 1e9 if n > 1 else np.array([[np.inf]])
    # one-atom cells: nearest neighbour is the periodic image along the shortest lattice vector
    min_d = float(dm.min()) if n > 1 else float(min(s.lattice.abc))
    return {"num_sites": n, "formula": s.composition.reduced_formula,
            "elements": sorted({e.symbol for e in s.composition.elements}),
            "min_distance_A": round(min_d, 4), "volume_per_atom_A3": round(s.volume / n, 4),
            "density_g_cm3": round(float(s.density), 4)}


@w.op("validate")
def validate(ctx, p, inputs):
    min_dist = p.get("min_distance", 0.5)
    allowed = set(p.get("allowed_elements") or [])
    max_sites = p.get("max_sites", 200)
    for i, d in enumerate(inputs):
        s = dict_to_pmg(d)
        m = metrics(s)
        reasons = []
        if m["min_distance_A"] < min_dist:
            reasons.append(f"atoms closer than {min_dist} A (min {m['min_distance_A']} A)")
        if allowed and not set(m["elements"]) <= allowed:
            reasons.append(f"elements {sorted(set(m['elements']) - allowed)} outside allowed {sorted(allowed)}")
        if m["num_sites"] > max_sites:
            reasons.append(f"{m['num_sites']} sites > max {max_sites}")
        if not 2.0 <= m["volume_per_atom_A3"] <= 200.0:
            reasons.append(f"volume per atom {m['volume_per_atom_A3']} A^3 outside [2, 200]")
        try:
            sga = SpacegroupAnalyzer(s, symprec=0.1)
            m["spacegroup"], m["spacegroup_number"] = sga.get_space_group_symbol(), sga.get_space_group_number()
            m["crystal_system"] = sga.get_crystal_system()
        except Exception:  # spglib can fail on badly distorted cells; symmetry is optional here
            m["spacegroup"] = None
        ctx.item(i, {"valid": not reasons, "reasons": reasons, "metrics": m})


@w.op("mutate")
def mutate(ctx, p, inputs):
    s = dict_to_pmg(inputs[0])
    kind = p["kind"]
    if kind == "substitute":
        s.replace(p["site"], p["element"])
    elif kind == "vacancy":
        s.remove_sites([p["site"]])
    elif kind == "strain":
        eps = p["strain"]
        s.apply_strain(eps if isinstance(eps, list) else [eps, eps, eps])
    elif kind == "rattle":
        rng = np.random.default_rng(p["seed"])
        disp = rng.normal(0.0, p["stdev"], size=(len(s), 3))
        for i in range(len(s)):
            s.translate_sites([i], disp[i], frac_coords=False, to_unit_cell=True)
    elif kind == "swap":
        i, j = p["i"], p["j"]
        si, sj = s[i].specie, s[j].specie
        s.replace(i, sj)
        s.replace(j, si)
    elif kind == "supercell":
        s.make_supercell(p["scale"])
    else:
        raise ValueError(f"unknown mutation {kind!r}")
    ctx.item(0, {"structure": pmg_to_dict(s)})


@w.op("compare")
def compare(ctx, p, inputs):
    a, b = dict_to_pmg(inputs[0]), dict_to_pmg(inputs[1])
    sm = StructureMatcher()
    same = bool(sm.fit(a, b))
    rms = sm.get_rms_dist(a, b) if same else None
    ctx.item(0, {"same_structure": same, "rms_displacement": None if rms is None else round(float(rms[0]), 5),
                 "a": metrics(a), "b": metrics(b)})


@w.op("symmetry")
def symmetry(ctx, p, inputs):
    for i, d in enumerate(inputs):
        s = dict_to_pmg(d)
        try:
            sga = SpacegroupAnalyzer(s, symprec=p.get("symprec", 0.1))
            sym = {"spacegroup": sga.get_space_group_symbol(), "number": sga.get_space_group_number(),
                   "crystal_system": sga.get_crystal_system()}
        except Exception as e:  # spglib can fail on badly distorted cells
            sym = {"spacegroup": None, "error": str(e)}
        ctx.item(i, {**sym, **metrics(s)})


@w.op("hull")
def hull(ctx, p, _inputs):
    entries = [PDEntry(Composition(e["composition"]), e["energy_eV"], name=e["id"]) for e in p["entries"]]
    pd = PhaseDiagram(entries)  # raises if an element has no elemental reference entry
    for i, e in enumerate(entries):
        ctx.item(i, {"id": e.name, "formation_energy_per_atom_eV": round(float(pd.get_form_energy_per_atom(e)), 6),
                     "e_above_hull_eV": round(float(pd.get_e_above_hull(e)), 6)})


@w.op("export")
def export(ctx, p, inputs):
    s = dict_to_pmg(inputs[0])
    fmt = p.get("format", "cif")
    ctx.item(0, {"format": fmt, "text": s.to(fmt="poscar" if fmt == "poscar" else fmt)})


w.serve({"model": None, "versions": versions("pymatgen", "ase", "spglib", "numpy")})
