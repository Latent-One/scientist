"""Materials Project worker: known crystal phases and their DFT data (network, CPU).

Op:
  search  params: chemical_system ("B-C-N"), include_subsystems (bool), max_e_above_hull (eV/atom),
                  limit (per request, lowest e_above_hull first)
          -> one item per material: id, formula, structure, DFT values, provenance

The API key is read from $MP_API_KEY or ~/.config/scientist/mp_api_key (never logged).
Values returned are database DFT results, not computed here, and are on Materials Project's
own energy scale: never compare them directly with MatterSim or MACE energies.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, pmg_to_dict, versions  # noqa: E402

key = os.environ.get("MP_API_KEY")
if not key:
    path = os.path.expanduser("~/.config/scientist/mp_api_key")
    key = open(path).read().strip() if os.path.exists(path) else None
if not key:
    raise SystemExit("no Materials Project API key: set MP_API_KEY or write ~/.config/scientist/mp_api_key")

from mp_api.client import MPRester  # noqa: E402

mpr = MPRester(key, mute_progress_bars=True)
try:
    db_version = mpr.get_database_version()
except Exception as e:  # version is provenance, not essential
    db_version = f"unknown ({type(e).__name__})"
w = Worker("mp")

FIELDS = ["material_id", "formula_pretty", "structure", "energy_per_atom", "formation_energy_per_atom",
          "energy_above_hull", "band_gap", "symmetry", "nsites", "theoretical", "is_stable"]


@w.op("search")
def search(ctx, p, _inputs):
    els = sorted(set(p["chemical_system"].split("-")))
    systems = ["-".join(s) for r in range(1, len(els) + 1) for s in itertools.combinations(els, r)] \
        if p.get("include_subsystems", True) else ["-".join(els)]
    docs = mpr.materials.summary.search(chemsys=systems, energy_above_hull=(0, float(p.get("max_e_above_hull", 0.1))),
                                        fields=FIELDS)
    docs = sorted(docs, key=lambda d: (d.energy_above_hull, d.nsites))[: int(p.get("limit", 30))]
    for i, d in enumerate(docs):
        ctx.item(i, {
            "material_id": str(d.material_id), "formula": d.formula_pretty, "nsites": d.nsites,
            "structure": pmg_to_dict(d.structure),
            "energy_per_atom_eV": d.energy_per_atom, "formation_energy_per_atom_eV": d.formation_energy_per_atom,
            "e_above_hull_eV": d.energy_above_hull, "band_gap_eV": d.band_gap, "is_stable": d.is_stable,
            "spacegroup": d.symmetry.symbol if d.symmetry else None, "theoretical": d.theoretical,
            "database_version": db_version, "url": f"https://next-gen.materialsproject.org/materials/{d.material_id}",
        })


@w.op("structures")
def structures(ctx, p, _inputs):
    """Structures by Materials Project id, in the order given."""
    ids = list(p["ids"])
    docs = {str(d.material_id): d for d in mpr.materials.summary.search(material_ids=ids, fields=["material_id", "structure", "formula_pretty", "energy_above_hull", "symmetry"])}
    for i, mid in enumerate(ids):
        d = docs.get(mid)
        if d is None:
            ctx.item_error(i, "not_found", f"{mid} not in Materials Project {db_version}")
            continue
        ctx.item(i, {"material_id": mid, "formula": d.formula_pretty, "structure": pmg_to_dict(d.structure),
                     "e_above_hull_eV": d.energy_above_hull, "spacegroup": d.symmetry.symbol if d.symmetry else None})


UNCONVENTIONAL = ("Cuprate", "Ferrite", "Heavy_fermion")


@w.op("known_superconductors")
def known_superconductors(ctx, p, _inputs):
    """Measured superconductors from 3DSC (SuperCon matched to MP structures), grouped by parent
    MP structure. Tc = median of measurements whose composition is within max_mismatch of it."""
    import pandas as pd
    df = pd.read_csv(os.path.expanduser(p["csv"]), header=1, low_memory=False)
    df = df[(df.tc > 0) & (df.totreldiff <= float(p.get("max_mismatch", 0.05)))]
    if not p.get("include_unconventional", False):
        df = df[~df.sc_class.astype(str).str.contains("|".join(UNCONVENTIONAL))]
    allowed = set(p.get("elements") or [])
    groups = []
    from pymatgen.core import Composition
    for mid, g in df.groupby("material_id_2"):
        els = {e.symbol for e in Composition(str(g.pretty_formula_2.iloc[0])).elements}
        if allowed and not els <= allowed:
            continue
        tc = float(g.tc.median())
        if tc < float(p.get("min_tc", 0.0)):
            continue
        groups.append((tc, mid, g))
    groups.sort(key=lambda x: -x[0])
    groups = groups[: int(p.get("limit", 20))]
    docs = {str(d.material_id): d for d in mpr.materials.summary.search(material_ids=[m for _, m, _ in groups],
                                                                        fields=["material_id", "structure", "energy_above_hull"])} if groups else {}
    for i, (tc, mid, g) in enumerate(groups):
        d = docs.get(mid)
        if d is None:
            ctx.item_error(i, "not_found", f"{mid} not in Materials Project {db_version}")
            continue
        ctx.item(i, {"material_id": mid, "parent_formula": str(g.pretty_formula_2.iloc[0]), "structure": pmg_to_dict(d.structure),
                     "tc_median_K": tc, "tc_min_K": float(g.tc.min()), "tc_max_K": float(g.tc.max()), "n_measurements": int(len(g)),
                     "measured_formulas": sorted(set(map(str, g.formula_sc)))[:5], "composition_mismatch_max": float(g.totreldiff.max()),
                     "sc_class": sorted(set(map(str, g.sc_class))), "e_above_hull_eV": d.energy_above_hull,
                     "source": "3DSC (SuperCon measurements matched to Materials Project), CC BY 4.0"})


w.serve({"model": "materials_project", "checkpoint": f"MP database {db_version}", "database_version": db_version,
         "versions": versions("mp-api", "pymatgen")})
