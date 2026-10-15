"""Machine-learned interatomic potential worker: MatterSim or MACE on the GPU.

Both models implement the same contract (docs/design.md, decision A6):
  single_point  structure -> energy, forces, stress            (atoms do not move)
  relax         structure -> relaxed structure + trajectory     (ASE optimizer moves atoms
                                                                 and, optionally, the cell)
usage: python potential_worker.py --model {mattersim,mace} [--dtype float64]
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, atoms_to_dict, dict_to_atoms, file_sha256, versions  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.optimize import BFGS, FIRE  # noqa: E402
from ase.units import GPa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mattersim", "mace"], required=True)
ap.add_argument("--dtype", default="float64", help="mace only; MACE recommends float64 for optimization")
args = ap.parse_args()

t0 = time.perf_counter()
if args.model == "mattersim":
    from mattersim.forcefield import MatterSimCalculator
    CKPT_NAME = "MatterSim-v1.0.0-1M.pth"
    calc = MatterSimCalculator(load_path=CKPT_NAME, device="cuda")
    ckpt_path = os.path.expanduser("~/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth")
    pkg = "mattersim"
    dtype = "float32"
else:
    from mace.calculators import mace_mp
    CKPT_NAME = "mace-mp-0 medium"
    calc = mace_mp(model="medium", device="cuda", default_dtype=args.dtype)
    # mace_mp("medium") resolves to the 2023-12-03 MACE-MP-0 medium checkpoint
    found = sorted(glob.glob(os.path.expanduser("~/.cache/mace/20231203mace128L1_epoch199model")))
    ckpt_path = found[0] if found else None
    pkg = "mace-torch"
    dtype = args.dtype
load_s = time.perf_counter() - t0

w = Worker(f"potential:{args.model}")


def props(atoms):
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    s = atoms.get_stress()  # eV/A^3, Voigt (xx, yy, zz, yz, xz, xy); ASE: positive = tensile
    n = len(atoms)
    return {"energy_eV": e, "energy_per_atom_eV": e / n,
            "max_force_eV_A": float(np.linalg.norm(f, axis=1).max()),
            "stress_GPa": (s / GPa).tolist(), "pressure_GPa": float(-s[:3].mean() / GPa),
            "volume_per_atom_A3": float(atoms.get_volume() / n), "forces_eV_A": f.tolist()}


@w.op("single_point")
def single_point(ctx, p, inputs):
    for i, d in enumerate(inputs):
        atoms = dict_to_atoms(d)
        atoms.calc = calc
        t = time.perf_counter()
        out = props(atoms)
        out["compute_s"] = time.perf_counter() - t
        ctx.item(i, out)


@w.op("relax")
def relax(ctx, p, inputs):
    fmax, steps = p.get("fmax", 0.05), p.get("steps", 300)
    relax_cell = p.get("relax_cell", True)
    opt_cls = {"FIRE": FIRE, "BFGS": BFGS}[p.get("optimizer", "FIRE")]
    for i, d in enumerate(inputs):
        atoms = dict_to_atoms(d)
        atoms.calc = calc
        start_pos = atoms.get_positions().copy()
        start = props(atoms)
        target = FrechetCellFilter(atoms) if relax_cell else atoms
        opt = opt_cls(target, logfile=None)
        traj = []

        def record():
            e = float(atoms.get_potential_energy())
            fm = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            row = {"step": opt.nsteps, "energy_per_atom_eV": e / len(atoms), "max_force_eV_A": fm,
                   "volume_per_atom_A3": float(atoms.get_volume() / len(atoms))}
            traj.append(row)
            ctx.progress(item=i, **row)

        opt.attach(record, interval=1)
        t = time.perf_counter()
        try:
            converged = bool(opt.run(fmax=fmax, steps=steps))
        except Exception as e:  # numerical blow-up is a per-structure observation, not a crash
            ctx.item_error(i, "relaxation_failed", f"{type(e).__name__}: {e}")
            continue
        end = props(atoms)
        disp = atoms.get_positions() - start_pos  # cartesian; includes cell-change contribution
        ctx.item(i, {"converged": converged, "steps": opt.nsteps, "compute_s": time.perf_counter() - t,
                     "fmax_target_eV_A": fmax, "relax_cell": relax_cell, "optimizer": p.get("optimizer", "FIRE"),
                     "initial": {k: v for k, v in start.items() if k != "forces_eV_A"},
                     "final": end, "max_displacement_A": float(np.linalg.norm(disp, axis=1).max()),
                     "trajectory": traj, "structure": atoms_to_dict(atoms)})


w.serve({"model": args.model, "checkpoint": CKPT_NAME, "dtype": dtype,
         "checkpoint_sha256": file_sha256(ckpt_path) if ckpt_path and os.path.exists(ckpt_path) else None,
         "load_s": round(load_s, 3), "device": torch.cuda.get_device_name(0),
         "versions": versions(pkg, "torch", "ase", "numpy")})
