"""Worker side of the scientist protocol (see docs/design.md, section E).

A worker is one long-lived Python process that owns one model (or a CPU toolkit).
It reads one JSON request per line on stdin and writes JSON lines on stdout:

    hello     once, after the model is loaded
    progress  zero or more per request (optimizer steps, diffusion batches)
    item      one per input (or per generated output), ok or error
    done      exactly one per request, with timing and GPU memory

Library output (tqdm bars, loguru, print) must never reach the protocol stream, so
__init__ duplicates file descriptor 1 for the protocol and points fd 1 at stderr.
"""
import json
import os
import sys
import time
import traceback

# If the scientist runtime dies (even by SIGKILL), the Linux kernel kills this worker too,
# so an orphan never keeps the GPU busy. prctl(PR_SET_PDEATHSIG=1, SIGKILL). Linux only.
try:
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    if os.getppid() == 1:  # the parent was already gone before prctl ran
        sys.exit(0)
except OSError:
    pass

# Claim the protocol stream at import time, before any model library loads and prints.
_PROTOCOL_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


class Ctx:
    """Handed to an op: lets it stream progress and per-item results."""

    def __init__(self, worker, rid):
        self.w, self.rid, self.items = worker, rid, 0

    def progress(self, **data):
        self.w.send({"type": "progress", "id": self.rid, **data})

    def item(self, index, output):
        self.items += 1
        self.w.send({"type": "item", "id": self.rid, "index": index, "ok": True, "output": output})

    def item_error(self, index, kind, message):
        self.items += 1
        self.w.send({"type": "item", "id": self.rid, "index": index, "ok": False,
                     "error": {"kind": kind, "message": message}})


class Worker:
    def __init__(self, name):
        self._out = _PROTOCOL_OUT
        self.name = name
        self.ops = {}

    def op(self, name):
        def register(fn):
            self.ops[name] = fn
            return fn
        return register

    def send(self, msg):
        self._out.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self._out.flush()

    @staticmethod
    def gpu_mb():
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_reserved() / 2**20, 1)
        except ImportError:
            pass
        return 0.0

    def serve(self, hello):
        self.send({"type": "hello", "worker": self.name, "pid": os.getpid(),
                   "ops": sorted(self.ops), "gpu_mb": self.gpu_mb(), **hello})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("type") == "shutdown":
                break
            rid = req.get("id", "?")
            ctx = Ctx(self, rid)
            t = time.perf_counter()
            fn = self.ops.get(req.get("op"))
            try:
                if fn is None:
                    raise ValueError(f"unknown op {req.get('op')!r}; this worker serves {sorted(self.ops)}")
                fn(ctx, req.get("params") or {}, req.get("inputs") or [])
                self.send({"type": "done", "id": rid, "wall_s": round(time.perf_counter() - t, 4),
                           "items": ctx.items, "gpu_mb": self.gpu_mb()})
            except Exception as e:  # report every failure as data; the process stays up
                kind = "oom" if "OutOfMemory" in type(e).__name__ else "tool_error"
                self.send({"type": "done", "id": rid, "wall_s": round(time.perf_counter() - t, 4),
                           "items": ctx.items, "gpu_mb": self.gpu_mb(),
                           "error": {"kind": kind, "message": f"{type(e).__name__}: {e}"[:2000],
                                     "traceback": traceback.format_exc()[-4000:]}})
                if kind == "oom":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass


# ---- canonical structure form shared with Rust (src/types.rs::Structure) ----

def atoms_to_dict(atoms):
    import numpy as np
    frac = np.mod(atoms.get_scaled_positions(wrap=True), 1.0)
    return {"lattice": atoms.cell.array.tolist(), "species": atoms.get_chemical_symbols(),
            "frac_coords": frac.tolist()}


def dict_to_atoms(d):
    from ase import Atoms
    return Atoms(symbols=d["species"], scaled_positions=d["frac_coords"], cell=d["lattice"], pbc=True)


def pmg_to_dict(s):
    import numpy as np
    return {"lattice": s.lattice.matrix.tolist(), "species": [site.specie.symbol for site in s],
            "frac_coords": np.mod(s.frac_coords, 1.0).tolist()}


def dict_to_pmg(d):
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(d["lattice"]), d["species"], d["frac_coords"])


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def versions(*packages):
    import importlib.metadata as m
    out = {"python": sys.version.split()[0]}
    for p in packages:
        try:
            out[p] = m.version(p)
        except m.PackageNotFoundError:
            out[p] = None
    return out
