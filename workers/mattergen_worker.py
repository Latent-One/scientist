"""MatterGen worker: diffusion model that samples crystal structures (GPU).

Op:
  generate  params: chemical_system ("B-C-N"), n, batch_size, guidance, seed,
                    checkpoint ("chemical_system")
            -> n structures (one item each)

Generation is not evaluation: outputs are proposals with fidelity "Generated".
"""
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, file_sha256, pmg_to_dict, versions  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo  # noqa: E402
from mattergen.generator import CrystalGenerator  # noqa: E402

DEFAULT_CKPT = "chemical_system"
t0 = time.perf_counter()
generators = {}


def generator(name):
    """Load each checkpoint once and keep it resident (decision A4)."""
    if name not in generators:
        info = MatterGenCheckpointInfo.from_hf_hub(name)
        g = CrystalGenerator(checkpoint_info=info, record_trajectories=False)
        g.prepare()
        generators[name] = g
    return generators[name]


generator(DEFAULT_CKPT)
load_s = time.perf_counter() - t0
w = Worker("mattergen")


def ckpt_file(name):
    g = generators[name]
    path = getattr(g.checkpoint_info, "checkpoint_path", None)
    return str(path) if path else None


@w.op("generate")
def generate(ctx, p, _inputs):
    name = p.get("checkpoint", DEFAULT_CKPT)
    g = generator(name)
    n = int(p["n"])
    batch = min(int(p.get("batch_size", 16)), n)
    g.properties_to_condition_on = {"chemical_system": p["chemical_system"]} if name == "chemical_system" else p.get("conditions")
    g.diffusion_guidance_factor = float(p.get("guidance", 2.0))
    # The diffusion step count is not a free parameter: the atom-type (discrete) corruption
    # was trained with N=1000 and the sampler asserts equality (pc_sampler.py:66, measured).
    g.seed = p.get("seed")
    g.progress_callback = lambda progress: ctx.progress(fraction=round(float(progress), 3))
    with tempfile.TemporaryDirectory() as out:
        structures = g.generate(batch_size=batch, num_batches=math.ceil(n / batch), output_dir=out)
    for i, s in enumerate(structures[:n]):
        ctx.item(i, {"structure": pmg_to_dict(s)})


ck = ckpt_file(DEFAULT_CKPT)
w.serve({"model": "mattergen", "checkpoint": DEFAULT_CKPT,
         "checkpoint_sha256": file_sha256(ck) if ck and os.path.exists(ck) else None,
         "load_s": round(load_s, 3), "versions": versions("mattergen", "torch", "pymatgen", "numpy")})
