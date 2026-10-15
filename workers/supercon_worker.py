"""Superconductivity estimators (ML surrogates trained on DFT electron-phonon data).

usage: python supercon_worker.py --model {alignn,betenet}

  alignn   ALIGNN 2.0 Alexandria models (alignn 2026.8.11): alex_supercon_Tc, _la (lambda), _wlog
           Validated on this machine 2026-10-08: test-split MAE 0.856 K vs 0.883 K reported.
  betenet  BETE-NET CSO ensemble, 100 models (henniggroup/BETE-NET @ 56bc7839): predicts the
           Eliashberg spectral function a2F(w); lambda, omega_log and Allen-Dynes Tc (mu*=0.09)
           follow from it with the authors' own notebook functions.
           Validated 2026-10-08: test-split Tc MAE 2.34 K vs ~2.5 K reported.

Op:
  tc   inputs: structures -> per structure: tc_K (+ lambda, omega_log_K; betenet adds tc_std_K, a2f)

Both predict DFT-level quantities, not measurements. On 12 measured materials both failed MgB2
(predicted 1-7 K vs 38 K measured): multi-gap superconductors are outside what they capture.
"""
import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, dict_to_atoms, versions  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["alignn", "betenet"], required=True)
ap.add_argument("--betenet-repo", default=os.path.expanduser("~/scientist-spike/supercon/BETE-NET"))
args = ap.parse_args()
device = "cuda"
t0 = time.perf_counter()

if args.model == "alignn":
    from alignn.graphs import Graph
    from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig
    from alignn.pretrained import ALIGNN2_MODELS, get_alignn2_model
    from jarvis.core.atoms import ase_to_atoms

    NAMES = {"tc_K": "alex_supercon_Tc", "lambda": "alex_supercon_la", "omega_log_K": "alex_supercon_wlog"}
    models = {}
    for key, name in NAMES.items():
        paths = get_alignn2_model(name)
        cfg = json.load(open(paths["config.json"]))
        m = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**cfg["model"]))
        m.load_state_dict(torch.load(paths["best_model.pt"], map_location=device, weights_only=True))
        models[key] = (m.to(device).eval(), cfg)

    def scalar(m, cfg, ja):
        g, lg = Graph.atom_dgl_multigraph(ja, neighbor_strategy="pure_torch", cutoff=float(cfg.get("cutoff", 5.0)),
                                          max_neighbors=int(cfg.get("max_neighbors", 12)),
                                          atom_features=cfg.get("atom_features", "cgcnn"),
                                          use_canonize=bool(cfg.get("use_canonize", False)))
        lat = torch.tensor(ja.lattice_mat).type(torch.get_default_dtype())
        with torch.no_grad():
            out = m((g.to(device), lg.to(device), lat.to(device)))
        if isinstance(out, dict):
            out = out.get("out", out.get("energy", next(iter(out.values()))))
        return float(out.detach().cpu().numpy().flatten()[0])

    def predict(d):
        ja = ase_to_atoms(dict_to_atoms(d))
        return {k: scalar(m, cfg, ja) for k, (m, cfg) in models.items()}

    identity = {"model": "alex_supercon_Tc", "checkpoint": ",".join(NAMES.values()),
                "test_mae_K": ALIGNN2_MODELS["alex_supercon_Tc"]["test_mae"]}
    pkg = "alignn"
else:
    torch.serialization.add_safe_globals([slice])  # e3nn 0.4.2 constants file; safe loading stays on
    sys.path.insert(0, args.betenet_repo)
    import ase.io
    import pandas as pd
    import torch_geometric as tg
    from utils.data import Freq_final, build_data, get_neighbors, get_target
    from utils.training import get_model

    nb = json.load(open(f"{args.betenet_repo}/notebooks/Pred_CSO.ipynb"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ns = {"np": np}
    for fname in ("cal_lamb", "cal_w_log", "cal_tc"):
        exec(re.search(rf"^def {fname}\(.*?(?=^\S)", src + "\nEND", re.S | re.M).group(0), ns)

    # num_neighbors is a training-set statistic baked into the model: compute it exactly as the
    # authors do, from the full training database (not from the structures being predicted).
    df = pd.read_json(f"{args.betenet_repo}/database.json")
    df.dropna(inplace=True)
    df["structure"] = [ase.io.read(f"{args.betenet_repo}/structures/{i}.cif") for i in df.index]
    df["target"] = df.apply(get_target, axis=1)
    df["data"] = df.apply(build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4, axis=1)
    num_neighbors = float(get_neighbors(df, df.index).mean())
    del df
    init = dict(in_dim=118, em_dim=64, irreps_in="64x0e", irreps_out=f"{len(Freq_final)}x0e", irreps_node_attr="64x0e",
                layers=2, mul=32, lmax=1, max_radius=4, num_neighbors=num_neighbors, reduce_output=True, p=0.0)
    ensemble = []
    for k in range(100):
        m, _, _ = get_model(init, device=device)
        m.load_state_dict(torch.load(f"{args.betenet_repo}/CSO/model_cso_{k}.pt", map_location=device))
        m.pool = True
        ensemble.append(m.eval())

    def summarize(a2f):
        lam = ns["cal_lamb"](Freq_final, a2f)
        wlog = ns["cal_w_log"](Freq_final, a2f, lam) / 0.08617
        tc = ns["cal_tc"](lam, wlog)
        ok = np.isfinite(tc) and lam > 0.09 * (1 + 0.62 * lam)
        return float(lam), float(wlog), float(tc) if ok else 0.0

    def predict_batch(ds):
        """All structures in one PyG batch per model (the model reads data.batch)."""
        graphs = [build_data(SimpleNamespace(structure=dict_to_atoms(d), target=np.zeros(len(Freq_final))),
                             embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4) for d in ds]
        batch = tg.data.Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            # each model gets a fresh copy: the network overwrites batch.x in place (118 -> 64 features)
            outs = np.stack([m(batch.clone()).cpu().numpy() for m in ensemble])  # (models, structures, freqs)
        res = []
        for j in range(len(ds)):
            lam, wlog, tc = summarize(outs[:, j].mean(axis=0))
            per = np.array([summarize(o)[2] for o in outs[:, j]])
            res.append({"tc_K": tc, "tc_std_K": float(per.std()), "lambda": lam, "omega_log_K": wlog,
                        "a2f": [round(float(x), 6) for x in outs[:, j].mean(axis=0)], "freq_meV": [float(x) for x in Freq_final]})
        return res

    identity = {"model": "bete-net-cso", "checkpoint": "henniggroup/BETE-NET@56bc7839 CSO x100",
                "num_neighbors": num_neighbors}
    pkg = "e3nn"

load_s = time.perf_counter() - t0
w = Worker(f"supercon:{args.model}")


@w.op("tc")
def tc(ctx, p, inputs):
    if args.model == "betenet":
        for i, r in enumerate(predict_batch(inputs)):
            ctx.item(i, r)
        return
    for i, d in enumerate(inputs):
        try:
            ctx.item(i, predict(d))
        except Exception as e:  # e.g. an element the model has never seen
            ctx.item_error(i, "prediction_failed", f"{type(e).__name__}: {e}")


w.serve({**identity, "load_s": round(load_s, 3), "dtype": str(torch.get_default_dtype()).replace("torch.", ""),
         "versions": versions(pkg, "torch", "ase")})
