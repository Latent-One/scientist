"""Molecule worker: ChemFM-1B (GPU, generates SMILES) + RDKit (CPU, deterministic chemistry).

Ops:
  chemfm_generate   params: n, seed, temperature, top_k, max_new_tokens, batch_size
                    -> one item per sampled string: raw text, valid, canonical SMILES, reason
  rdkit_validate    inputs: [smiles] -> canonical SMILES, validity, elements, atom counts
  rdkit_descriptors inputs: [smiles] -> computed descriptors (MolWt, LogP, TPSA, ...)

Two traps found while testing on this machine (2026-10-07):
  - config.json bos/eos ids (1, 2) are the element tokens "He"/"Li"; the real separator is
    <eos> (id 265). A lone <eos> at position 0 predicts <eos> with p=1.0, so sampling starts
    from the prefix "C<eos>" (molecules were packed and <eos>-separated in pretraining).
  - RDKit parses "" as a valid molecule with zero atoms; empty molecules are rejected here.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, versions  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402
from rdkit import Chem, RDLogger, rdBase  # noqa: E402
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
MODEL = "ChemFM/ChemFM-1B"
PREFIX = "C"

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
load_s = time.perf_counter() - t0
w = Worker("chem")


def check(smiles):
    """Parse with RDKit. Returns (mol or None, canonical or None, reason or None)."""
    if not smiles:
        return None, None, "empty string"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "RDKit could not parse it (bad syntax or impossible valence)"
    if mol.GetNumAtoms() == 0:
        return None, None, "molecule has zero atoms"
    return mol, Chem.MolToSmiles(mol), None


def summary(mol):
    return {"elements": sorted({a.GetSymbol() for a in mol.GetAtoms()}),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol)}


@w.op("chemfm_generate")
def chemfm_generate(ctx, p, _inputs):
    n = int(p["n"])
    batch = min(int(p.get("batch_size", 128)), n)
    prefix = tok(PREFIX)["input_ids"] + [tok.eos_token_id]
    torch.manual_seed(int(p.get("seed", 0)))
    idx = 0
    while idx < n:
        b = min(batch, n - idx)
        inputs = torch.tensor([prefix] * b, dtype=torch.long, device="cuda")
        with torch.no_grad():
            seqs = model.generate(inputs, do_sample=True, temperature=float(p.get("temperature", 1.0)),
                                  top_k=int(p.get("top_k", 0)), max_new_tokens=int(p.get("max_new_tokens", 128)),
                                  eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        for row in seqs[:, len(prefix):]:
            raw = tok.decode(row, skip_special_tokens=True).replace(" ", "")
            mol, canon, reason = check(raw)
            out = {"raw": raw, "valid": mol is not None, "smiles": canon, "reason": reason}
            if mol is not None:
                out.update(summary(mol))
            ctx.item(idx, out)
            idx += 1
        ctx.progress(fraction=round(idx / n, 3))


@w.op("rdkit_validate")
def rdkit_validate(ctx, p, inputs):
    for i, s in enumerate(inputs):
        mol, canon, reason = check(s)
        out = {"input": s, "valid": mol is not None, "smiles": canon, "reason": reason}
        if mol is not None:
            out.update(summary(mol))
        ctx.item(i, out)


@w.op("rdkit_descriptors")
def rdkit_descriptors(ctx, p, inputs):
    for i, s in enumerate(inputs):
        mol, canon, reason = check(s)
        if mol is None:
            ctx.item_error(i, "invalid_molecule", reason)
            continue
        ctx.item(i, {"smiles": canon, **summary(mol),
                     "mol_wt_g_mol": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
                     "tpsa_A2": rdMolDescriptors.CalcTPSA(mol), "h_donors": Lipinski.NumHDonors(mol),
                     "h_acceptors": Lipinski.NumHAcceptors(mol), "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                     "rings": rdMolDescriptors.CalcNumRings(mol),
                     "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                     "fraction_sp3": rdMolDescriptors.CalcFractionCSP3(mol), "qed": QED.qed(mol)})


w.serve({"model": MODEL, "checkpoint": MODEL, "dtype": "bfloat16", "load_s": round(load_s, 3),
         "rdkit": rdBase.rdkitVersion, "versions": versions("transformers", "torch", "rdkit")})
