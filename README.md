# scientist

A local instrument for materials, chemistry and superconductor investigations, built and tested on an NVIDIA DGX Spark
(GB10, 128 GB unified memory). Untested on other hardware.
You are the investigator. A local LLM planner (Qwen3.6-35B-A3B) and numerical searchers
propose; a Rust runtime validates, runs the models, and records everything.
Design and every decision: docs/design.md.

## Setup

1. Build: `cargo install --path .` (Rust 2024 edition).
2. Config: `cp scientist.example.toml scientist.toml`, then point each worker's `python` at an environment that has its
   model installed (one virtual environment per model; see Models below). `scientist.toml` is git-ignored.
3. Planner: an OpenAI-compatible server (tested: vLLM serving Qwen3.6-35B-A3B, see `scripts/serve_planner.sh`).
4. Optional data:
   - Materials Project key: `MP_API_KEY` or `~/.config/scientist/mp_api_key` (free account at materialsproject.org).
   - 3DSC table of measured Tc (`data_3dsc` in the config): https://github.com/aimat-lab/3DSC

## Start

    scripts/serve_planner.sh      # once per boot: the planner model (~4 min to load)
    cd ~/anywhere/my-project
    scientist                     # type what you want to investigate

Like Codex or Claude Code: each directory is its own experiment. The first time, your first
message becomes the experiment (the planner researches the literature, drafts it, you confirm with y),
and the planner starts working on it. Run `scientist` there again later to resume.
`scientist "build a (5,5) nanotube and relax it"` starts with that message.

Inside: type plain language, or /commands (/help). Shift+Tab toggles auto-approve.
Ctrl-C interrupts the agent at once; on an empty prompt, twice exits.

Every step also works from the shell for scripting: `scientist relax c1 --with mace --learn`,
`scientist status`, `scientist evidence c1`, `scientist learn relaxation`.

## Superconductors

    scientist superconductors --elements Nb,Sn --min-tc 10   # known superconductors, MEASURED Tc (SuperCon via 3DSC)
    scientist tc c1 c2                                       # ML Tc estimate: ALIGNN + BETE-NET, verdict LIKELY/possible/unlikely
    scientist benchmark                                      # re-check the Tc models on 12 measured materials (PASS/REGRESSION)
    scientist search define --strategy evolutionary --substitution --objective max:tc --objective min:e_above_hull --pool c2,c4

Tc estimates are ML surrogates of DFT, validated here at ~2.4 K error on conventional superconductors; they miss
multi-gap superconductors (MgB2). Anything above 39 K (MgB2, the ambient-pressure conventional record) is flagged as
extrapolation. Measured Tc from 3DSC is matched to structures by composition only: check stability (hull) and space
group before trusting the pairing. Only a measurement of zero resistance AND magnetic-field expulsion proves superconductivity.

## Literature

`scientist papers <query> [--since 2015]` searches OpenAlex (published, with citation counts) and arXiv
(preprints) in parallel, ~1.5 s end to end. Papers get ids p1, p2…; `scientist inspect p3` shows the abstract.
Abstracts are the authors' claims, kept separate from your observations. No API key needed.

## Materials Project

`scientist known --system B-C-N` imports known phases (needs the key in ~/.config/scientist/mp_api_key).
Their DFT values are recorded as database observations; relax them with your evaluator before comparing.

## What runs where

    scientist (Rust, CPU)  -> SQLite event log + content-addressed artifacts in ~/scientist-data
       |-- HTTP  -> vLLM container "planner" (Qwen3.6-35B-A3B)
       |-- pipes -> Python workers, one per model, each in its own environment:
                    structure (pymatgen/ASE, CPU), mattersim, mace, mattergen, chem (ChemFM + RDKit)

Workers stay loaded for the life of the scientist process (`scientist chat` keeps them warm).
Ctrl-C once: finish the current job and stop. Twice: cancel it (completed items are kept).

## Autonomy levels (`scientist level N`)

    0 manual            planner off
    1 assisted          planner proposes, you approve every tool run      (default)
    2 search-assisted   you approve searches, generation, big evaluations
    3 planner-assisted  you approve generation, new searches, judgments
    4 autonomous        inside budget and stop rules

## Commands

    new use list status chat ask run log inspect evidence note interpret hypothesis
    build import known papers export generate relax point mutate sweep compare describe hull constraints
    search {define,run,show} promote reject budget level workers learn notebook fork terminate

`scientist <command> --help` explains each one. In chat, the same commands start with `/`.

## Data and models

Not included in this repository; each is downloaded separately under its own license.

| Component | Used for | License (as published) |
|---|---|---|
| [3DSC](https://github.com/aimat-lab/3DSC) (Sommer et al., Sci. Data 10, 816, 2023) | measured Tc; `benchmarks/supercon*.toml` are derived from it | CC BY 4.0 |
| [Materials Project](https://materialsproject.org) | known structures and DFT data | see its terms of use |
| [MatterGen](https://github.com/microsoft/mattergen) | crystal generation | MIT |
| [MatterSim](https://github.com/microsoft/mattersim) | ML interatomic potential | MIT |
| [MACE](https://github.com/ACEsuit/mace) (MACE-MP-0) | ML interatomic potential | MIT (code; check the weights' own license) |
| [ALIGNN](https://github.com/atomgptlab/alignn) | Tc, lambda, omega_log surrogate | MIT |
| [BETE-NET](https://github.com/henniggroup/BETE-NET) | a2F(w) -> Tc surrogate | Apache-2.0 |
| [ChemFM-1B](https://huggingface.co/ChemFM/ChemFM-1B) + RDKit | molecule generation and checks | MIT; BSD-3-Clause |
| [OpenAlex](https://openalex.org), [arXiv API](https://info.arxiv.org/help/api/) | literature search | see their terms |

The benchmark files contain measured Tc values from 3DSC/SuperCon and Materials Project identifiers. When you
use them, cite 3DSC and the NIMS SuperCon database.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later); see `LICENSE`. If you run a modified version
as a network service, you must offer its users the corresponding source code.
