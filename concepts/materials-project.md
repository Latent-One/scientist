# Materials Project (known phases from a database)

WHAT IT IS
A public database (next-gen.materialsproject.org) of ~150k inorganic crystals, each computed
with DFT by the same workflow: relaxed structure, total energy, formation energy, energy above
the MP convex hull, band gap. Entries marked "experimentally reported" match a structure in the
ICSD (a database of measured crystal structures); "theoretical" ones were only computed.

IN THIS SYSTEM
  scientist known --system B-C-N --max-hull 50 --limit 30
fetches phases of B-C-N and every subsystem (B, C, N, B-C, B-N, C-N). Each becomes a candidate
(origin: materials_project mp-…), and MP's values are recorded as an observation with fidelity
Dft and source "database, NOT computed here". The raw response is cached with the job.

THE RULE THAT MATTERS
MP energies are on MP's own DFT scale (its current data mixes functionals). MatterSim and MACE
have their own scales. Never compare a database energy with a model energy. Instead:
  1. look up known phases,  2. relax them with YOUR evaluator,  3. scientist hull --with <it>.
Now your hull contains the real competing phases, and "on the hull" means much more.

WHAT IT IS GOOD FOR
- Elemental references (alpha-boron, solid N2) for formation energies, instead of guesses.
- Checking whether a "new" candidate is actually a known material.
- A higher-fidelity reference to see how far your ML potentials are from DFT on known phases.
