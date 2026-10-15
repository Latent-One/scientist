# Fidelity ladder

Each rung is more trustworthy and more expensive. Every observation here records its rung.

  Generated        a model proposed it                         MatterGen, ChemFM
  StructuralCheck  geometry plausible: distances, density       pymatgen (CPU, ms)
  Descriptor       deterministic molecular descriptors          RDKit (CPU, ms)
  MlPotential      energy/forces from a learned potential       MatterSim, MACE (GPU, seconds)
  Dft              quantum-mechanical calculation               not installed
  Experiment       physical measurement                         not applicable here

RULES THE RUNTIME ENFORCES
- Generation is not evidence: `promote` refuses candidates with no evaluation.
- Cheap filters run before expensive ones (the structural check is automatic).
- `scientist evidence <id>` states the highest rung reached and what is missing.
