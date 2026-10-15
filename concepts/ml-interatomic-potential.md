# Machine-learned interatomic potentials (MatterSim, MACE)

WHAT IT IS
A neural network that maps atom species and positions to a total energy. It is trained to
reproduce energies, forces and stresses computed by DFT (a quantum-mechanics method) on
hundreds of thousands to millions of structures. Inside: a graph of atoms within a cutoff
radius (including periodic images); messages pass between neighbours; each atom gets an
energy contribution; they sum to E. Forces come from differentiating E.

WHY USE IT
DFT on a 16-atom cell takes minutes to hours on CPUs; these models take milliseconds on the
GPU (measured here: MatterSim relaxes a 16-atom carbon cell in ~1.8 s for 31 steps).

LIMITS (why its numbers are "fidelity: ML potential")
- It can only be as good as its training data and DFT settings (mostly PBE functional).
- Far from training data (unusual bonding, extreme compression, rare elements) errors grow
  and the model does not say so.
- MatterSim and MACE-MP-0 overlap in training data (Materials Project), so agreement
  between them is weaker evidence than it looks.

IN THIS SYSTEM
Each runs in its own Python process (workers/potential_worker.py) with its own environment;
the checkpoint hash and package versions are recorded in every observation.
