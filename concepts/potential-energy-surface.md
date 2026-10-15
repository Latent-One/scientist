# Potential energy surface (PES)

WHAT IT IS
A function E(R): the energy of a fixed set of atoms as a function of all their positions
R (3N numbers for N atoms, plus the cell). Low regions are stable arrangements; the bottom
of a valley is a local minimum; the lowest of all valleys is the global minimum.

HOW IT IS OBTAINED HERE
Nobody computes the whole surface. A model (MatterSim, MACE) returns E and its gradient at
one point R per call. An optimizer walks downhill using that gradient (see `relaxation`).

WHY IT MATTERS FOR INTERPRETATION
- Relaxation finds the NEAREST valley, not necessarily the deepest. Two starting
  structures with the same atoms can end in different valleys with different energies.
- Different models are different approximations to the true quantum-mechanical surface;
  their valleys can sit at different places and depths (see `scientist evidence`).

TRY
  scientist mutate c1 --rattle 0.15 --seed 1     # push atoms out of the valley
  scientist relax c2 --learn                     # watch energy fall back
