# Relaxation (geometry optimization)

WHAT IT IS
An iterative search for a local minimum of the potential energy surface:
  1. evaluate energy, forces (and stress) at the current geometry   [GPU, model]
  2. move atoms (and the cell) a step along the forces               [CPU, ASE optimizer]
  3. repeat until max |force| < fmax, or the step limit is reached.
Here the optimizer is FIRE (Fast Inertial Relaxation Engine) from ASE; with the cell free,
ASE's FrechetCellFilter lets the stress change the cell shape and size too.

WHAT TO READ IN THE OUTPUT (`--learn` shows all of it)
- steps: how far from a minimum the start was.
- energy change (meV/atom): how much the model "prefers" the relaxed geometry.
- volume change: whether the starting density was off.
- converged: if false, the energy is NOT a minimum; do not rank it.

WHAT IT DOES NOT TELL YOU
- Whether this is the lowest arrangement (only the nearest valley).
- Whether the structure is stable against vibrations (no phonon calculation here).
- Whether it is stable against decomposing into other compounds (see `convex-hull`).

TRY
  scientist relax c1 --with mattersim --learn
  scientist relax c1 --with mattersim --fmax 0.1 --learn   # looser stop: fewer steps
