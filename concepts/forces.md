# Forces on atoms

WHAT IT IS
The force on atom i is minus the slope of the energy with respect to its position:
F_i = -dE/dr_i, a 3-vector in eV/Å. It points in the direction that lowers the energy.

HOW IT IS OBTAINED HERE
MatterSim and MACE compute E with a neural network, then PyTorch automatic
differentiation computes dE/dr for every atom in the same GPU pass. Stress (the slope with
respect to the cell shape) comes the same way and gives the pressure.

READING THE NUMBERS
- max|F| near 0 (here: below the fmax target, default 0.02 eV/Å) = at a stationary point.
- In a perfect high-symmetry crystal (diamond), forces are exactly 0 by symmetry even when
  the cell size is wrong; then only the stress/pressure is non-zero.
- Large forces after generation mean the generated geometry is not at a minimum yet.

TRY
  scientist point c1 && scientist inspect <observation id>   # forces and stress
