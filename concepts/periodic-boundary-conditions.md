# Periodic boundary conditions

WHAT IT IS
The simulation treats the unit cell as repeating forever: an atom near one face also
"sees" the atoms of the neighbouring copies of the cell. Mathematically, every position
r is equivalent to r + n1·a + n2·b + n3·c for integers n1, n2, n3.

WHY
A real crystal has ~10^23 atoms. Periodicity lets a calculation with 2–40 atoms describe
the bulk interior of an infinite crystal, with no surfaces.

CONSEQUENCES
- A 2D sheet (graphene) is modelled as a periodic stack with empty space (vacuum) between
  layers; the vacuum must be large enough that layers do not interact.
- An isolated molecule is put in a large box for the same reason.
- A defect (vacancy, dopant) in a small cell repeats too: one dopant in a 2-atom cell is
  50% doping, not "one dopant". Use a supercell to dilute it.

IN THIS SYSTEM
The structure worker builds the neighbour list including periodic images before every
model evaluation (inside MatterSim / MACE). `scientist learn unit-cell` for the cell itself.
