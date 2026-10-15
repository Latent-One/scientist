# Unit cell, lattice vectors, fractional coordinates

WHAT IT IS
A crystal is one small box of atoms repeated in all three directions. The box is the
unit cell. It is defined by three lattice vectors a, b, c (rows of `lattice` in every
structure file here, in Å = 1e-10 m). Atom positions are stored as fractional
coordinates: (0.25, 0.25, 0.25) means 0.25·a + 0.25·b + 0.25·c.

WHY FRACTIONAL
Stretching the cell (strain) changes a, b, c but not the fractional coordinates, so the
same numbers describe the atom arrangement at any cell size.

IN THIS SYSTEM
`scientist inspect c1` prints the cell lengths, angles and every site. Diamond here is a
2-atom "primitive" cell: the smallest box that repeats to the full crystal. The cubic
conventional cell has 8 atoms; both describe the same crystal (and hash differently).

TRY
  scientist build --prototype diamond --formula C
  scientist inspect c1
  scientist mutate c1 --supercell 2x2x2 && scientist inspect c2   # 16 atoms, same crystal
