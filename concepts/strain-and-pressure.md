# Strain and pressure

WHAT IT IS
Strain is a fractional change of the cell: +2% isotropic strain multiplies every lattice
vector by 1.02 (volume by ~1.061). Pressure is the force per area the crystal exerts on its
cell walls, in GPa (1 GPa ≈ 10,000 atmospheres). Here: positive pressure = compressed
crystal pushing outward; negative = stretched crystal pulling inward.

HOW IT IS OBTAINED HERE
`scientist sweep` applies strains with the structure worker, holds each cell fixed, lets the
atoms relax inside it, and records energy and pressure per point.

READING A SWEEP
- The energy minimum and the zero-pressure point should coincide (the model's preferred cell).
- The curvature at the minimum measures stiffness (bulk modulus): steep = hard to compress.
  Diamond is among the stiffest known materials.
- Asymmetry (stretching costs less than compressing) is normal: atoms repel strongly up close.

TRY
  scientist sweep c1 --from -4 --to 4 --points 9
