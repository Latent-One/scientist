# Carbon nanotubes and chirality (n, m)

WHAT IT IS
A single-wall carbon nanotube is a graphene sheet rolled into a seamless cylinder. Which way
it is rolled is given by the chiral vector C = n·a1 + m·a2 on the graphene lattice: the tube's
circumference is exactly that vector.
  armchair  (n, n)   e.g. (5,5)
  zigzag    (n, 0)   e.g. (9,0)
  chiral    others   e.g. (6,5)
Diameter d = 0.246 nm · sqrt(n² + nm + m²) / π   ((5,5): 0.68 nm, (6,5): 0.75 nm).

WHY THE ATOM COUNT JUMPS
The repeating cell along the tube axis must close up on the lattice. For chiral tubes that
period can be long: (5,5) repeats every 2.46 Å with 20 atoms; (6,5) repeats every 40.6 Å with
364 atoms (both measured from ASE's builder here). Cost of a relaxation grows with atom count.

ELECTRONIC CHARACTER (tight-binding rule of thumb, not computed here)
(n − m) divisible by 3 → metallic or very small gap (all armchair tubes are metallic);
otherwise semiconducting, with a gap roughly ∝ 1/diameter. (6,5): n − m = 1 → semiconducting.
MatterSim and MACE predict energy and geometry only; they cannot tell you the band gap.
That needs an electronic-structure method (DFT or tight-binding), not installed in V0.

STRAIN ENERGY
Rolling costs energy: smaller diameter = more curvature = higher energy per atom than flat
graphene. Relaxing tubes of different diameters with the same model and comparing E/atom
against graphene shows this directly (same composition, so energies are comparable).

TRY
  scientist build --prototype nanotube --formula C --chirality 5,5
  scientist build --prototype nanotube --formula C --chirality 9,0
  scientist build --prototype graphene --formula C
  scientist relax c1 c2 c3 --with mattersim && scientist compare c1 c3
