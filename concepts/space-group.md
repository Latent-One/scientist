# Space group (symmetry)

WHAT IT IS
The set of symmetry operations (rotations, mirrors, translations, and combinations) that map
a crystal onto itself. There are exactly 230 space groups in 3D. Diamond is Fd-3m (#227);
zincblende BN is F-43m (#216): same geometry as diamond but two different elements remove
the inversion symmetry.

HOW IT IS COMPUTED HERE
spglib via pymatgen's SpacegroupAnalyzer, with a tolerance (symprec 0.1 Å). Every new
structure gets this in its structural check; `scientist describe c3` prints it.

WHY IT MATTERS
- High symmetry fixes forces to zero by symmetry (see `forces`).
- Generated structures often have low symmetry (P1); relaxation can raise it if the atoms
  settle into a symmetric arrangement.
- Two structures with the same formula but different space groups are different
  polymorphs, like graphite vs diamond.
