# Lattice constant

WHAT IT IS
The edge length of the conventional cubic cell (for cubic crystals), in Å. Diamond's
measured value is 3.567 Å at room temperature.

WHAT A MODEL GIVES
Each model has its own preferred lattice constant: the cell size where its energy is
lowest and its predicted pressure is zero. The difference from experiment mixes model
error with temperature (models here have no temperature; experiments do).

IN THIS SYSTEM
A relaxation with the cell free finds the model's preferred size. A strain sweep
(`scientist sweep`) maps energy around it: a U-shaped curve whose minimum is that size.

TRY
  scientist sweep c1 --from -4 --to 4 --points 9 --with mattersim
  scientist sweep c1 --from -4 --to 4 --points 9 --with mace     # does the minimum move?
