# Convex hull and energy above hull

WHAT IT IS
Plot formation energy (y) against composition (x) for every phase you know. The lower
convex hull is the lowest line you can stretch underneath all points. A compound ON the hull
cannot lower its energy by splitting into other known phases. A compound ABOVE it can: the
vertical distance, E_above_hull (meV/atom), is the energy released by decomposing into the
hull phases at that composition.

HOW IT IS COMPUTED HERE
pymatgen's PhaseDiagram, in the structure worker, over every structure relaxed in THIS
experiment with one evaluator (`scientist hull --with mattersim`).

HOW TO READ IT
- 0 meV/atom: on this experiment's hull. Phases that are missing from the experiment cannot
  compete, so "on the hull here" can still be unstable in reality. The Materials Project
  hull includes ~150k known phases; this one includes what you computed.
- Small positive values (< ~50 meV/atom) are often still synthesizable (metastable).
- The hull inherits the model's errors and the reference choices (`formation-energy`).

TRY
  scientist hull --with mattersim && scientist hull --with mace   # do they agree?
