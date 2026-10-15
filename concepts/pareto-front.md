# Pareto front (multi-objective search)

WHAT IT IS
With several objectives (say: low formation energy AND high volume), candidate A dominates B
if A is at least as good on every objective and strictly better on one. The Pareto front is
the set of candidates nothing dominates. Every point on it is a different trade-off.

WHY NOT ONE SCORE
Collapsing objectives into a weighted sum hides the trade-off and silently discards
candidates that are excellent on one axis. The plan (section 20) keeps physical observables
separate; you decide the trade-off after seeing the front.

IN THIS SYSTEM
src/search.rs ranks candidates by non-dominated sorting (rank 0 = front). The evolutionary
searcher prefers parents with lower rank. Near-identical minima (same formula, energy within
1 meV/atom, volume within 0.5%) are shown once.

TRY
  scientist search define --strategy evolutionary --objective min:energy_per_atom \
      --objective max:volume_per_atom --pool c2 --rounds 2
  scientist search run s1 --rounds 2
