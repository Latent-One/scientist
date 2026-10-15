# Formation energy

WHAT IT IS
For a compound with composition x_i (fraction of each element i) and energy per atom E:
  E_form = E - sum_i x_i * mu_i
where mu_i is the energy per atom of element i in its reference phase (the pure element
in a stable form), computed with the SAME model. Negative E_form: forming the compound from
its elements releases energy.

WHY IT EXISTS
Raw energies per atom are not comparable across compositions: a nitrogen atom and a carbon
atom carry very different baseline energies. Subtracting the elemental references puts
compounds of different compositions on one scale.

THE CATCH
The answer depends on the references you chose. Boron's true ground state (alpha-boron,
12-atom icosahedra) and nitrogen's (N2 molecules packed in a crystal) are not simple
prototypes. A simpler stand-in shifts every formation energy containing that element.
`scientist hull` refuses to run until each element has a relaxed elemental phase, and the
choice is recorded in the history.

TRY
  scientist build --prototype diamond --formula C && scientist relax <id>
  scientist hull --with mattersim
