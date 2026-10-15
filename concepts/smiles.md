# SMILES (molecule line notation)

WHAT IT IS
A text encoding of a molecule's graph: atoms as element symbols, bonds implied by adjacency
(= double, # triple), branches in parentheses, ring closures as matching digits, lowercase
for aromatic atoms. Ethanol: CCO. Benzene: c1ccccc1. Hydrogens are usually implicit.

WHY "CANONICAL"
One molecule has many valid SMILES strings. RDKit's canonical form picks one, so identical
molecules hash identically here. The canonical form can change between RDKit versions, which
is why the version is stored with every molecule artifact.

IN THIS SYSTEM
ChemFM writes SMILES token by token; RDKit parses them, rejects impossible valences and empty
strings, and computes descriptors (`scientist describe`). No 3D coordinates exist until a tool
embeds them.
