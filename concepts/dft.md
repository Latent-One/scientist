# DFT (density functional theory)

WHAT IT IS
A quantum-mechanical method that computes the ground-state energy of electrons in the field of
fixed nuclei, by solving for the electron density instead of the many-electron wavefunction.
An approximation called the exchange-correlation functional (PBE is the common default) is
needed; it is the main source of DFT error (PBE overestimates lattice constants by ~1% and
underestimates band gaps substantially).

WHY IT MATTERS HERE
MatterSim and MACE are trained to imitate DFT (mostly PBE) results. They inherit DFT's
systematic errors and add their own. DFT is the next rung on the fidelity ladder above them.

IN THIS SYSTEM
No DFT engine is installed (V0). `scientist evidence` lists it as missing for every candidate,
so ML-potential results are never presented as verified.
