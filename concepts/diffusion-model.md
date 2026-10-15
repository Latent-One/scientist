# Diffusion generative model (MatterGen)

WHAT IT IS
A neural network trained to reverse a noising process. Training: take real crystals, add
noise to atom positions, cell and element types step by step until they are random; learn
to undo one step. Generation: start from random noise and apply the learned denoising 1000
times. MatterGen does this jointly for coordinates, cell, and element identities.

CONDITIONING
The `chemical_system` checkpoint was fine-tuned so you can ask for a set of elements
("B-C-N"). Classifier-free guidance (factor 2.0 here) trades diversity for adherence to the
condition.

WHAT IT GIVES YOU (and does not)
- Plausible-looking crystals near its training distribution (Materials Project / Alexandria).
- No guarantee of stability: outputs need relaxation and a hull check. Measured here: two
  B-C-N samples dropped 130–170 meV/atom on relaxation.
- Cost on this GPU: ~30–55 s per structure (1000 denoising steps; the step count is fixed by
  training and cannot be reduced).
