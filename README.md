# Liturgy of Tunneling

Experimental TORX probabilistic-circuit and ARTIQ scheduling research for Genesis Conductor / Yennefer Daemon.

> **Research status:** The legacy `liturgy_v_omega_plus_1_notebook.py` script is **not validated for hardware execution or quantum-physics claims**. It includes ARTIQ scheduling and RF-safety defects. Do not run it on powered hardware.

## Research integration — G2 compactification, reduced dynamics and FDT

See [RFC-0001: Reduction, memory and interpretive symmetry](docs/RFC-0001-reduction-memory-and-analogy.md). It explicitly separates:

1. Conditional M-theory compactification on torsion-free G2 seven-manifolds.
2. Quantum reduced dynamics, decoherence and conjectural ER=EPR / MWI interpretations.
3. Classical projection, memory, and *conditionally applicable* fluctuation-dissipation.
4. CPU-only stochastic reference laws and digital skip scheduling.
5. DID/OSDD and 27 Club narratives, retained only as non-causal analogies.

Neither the reference model nor the preexisting notebook simulates G2 compactification, quantum decoherence recovery, quantum many-body scarring, clinical identity, or age-dependent mortality.

## CPU-only verification

This branch adds standard-library Python modules under `liturgy/`, with tests that do not import ARTIQ or TORX:

```sh
python -m unittest discover -s tests -v
```

- `liturgy/reference_dynamics.py`: exact discrete-time Ornstein–Uhlenbeck transitions and analytic second moments under a stated equilibrium assumption.
- `liturgy/schedule.py`: validated, deterministic, count-exact digital skip indices; `skip_indices(137,17,3) == (17,34,51)`.
- `tests/test_reference_models.py`: CPU-only mathematical, stochastic reproducibility, boundary, and schedule tests.

**Release gates:** Mathematical and code-level CPU verification precedes TORX API integration; independent ARTIQ timing and RF qualification are required before hardware use. No hardware deployment is authorized by the addition of these files.
