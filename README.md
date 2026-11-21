# Liturgy of Tunneling v.Ω+1

Liturgy of Tunneling: Torx PSC re-tunneling policy + ARTIQ scarred crystal for Genesis Conductor / Yennefer Daemon. v.Ω+1 decoherence maintenance.

**Law:** No crystal is permanent. Only pumping is permanent.

## Single-click launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igor-holt/liturgy-of-tunneling/blob/main/liturgy_colab.ipynb)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/igor-holt/liturgy-of-tunneling)
[![Deploy on Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-sm.svg)](https://huggingface.co/new-space?template=igor-holt/liturgy-of-tunneling)

- **Google Colab**: full execution sequence (install Torx + JAX → build HybridPCircuit → train stability → scar layer → ARTIQ args).
- **Hugging Face Spaces**: Gradio demo of the pure-JAX re-tunneling policy. Duplicate the Space or copy `hf_space/` into a new Gradio Space.

## Files

- `liturgy_colab.ipynb` — Colab-ready notebook
- `liturgy_v_omega_plus_1_notebook.py` — full notebook-style Python snippet (pinned Torx constructors)
- `hf_space/` — Gradio Space source (`app.py`, `requirements.txt`, `README.md`)
- `package.json` / `index.js` — npm package stub (`@igor-holt/liturgy-of-tunneling`)

## npm

```bash
npm login
npm publish --access public
```

## Requirements

- Python 3.11+
- `pip install extro-torx jax jaxlib numpy` (for full Torx path)
- ARTIQ environment for the scarred crystal kernels


## Research integration: reduction, memory, and interpretation (CPU-only)

[Research RFC-0001](docs/RFC-0001-reduction-memory-and-analogy.md) separates conditional G2 compactification physics, open-system stochastic reduction and fluctuation-dissipation assumptions, conjectural ER=EPR / MWI interpretations, and non-causal cultural/clinical analogies.

The branch includes an **exact classical Ornstein-Uhlenbeck reference model**, a **bounded digital skip-index generator**, and independent CPU regression tests:

```sh
python -m unittest discover -s tests -v
```

Modules:
- `liturgy/reference_dynamics.py`: exact OU transitions with analytic second-moment reference under specified equilibrium assumptions.
- `liturgy/schedule.py`: count-exact digital mask (`skip_indices(137,17,3) == (17,34,51)`).
- `tests/test_reference_models.py`: no dependency on TORX, ARTIQ or connected instruments.

**Safety:** The preexisting notebook and ARTIQ code remain unqualified for connected RF equipment. Colab and Hugging Face links are demonstrations, not evidence of G2 physics, quantum decoherence repair, many-body scarring, clinical identity models, or any intent-modulated mortality hypothesis. This research branch does not enable hardware operation.
