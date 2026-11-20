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
