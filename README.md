# Transformer Surprise

A small research repository for testing whether a frozen transformer can provide rich nonlinear representations while a tiny online state-space sidecar learns which activation trajectories become expected during deployment.

## Motivation

The transformer stays frozen and unchanged. Selected intermediate hidden states are immediately compressed with a deterministic fixed projection, then an online recursive least-squares model predicts the next projected activation. Prediction residuals expose deviation; experience and confidence remain separate so the default epistemic state is `unknown` rather than accidentally `expected`.

This first version primarily supports temporal prediction:

```text
z_t(layer) -> predict z_(t+1)(layer)
```

The activation helpers leave room for depth prediction and residual monitoring later.

## Non-goals

This is not Bayesian inference over transformer weights, transformer fine-tuning, anomaly detection trained on the original pretraining distribution, a replacement for attention, or a production system.

## Design principles

- freeze the transformer;
- observe rather than perturb;
- compress immediately;
- learn expectations online;
- keep compute negligible;
- remain hardware-agnostic.

## Install and run

Python 3.9+ is supported. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
pytest
```

The first model run downloads `EleutherAI/pythia-160m` from Hugging Face. Change `model_name`, `layers`, or sidecar settings in `configs/default.yaml`; `device: auto` selects CUDA when available and otherwise CPU. No explicit `.cuda()` call is used.

## Colab

Upload or clone this repository into a Colab runtime, then open `notebooks/01_prototype.ipynb`. Run the setup cell first; it installs the local package and notebook dependencies. The model download is cached by the runtime after the first run. The notebook uses the same CPU/CUDA auto-selection and does not require a specific GPU.

## Architecture

`model.py` owns frozen Hugging Face inference and `output_hidden_states=True`. `activations.py` selects compact final-token observations. `projection.py` applies a seeded, row-normalized random matrix registered as a PyTorch buffer. `state_space.py` scores then updates a tiny RLS transition model with `A` and `b` represented by one augmented weight matrix. `surprise.py` provides L2 and cosine baselines and leaves the API open for covariance-based normalized innovation later.

A result reports `error`, `confidence`, `experience`, and `status` independently. The initial status is `unknown`; after enough observations, low error is `expected` and high error is `surprising` under a simple threshold. This toy threshold is not a calibrated Bayesian posterior.

## Research caution

The prototype exposes layer-wise signals for inspection. It does not demonstrate early-layer or late-layer semantic specialization, and the toy sentence set is not evidence of a scientific effect.
