# Public release boundary

This repository contains the local Ollama control-plane code, Modelfiles, evaluation helpers
and training utilities that are safe to share. It contains no model weights, API credentials,
private deployment configuration, hidden benchmark payloads or customer data.

Model weights remain local. Set `HYDRA_MODELS_DIR` and `HYDRA_TRAIN_DIR` when using optional
training configurations, or use the local Ollama inventory documented in `docs/OLLAMA_MODELS.md`.
The former hosted Hydra API and portal are retired; the supported product path is local Ollama
and private/local router development.
