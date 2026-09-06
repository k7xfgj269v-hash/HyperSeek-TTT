# HyperSeek-TTT

Toy-scale reimplementation of the DeepSeek V4-Pro architecture with 2026 test-time learning (In-Place TTT). The pipeline runs end-to-end; real efficacy validation is deferred to scale-up.

[中文版 README](README.zh-CN.md)

> Independent reimplementation. Not affiliated with DeepSeek-AI or the authors of the referenced papers. All code is original; references to DeepSeek V4-Pro and arXiv papers are for attribution only. Released under the MIT license.

Architecture deep-dives and the experiment log live in [`docs/`](docs/): [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design and [EXPERIMENTS.md](docs/EXPERIMENTS.md) for monitoring, ablations and known pitfalls.

## Features

- **MLA backbone** — DeepSeek-V2-style multi-head latent attention with decoupled RoPE.
- **DeepSeekMoE** — routed + shared experts, sigmoid + bias router, no auxiliary loss.
- **mHC-lite** — multi-copy residual hyper-connections via Birkhoff–von Neumann decomposition (exactly doubly stochastic).
- **In-Place TTT** — top-layer memory adapter with hidden-delta inner loss and per-sample closed-form fast weights.
- **Opt-in gated memory** — write / learned-forget gates plus a norm guard, initialised exactly at the baseline point.
- **Multi-Token Prediction** — auxiliary MTP heads with an embedded DeepSeekBlock.
- **Muon + AdamW** — optimiser split (ndim ≥ 2 → Muon; biases / norms / embeddings → AdamW).
- **Chunked-recall training mixture** — memory steps give the gate real learning pressure; 3-mode fair-compare harness.

## Quick Start

```bash
cd HyperSeek-TTT

# Setup (venv in the repo root; --without-scm-ignore-files keeps the repo .gitignore intact)
python3.14 -m venv . --without-scm-ignore-files
./bin/pip install -r requirements.txt

# Main training (auto MPS; steps / batches / lr via cfg.train_*)
./bin/python train.py

# Inference + expert-specialisation test
./bin/python eval.py

# Long-context / episode-memory harness (3-mode fair compare)
./bin/python memory_eval.py --task kv     --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task needle --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task rule   --episodes 100 --chunk-len 32

# Hidden-secret recall training (3-mode fair compare)
./bin/python memory_train.py --steps 300 --batch-size 16 --secret-len 1
./bin/python memory_train.py --steps 100 --batch-size 8  --secret-len 4

# Gated-memory ablation (write/forget gates + norm bound, init == baseline)
./bin/python memory_train.py --modes persistent --steps 100 --gated --max-mem-norm 5.0
```

## Testing

```bash
./bin/python -m pytest tests/
```

## Project Structure

```
HyperSeek-TTT/
├── config.py       Config dataclass (all hyper-parameters)
├── model.py        Architecture (all nn.Module)
├── data.py         Tokenizer, arithmetic dataset, episode generators
├── optim.py        Muon optimizer + parameter split
├── train.py        Training entrypoint (NTP + chunked-recall mixture)
├── eval.py         Inference + expert-specialisation test
├── memory_train.py Hidden-secret recall training + 3-mode fair compare
├── memory_eval.py  Episode memory + chunked-recall ablation
├── runlog.py       JSONL metric logger (logs/*.jsonl)
├── transformer.py  v0 baseline (MiniGPT), kept as reference
├── docs/           Architecture + experiment/engineering notes
├── tests/          pytest regression suite
└── requirements.txt
```

## License

MIT License — see [LICENSE](LICENSE).
