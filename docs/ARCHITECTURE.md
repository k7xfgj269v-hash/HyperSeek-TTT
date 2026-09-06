# HyperSeek-TTT — Architecture

Deep-dive on the model design. For the public-facing summary, examples and setup, see [../README.md](../README.md); for experiments, monitoring and known pitfalls see [EXPERIMENTS.md](EXPERIMENTS.md).

> Independent reimplementation. Not affiliated with DeepSeek-AI or the authors of the referenced papers. All code is original; references to DeepSeek V4-Pro and arXiv papers are for attribution only. Released under the MIT license.

## Architecture Overview

**Backbone**
- **MLA** — Multi-head Latent Attention with decoupled RoPE (DeepSeek-V2).
- **DeepSeekMoE** — routed + shared experts, sigmoid + bias router, no auxiliary loss (V2+V3).
- **mHC-lite** — multi-copy residual connections via Birkhoff–von Neumann decomposition; the `A` matrix is exactly doubly stochastic (arXiv 2601.05732).
- RMSNorm / RoPE / SwiGLU.

**Top-level**
- **MTP** — Multi-Token Prediction as an auxiliary loss; 1 head with an embedded, complete DeepSeekBlock (V3).
- **InPlaceTTT** — single linear adapter + gate; hidden-delta NTP inner loss; per-sample fast weights `(B, D, D)`; optional persistent `W_mem` accumulated across chunks (arXiv 2604.06169). Opt-in `ttt_gated_memory`: a write gate (weights inner-loss positions) plus a learned forget gate that replaces the fixed retention (Titans style; initialised exactly at the baseline point). `ttt_max_mem_norm` serves as a drift bound for streaming.
- **Muon + AdamW** — tensors with `ndim >= 2` are trained by Muon; biases / RMSNorm weights / token embeddings by AdamW (arXiv 2502.16982).

**Configuration**

Concrete parameters live in `config.py` → `Config` (a dataclass). Every scale-dependent field (`d_model` / `n_layers` / `kv_lora_rank` / `moe_inter_dim` / training hyper-parameters …) is centralised there; variants are created with `dataclasses.replace(cfg, ...)`, so a scale-up touches exactly one place.

**Deliberately omitted:** FP4 training; DSA / CSA / HCA / Hybrid Attention (V4-Pro's long-context trio — meaningless at toy scale).

## Memory module: InPlaceTTT vs ATLAS

InPlaceTTT is the default; `AtlasMemory` is not in the source tree. If needed, add it as a new class per the [arXiv 2505.23735](https://arxiv.org/abs/2505.23735) specification.

| | InPlaceTTT (current) | ATLAS (alternative) |
|---|---|---|
| Inner loss | hidden-delta MSE: `\|\|W·h_t − (h_{t+1} − h_t)\|\|²` | associative memory: `\|\|M(k_t) − v_t\|\|²` |
| Internal structure | single linear + gate | M (2-layer MLP) + W_k / W_v + gate |
| Parameter magnitude | once `d_model²` | roughly 5× `d_model²` |
| Design philosophy | reuse existing structure + fast weight | add a standalone memory module |
| Paper | 2026-04 | 2025-05 |

**Choose ATLAS when**
- The task has a clear key–value association structure (query-style recall, explicit lookup).
- Key hits are discrete and association patterns are easier to learn than a hidden delta.
- More storage capacity is required (M is a 2-layer MLP, more expressive than a single linear).
- A direct comparison against the ATLAS paper baseline is desired.

**Choose InPlaceTTT when**
- General NTP-style tasks.
- Parameter efficiency matters (the memory branch uses ~80% fewer parameters).
- You want to follow the 2026 mainstream line (Behrouz et al. successors).

## References

- DeepSeek-V2: arXiv 2405.04434
- DeepSeek-V3 Technical Report: arXiv 2412.19437
- DeepSeek-V4-Pro: 2026-04-24 release ([HF Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro))
- mHC (manifold-constrained): arXiv 2512.24880
- **mHC-lite** (used in this project): arXiv 2601.05732
- Muon Optimizer: arXiv 2502.16982
- ATLAS (alternative memory branch): arXiv 2505.23735
- Titans / MIRAS: arXiv 2501.00663
- TNT (ATLAS training efficiency, ICLR 2026): arXiv 2511.07343
- **In-Place TTT** (used in this project): arXiv 2604.06169
- LLMs-from-scratch MLA reference: github.com/rasbt/LLMs-from-scratch
