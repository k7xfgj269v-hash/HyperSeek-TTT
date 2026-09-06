# HyperSeek-TTT — Experiments & Engineering Notes

Research log: training monitoring, experiment design, scale-up notes, known limitations and hidden constraints. For the architecture deep-dive see [ARCHITECTURE.md](ARCHITECTURE.md); for the public summary see [../README.md](../README.md).

> Status caveat: the current code is **pipeline verification only, not an efficacy check**. The real value of InPlaceTTT test-time learning, mHC replica utilisation, the MTP gain or the Muon-vs-AdamW advantage is **not visible at toy scale**. Only a scale-up can validate it.

## Training Monitoring

Every `cfg.train_mem_every` steps, `train.py` mixes in one chunked-recall memory step (the secret sits in chunks outside the query attention; only `W_mem` can carry it — this gives the gate real learning pressure). One line is printed every 100 steps; every step is appended to `logs/*.jsonl` (`step / kind / loss / route / gate / amp_avg / amp_max`):

```
Step  XXXX loss X.XXXX route X.XXXX gate X.XXXX mhc_amp avg X.XXX max X.XXX
                                         ↑ only backbone HC counted, MTP-internal excluded
```

Monitoring fields exposed by each module (updated live during forward):

| Module | Field | Purpose |
|---|---|---|
| HyperConnect | `last_amp` | signal amplification, V4 target ≈ 1.6 |
| HyperConnect | `last_alpha`, `last_beta`, `last_lambda` | normalised weight distribution |
| HyperConnect | `last_replica_norms` | `n_hc=4` replica utilisation |
| MoE | `last_load`, `last_router_logits` | expert load |
| InPlaceTTT | `last_gate`, `last_L` | gate mean + inner loss |
| InPlaceTTT | `last_delta_norm`, `last_h_norm`, `last_update_norm`, `last_mem_norm` | memory / update magnitude |

The four indicators to watch together (every 100 steps):

| Indicator | What to look for |
|---|---|
| **`loss`** | monotonically decreasing; a spike or a long plateau is an alarm |
| **`gate`** (`InPlaceTTT.last_gate`) | **key indicator of whether TTT is really learning**. During training `> 0.1` → TTT is active; staying `≈ 0.02` → idle |
| **`route`** (router-supervised CE) | at toy scale converges to 1.0–1.5; if it never falls the router is not differentiating |
| **`mhc_amp avg / max`** | `avg ≈ 1.6` (V4 target), `max < 5`; `max > 10` → mHC `alpha`/`beta`/`lambda` drifted, amplification out of control |

## Experiment Design: memory_train 3-mode Fair Compare

`memory_train.py` trains the task "secret hidden in context, query outputs the secret" under three modes:

| Mode | `W_mem` behaviour | `atlas.W` learning path |
|---|---|---|
| `none` | fixed `W_mem = 0`, neither read nor write | learns via query forward; atlas degenerates to a static linear + gate |
| `transient` | per chunk, inner update from a zero base; no accumulation across chunks | backprop through the functional path |
| `persistent` | `W_mem` accumulated across chunks, reset before an episode starts | backprop through the functional path |

All three modes go through the same `build_functional_memory` (functional path), so the only difference is "whether `W_mem` accumulates across chunks" — making the **marginal benefit of persistent fast weights** actually measurable.

`memory_eval.py` runs three long-context tasks (kv_recall / needle / rule) × three modes as an ablation.

### Gated memory ablation

The TTT double-gate is implemented as `ttt_gated_memory` (a read gate already existed; the write gate weights inner-loss positions; the forget gate replaces the fixed retention). On a 100-step A/B at toy scale there is **no measurable difference** to the baseline — by design, because the init sits exactly at the baseline point. The efficacy check is deferred to scale-up.

## Scale-up Notes

### Parameter tuning order

1. **Confirm depth `n_layers` first** — weak multi-task preservation fundamentally comes from flat models. First scale-up step: increase depth. The MTP share shrinks automatically with depth.
2. **Keep `n_mtp_heads = 1`** — V3 / V4 are both depth=1; do not touch it.
3. **A `seq_len` change requires model reconstruction** — `rope_cos` / `causal_mask` are buffers, not dynamically extensible.
4. **Scale the MoE block with model size** — scale `n_routed_experts` / `moe_inter_dim` proportionally.

> Note: item 3 predates the lazy length decoupling (`_rope_mask`), which now grows the caches on demand — see *Hidden constraints*. The two statements coexist in the source doc history; flag if item 3 should be retired.

### Possible extension directions (hooks exist, new logic needed)

- ~~**TTT double gate**~~ — **implemented** as `ttt_gated_memory` (see above). On a toy 100-step A/B there is no measurable difference to baseline (by design: init == baseline); efficacy check deferred to scale-up.
- **Reasoning token format** — toy uses inline CoT; at scale-up add `<think>...</think>` as explicit segment tokens.

### Checked and deferred directions

- **Adaptive chunk size by information density** — breaks the fixed-length chunk alignment that the batched per-sample fast weights rely on; the goal (not writing redundant content) is already covered per token by the write gate rather than per block. Re-evaluate at scale-up with real compute pressure.
- **Per-layer differentiated update frequency** — the premise does not hold here: there is exactly one top-level TTT adapter, no per-layer fast weights. Multi-level TTT would be a new architecture experiment, not an optimisation of the current code.
- **Contrastive auxiliary loss / information bottleneck** — not measurable at toy scale (even more direct interventions sit in the noise; see the A/B results). Only meaningful with a scale-up measurement rig.
- **Fast-weight quantization (4-bit)** — the fast-weight state is `d_model² = 64 KB`; quantisation would add nothing but complexity here (and MPS has no ready 4-bit kernels). Only relevant at large `d_model`.

## Known Limitations

1. **Toy arithmetic alone cannot validate test-time learning** — on closed, deterministic tasks the gate idles. Since the chunked-recall mixture in `train.py`, the gate gets real pressure (measured: 0.018 → ~0.1 in 400 steps); reliable claims still need BPE + natural language.
2. **Grafting onto DeepSeek-V2-Lite is extremely costly** — the dimensions do not match, the tokenizer is completely different and the architectures differ strongly (V2-Lite has no mHC / MTP / InPlaceTTT). "Grafting" ≈ rewriting the project.
3. **`n_layers` too flat** — currently a toy safety limit, but multi-task preservation may still fail; the MTP-head share stays large and MTP perturbs training dynamics. Scaling up to ≥ 12 layers mitigates automatically.
4. **Character-based tokenizer, arithmetic only** — natural language requires a BPE 32K–50K, i.e. rewriting `data.py` entirely.
5. **mHC-lite is sensitive to `n_hc`** — `n_hc=4` → 24 perms is safe; `n_hc ≥ 8` → buffer explosion, K-cap subsampling needed (per the paper, a subset of `n_hc!`).
6. ~~**`make_long_effective_mask`**~~ — dead placeholder without callers; removed.
7. ~~**InPlaceTTT inner step shared over samples**~~ — **fixed**: the inner gradient now has a closed form per sample (`_inner_grad`, `(B, D, D)` fast weights); a regression test enforces batch == serial equivalence.
8. **MTP with a complete embedded DeepSeekBlock** — consistent with the V3 paper, but at toy scale the MTP share is large and disturbs backbone training diagnostics. Scale-up mitigates automatically.

## Hidden Constraints (stumbling blocks)

- `model(idx, return_mtp=True)` must be given `mtp_tokens=next_tok_ids` (the fallback is a raise, no longer silent-wrong).
- `rope_cos` / `causal_mask` have grown lazily since the length decoupling (`_rope_mask`); any length runs, but beyond the trained length RoPE gives no quality guarantee.
- A persistent `W_mem` **write** via `InPlaceTTT.forward` requires batch 1 (per-sample fast weights cannot be merged into one buffer) — otherwise a `ValueError`; read-only runs batched.
- InPlaceTTT's `persistent_memory` is fixed at construction time (`cfg.ttt_persistent_memory`) and cannot be switched mid-training; `memory_train.py` creates a fresh model per mode (`dataclasses.replace(cfg, ...)`).
- Main `train.py` uses the attribute name `model.atlas` for the InPlaceTTT instance (historical naming; state-dict compatibility is not broken; the actual type is InPlaceTTT).
