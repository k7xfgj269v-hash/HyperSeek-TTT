# HyperSeek-TTT

[English README](README.md)

> 独立复现。与 DeepSeek-AI 或所引用论文的作者无关。所有代码均为原创；对 DeepSeek V4-Pro 和 arXiv 论文的引用仅用于注明出处。以 MIT 许可证发布。

DeepSeek V4-Pro 架构 + 2026 Test-Time Learning（In-Place TTT）的玩具级复现。流水线端到端可跑；真正的有效性验证有待扩规模（scale-up）完成。

## 架构总览

**骨干（Backbone）**：
- **MLA** — 带解耦 RoPE 的多头潜在注意力（DeepSeek-V2）
- **DeepSeekMoE** — routed + shared experts，sigmoid + bias router，无 aux loss（V2+V3）
- **mHC-lite** — 通过 Birkhoff-von Neumann 分解实现的多副本残差，A 矩阵严格双随机（arXiv 2601.05732）
- RMSNorm / RoPE / SwiGLU

**顶层（Top-Ebene）**：
- **MTP** — 多 Token 预测作为辅助损失，1 个 head，内嵌完整 DeepSeekBlock（V3）
- **InPlaceTTT** — 单一 Linear 适配器 + Gate；hidden-delta NTP inner loss；per-Sample Fast Weights (B, D, D)；可选的持久化 `W_mem` 跨 chunk 累积（arXiv 2604.06169）。可选 `ttt_gated_memory`：Write-Gate（加权 Inner-Loss 位置）+ 学习式 Forget-Gate 以替代固定保留率（Titans 风格，初始化恰在 baseline 点上）；`ttt_max_mem_norm` 作为 Streaming 的漂移上限
- **Muon + AdamW** — ndim≥2 用 Muon，bias / RMSNorm weight / token_emb 用 AdamW（arXiv 2502.16982）

**具体参数见 `config.py` → `Config`**（dataclass）——所有随规模变化的字段（d_model / n_layers / kv_lora_rank / moe_inter_dim / 训练超参……）都集中在此；通过 `dataclasses.replace(cfg, ...)` 实现变体，scale-up 时只需改一处。

刻意省略：FP4 训练、DSA / CSA / HCA / Hybrid Attention（V4-Pro 的长上下文三件套，在玩具规模下无意义）。

## 文件结构

```
HyperSeek-TTT/
├── config.py       Config dataclass（全部超参）
├── model.py        架构（全部 nn.Module）
├── data.py         Tokenizer，算术数据集，Episode 生成器
├── optim.py        Muon 优化器 + 参数切分
├── train.py        训练入口（NTP + chunked-recall 混合）
├── eval.py         推理 + 专家专化测试
├── memory_train.py Hidden-Secret Recall 训练 + 3 模式公平对比
├── memory_eval.py  Episode Memory + Chunked Recall 消融
├── runlog.py       JSONL 指标记录器（logs/*.jsonl）
├── transformer.py  v0 基线（MiniGPT），作为参考保留
├── tests/          pytest 回归测试套件
└── requirements.txt
```

## 使用

```bash
cd HyperSeek-TTT

# Setup（venv 在仓库根目录；--without-scm-ignore-files 保护 .gitignore）
python3.14 -m venv . --without-scm-ignore-files
./bin/pip install -r requirements.txt

# 测试
./bin/python -m pytest tests/

# 主训练（自动 MPS；步数 / 批大小 / lr 见 cfg.train_*）
./bin/python train.py

# 推理 + 专家专化测试
./bin/python eval.py

# 长上下文 / Episode-Memory 验证框架（3 模式公平对比）
./bin/python memory_eval.py --task kv     --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task needle --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task rule   --episodes 100 --chunk-len 32

# Hidden-Secret Recall 训练（3 模式公平对比）
./bin/python memory_train.py --steps 300 --batch-size 16 --secret-len 1
./bin/python memory_train.py --steps 100 --batch-size 8  --secret-len 4

# Gated-Memory 消融（Write-/Forget-Gate + 范数上限，init == baseline）
./bin/python memory_train.py --modes persistent --steps 100 --gated --max-mem-norm 5.0
```

## 训练监控

`train.py` 每 `cfg.train_mem_every` 步混入一个 chunked-recall
Memory 步骤（Secret 位于 Query 注意力范围之外的 chunk 中；只有
W_mem 能承载它——这给了 Gate 真正的学习压力）。每 100
步打印一行，每步写入 `logs/*.jsonl`
（`step / kind / loss / route / gate / amp_avg / amp_max`）：

```
Step  XXXX loss X.XXXX route X.XXXX gate X.XXXX mhc_amp avg X.XXX max X.XXX
                                         ↑ 仅统计骨干 HC，MTP 内部不计
```

各模块暴露的监控字段（在前向时实时更新）：

| 模块 | 字段 | 用途 |
|---|---|---|
| HyperConnect | `last_amp` | 信号放大，V4 目标 ≈ 1.6 |
| HyperConnect | `last_alpha`, `last_beta`, `last_lambda` | 归一化权重分布 |
| HyperConnect | `last_replica_norms` | n_hc=4 副本利用率 |
| MoE | `last_load`, `last_router_logits` | 专家负载 |
| InPlaceTTT | `last_gate`, `last_L` | Gate 均值 + Inner Loss |
| InPlaceTTT | `last_delta_norm`, `last_h_norm`, `last_update_norm`, `last_mem_norm` | Memory / 更新强度 |

## 实验设计：memory_train 3 模式公平对比

`memory_train.py` 训练任务"将 Secret 隐藏在 Context 中，Query 输出 Secret"，有三种模式：

| 模式 | W_mem 行为 | atlas.W 学习路径 |
|---|---|---|
| `none` | 固定 `W_mem = 0`，既不读也不写 | 通过 Query-Forward 学习；atlas 退化为静态 Linear + Gate |
| `transient` | 每个 chunk 从零起始做 Inner Update；不跨 chunk 累积 | 通过 Functional 路径反向传播 |
| `persistent` | W_mem 跨 chunk 累积，Episode 开始前重置 | 通过 Functional 路径反向传播 |

三种模式统一走 `build_functional_memory`（Functional 路径）。这样差异只在于"W_mem 是否跨 chunk 累积"，持久化 Fast Weights 的**边际收益**也就真正可测。

`memory_eval.py` 对三种长上下文任务（kv_recall / needle / rule）× 三种模式做消融。

## InPlaceTTT vs ATLAS

当前 InPlaceTTT 为默认；`AtlasMemory` 不在源码树中。需要时按 [arXiv 2505.23735](https://arxiv.org/abs/2505.23735) 规范作为新类添加。

| | InPlaceTTT（当前） | ATLAS（备选） |
|---|---|---|
| Inner Loss | hidden-delta MSE：`\|\|W·h_t - (h_{t+1}-h_t)\|\|²` | 联想记忆：`\|\|M(k_t) - v_t\|\|²` |
| 内部结构 | 单一 Linear + Gate | M（2 层 MLP）+ W_k / W_v + Gate |
| 参数规模 | 一次 `d_model²` | 约 5× `d_model²` |
| 设计理念 | 复用现有结构 + Fast Weight | 新增独立 Memory 模块 |
| 论文 | 2026-04 | 2025-05 |

**何时推荐 ATLAS**：
- 任务有清晰的键值（Key-Value）联想结构（Query 式召回、显式查找）
- Key 命中是离散的，联想模式比 hidden delta 更容易学习
- 需要更大的存储容量（M 是 2 层 MLP，比单一 Linear 表达力更强）
- 希望与 ATLAS 论文基线直接对比

**何时推荐 InPlaceTTT**：
- 通用 NTP 式任务
- 参数效率优先（Memory 分支少约 80% 参数）
- 接轨 2026 主流（Behrouz 等人的后继工作）

## 扩规模（Scale-up）注意事项

**目前仅流水线验证，未做有效性检验** —— InPlaceTTT Test-Time Learning、mHC 副本利用率、MTP 收益、Muon vs AdamW 优势等的真实价值在玩具规模上**全部无法显现**。只有 scale-up 才能验证。

### 参数调优顺序

1. **先确认深度 `n_layers`** — 多任务保持能力不足根本上来自浅层模型。scale-up 第一步：增加深度。MTP 占比随深度自动下降。
2. **保持 `n_mtp_heads = 1`** — V3 / V4 都是 depth=1，不要动。
3. **`seq_len` 改动需要重建模型** — `rope_cos / causal_mask` 是 buffer，不能动态扩展。
4. **按规模调整 MoE 块** — `n_routed_experts` / `moe_inter_dim` 与模型大小成比例扩展。

### 可能的扩展方向（已有 hook，需新逻辑）

- ~~**TTT 双 Gate**~~ — **已实现**为 `ttt_gated_memory`（Read-Gate 本已存在；Write-Gate 加权 Inner-Loss 位置，Forget-Gate 取代固定保留）。玩具规模的 100 步 A/B 中与 baseline 无可测差异（by design: init == baseline）——有效性检验留给 scale-up
- **Reasoning Token 格式**：玩具用内联 CoT；scale-up 时以 `<think>...</think>` 作为显式段 Token 添加

### 已检验并暂缓的方向

- **按信息密度自适应的 chunk 大小** — 会破坏批处理 per-Sample Fast Weights 所依赖的固定长度 chunk 对齐；目标（不写入冗余内容）已由 Write-Gate 按 token 而非按 block 覆盖。只在 scale-up 有真实算力压力时重新评估。
- **逐层差异化更新频率** — 前提在此不成立：只有一个顶层 TTT 适配器，没有逐层 Fast Weights。多级 TTT 是新的架构实验，而非对现有实现的优化。
- **对比性附加损失 / 信息瓶颈** — 玩具规模无法测量（连更直接的干预都落在噪声里，见 A/B 结果）；只有 scale-up 测量条件成熟后才值得。
- **Fast-Weight 量化（4-bit）** — Fast-Weight 状态是 d_model² = 64 KB；量化在此只会增加复杂度而无收益（且 MPS 没有现成的 4-bit kernel）。只有 d_model 足够大才相关。

### 训练监控（每 100 步盯这四个）

| 指标 | 关注点 |
|---|---|
| **`loss`** | 单调下降；尖峰 / 长平台要警惕 |
| **`gate`**（`InPlaceTTT.last_gate`） | **判断 TTT 是否真正在学的关键指标**。训练中 > 0.1 → TTT 生效；停留在 ≈ 0.02 → 空转 |
| **`route`**（Router 监控的 CE） | 玩具规模收敛到 1.0–1.5；不下降 → Router 未分化 |
| **`mhc_amp avg / max`** | avg ≈ 1.6（V4 目标），max < 5；max > 10 → mHC alpha / beta / lambda 漂移，信号放大失控 |

监控字段均已暴露；scale-up 时对多步曲线作图最为直观。

## 设计缺陷

1. **仅靠玩具算术不能验证 Test-Time Learning** — 在封闭确定性任务上 Gate 会空转。自 `train.py` 加入 chunked-recall 混合后 Gate 获得了真正压力（实测：400 步从 0.018 → ~0.1）；要得到可靠的结论仍需要 BPE + 自然语言。

2. **嫁接（Anpfropfung）到 DeepSeek-V2-Lite 成本极高** — 维度全部对不上，Tokenizer 完全不同，架构差异巨大（V2-Lite 没有 mHC / MTP / InPlaceTTT）。"嫁接"≈ 重写整个项目。

3. **n_layers 太浅** — 目前是玩具安全性上限，但多任务保持仍可能失效；MTP head 占比仍大，训练动态受 MTP 干扰。scale-up 到 ≥ 12 层可自动缓解。

4. **基于字符的 Tokenizer，只有算术** — 自然语言需要 BPE 32K–50K，整个 `data.py` 需要重写。

5. **mHC-lite 对 `n_hc` 敏感** — `n_hc=4` → 24 种排列安全；`n_hc≥8` → buffer 爆炸，需要 K-cap 子采样（按论文为 n_hc! 的子集）。

6. ~~**`make_long_effective_mask`**~~ — 无调用方的死占位符，已删除。

7. ~~**InPlaceTTT Inner Step 在样本间共享**~~ — **已修复**：Inner 梯度按样本的闭式解（`_inner_grad`, (B, D, D) Fast Weights）；回归测试强制 batch==serial 等价。

8. **MTP 内嵌完整 DeepSeekBlock** — 与 V3 论文一致，但玩具规模下 MTP 占比大，干扰骨干训练诊断。scale-up 自动缓解。

## 隐藏约束（坑）

- `model(idx, return_mtp=True)` 必须传入 `mtp_tokens=next_tok_ids`（fallback 是 raise，不再是静默错误）
- `rope_cos / causal_mask` 自长度解耦后懒增长（`_rope_mask`）；任意长度都能跑，但超出训练长度后 RoPE 不保证质量
- 持久化 `W_mem` 的**写入**经 `InPlaceTTT.forward` 要求 batch 为 1（per-Sample Fast Weights 无法合并进一个 buffer）——否则 ValueError；只读可 batched
- InPlaceTTT 的 `persistent_memory` 在构造时确定（`cfg.ttt_persistent_memory`），训练中途不能切换；`memory_train.py` 每个模式新建一个 Model（`dataclasses.replace(cfg, ...)`）
- 主 `train.py` 对 InPlaceTTT 实例使用属性名 `model.atlas`（命名源于历史，state_dict 兼容性未破坏；实际类型是 InPlaceTTT）

## 参考文献

- DeepSeek-V2: arXiv 2405.04434
- DeepSeek-V3 Technical Report: arXiv 2412.19437
- DeepSeek-V4-Pro: 2026-04-24 发布（[HF Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)）
- mHC (manifold-constrained): arXiv 2512.24880
- **mHC-lite**（本项目）：arXiv 2601.05732
- Muon 优化器: arXiv 2502.16982
- ATLAS（备选 Memory 分支）：arXiv 2505.23735
- Titans / MIRAS: arXiv 2501.00663
- TNT（ATLAS 训练效率，ICLR 2026）：arXiv 2511.07343
- **In-Place TTT**（本项目）：arXiv 2604.06169
- LLMs-from-scratch MLA 参考：github.com/rasbt/LLMs-from-scratch
