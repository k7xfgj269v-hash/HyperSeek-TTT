# HyperSeek-TTT

DeepSeek V4-Pro 架构 + 2026 Test-Time Learning（In-Place TTT）的玩具级复现。流水线端到端可跑；真正的有效性验证有待扩规模（scale-up）完成。

[English README](README.md)

> 独立复现。与 DeepSeek-AI 或所引用论文的作者无关。所有代码均为原创；对 DeepSeek V4-Pro 和 arXiv 论文的引用仅用于注明出处。以 MIT 许可证发布。

架构深挖与实验日志在 [`docs/`](docs/) 下：[ARCHITECTURE.md](docs/ARCHITECTURE.md) 看设计，[EXPERIMENTS.md](docs/EXPERIMENTS.md) 看监控、消融与已知坑。

## 功能特性

- **MLA 骨干** — DeepSeek-V2 风格的多头潜在注意力，带解耦 RoPE。
- **DeepSeekMoE** — routed + shared experts，sigmoid + bias router，无 aux loss。
- **mHC-lite** — 经 Birkhoff–von Neumann 分解的多副本残差超连接（严格双随机）。
- **In-Place TTT** — 顶层记忆适配器，hidden-delta 内损失 + 逐样本闭式 Fast Weights。
- **可选门控记忆** — write / 学习式 forget 门 + 范数护栏，初始化恰在 baseline 点上。
- **多 Token 预测** — 内嵌 DeepSeekBlock 的辅助 MTP head。
- **Muon + AdamW** — 优化器切分（ndim ≥ 2 → Muon；bias / norm / embedding → AdamW）。
- **chunked-recall 训练混合** — Memory 步骤给 Gate 真实学习压力；3 模式公平对比框架。

## 快速开始

```bash
cd HyperSeek-TTT

# Setup（venv 在仓库根目录；--without-scm-ignore-files 保护 .gitignore）
python3.14 -m venv . --without-scm-ignore-files
./bin/pip install -r requirements.txt

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

## 测试

```bash
./bin/python -m pytest tests/
```

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
├── docs/           架构与实验/工程笔记
├── tests/          pytest 回归测试套件
└── requirements.txt
```

## License

MIT License — 见 [LICENSE](LICENSE)。
