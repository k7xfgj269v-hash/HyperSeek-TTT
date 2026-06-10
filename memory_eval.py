import argparse
import os
import random
from dataclasses import replace

import torch

from data import TABLE, decode, encode, kv_recall_episode, needle_episode, rule_episode
from model import CKPT, DeepSeekMini, cfg
from memory_train import build_functional_memory, logits_with_memory


def make_model(device, persistent_memory=False, load_ckpt=True):
    run_cfg = replace(cfg, ttt_persistent_memory=persistent_memory)
    model = DeepSeekMini(run_cfg).to(device).eval()
    if load_ckpt and os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=device))
    return model


def generate_text_with_memory(model, prompt, n_tokens, W_mem, device):
    out = prompt
    for _ in range(n_tokens):
        logits = logits_with_memory(model, encode(out), W_mem, device)
        nxt = int(logits[0, -1].argmax(-1).item())
        out += decode([nxt])
    return out[len(prompt):]


def eval_episode(model, context, query, target, device, chunk_len, mode):
    model.reset_memory()
    if mode == "none":
        W_mem = torch.zeros_like(model.atlas.W)
    elif mode == "transient":
        W_mem = build_functional_memory(
            model, context, device, chunk_len,
            create_graph=False, accumulate=False,
        )
    elif mode == "persistent":
        W_mem = build_functional_memory(
            model, context, device, chunk_len,
            create_graph=False, accumulate=True,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    with torch.no_grad():
        pred = generate_text_with_memory(model, query, len(target), W_mem.detach(), device)

    gate = getattr(model.atlas, "last_gate", 0.0) or 0.0
    mem = W_mem.detach().norm().item()
    token_ok = sum(int(a == b) for a, b in zip(pred, target))
    return pred == target, token_ok, len(target), gate, mem


TASKS = {
    "kv": kv_recall_episode,
    "needle": needle_episode,
    "rule": rule_episode,
}


def run_task(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    persistent_model = make_model(device, persistent_memory=True, load_ckpt=not args.no_ckpt)
    transient_model = make_model(device, persistent_memory=False, load_ckpt=not args.no_ckpt)
    task_fn = TASKS[args.task]
    modes = ["none", "transient", "persistent"] if args.mode == "all" else [args.mode]
    models = {
        "none": transient_model,
        "transient": transient_model,
        "persistent": persistent_model,
    }
    stats = {mode: {"ok": 0, "tok": 0, "total_tok": 0, "gate": 0.0, "mem": 0.0} for mode in modes}

    chunk_len = min(args.chunk_len, cfg.seq_len)
    for _ in range(args.episodes):
        context, query, target = task_fn(secret_len=args.secret_len)
        for mode in modes:
            ok, tok, total_tok, gate, mem = eval_episode(
                models[mode],
                context,
                query,
                target,
                device,
                chunk_len,
                mode,
            )
            stats[mode]["ok"] += int(ok)
            stats[mode]["tok"] += tok
            stats[mode]["total_tok"] += total_tok
            stats[mode]["gate"] += gate or 0.0
            stats[mode]["mem"] += mem or 0.0

    print(
        f"task={args.task} episodes={args.episodes} chunk_len={chunk_len} "
        f"secret_len={args.secret_len} device={device}"
    )
    for mode in modes:
        acc = stats[mode]["ok"] / args.episodes
        tok = stats[mode]["tok"] / stats[mode]["total_tok"]
        gate = stats[mode]["gate"] / args.episodes
        mem = stats[mode]["mem"] / args.episodes
        print(f"{mode:10s} exact={acc:.3f} token={tok:.3f} gate={gate:.4f} mem_norm={mem:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Episode memory / chunked-context evaluation.")
    parser.add_argument("--task", choices=sorted(TASKS), default="kv")
    parser.add_argument("--mode", choices=["all", "none", "transient", "persistent"], default="all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--chunk-len", type=int, default=32)
    parser.add_argument("--secret-len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--no-ckpt", action="store_true")
    args = parser.parse_args()
    run_task(args)


if __name__ == "__main__":
    main()
