import argparse
import os
import random
from types import SimpleNamespace

import torch

from data import TABLE, decode, encode
from model import CKPT, DeepSeekMini, cfg
from memory_train import build_functional_memory, logits_with_memory


DIGITS = "0123456789"


def make_model(device, persistent_memory=False, load_ckpt=True):
    run_cfg = SimpleNamespace(**{k: v for k, v in vars(cfg.__class__).items() if not k.startswith('__')})
    for k, v in vars(cfg).items():
        setattr(run_cfg, k, v)
    run_cfg.ttt_persistent_memory = persistent_memory
    model = DeepSeekMini(run_cfg).to(device).eval()
    if load_ckpt and os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=device))
    return model


def split_chunks(text, chunk_len):
    ids = encode(text)
    return [ids[i:i + chunk_len] for i in range(0, len(ids), chunk_len)]


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
        with torch.enable_grad():
            W_mem = build_functional_memory(
                model, context, device, chunk_len,
                create_graph=False, accumulate=False,
            )
    elif mode == "persistent":
        with torch.enable_grad():
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


def random_digits(n):
    return "".join(random.choice(DIGITS) for _ in range(n))


def kv_recall_episode(n_pairs=8, secret_len=1):
    n_pairs = min(n_pairs, len(DIGITS))
    keys = random.sample(DIGITS, n_pairs)
    vals = [random_digits(secret_len) for _ in keys]
    pairs = list(zip(keys, vals))
    random.shuffle(pairs)
    q_key, q_val = random.choice(pairs)
    context = ",".join(f"{k}={v}" for k, v in pairs) + ","
    query = f"{q_key}="
    return context, query, q_val


def needle_episode(n_fillers=20, secret_len=1):
    secret = random_digits(secret_len)
    fillers = [f"{random.choice(DIGITS)}+{random.choice(DIGITS)}={random.choice(DIGITS)}" for _ in range(n_fillers)]
    pos = random.randrange(len(fillers) + 1)
    fillers.insert(pos, f"9={secret}")
    context = ",".join(fillers) + ","
    query = "9="
    return context, query, secret


def rule_episode(n_rules=6, secret_len=1):
    lhs = random.sample(DIGITS, n_rules)
    rhs = [random_digits(secret_len) for _ in lhs]
    rules = list(zip(lhs, rhs))
    random.shuffle(rules)
    q_key, q_val = random.choice(rules)
    context = ",".join(f"{k}={v}" for k, v in rules) + ","
    query = f"{q_key}="
    return context, query, q_val


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
