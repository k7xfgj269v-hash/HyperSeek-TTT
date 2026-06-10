import argparse
import random
from dataclasses import replace

import torch
import torch.nn.functional as F

from data import TABLE, encode, decode, secret_episode, split_chunks
from model import DeepSeekMini, cfg
from optim import Muon, split_params
from runlog import JsonlLogger


MODES = ["none", "transient", "persistent"]


def make_model(device, seq_len, mode):
    run_cfg = replace(cfg, seq_len=seq_len, ttt_persistent_memory=(mode == "persistent"))
    return DeepSeekMini(run_cfg).to(device)


def set_mode(model, mode):
    if mode == "none":
        model.set_memory_enabled(False, write_enabled=False)
    elif mode == "transient":
        model.set_memory_enabled(True, write_enabled=False)
    elif mode == "persistent":
        model.set_memory_enabled(True, write_enabled=True)
    else:
        raise ValueError(f"unknown mode: {mode}")


def build_functional_memory(model, context, device, chunk_len, create_graph, accumulate=True):
    """Build fast-weight W_mem from context, chunked, via InPlaceTTT.update_memory.

    accumulate=True: persistent — W_mem 跨 chunk 累积
    accumulate=False: transient — 每 chunk 重新从 zero base 算，返回最后 chunk 的 W_mem
    """
    atlas = model.atlas
    W_mem = torch.zeros_like(atlas.W)
    for chunk in split_chunks(context, chunk_len):
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        h = model.backbone_hidden(x)
        base = W_mem if accumulate else torch.zeros_like(atlas.W)
        W_mem = atlas.update_memory(h, base, differentiable=create_graph)
    return W_mem


def logits_with_memory(model, ids, W_mem, device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    h = model.backbone_hidden(x)
    return model.logits_from_hidden(model.atlas.apply_memory(h, W_mem))


def target_loss_with_memory(model, query, secret, W_mem, device):
    prompt = query + secret[:-1]
    target = torch.tensor(encode(secret), dtype=torch.long, device=device)
    logits = logits_with_memory(model, encode(prompt), W_mem, device)
    logits = logits[0, -len(secret):, :]
    return F.cross_entropy(logits, target)


@torch.no_grad()
def generate_secret(model, query, secret_len, device):
    out = query
    model.set_memory_enabled(True, write_enabled=False)
    for _ in range(secret_len):
        x = torch.tensor([encode(out)], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax(-1).item())
        out += decode([nxt])
    return out[len(query):]


def generate_secret_with_memory(model, query, secret_len, W_mem, device):
    out = query
    for _ in range(secret_len):
        logits = logits_with_memory(model, encode(out), W_mem, device)
        nxt = int(logits[0, -1].argmax(-1).item())
        out += decode([nxt])
    return out[len(query):]


def train_one_mode(args, mode, device, logger=None):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = make_model(device, args.seq_len, mode)
    model.train()
    muon_params, adamw_params = split_params(model)
    opt_muon = Muon(muon_params, lr=cfg.muon_lr)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args.lr)
    chunk_len = min(args.chunk_len, args.seq_len)

    for step in range(1, args.steps + 1):
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        last_mem_norm = 0.0
        for _ in range(args.batch_size):
            context, query, secret = secret_episode(args.secret_len, args.fillers)
            model.reset_memory()
            if mode == "none":
                W_mem = torch.zeros_like(model.atlas.W)
            elif mode == "transient":
                W_mem = build_functional_memory(
                    model, context, device, chunk_len,
                    create_graph=True, accumulate=False,
                )
            elif mode == "persistent":
                W_mem = build_functional_memory(
                    model, context, device, chunk_len,
                    create_graph=True, accumulate=True,
                )
            last_mem_norm = W_mem.detach().norm().item()
            total_loss = total_loss + target_loss_with_memory(model, query, secret, W_mem, device)
        loss = total_loss / args.batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=args.grad_clip)
        opt_muon.step()
        opt_adamw.step()

        gate = getattr(model.atlas, "last_gate", 0.0) or 0.0
        if logger is not None:
            logger.log(mode=mode, step=step, loss=round(loss.item(), 4),
                       gate=round(gate, 4), mem=round(last_mem_norm, 4))
        if step % args.log_every == 0 or step == 1:
            print(f"{mode:10s} step={step:4d} loss={loss.item():.4f} gate={gate:.4f} mem={last_mem_norm:.4f}")

    return model


@torch.no_grad()
def eval_model(args, model, mode, device):
    model.eval()
    chunk_len = min(args.chunk_len, args.seq_len)
    exact = 0
    token_ok = 0
    total_tokens = args.eval_episodes * args.secret_len
    gate_sum = 0.0
    mem_sum = 0.0

    for _ in range(args.eval_episodes):
        context, query, secret = secret_episode(args.secret_len, args.fillers)
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
        pred = generate_secret_with_memory(model, query, args.secret_len, W_mem.detach(), device)
        exact += int(pred == secret)
        token_ok += sum(int(a == b) for a, b in zip(pred, secret))
        gate_sum += getattr(model.atlas, "last_gate", 0.0) or 0.0
        mem_sum += W_mem.detach().norm().item()

    return {
        "exact": exact / args.eval_episodes,
        "token": token_ok / total_tokens,
        "gate": gate_sum / args.eval_episodes,
        "mem": mem_sum / args.eval_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train hidden-secret recall with memory ablations.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--chunk-len", type=int, default=64)
    parser.add_argument("--fillers", type=int, default=40)
    parser.add_argument("--secret-len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"device={device} steps={args.steps} batch={args.batch_size} "
        f"seq_len={args.seq_len} chunk_len={min(args.chunk_len, args.seq_len)} "
        f"secret_len={args.secret_len}"
    )

    logger = JsonlLogger('memory_train', root=args.log_dir)
    results = {}
    for mode in args.modes:
        model = train_one_mode(args, mode, device, logger=logger)
        results[mode] = eval_model(args, model, mode, device)
        logger.log(mode=mode, kind='eval', **{k: round(v, 4) for k, v in results[mode].items()})

    print("summary")
    for mode in args.modes:
        r = results[mode]
        print(
            f"{mode:10s} exact={r['exact']:.3f} token={r['token']:.3f} "
            f"gate={r['gate']:.4f} mem_norm={r['mem']:.4f}"
        )
    print(f"metrics {logger.path}")
    logger.close()


if __name__ == "__main__":
    main()
