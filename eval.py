import os
import torch

from model import cfg, CKPT, DeepSeekMini, MoE
from data import TABLE, PAD, OPS, encode, decode, gen_sample


@torch.no_grad()
def generate(model, prompt, max_new=10, device='cpu'):
    model.eval()
    idx = encode(prompt)
    x = torch.tensor([idx], dtype=torch.long, device=device)
    for _ in range(max_new):
        if x.size(1) >= cfg.seq_len:
            break
        logits, _ = model(x)
        nxt = logits[0, -1].argmax(-1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == TABLE['$'] or nxt == PAD:
            break
    return decode(x[0].tolist())


@torch.no_grad()
def expert_specialization(model):
    model.eval()
    for op_id, op_name in enumerate(OPS):
        samples = []
        while len(samples) < 64:
            s, op = gen_sample()
            if op == op_id:
                samples.append(s)
        encoded = [encode(s) for s in samples]
        padded = [
            (e[:cfg.seq_len] if len(e) >= cfg.seq_len else e + [PAD] * (cfg.seq_len - len(e)))
            for e in encoded
        ]
        x_test = torch.tensor(padded, dtype=torch.long)
        model(x_test[:, :-1])
        counts = torch.zeros(cfg.n_routed_experts)
        for m in model.modules():
            if isinstance(m, MoE) and not m.is_mtp:
                counts += m.last_load.cpu()
        print(f"  op {op_name} load {counts.tolist()}")


if __name__ == "__main__":
    model = DeepSeekMini(cfg)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location='cpu'))
        print(f"loaded {CKPT}")
    else:
        print(f"WARNING: no checkpoint at {CKPT}, using random init")

    print("Generieren mit DeepSeekMini")
    for prompt in ["12+34=", "56+78=", "10+9=", "75-23=", "90-45=", "7*8=", "9*6=", "63/9=", "48/6="]:
        result = generate(model, prompt, max_new=25)
        print(f"'{prompt}' -> '{result}'")

    print("\nExpert Specialisierung")
    expert_specialization(model)
