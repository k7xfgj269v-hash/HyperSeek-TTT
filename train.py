import os
import torch
import torch.nn.functional as F

from model import cfg, CKPT, DeepSeekMini, MoE, HyperConnect
from data import PAD, make_batch, make_recall_batch
from optim import Muon, split_params


def memory_step(model, batch_size, chunk_len, n_pairs, secret_len, device):
    """Chunked kv-recall step through the unified InPlaceTTT primitives.

    The secret lives in context chunks the query forward never attends to;
    only W_mem can carry it, so the gate gets real pressure to open.
    """
    x, ctx_len = make_recall_batch(batch_size, n_pairs, secret_len, device=device)
    atlas = model.atlas
    W_mem = torch.zeros(batch_size, *atlas.W.shape, device=device)
    for s in range(0, ctx_len, chunk_len):
        h = model.backbone_hidden(x[:, s:s + chunk_len])
        W_mem = atlas.update_memory(h, W_mem, differentiable=True)
    h_q = model.backbone_hidden(x[:, ctx_len:-1])
    logits = model.logits_from_hidden(atlas.apply_memory(h_q, W_mem))
    logits = logits[:, -secret_len:]
    target = x[:, -secret_len:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def train(model, n_steps=None, batch_size=None, lr=None, device='cpu', route_w=None):
    if n_steps is None: n_steps = cfg.train_n_steps
    if batch_size is None: batch_size = cfg.train_batch_size
    if lr is None: lr = cfg.train_lr
    if route_w is None: route_w = cfg.train_route_w
    model.to(device).train()
    muon_params, adamw_params = split_params(model)
    opt_muon = Muon(muon_params, lr=cfg.muon_lr)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=lr)
    mem_every = getattr(cfg, 'train_mem_every', 0)
    for step in range(n_steps):
        loss_route = torch.tensor(0.0, device=device)
        if mem_every and step % mem_every == mem_every - 1:
            loss = memory_step(
                model, cfg.train_mem_batch, cfg.train_mem_chunk,
                cfg.train_mem_pairs, cfg.train_mem_secret_len, device,
            )
        else:
            x, ops = make_batch(batch_size, cfg.seq_len, device=device)
            inp = x[:, :-1]
            tgt = x[:, 1:]
            logits, mtp_list = model(inp, return_mtp=True, mtp_tokens=x[:, 1:])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1), ignore_index=PAD)
            if mtp_list is not None and inp.size(1) > 1:
                tgt2 = x[:, 2:]
                mtp_logits = mtp_list[0]
                L = min(mtp_logits.size(1), tgt2.size(1))
                if L > 0:
                    loss_mtp = F.cross_entropy(
                        mtp_logits[:, :L].reshape(-1, cfg.vocab_size),
                        tgt2[:, :L].reshape(-1),
                        ignore_index=PAD,
                    )
                    loss = loss + cfg.mtp_loss_weight * loss_mtp
            n_moe = 0
            for m in model.modules():
                if isinstance(m, MoE) and not m.is_mtp and m.last_router_logits is not None:
                    rl = m.last_router_logits
                    target = ops.unsqueeze(-1).expand(-1, rl.size(1))
                    loss_route = loss_route + F.cross_entropy(rl.reshape(-1, rl.size(-1)), target.reshape(-1))
                    n_moe += 1
            if n_moe > 0:
                loss_route = loss_route / n_moe
                loss = loss + route_w * loss_route
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adamw_params, max_norm=cfg.train_grad_clip)
        opt_muon.step()
        opt_adamw.step()
        # Bug 4: 监督路由 CE 与 bias 均衡冲突，本实验停用 update_router_bias
        # for m in model.modules():
        #     if isinstance(m, MoE) and not m.is_mtp:
        #         m.update_router_bias(cfg.bias_update_speed)
        if step % 100 == 0:
            amps = [hc.last_amp for blk in model.blocks for hc in (blk.hc_attn, blk.hc_ffn) if hc.last_amp is not None]
            avg_amp = sum(amps) / len(amps) if amps else 0.0
            max_amp = max(amps) if amps else 0.0
            gate = model.atlas.last_gate if model.atlas.last_gate is not None else 0.0
            print(f"Step {step:4d} loss {loss.item():.4f} route {loss_route.item():.4f} gate {gate:.4f} mhc_amp avg {avg_amp:.3f} max {max_amp:.3f}")
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")
    return model


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"using device: {device}")
    model = DeepSeekMini(cfg)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location='cpu'))
        print(f"loaded {CKPT}")
        if os.environ.get('CONTINUE'):
            train(model, device=device)
    else:
        train(model, device=device)
