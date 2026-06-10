import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace

from config import Config
from model import DeepSeekMini, InPlaceTTT


def small_cfg(**overrides):
    base = Config(
        d_model=32, n_heads=2, n_layers=2, seq_len=48,
        kv_lora_rank=16, q_lora_rank=16,
        qk_nope_dim=8, qk_rope_dim=8, v_head_dim=8,
        n_routed_experts=2, n_shared_experts=1, n_activated_experts=1,
        moe_inter_dim=32, n_hc=2, atlas_window=8,
    )
    return replace(base, **overrides)


def autograd_inner_grad(atlas, h_b):
    """Reference: the original pooled-autograd inner update for one sample."""
    W = atlas._state_W().detach().requires_grad_(True)
    D = h_b.shape[-1]
    w = min(atlas.window, h_b.shape[1] - 1)
    with torch.enable_grad():
        h_curr = h_b[:, -w - 1:-1, :].reshape(-1, D).detach()
        h_next = h_b[:, -w:, :].reshape(-1, D).detach()
        pred = F.linear(h_curr, W)
        L = ((pred - (h_next - h_curr)) ** 2).sum(-1).mean()
        g = torch.autograd.grad(L, [W])[0]
    norm = g.norm() + 1e-12
    scale = (1.0 / norm).clamp(max=1.0)
    return g * scale, L


def test_closed_form_matches_autograd():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8)
    with torch.no_grad():
        atlas.W.normal_(0, 0.1)
    h = torch.randn(4, 24, 16)
    g, L = atlas._inner_grad(h, atlas._state_W())
    for b in range(h.shape[0]):
        g_ref, L_ref = autograd_inner_grad(atlas, h[b:b + 1])
        assert torch.allclose(g[b], g_ref, atol=1e-6), f"sample {b} grad mismatch"
        assert torch.allclose(L[b], L_ref, atol=1e-6), f"sample {b} loss mismatch"


def test_ttt_batch_equals_serial():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8)
    with torch.no_grad():
        atlas.W.normal_(0, 0.1)
        atlas.gate.weight.normal_(0, 0.1)
    h = torch.randn(4, 24, 16)
    out_batch = atlas(h)
    out_serial = torch.cat([atlas(h[b:b + 1]) for b in range(h.shape[0])])
    assert torch.allclose(out_batch, out_serial, atol=1e-5)


def test_full_model_batch_equals_serial():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DeepSeekMini(cfg).eval()
    with torch.no_grad():
        model.atlas.W.normal_(0, 0.1)
        model.atlas.gate.weight.normal_(0, 0.1)
        model.atlas.gate.bias.zero_()
    x = torch.randint(0, cfg.vocab_size, (4, 33))
    with torch.no_grad():
        out_batch, _ = model(x)
        out_serial = torch.cat([model(x[b:b + 1])[0] for b in range(x.shape[0])])
    assert torch.allclose(out_batch, out_serial, atol=1e-5)


def test_persistent_write_requires_batch_one():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8, persistent_memory=True)
    atlas.set_memory_enabled(True, write_enabled=True)
    with pytest.raises(ValueError):
        atlas(torch.randn(2, 24, 16))


def test_persistent_write_and_reset():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8, persistent_memory=True)
    atlas.set_memory_enabled(True, write_enabled=True)
    h = torch.randn(1, 24, 16)
    atlas(h)
    assert atlas.W_mem.norm().item() > 0
    atlas.reset_memory()
    assert atlas.W_mem.norm().item() == 0


def test_non_persistent_forward_is_stateless():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8)
    h = torch.randn(2, 24, 16)
    out1 = atlas(h)
    out2 = atlas(h)
    assert torch.allclose(out1, out2)
    assert atlas.W_mem.norm().item() == 0


def test_outer_grad_flows_to_slow_weights():
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=16, window=8)
    h = torch.randn(2, 24, 16)
    out = atlas(h)
    out.sum().backward()
    assert atlas.W.grad is not None
    assert atlas.gate.weight.grad is not None
    assert atlas.W.grad.norm().item() > 0
