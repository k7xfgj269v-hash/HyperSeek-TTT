import torch

from model import DeepSeekMini, InPlaceTTT

from tests.test_ttt import small_cfg


def make_gated(d_model=16, **kw):
    torch.manual_seed(0)
    atlas = InPlaceTTT(d_model=d_model, window=8, gated_memory=True, **kw)
    with torch.no_grad():
        atlas.W.normal_(0, 0.1)
    return atlas


def test_gated_creates_params_only_when_enabled():
    plain = InPlaceTTT(d_model=16, window=8)
    gated = InPlaceTTT(d_model=16, window=8, gated_memory=True)
    assert not hasattr(plain, 'write_gate') and not hasattr(plain, 'forget_gate')
    assert hasattr(gated, 'write_gate') and hasattr(gated, 'forget_gate')


def test_forget_gate_initializes_at_retention():
    atlas = make_gated()
    h = torch.randn(3, 24, 16)
    r = atlas._retention(h)
    assert r.shape == (3, 1, 1)
    assert torch.allclose(r, torch.full_like(r, atlas.retention), atol=1e-6)


def test_write_gate_initializes_near_one():
    atlas = make_gated()
    s = torch.sigmoid(atlas.write_gate(torch.randn(4, 8, 16)))
    assert (s > 0.95).all()


def test_mem_norm_guard():
    atlas = make_gated(max_mem_norm=0.005)
    h = torch.randn(2, 24, 16)
    W_mem = torch.zeros(2, 16, 16)
    for _ in range(5):
        W_mem = atlas.update_memory(h, W_mem)
    assert (W_mem.flatten(1).norm(dim=1) <= 0.005 + 1e-6).all()

    atlas_free = make_gated(max_mem_norm=0.0)
    W_free = torch.zeros(2, 16, 16)
    for _ in range(5):
        W_free = atlas_free.update_memory(h, W_free)
    assert (W_free.flatten(1).norm(dim=1) > 0.005).all()


def test_gated_batch_equals_serial():
    atlas = make_gated()
    with torch.no_grad():
        atlas.gate.weight.normal_(0, 0.1)
        atlas.write_gate.weight.normal_(0, 0.1)
        atlas.forget_gate.weight.normal_(0, 0.1)
    h = torch.randn(4, 24, 16)
    out_batch = atlas(h)
    out_serial = torch.cat([atlas(h[b:b + 1]) for b in range(h.shape[0])])
    assert torch.allclose(out_batch, out_serial, atol=1e-5)


def test_gates_receive_gradients():
    atlas = make_gated()
    h = torch.randn(2, 24, 16, requires_grad=True)
    out = atlas(h)
    out.sum().backward()
    assert atlas.write_gate.weight.grad is not None
    assert atlas.forget_gate.weight.grad is not None
    assert atlas.gate.weight.grad is not None


def test_gated_full_model_runs_and_trains():
    torch.manual_seed(0)
    cfg = small_cfg(ttt_gated_memory=True, ttt_max_mem_norm=5.0)
    model = DeepSeekMini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 33))
    logits, _ = model(x)
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert model.atlas.forget_gate.weight.grad is not None


def test_gated_update_memory_differentiable_path():
    atlas = make_gated()
    h = torch.randn(2, 24, 16)
    W_mem = atlas.update_memory(h, torch.zeros(2, 16, 16), differentiable=True)
    W_mem.square().sum().backward()
    assert atlas.write_gate.weight.grad is not None and atlas.write_gate.weight.grad.norm() > 0
    assert atlas.forget_gate.weight.grad is None or atlas.forget_gate.weight.grad.norm() == 0
