import torch
import torch.nn.functional as F

from data import make_recall_batch, secret_episode, split_chunks
from memory_train import build_functional_memory, logits_with_memory
from model import DeepSeekMini
from train import memory_step

from tests.test_ttt import small_cfg


def old_functional_memory_update(model, h, W_mem, create_graph):
    """The pre-refactor autograd implementation, kept as parity reference."""
    atlas = model.atlas
    B, T, D = h.shape
    w = min(atlas.window, T - 1)
    if w < 1:
        return W_mem
    W_state = atlas.W + W_mem
    h_curr = h[:, -w - 1:-1, :].reshape(-1, D)
    h_next = h[:, -w:, :].reshape(-1, D)
    pred = F.linear(h_curr, W_state)
    target = h_next - h_curr
    loss = ((pred - target) ** 2).sum(-1).mean()
    g_W = torch.autograd.grad(loss, W_state, create_graph=create_graph, retain_graph=create_graph)[0]
    scale = (1.0 / (g_W.norm() + 1e-12)).clamp(max=1.0)
    return atlas.retention * W_mem - atlas.inner_lr * g_W * scale


def old_build_functional_memory(model, context, device, chunk_len, create_graph, accumulate):
    W_mem = torch.zeros_like(model.atlas.W)
    for chunk in split_chunks(context, chunk_len):
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        h = model.backbone_hidden(x)
        base = W_mem if accumulate else torch.zeros_like(W_mem)
        W_mem = old_functional_memory_update(model, h, base, create_graph=create_graph)
    return W_mem


def make_test_model(**overrides):
    torch.manual_seed(0)
    model = DeepSeekMini(small_cfg(**overrides))
    with torch.no_grad():
        model.atlas.W.normal_(0, 0.05)
        model.atlas.gate.weight.normal_(0, 0.05)
    return model


def grads_of(model, W_mem, probe):
    model.zero_grad()
    (W_mem.reshape(probe.shape) * probe).sum().backward()
    return (
        model.atlas.W.grad.clone(),
        model.token_emb.weight.grad.clone(),
        model.blocks[0].attn.W_DQ.weight.grad.clone(),
    )


def test_functional_parity_values_and_outer_grads():
    import random
    model = make_test_model()
    random.seed(1)
    context, _, _ = secret_episode(secret_len=2, n_fillers=10)
    assert len(context) % 13 == 0, "context must split into equal chunks for the transient mode"
    probe = torch.randn(model.atlas.W.shape)
    for accumulate in (True, False):
        W_old = old_build_functional_memory(model, context, 'cpu', 13, True, accumulate)
        g_old = grads_of(model, W_old, probe)
        W_new = build_functional_memory(model, context, 'cpu', 13, True, accumulate=accumulate)
        g_new = grads_of(model, W_new, probe)
        assert torch.allclose(W_old, W_new.reshape(W_old.shape), atol=1e-5)
        for a, b in zip(g_old, g_new):
            assert torch.allclose(a, b, atol=1e-4), f"outer grad mismatch (accumulate={accumulate})"
    model.zero_grad()


def test_logits_with_memory_matches_manual():
    from data import encode
    model = make_test_model().eval()
    W_mem = torch.randn_like(model.atlas.W) * 0.01
    ids = encode("1+2=3,9=")
    with torch.no_grad():
        out = logits_with_memory(model, ids, W_mem, 'cpu')
        x = torch.tensor([ids], dtype=torch.long)
        h = model.backbone_hidden(x)
        delta = F.linear(h, model.atlas.W + W_mem)
        gate = torch.sigmoid(model.atlas.gate(h))
        ref = model.logits_from_hidden(h + gate * delta)
    assert torch.allclose(out, ref, atol=1e-6)


def test_memory_step_backprops_through_chunks():
    model = make_test_model()
    loss = memory_step(model, batch_size=4, chunk_len=16, n_pairs=6, secret_len=2, device='cpu')
    assert torch.isfinite(loss)
    loss.backward()
    assert model.atlas.W.grad is not None and model.atlas.W.grad.norm() > 0
    assert model.atlas.gate.weight.grad is not None
    assert model.token_emb.weight.grad is not None and model.token_emb.weight.grad.norm() > 0


def test_make_recall_batch_shapes():
    x, ctx_len = make_recall_batch(5, n_pairs=6, secret_len=2)
    assert ctx_len == 6 * 5
    assert x.shape == (5, ctx_len + 2 + 2)


def test_train_mixture_smoke(tmp_path, monkeypatch):
    import train as train_module
    monkeypatch.setattr(train_module, 'CKPT', str(tmp_path / 'smoke.pt'))
    model = make_test_model(n_routed_experts=4)
    train_module.train(model, n_steps=4, batch_size=4, device='cpu')
    assert (tmp_path / 'smoke.pt').exists()
