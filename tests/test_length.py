import torch

from model import DeepSeekMini

from tests.test_ttt import small_cfg


def test_input_longer_than_cfg_seq_len_runs():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DeepSeekMini(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len * 2 + 5))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (2, cfg.seq_len * 2 + 5, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backbone_prefix_consistency_across_growth():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DeepSeekMini(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len * 2))
    with torch.no_grad():
        h_short = model.backbone_hidden(x[:, :30])
        h_full = model.backbone_hidden(x)
    assert torch.allclose(h_full[:, :30], h_short, atol=1e-5)


def test_rope_cache_growth_preserves_prefix():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DeepSeekMini(cfg).eval()
    cos_before = model.rope_cos.clone()
    n_before = cos_before.size(0)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len * 3))
    with torch.no_grad():
        model.backbone_hidden(x)
    assert model.rope_cos.size(0) >= cfg.seq_len * 3
    assert torch.allclose(model.rope_cos[:n_before], cos_before, atol=1e-6)
    assert model.causal_mask.size(0) >= cfg.seq_len * 3


def test_mtp_path_with_long_input():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DeepSeekMini(cfg).eval()
    T = cfg.seq_len + 17
    x = torch.randint(0, cfg.vocab_size, (2, T + 1))
    with torch.no_grad():
        logits, mtp = model(x[:, :-1], return_mtp=True, mtp_tokens=x[:, 1:])
    assert logits.shape[1] == T
    assert mtp is not None and torch.isfinite(mtp[0]).all()
