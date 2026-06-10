import torch

from model import DeepSeekMini
from optim import split_params

from tests.test_ttt import small_cfg


def test_muon_never_gets_rank_deficient_params():
    model = DeepSeekMini(small_cfg())
    muon_params, adamw_params = split_params(model)
    for p in muon_params:
        assert p.ndim >= 2 and min(p.shape) > 1
    adamw_ids = {id(p) for p in adamw_params}
    assert id(model.atlas.gate.weight) in adamw_ids
    assert id(model.atlas.gate.bias) in adamw_ids
    assert id(model.token_emb.weight) in adamw_ids
    assert id(model.atlas.W) not in adamw_ids


def test_split_covers_all_trainable_params():
    model = DeepSeekMini(small_cfg())
    muon_params, adamw_params = split_params(model)
    split_ids = {id(p) for p in muon_params} | {id(p) for p in adamw_params}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert split_ids == all_ids
