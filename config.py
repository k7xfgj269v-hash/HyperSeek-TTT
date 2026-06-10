from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 18

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    seq_len: int = 96

    kv_lora_rank: int = 32
    q_lora_rank: int = 64

    qk_nope_dim: int = 16
    qk_rope_dim: int = 16
    v_head_dim: int = 32

    n_routed_experts: int = 4
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_inter_dim: int = 128

    n_hc: int = 4

    n_mtp_heads: int = 1
    mtp_loss_weight: float = 0.1

    atlas_window: int = 32
    atlas_inner_lr: float = 1e-2
    atlas_retention: float = 0.99
    ttt_persistent_memory: bool = False

    muon_lr: float = 1e-2
    train_lr: float = 5e-4
    train_n_steps: int = 2000
    train_batch_size: int = 32
    train_route_w: float = 0.1
    train_grad_clip: float = 1.0

    train_mem_every: int = 4
    train_mem_batch: int = 8
    train_mem_chunk: int = 32
    train_mem_pairs: int = 8
    train_mem_secret_len: int = 2


cfg = Config()
