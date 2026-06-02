import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyperseek_ttt.pt')


class Config:
    vocab_size = 18

    d_model = 128
    n_heads = 4
    n_layers = 4
    seq_len = 96

    kv_lora_rank = 32
    q_lora_rank = 64

    qk_nope_dim = 16
    qk_rope_dim = 16
    v_head_dim = 32

    n_routed_experts = 4
    n_shared_experts = 1
    n_activated_experts = 2
    moe_inter_dim = 128

    n_hc = 4

    n_mtp_heads = 1
    mtp_loss_weight = 0.1

    atlas_window = 32
    atlas_inner_lr = 1e-2
    atlas_retention = 0.99
    ttt_persistent_memory = False

    muon_lr = 1e-2
    train_lr = 5e-4
    train_n_steps = 2000
    train_batch_size = 32
    train_route_w = 0.1
    train_grad_clip = 1.0


cfg = Config()


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated_ = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if x.dim() == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 3:
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)
    return x * cos_full + rotated_ * sin_full


def make_long_effective_mask(T, S, R, window_size, device):
    raise NotImplementedError(
        "make_long_effective_mask is a long-context placeholder. "
        "The old implementation had incompatible S/R/T shapes and must be "
        "redesigned before use."
    )


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute(seq_len, d_rope, base=10000.0):
    half = d_rope // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_nope_dim = cfg.qk_nope_dim
        self.qk_rope_dim = cfg.qk_rope_dim
        self.q_head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.v_head_dim = cfg.v_head_dim
        self.W_DQ = nn.Linear(cfg.d_model, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.W_UQ = nn.Linear(cfg.q_lora_rank, self.n_heads * self.q_head_dim, bias=False)
        self.W_DKV = nn.Linear(cfg.d_model, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.W_UK = nn.Linear(cfg.kv_lora_rank, self.n_heads * self.qk_nope_dim, bias=False)
        self.W_UV = nn.Linear(cfg.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)
        self.W_KR = nn.Linear(cfg.d_model, self.qk_rope_dim, bias=False)
        self.W_O = nn.Linear(self.n_heads * self.v_head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, msk=None):
        B, T, _ = x.shape
        H = self.n_heads
        q = self.W_UQ(self.q_norm(self.W_DQ(x)))
        q = q.view(B, T, H, self.q_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = q.transpose(1, 2)
        c_kv = self.kv_norm(self.W_DKV(x))
        k_nope = self.W_UK(c_kv).view(B, T, H, self.qk_nope_dim)
        v = self.W_UV(c_kv).view(B, T, H, self.v_head_dim)
        k_rope = self.W_KR(x)
        k_rope = apply_rope(k_rope, cos, sin)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, H, -1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.q_head_dim)
        if msk is not None:
            scores = scores.masked_fill(msk == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H * self.v_head_dim)
        return self.W_O(out)


class Expert(nn.Module):
    def __init__(self, d_model, d_inter):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_inter, bias=False)
        self.W_up = nn.Linear(d_model, d_inter, bias=False)
        self.W_down = nn.Linear(d_inter, d_model, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.n_shared = cfg.n_shared_experts
        self.top_k = cfg.n_activated_experts
        self.routed = nn.ModuleList([Expert(cfg.d_model, cfg.moe_inter_dim) for _ in range(self.n_routed)])
        self.shared = nn.ModuleList([Expert(cfg.d_model, cfg.moe_inter_dim) for _ in range(self.n_shared)])
        self.router = nn.Linear(cfg.d_model, self.n_routed, bias=False)
        self.register_buffer('bias', torch.zeros(self.n_routed))
        self.register_buffer('last_load', torch.zeros(self.n_routed))
        self.last_router_logits = None
        self.is_mtp = False

    def forward(self, x):
        B, T, D = x.shape
        shared_out = torch.zeros_like(x)
        for exp in self.shared:
            shared_out = shared_out + exp(x)
        router_logits = self.router(x)
        self.last_router_logits = router_logits
        scores = torch.sigmoid(router_logits)
        topk_idx = (scores + self.bias).topk(self.top_k, dim=-1).indices
        topk_w = scores.gather(-1, topk_idx)
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)
        routed_out = torch.zeros_like(x)
        for i, expert in enumerate(self.routed):
            sel_i = (topk_idx == i)
            if not sel_i.any():
                continue
            weight_i = (sel_i.float() * topk_w).sum(-1, keepdim=True)
            routed_out = routed_out + expert(x) * weight_i
        with torch.no_grad():
            load = torch.stack([(topk_idx == i).float().sum() for i in range(self.n_routed)])
            self.last_load.copy_(load)
        return shared_out + routed_out

    @torch.no_grad()
    def update_router_bias(self, gamma):
        mean_load = self.last_load.mean()
        self.bias -= gamma * (self.last_load - mean_load).sign()


class HyperConnect(nn.Module):
    """mHC-lite: Birkhoff-von Neumann decomposition.

    A = sum_k softmax(lambda)_k * P_k where P_k are all n_hc! permutation
    matrices. Convex combination of doubly stochastic matrices is itself
    exactly doubly stochastic. Avoids 5-step Sinkhorn iteration entirely.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_hc = cfg.n_hc
        import math
        import itertools
        n_perms = math.factorial(self.n_hc)
        self.n_perms = n_perms
        perms = []
        for sigma in itertools.permutations(range(self.n_hc)):
            perms.append(torch.eye(self.n_hc)[list(sigma)])
        self.register_buffer('perms', torch.stack(perms))
        lam = torch.zeros(n_perms)
        lam[0] = 4.0
        self.lambda_raw = nn.Parameter(lam)
        self.alpha = nn.Parameter(torch.full((self.n_hc,), 1.0 / self.n_hc))
        self.beta = nn.Parameter(torch.full((self.n_hc,), 1.0 / self.n_hc))
        self.last_alpha = None
        self.last_beta = None
        self.last_lambda = None
        self.last_amp = None
        self.last_replica_norms = None

    def get_A(self):
        w = F.softmax(self.lambda_raw, dim=0)
        return (w.view(-1, 1, 1) * self.perms).sum(dim=0)

    def combine(self, X):
        a = F.softmax(self.alpha, dim=-1)
        with torch.no_grad():
            self.last_alpha = a.detach().cpu()
        return torch.einsum('i,btid->btd', a, X)

    def forward(self, X, y):
        A = self.get_A()
        b = F.softmax(self.beta, dim=-1)
        X_mixed = torch.einsum('ij,btjd->btid', A, X)
        out = X_mixed + b.view(1, 1, -1, 1) * y.unsqueeze(-2)
        with torch.no_grad():
            in_norm = X.norm()
            out_norm = out.norm()
            self.last_amp = (out_norm / (in_norm + 1e-9)).item()
            self.last_beta = b.detach().cpu()
            self.last_lambda = F.softmax(self.lambda_raw, dim=0).detach().cpu()
            self.last_replica_norms = torch.linalg.vector_norm(out, dim=(0, 1, 3)).detach().cpu()
        return out


class DeepSeekBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attn = MLA(cfg)
        self.ffn = MoE(cfg)
        self.hc_attn = HyperConnect(cfg)
        self.hc_ffn = HyperConnect(cfg)

    def forward(self, X, cos, sin, msk=None):
        x = self.hc_attn.combine(X)
        y = self.attn(self.attn_norm(x), cos, sin, msk)
        X = self.hc_attn(X, y)
        x = self.hc_ffn.combine(X)
        y = self.ffn(self.ffn_norm(x))
        X = self.hc_ffn(X, y)
        return X


class MTPHead(nn.Module):
    def __init__(self, cfg, token_emb):
        super().__init__()
        self.n_hc = cfg.n_hc
        self.token_emb = token_emb
        self.norm_h = RMSNorm(cfg.d_model)
        self.norm_e = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
        self.block = DeepSeekBlock(cfg)
        self.block.ffn.is_mtp = True
        self.final_norm = RMSNorm(cfg.d_model)

    def forward(self, h, next_tok_ids, cos, sin, mask=None):
        e = self.token_emb(next_tok_ids)
        combined = torch.cat([self.norm_h(h), self.norm_e(e)], dim=-1)
        h2 = self.proj(combined)
        H = h2.unsqueeze(-2).expand(-1, -1, self.n_hc, -1).contiguous()
        H = self.block(H, cos, sin, mask)
        h2 = H.mean(-2)
        h2 = self.final_norm(h2)
        return h2


class InPlaceTTT(nn.Module):
    """In-Place Test-Time Training adapter (arXiv 2604.06169).

    A single Linear that is fast-updated during forward using a
    hidden-space NTP-like loss on the most recent window. Simpler
    than AtlasMemory (1 Parameter vs 4 sub-modules, ~80% fewer params).
    """

    def __init__(
        self,
        d_model,
        window=32,
        inner_lr=1e-2,
        retention=0.99,
        persistent_memory=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.inner_lr = inner_lr
        self.retention = retention
        self.persistent_memory = persistent_memory
        self.memory_enabled = persistent_memory
        self.write_enabled = persistent_memory

        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.zeros_(self.W)
        self.register_buffer('W_mem', torch.zeros(d_model, d_model), persistent=False)

        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

        self.last_gate = None
        self.last_L = None
        self.last_delta_norm = None
        self.last_h_norm = None
        self.last_update_norm = None
        self.last_mem_norm = None

    @torch.no_grad()
    def reset_memory(self):
        self.W_mem.zero_()

    @torch.no_grad()
    def detach_memory(self):
        self.W_mem.detach_()

    def set_memory_enabled(self, enabled=True, write_enabled=None):
        self.memory_enabled = enabled
        self.write_enabled = enabled if write_enabled is None else write_enabled

    def _state_W(self):
        if self.persistent_memory and self.memory_enabled:
            return self.W + self.W_mem
        return self.W

    def forward(self, h):
        B, T, D = h.shape
        w = min(self.window, T - 1)
        W_state = self._state_W()
        if w < 1:
            gate_val = torch.sigmoid(self.gate(h))
            delta = F.linear(h, W_state)
            out = h + gate_val * delta
            with torch.no_grad():
                self.last_gate = gate_val.mean().item()
                self.last_L = 0.0
                self.last_delta_norm = (gate_val * delta).norm().item()
                self.last_h_norm = h.norm().item()
                self.last_update_norm = 0.0
                self.last_mem_norm = self.W_mem.norm().item()
            return out

        with torch.enable_grad():
            W = W_state.detach().requires_grad_(True)
            h_curr = h[:, -w - 1:-1, :].reshape(-1, D).detach()
            h_next = h[:, -w:, :].reshape(-1, D).detach()
            pred = F.linear(h_curr, W)
            target = h_next - h_curr
            L = ((pred - target) ** 2).sum(-1).mean()
            grads = torch.autograd.grad(L, [W], create_graph=False)

        g_W = grads[0]
        norm = g_W.norm() + 1e-12
        scale = (1.0 / norm).clamp(max=1.0)
        g_W = g_W * scale

        if self.persistent_memory and self.memory_enabled:
            W_mem_t = self.retention * self.W_mem - self.inner_lr * g_W
            W_t = self.W + W_mem_t
            if self.write_enabled:
                with torch.no_grad():
                    self.W_mem.copy_(W_mem_t.detach())
        else:
            W_t = self.retention * self.W - self.inner_lr * g_W
        delta = F.linear(h, W_t)
        gate_val = torch.sigmoid(self.gate(h))

        with torch.no_grad():
            self.last_gate = gate_val.mean().item()
            self.last_L = L.detach().item()
            self.last_delta_norm = (gate_val * delta).norm().item()
            self.last_h_norm = h.norm().item()
            self.last_update_norm = (self.inner_lr * g_W).norm().item()
            self.last_mem_norm = self.W_mem.norm().item()

        return h + gate_val * delta


class DeepSeekMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        cos, sin = precompute(cfg.seq_len, cfg.qk_rope_dim)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        m = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
        self.register_buffer('causal_mask', m, persistent=False)
        self.blocks = nn.ModuleList([DeepSeekBlock(cfg) for _ in range(cfg.n_layers)])
        self.atlas = InPlaceTTT(
            d_model=cfg.d_model,
            window=cfg.atlas_window,
            inner_lr=cfg.atlas_inner_lr,
            retention=cfg.atlas_retention,
            persistent_memory=getattr(cfg, 'ttt_persistent_memory', False),
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.mtp = nn.ModuleList([MTPHead(cfg, self.token_emb) for _ in range(cfg.n_mtp_heads)])

    def reset_memory(self):
        if hasattr(self.atlas, 'reset_memory'):
            self.atlas.reset_memory()

    def detach_memory(self):
        if hasattr(self.atlas, 'detach_memory'):
            self.atlas.detach_memory()

    def set_memory_enabled(self, enabled=True, write_enabled=None):
        if hasattr(self.atlas, 'set_memory_enabled'):
            self.atlas.set_memory_enabled(enabled, write_enabled)

    def backbone_hidden(self, idx):
        B, T = idx.shape
        max_T = self.rope_cos.size(0)
        if T > max_T:
            raise ValueError(f"input length {T} exceeds model context length {max_T}")
        x = self.token_emb(idx)
        X = x.unsqueeze(-2).expand(-1, -1, self.cfg.n_hc, -1).contiguous()
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            X = block(X, cos, sin, mask)
        return X.mean(-2)

    def logits_from_hidden(self, x):
        h = self.final_norm(x)
        return self.head(h)

    def forward(self, idx, return_mtp=False, mtp_tokens=None):
        T = idx.shape[1]
        x = self.backbone_hidden(idx)
        x = self.atlas(x)
        logits = self.logits_from_hidden(x)
        if not return_mtp or len(self.mtp) == 0:
            return logits, None
        if mtp_tokens is None:
            raise ValueError("return_mtp=True requires mtp_tokens (next-token ids of same length as idx)")
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        mask = self.causal_mask[:T, :T]
        h = self.final_norm(x)
        mtp_hidden = []
        for head in self.mtp:
            h2 = head(h, mtp_tokens, cos, sin, mask)
            mtp_hidden.append(self.head(h2))
        return logits, mtp_hidden
