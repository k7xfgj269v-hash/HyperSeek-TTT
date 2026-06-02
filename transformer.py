import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

class Config:
    vocab_size = 13      # 字符表大小：0-9 + '+' '=' '$'(结束符)
    d_model    = 64      # 每个 token 的向量维度
    n_heads    = 4       # 多头数量（d_model 必须能整除 n_heads）
    n_layers   = 2       # 堆几层 Block
    d_ff      = 256     # 前馈网络的隐藏层维度
    seq_len    = 20      # 输入序列的最大长度
    dropout    = 0.1     # Dropout 概率

cfg = Config()
def scaled_dot_product_attention(Q, K, V, mask=None):
      d_k = Q.size(-1)
      scores = Q @ K.transpose(-2,-1)
      scores = scores / math.sqrt(d_k)
      if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
      attn = F.softmax(scores, dim=-1)
      out = attn @ V
      return out
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, n_heads, dropout=0.1):
       super().__init__()
       assert d_model % n_heads == 0 
       self.n_heads=n_heads
       self.d_k = d_model // n_heads
       self.W_Q = nn.Linear(d_model, d_model)
       self.W_K = nn.Linear(d_model, d_model)
       self.W_V = nn.Linear(d_model, d_model)
       self.W_O = nn.Linear(d_model, d_model)
       self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T , _ = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1,2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1,2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1,2)
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(B, T, -1)
        out = self.W_O(out)
        return self.dropout(out)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x),mask)
        x = x + self.ffn(self.ln2(x))
        return x
class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
        self.register_buffer('causal_mask',mask)
    def forward(self, idx):
        B, T =idx.shape
        tok= self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x,mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
if __name__ =="__main__":
    x = torch.randn(2, 10, cfg.d_model)
    mha = MultiHeadAttention(cfg.d_model, cfg.n_heads)
    out = mha(x)
    print("MHA output shape:", out.shape)
    ffn = FeedForward(cfg.d_model, cfg.d_ff)
    out = ffn(x)
    print("FFN output shape:", out.shape)
    print("---MiniGpT überprüfen---")
    model = MiniGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    print("Input shape:", idx.shape)
    logits = model(idx)
    print("Logits shape:", logits.shape)