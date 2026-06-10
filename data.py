import random
import torch


TABLE = {**{str(i): i for i in range(10)}, '+': 10, '-': 11, '*': 12, '/': 13, '=': 14, '$': 15, ',': 16, '<pad>': 17}
PAD = TABLE['<pad>']
INV = {v: k for k, v in TABLE.items()}
OPS = ['+', '-', '*', '/']
OP_DIST = [0, 0, 0, 1, 2, 3]


def gen_sample():
    op = random.choice(OP_DIST)
    if op == 0:
        a, b = random.randint(0, 99), random.randint(0, 99)
        c = a + b
        oa, ta = a % 10, a // 10
        ob, tb = b % 10, b // 10
        ones = oa + ob
        carry = 1 if ones >= 10 else 0
        tens = carry + ta + tb
        s = f"{a}+{b}={oa}+{ob}={ones},{carry}+{ta}+{tb}={tens},{c}$"
    elif op == 1:
        a, b = random.randint(0, 99), random.randint(0, 99)
        if a < b:
            a, b = b, a
        c = a - b
        oa, ta = a % 10, a // 10
        ob, tb = b % 10, b // 10
        if oa >= ob:
            borrow = 0
            ones = oa - ob
            ones_part = f"{oa}-{ob}={ones}"
        else:
            borrow = 1
            ones = 10 + oa - ob
            ones_part = f"10+{oa}-{ob}={ones}"
        tens = ta - tb - borrow
        s = f"{a}-{b}={ones_part},{ta}-{tb}-{borrow}={tens},{c}$"
    elif op == 2:
        a, b = random.randint(0, 9), random.randint(0, 9)
        c = a * b
        s = f"{a}*{b}={c}$"
    else:
        b = random.randint(1, 9)
        c = random.randint(0, 9)
        a = b * c
        s = f"{a}/{b}={c}$"
    return s, op


def encode(s):
    return [TABLE[c] for c in s]


def decode(ids):
    return ''.join(INV.get(i, '?') for i in ids)


def make_batch(batch_size, max_len, device='cpu'):
    samples = [gen_sample() for _ in range(batch_size)]
    encoded = [encode(s) for s, _ in samples]
    ops = [op for _, op in samples]
    padded = []
    for e in encoded:
        e = e[:max_len] if len(e) > max_len else e + [PAD] * (max_len - len(e))
        padded.append(e)
    return torch.tensor(padded, dtype=torch.long, device=device), torch.tensor(ops, dtype=torch.long, device=device)


def split_chunks(text, chunk_len):
    ids = encode(text)
    return [ids[i:i + chunk_len] for i in range(0, len(ids), chunk_len)]


DIGITS = "0123456789"


def random_digits(n):
    return "".join(random.choice(DIGITS) for _ in range(n))


def filler_item():
    a = random.choice(DIGITS)
    b = random.choice(DIGITS)
    c = str((int(a) + int(b)) % 10)
    return f"{a}+{b}={c}"


def secret_episode(secret_len=4, n_fillers=40):
    secret = random_digits(secret_len)
    fillers = [filler_item() for _ in range(n_fillers)]
    pos = random.randrange(len(fillers) + 1)
    fillers.insert(pos, f"9={secret}")
    context = ",".join(fillers) + ","
    query = "9="
    return context, query, secret


def kv_recall_episode(n_pairs=8, secret_len=1):
    n_pairs = min(n_pairs, len(DIGITS))
    keys = random.sample(DIGITS, n_pairs)
    vals = [random_digits(secret_len) for _ in keys]
    pairs = list(zip(keys, vals))
    random.shuffle(pairs)
    q_key, q_val = random.choice(pairs)
    context = ",".join(f"{k}={v}" for k, v in pairs) + ","
    query = f"{q_key}="
    return context, query, q_val


def needle_episode(n_fillers=20, secret_len=1):
    secret = random_digits(secret_len)
    fillers = [f"{random.choice(DIGITS)}+{random.choice(DIGITS)}={random.choice(DIGITS)}" for _ in range(n_fillers)]
    pos = random.randrange(len(fillers) + 1)
    fillers.insert(pos, f"9={secret}")
    context = ",".join(fillers) + ","
    query = "9="
    return context, query, secret


def rule_episode(n_rules=6, secret_len=1):
    lhs = random.sample(DIGITS, n_rules)
    rhs = [random_digits(secret_len) for _ in lhs]
    rules = list(zip(lhs, rhs))
    random.shuffle(rules)
    q_key, q_val = random.choice(rules)
    context = ",".join(f"{k}={v}" for k, v in rules) + ","
    query = f"{q_key}="
    return context, query, q_val


def make_recall_batch(batch_size, n_pairs=8, secret_len=2, device='cpu'):
    """Fixed-length kv-recall episodes so chunk boundaries align across the
    batch: every sample is n_pairs * (3 + secret_len) context tokens, then
    "k=" + secret. Returns (tokens, ctx_len)."""
    n_pairs = min(n_pairs, len(DIGITS))
    rows = []
    for _ in range(batch_size):
        context, query, secret = kv_recall_episode(n_pairs, secret_len)
        rows.append(encode(context + query + secret))
    ctx_len = n_pairs * (3 + secret_len)
    return torch.tensor(rows, dtype=torch.long, device=device), ctx_len
