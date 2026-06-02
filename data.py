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
