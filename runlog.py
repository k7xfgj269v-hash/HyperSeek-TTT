import json
import os
import time


class JsonlLogger:
    """Append-only JSONL metrics log; one file per run under logs/."""

    def __init__(self, name, root=None):
        root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(root, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(root, f'{name}_{stamp}.jsonl')
        self._f = open(self.path, 'a')

    def log(self, **kv):
        kv.setdefault('t', round(time.time(), 3))
        self._f.write(json.dumps(kv) + '\n')
        self._f.flush()

    def close(self):
        self._f.close()
