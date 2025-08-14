from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import json
import numpy as np

@dataclass
class ModelUpdate:
    client_id: str
    round_id: int
    weight_delta: List[float]  # flattened weights
    bias_delta: float
    num_samples: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ModelUpdate":
        d = json.loads(s)
        return ModelUpdate(**d)

def save_update_json(update: ModelUpdate, path: str):
    with open(path, "w") as f:
        f.write(update.to_json())

def save_update_npy(update: ModelUpdate, path: str):
    arr = np.array(update.weight_delta + [update.bias_delta, update.num_samples], dtype=float)
    np.save(path, arr)

def load_update_npy(path: str) -> ModelUpdate:
    arr = np.load(path, allow_pickle=False)
    *w, b, n = arr.tolist()
    return ModelUpdate(client_id="unknown", round_id=-1, weight_delta=w, bias_delta=b, num_samples=int(n))
