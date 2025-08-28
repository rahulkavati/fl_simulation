#!/usr/bin/env python3
"""
Smart Switch Aggregator — TenSEAL CKKS (Encrypted FedAvg)
---------------------------------------------------------

Loads params-only TenSEAL context (no SK), reads encrypted client JSONs,
validates round/layout/ctx_ref consistency, performs weighted FedAvg in
the encrypted domain, and writes one aggregated ciphertext JSON.

Usage:
  python smart_switch_tenseal.py \
    --fedl_dir updates/encrypted \
    --ctx_b64 Huzaif/keys/params.ctx.b64 \
    --out_dir Sriven/outbox

"""

import argparse
import base64
import json
import os
import sys
import time
from typing import List, Dict

try:
    import tenseal as ts
except Exception as e:
    print("[FATAL] TenSEAL not installed. Run: pip install tenseal")
    print("        Import error:", e)
    sys.exit(1)

def load_ctx_from_b64(path: str) -> ts.Context:
    if not os.path.exists(path):
        raise FileNotFoundError(f"params.ctx.b64 not found: {path}")
    with open(path, "rb") as f:
        b64 = f.read()
    raw = base64.b64decode(b64)
    ctx = ts.context_from(raw)
    if ctx.has_secret_key():
        ctx.clear_secret_key()
    return ctx

def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s)

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def read_jsons(fedl_dir: str) -> List[dict]:
    out = []
    if os.path.isfile(fedl_dir):
        with open(fedl_dir, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    else:
        for fn in sorted(os.listdir(fedl_dir)):
            if fn.lower().endswith(".json"):
                with open(os.path.join(fedl_dir, fn), "r", encoding="utf-8") as f:
                    out.append(json.load(f))
    return out

def validate_and_prepare(ctx: ts.Context, clients: List[dict]):
    if not clients:
        raise RuntimeError("No client JSONs found")

    round_ids = {c.get("round_id") for c in clients}
    if len(round_ids) != 1:
        raise RuntimeError(f"Inconsistent round_id: {round_ids}")
    round_id = round_ids.pop()

    layouts = {json.dumps(c.get("layout", {}), sort_keys=True) for c in clients}
    if len(layouts) != 1:
        raise RuntimeError("Inconsistent layout across clients")
    layout = clients[0].get("layout")

    ctx_refs = {c.get("ctx_ref") for c in clients}
    if len(ctx_refs) != 1:
        raise RuntimeError("Inconsistent ctx_ref across clients")
    ctx_ref = ctx_refs.pop()

    prepared = []
    for c in clients:
        try:
            ct = ts.ckks_vector_from(ctx, b64_to_bytes(c["ciphertext"]))
            prepared.append((ct, int(c.get("num_samples", 0))))
        except Exception:
            continue

    if not prepared:
        raise RuntimeError("No valid ciphertexts parsed from clients")

    return round_id, layout, ctx_ref, prepared

def aggregate(prepared, ctx):
    # prepared: List[Tuple[ts.CKKSVector, int]]
    total_samples = sum(n for _, n in prepared if n > 0)
    if total_samples <= 0:
        raise RuntimeError("total_samples must be > 0")

    acc = None
    inv_total = 1.0 / float(total_samples)

    for ct, n in prepared:
        if n <= 0:
            continue
        # weight *inside* each vector before adding to avoid large-scale on the accumulator
        w = float(n) * inv_total  # n / total_samples
        v = ct * w                # TenSEAL: scalar multiply returns a new vector
        acc = v if acc is None else (acc + v)

    if acc is None:
        raise RuntimeError("No positive-sample client vectors found")
    return acc


def write_output(out_dir: str, round_id: int, layout: dict, ctx_ref: str, agg_ct: ts.CKKSVector):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"agg_round_{round_id}.json")
    out = {
        "round_id": round_id,
        "layout": layout,
        "ciphertext": bytes_to_b64(agg_ct.serialize()),
        "ctx_ref": ctx_ref,
        "created_at": int(time.time())
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[Switch] Wrote aggregated ciphertext: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedl_dir", required=True, help="Dir or JSON file with encrypted updates")
    ap.add_argument("--ctx_b64", required=True, help="Path to params.ctx.b64")
    ap.add_argument("--out_dir", required=True, help="Output dir")
    return ap.parse_args()

def main():
    args = parse_args()
    ctx = load_ctx_from_b64(args.ctx_b64)
    clients = read_jsons(args.fedl_dir)
    round_id, layout, ctx_ref, prepared = validate_and_prepare(ctx, clients)
    agg_ct = aggregate(prepared, ctx)
    write_output(args.out_dir, round_id, layout, ctx_ref, agg_ct)
    print(f"[Switch] Done. Round {round_id}, Clients {len(prepared)}")

if __name__ == "__main__":
    main()
