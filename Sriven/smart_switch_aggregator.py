#!/usr/bin/env python3
"""
Smart Switch Aggregator — TenSEAL CKKS (Encrypted FedAvg)

Consumes encrypted client JSONs produced by your encryptor:
{
  "client_id": "...",
  "round_id": 5,
  "num_samples": 128,
  "layout": {"weights": N, "bias": 1},
  "ciphertext": "<base64>",
  "ctx_ref": "v1"
}

Validates round/layout/ctx_ref, performs weighted FedAvg entirely in the
encrypted domain (no secret key on the switch), and writes aggregated JSON(s):

{
  "aggregator": "smart-switch",
  "round_id": 5,
  "ctx_ref": "v1",
  "layout": {"weights": N, "bias": 1},
  "total_clients": 3,
  "total_samples": 320,
  "client_ids": ["clientA","clientB","clientC"],  # optional; remove if you prefer
  "ciphertext": "<base64 aggregated ckks_vector>"
}
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional
from collections import defaultdict

try:
    import tenseal as ts
except Exception as e:
    print("[FATAL] TenSEAL not installed. Run: pip install tenseal")
    print("        Import error:", e)
    raise SystemExit(1)


# ---------- Helpers ----------

def load_params_context(ctx_path: str):
    """Load a params/public TenSEAL context from .ctx or .b64 (no secret key needed)."""
    p = Path(ctx_path)
    if not p.exists():
        raise SystemExit(f"Context not found: {ctx_path}")
    raw = p.read_bytes()
    if p.suffix == ".b64":
        try:
            raw = base64.b64decode(raw)
        except Exception as e:
            raise SystemExit(f"Failed to base64-decode context {ctx_path}: {e}")
    return ts.context_from(raw)


def load_client_update(path: Path) -> Optional[Dict[str, Any]]:
    """Read and minimally validate a client update JSON."""
    try:
        obj = json.loads(path.read_text())
        for k in ("client_id", "round_id", "num_samples", "layout", "ciphertext", "ctx_ref"):
            if k not in obj:
                raise ValueError(f"missing key '{k}'")
        obj["round_id"] = int(obj["round_id"])
        obj["num_samples"] = int(obj["num_samples"])
        return obj
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None


def vec_length_from_layout(layout: Dict[str, int]) -> int:
    """Expected ciphertext slot count from layout."""
    w = int(layout.get("weights", 0))
    b = int(layout.get("bias", 0))
    return w + b


def deserialize_ct(ctx, b64_ct: str):
    raw = base64.b64decode(b64_ct.encode())
    return ts.ckks_vector_from(ctx, raw)


# ---------- Core aggregation (updated to avoid scale issues) ----------

def aggregate_round(ctx, items: List[Dict[str, Any]], round_id: int) -> Dict[str, Any]:
    """
    Robust FedAvg in the encrypted domain.

    Key differences vs. the earlier version:
      - Two-pass approach: first compute total_samples, then sum each client
        as (ct * (num_samples / total_samples)).
      - Avoids in-place mul (e.g., ct.mul_ / *=), which can trigger CKKS
        "scale out of bounds".
      - Avoids multiplying by large then dividing by tiny scalar later.
    """
    if not items:
        raise ValueError("No client items to aggregate")

    ctx_ref = items[0]["ctx_ref"]
    layout = items[0]["layout"]
    exp_len = vec_length_from_layout(layout)

    # ---- first pass: validate & compute total_samples
    total_samples = 0
    client_ids: List[str] = []
    for obj in items:
        if obj["round_id"] != round_id:
            raise ValueError(f"Mixed rounds in batch: found {obj['round_id']} in round {round_id}")
        if obj["ctx_ref"] != ctx_ref:
            raise ValueError(f"ctx_ref mismatch: {obj['ctx_ref']} != {ctx_ref}")
        if obj["layout"] != layout:
            raise ValueError("layout mismatch across client updates")

        n = int(obj["num_samples"])
        if n <= 0:
            print(f"[warn] num_samples<=0 for client {obj.get('client_id')}, skipping")
            continue

        total_samples += n
        client_ids.append(str(obj["client_id"]))

    if total_samples == 0:
        raise ValueError("No valid updates after filtering (total_samples=0)")

    # ---- second pass: weighted sum with normalized weights (one small multiply per client)
    agg_vec = None
    used_ids: List[str] = []
    for obj in items:
        n = int(obj["num_samples"])
        if n <= 0:
            continue

        # weight in (0,1]; keeps scale moderate and stable
        weight = float(n) / float(total_samples)

        ct = deserialize_ct(ctx, obj["ciphertext"])

        # sanity-check length (TenSEAL CKKSVector lacks __len__, use size())
        try:
            ct_len = ct.size()
        except AttributeError:
            ct_len = getattr(ct, "shape", [None])[0]
        if ct_len != exp_len:
            raise ValueError(f"ciphertext length {ct_len} != expected {exp_len}")

        # Non-inplace multiply to avoid scale issues
        term = ct * weight

        if agg_vec is None:
            agg_vec = term
        else:
            # Non-inplace add keeps things simple and avoids relying on mutating internal state
            agg_vec = agg_vec + term

        used_ids.append(str(obj["client_id"]))

    if agg_vec is None:
        raise ValueError("No terms produced for aggregation")

    agg_ct_b64 = base64.b64encode(agg_vec.serialize()).decode()

    return {
        "aggregator": "smart-switch",
        "round_id": int(round_id),
        "ctx_ref": ctx_ref,
        "layout": layout,
        "total_clients": len(used_ids),
        "total_samples": int(total_samples),
        "client_ids": used_ids,  # remove if you prefer not to expose
        "ciphertext": agg_ct_b64
    }


def group_by_round(in_dir: Path, round_filter: Optional[int], ctx_ref_filter: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    buckets: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in sorted(in_dir.rglob("*.json")):
        obj = load_client_update(p)
        if obj is None:
            continue
        if round_filter is not None and obj["round_id"] != round_filter:
            continue
        if ctx_ref_filter is not None and obj["ctx_ref"] != ctx_ref_filter:
            continue
        buckets[obj["round_id"]].append(obj)
    return dict(buckets)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[ok] wrote {path}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser("Encrypted FedAvg aggregator (switch)")
    ap.add_argument("--in-dir", required=True, help="folder with encrypted client JSONs")
    ap.add_argument("--ctx", required=True, help="params/public TenSEAL context (.ctx or .ctx.b64)")
    ap.add_argument("--round", type=int, default=None, help="aggregate a specific round only")
    ap.add_argument("--ctx-ref", default=None, help="(optional) only accept this ctx_ref")
    ap.add_argument("--out", help="output file when using --round")
    ap.add_argument("--out-dir", help="output folder when aggregating multiple rounds (default: <in-dir>_agg)")
    ap.add_argument("--outfile-template", default="agg_round_{round_id}.json",
                    help="filename template for per-round outputs")
    args = ap.parse_args()

    ctx = load_params_context(args.ctx)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Error: {in_dir} is not a directory")

    out_dir = Path(args.out_dir) if args.out_dir else in_dir.with_name(in_dir.name + "_agg")

    buckets = group_by_round(in_dir, args.round, args.ctx_ref)
    if not buckets:
        if args.round is not None:
            raise SystemExit(f"No client updates found for round {args.round} (ctx_ref={args.ctx_ref}) in {in_dir}")
        raise SystemExit(f"No client updates found in {in_dir} (ctx_ref={args.ctx_ref})")

    if args.round is not None:
        items = buckets.get(args.round, [])
        if not items:
            raise SystemExit(f"No client updates for round {args.round}")
        agg = aggregate_round(ctx, items, args.round)
        out_path = Path(args.out) if args.out else (out_dir / args.outfile_template.format(round_id=args.round))
        write_json(agg, out_path)
        return

    # Multi-round mode
    for r, items in sorted(buckets.items()):
        try:
            agg = aggregate_round(ctx, items, r)
            out_path = out_dir / args.outfile_template.format(round_id=r)
            write_json(agg, out_path)
        except Exception as e:
            print(f"[skip round {r}] {e}")

    print("[done] finished aggregating")


if __name__ == "__main__":
    main()
