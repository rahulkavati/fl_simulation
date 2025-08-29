#!/usr/bin/env python3
import argparse, base64, json, os
from pathlib import Path
import tenseal as ts

def load_full_context(path: str = "Huzaif/keys/secret.ctx") -> ts.Context:
    with open(path, "rb") as f:
        return ts.context_from(f.read())

def encrypt_payload(ctx: ts.Context, payload: dict, ctx_ref: str):
    # pack weights + bias into ONE ciphertext
    w = list(map(float, payload["weight_delta"]))
    b = float(payload["bias_delta"])
    vec = w + [b]

    ct = ts.ckks_vector(ctx, vec).serialize()
    return {
        "client_id": payload["client_id"],
        "round_id": int(payload["round_id"]),
        "num_samples": int(payload["num_samples"]),
        "layout": {"weights": len(w), "bias": 1},
        "ciphertext": base64.b64encode(ct).decode(),
        "ctx_ref": ctx_ref  # version tag like "v1"
    }

def process_file(ctx, in_path: Path, out_path: Path, ctx_ref: str):
    with in_path.open("r") as f:
        payload = json.load(f)
    enc = encrypt_payload(ctx, payload, ctx_ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(enc, f, indent=2)

def main():
    p = argparse.ArgumentParser("Encrypt FL update(s) with TenSEAL CKKS")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="inp", help="single input JSON file")
    g.add_argument("--in-dir", dest="in_dir", help="folder with JSON files")
    p.add_argument("--out", dest="outp", help="single output JSON file (when using --in)")
    p.add_argument("--out-dir", dest="out_dir", help="output folder (when using --in-dir)")
    p.add_argument("--ctx", default="keys/secret.ctx", help="path to TenSEAL context (.ctx)")
    p.add_argument("--ctx-ref", default="v1", help="context version tag to embed")
    args = p.parse_args()

    ctx = load_full_context(args.ctx)

    # Single file mode
    if args.inp:
        if not args.outp:
            raise SystemExit("Error: --out is required when using --in")
        process_file(ctx, Path(args.inp), Path(args.outp), args.ctx_ref)
        print(f"[ok] wrote {args.outp}")
        return

    # Directory mode
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir.with_name(in_dir.name + "_enc")
    if not in_dir.is_dir():
        raise SystemExit(f"Error: {in_dir} is not a directory")

    count = 0
    for in_path in sorted(in_dir.rglob("*.json")):
        rel = in_path.relative_to(in_dir)
        out_path = out_dir / rel
        # keep same filename; optionally append suffix:
        # out_path = out_path.with_suffix(".enc.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process_file(ctx, in_path, out_path, args.ctx_ref)
            print(f"[ok] {in_path} -> {out_path}")
            count += 1
        except Exception as e:
            print(f"[skip] {in_path}: {e}")

    print(f"[done] encrypted {count} files into {out_dir}")

if __name__ == "__main__":
    main()
