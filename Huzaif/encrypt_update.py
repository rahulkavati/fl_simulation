#!/usr/bin/env python3
import argparse, base64, json, os, tenseal as ts

def load_full_context(path="keys/secret.ctx") -> ts.Context:
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
        "ctx_ref": ctx_ref   # just a version tag like "v1"
    }

def main():
    p = argparse.ArgumentParser("Encrypt FL update (compact, no keys)")
    p.add_argument("--in", required=True, dest="inp")
    p.add_argument("--out", required=True, dest="outp")
    p.add_argument("--ctx", default="keys/secret.ctx")
    p.add_argument("--ctx-ref", default="v1")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outp), exist_ok=True)
    ctx = load_full_context(args.ctx)

    with open(args.inp, "r") as f:
        payload = json.load(f)

    enc = encrypt_payload(ctx, payload, args.ctx_ref)

    with open(args.outp, "w") as f:
        json.dump(enc, f, indent=2)

    print(f"[ok] wrote {args.outp}")

if __name__ == "__main__":
    main()