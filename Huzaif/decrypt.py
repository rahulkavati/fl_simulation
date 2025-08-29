#!/usr/bin/env python3
import argparse, base64, json, os, sys, tenseal as ts

def load_full_context(path="keys/secret.ctx") -> ts.Context:
    with open(path, "rb") as f:
        return ts.context_from(f.read())

def decrypt_payload(ctx: ts.Context, enc: dict):
    # basic schema checks
    for k in ["ciphertext", "layout"]:
        if k not in enc:
            raise ValueError(f"missing field: {k}")
    if "weights" not in enc["layout"] or "bias" not in enc["layout"]:
        raise ValueError("layout must have 'weights' and 'bias'")

    # decode and reconstruct the vector
    ct_bytes = base64.b64decode(enc["ciphertext"])
    ct_vec = ts.ckks_vector_from(ctx, ct_bytes)

    # decrypt to Python floats
    values = ct_vec.decrypt()  # list[float]
    w_len = int(enc["layout"]["weights"])
    b_len = int(enc["layout"]["bias"])

    if b_len != 1:
        raise ValueError(f"expected bias length 1, got {b_len}")
    if len(values) < w_len + b_len:
        raise ValueError(
            f"ciphertext length {len(values)} < layout size {w_len + b_len}"
        )

    weights = values[:w_len]
    bias = values[w_len]  # single float

    # build a cleartext payload mirroring your input shape
    out = {
        "client_id": enc.get("client_id"),
        "round_id": int(enc.get("round_id", 0)),
        "num_samples": int(enc.get("num_samples", 0)),
        "weight_delta": weights,
        "bias_delta": float(bias),
        "ctx_ref": enc.get("ctx_ref"),
        "created_at": enc.get("created_at"),
    }
    return out

def main():
    p = argparse.ArgumentParser("Decrypt FL update (compact, from one CKKS ciphertext)")
    p.add_argument("--in", required=True, dest="inp", help="path to encrypted JSON")
    p.add_argument("--out", required=True, dest="outp", help="where to write clear JSON")
    p.add_argument("--ctx", default="keys/secret.ctx", help="full TenSEAL context (with secret key)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outp) or ".", exist_ok=True)
    ctx = load_full_context(args.ctx)

    # IMPORTANT: context must include secret key; otherwise decrypt() will fail.
    if not ctx.secret_key():
        print("[error] provided context has no secret key; cannot decrypt.", file=sys.stderr)
        sys.exit(1)

    with open(args.inp, "r") as f:
        enc = json.load(f)

    dec = decrypt_payload(ctx, enc)

    with open(args.outp, "w") as f:
        json.dump(dec, f, indent=2)

    print(f"[ok] wrote {args.outp}")

if __name__ == "__main__":
    main()