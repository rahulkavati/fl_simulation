# pos/fl_simulation/Huzaif/prepare_ctx.py
#!/usr/bin/env python3
import base64, os
import tenseal as ts

def create_ctx(poly=4096, chain=(40, 30, 30), scale=2**30):
    """
    Create a CKKS TenSEAL context.
    For N=4096 the total coeff modulus bits must be <= 109.
    Default chain (40,30,30) = 100 bits -> OK.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly,
        coeff_mod_bit_sizes=list(chain)
    )
    ctx.global_scale = scale
    # Do NOT generate galois/relin keys for add + scalar-mult use cases.
    return ctx

def main():
    os.makedirs("keys", exist_ok=True)
    ctx = create_ctx()  # 4096 / [40,30,30] / 2**30

    # params-only (public) — small
    params_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False
    )
    with open("keys/params.ctx.b64", "w") as f:
        f.write(base64.b64encode(params_bytes).decode())

    # full (private) — includes secret key
    full_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=True,
        save_galois_keys=False,
        save_relin_keys=False
    )
    with open("keys/secret.ctx", "wb") as f:
        f.write(full_bytes)

    print("[ok] wrote keys/params.ctx.b64 (public) and keys/secret.ctx (private)")

if __name__ == "__main__":
    main()