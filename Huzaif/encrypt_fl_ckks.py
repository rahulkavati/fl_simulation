import json
import base64
import time
from typing import Dict, Any, List

import tenseal as ts


fl_result = {
    "client_id": "client_0",
    "round_id": 0,
    "weight_delta": [
        -2.2558427984753036,
        0.3159346218385186,
        -11.706958452043235,
        -15.634276570961564
    ],
    "bias_delta": 268.6885793948166,
    "num_samples": 200
}


def create_ckks_context(poly_modulus_degree: int = 8192,
                        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
                        scale: float = 2**40) -> ts.Context:
    """
    Create a CKKS context suitable for inference/training deltas.
    poly_modulus_degree=8192 is a solid default for small vectors.
    coeff_mod_bit_sizes = [60, 40, 40, 60] gives ~140-bit total.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    ctx.global_scale = scale
    # Keys: public for encryption, secret for local decrypt/validation, galois for rotations if needed later.
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def encrypt_fl_json(ckks_ctx: ts.Context, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encrypt weight_delta (vector) and bias_delta (scalar as 1D vector) using CKKS.
    Non-numeric metadata (client_id, round_id, num_samples) stays in plaintext.
    Returns a JSON-safe dict where ciphertexts and context are base64-encoded.
    """
    # ----- Prepare plaintexts -----
    weights = payload["weight_delta"]
    bias = [float(payload["bias_delta"])]  # CKKS encrypts vectors; use single-element vector for scalar.

    # ----- Timing: Encryption -----
    t0 = time.perf_counter()
    enc_weights = ts.ckks_vector(ckks_ctx, weights)
    enc_bias = ts.ckks_vector(ckks_ctx, bias)
    t1 = time.perf_counter()
    encryption_time_s = t1 - t0

    # ----- Serialize public context and ciphertexts (secret key excluded) -----
    # Server typically needs: context (without secret key) + ciphertexts.
    public_ctx_bytes = ckks_ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True
    )
    weights_bytes = enc_weights.serialize()
    bias_bytes = enc_bias.serialize()

    # Base64 to make JSON-safe
    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode("utf-8")

    out = {
        "client_id": payload["client_id"],
        "round_id": payload["round_id"],
        "num_samples": payload["num_samples"],
        "ciphertexts": {
            "weight_delta_ckks": b64(weights_bytes),
            "bias_delta_ckks": b64(bias_bytes)
        },
        "ckks_context_public": b64(public_ctx_bytes),
        "bonus_time_check": {
            "encryption_time_seconds": encryption_time_s
        }
    }
    return out, enc_weights, enc_bias


def local_decrypt_check(ckks_ctx: ts.Context,
                        enc_weights: ts.CKKSVector,
                        enc_bias: ts.CKKSVector) -> Dict[str, Any]:
    """
    Optional local verification (requires secret key in the same context).
    Measures decryption time and returns recovered plaintexts.
    """
    t0 = time.perf_counter()
    dec_weights = enc_weights.decrypt()
    dec_bias = enc_bias.decrypt()
    t1 = time.perf_counter()
    decryption_time_s = t1 - t0

    return {
        "decrypted_weight_delta": dec_weights,
        "decrypted_bias_delta": dec_bias[0] if len(dec_bias) == 1 else dec_bias,
        "bonus_time_check": {
            "decryption_time_seconds": decryption_time_s
        }
    }


def main():
    # 1) Build context (with secret key kept locally)
    ctx = create_ckks_context()

    # 2) Encrypt
    encrypted_json, enc_w, enc_b = encrypt_fl_json(ctx, fl_result)

    # 3) Optional: local decrypt check (useful for tests; don't ship secret key)
    decrypt_report = local_decrypt_check(ctx, enc_w, enc_b)

    # 4) Print results
    print("\n=== Encrypted Payload (JSON-safe) ===")
    print(json.dumps(encrypted_json, indent=2))

    print("\n=== Local Decrypt Sanity Check (not to be sent to server) ===")
    print(json.dumps(decrypt_report, indent=2))

    # 5) (Optional) Save to disk
    with open("encrypted_round.json", "w") as f:
        json.dump(encrypted_json, f, indent=2)
    print("\nSaved: encrypted_round.json")

    # If you need to persist the FULL context with secret key for local testing:
    full_ctx_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=True,      # keep this file private!
        save_galois_keys=True,
        save_relin_keys=True
    )
    with open("ckks_context_full.bin", "wb") as f:
        f.write(full_ctx_bytes)
    print("Saved: ckks_context_full.bin (contains SECRET key; NEVER share)")

if __name__ == "__main__":
    main()
