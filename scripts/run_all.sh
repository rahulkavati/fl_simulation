#!/usr/bin/env bash
set -e

# 1) Generate synthetic client datasets
python data/simulate_health_data.py --clients 5 --outdir data/clients

# 2) Rahul: run federated client sim (produces plaintext updates + a simple server apply)
python rahul/client_simulation.py --rounds 3 --lr 0.1 --batch_size 128

# 3) Huzaif: encrypt updates (CKKS via TenSEAL if installed; otherwise DRY-RUN base64)
python huzaif/router_encrypt.py --in_dir data/updates --out_dir data/updates_encrypted

# 4) Sriven: aggregate encrypted updates per round (homomorphic add if TenSEAL available)
python sriven/switch_aggregate.py --in_dir data/updates_encrypted --out_path data/aggregations/round_agg.enc.json

# 5) Cloud: decrypt aggregated ciphertext and update global model
python cloud/server_update.py --agg_dir data/aggregations --global_model data/global_model.json
