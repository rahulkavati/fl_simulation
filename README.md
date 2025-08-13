# Federated Learning Simulation (Rahul ↔ Huzaif ↔ Sriven)

This is a minimal, self-contained simulation of a federated learning workflow using synthetic health data.
It produces raw model updates per client (smartwatch) per round, saved as JSON and NumPy `.npy` files — ready for Huzaif's CKKS encryption and Sriven's aggregation.

## Quick start
python -m fl_sim.run_federated --n_devices 5 --rounds 3 --local_steps 2 --outdir runs/demo

