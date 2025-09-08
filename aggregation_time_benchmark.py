"""
Aggregation Time Benchmark (Server-side)

Generates a single PNG where:
- X-axis: rounds (1..N)
- Y-axis: aggregation time per round (seconds)
- One line per client count

Client counts tested: 10, 50, 100, 200, 400, 800, 1600
Rounds: default 40 (configurable via --rounds)

This is a self-contained benchmark that simulates a weighted aggregation step
(e.g., FedAvg) with realistic vector sizes. It measures wall-clock time for the
aggregation function per round and plots the results.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import csv


def simulate_server_aggregation(num_clients: int, vector_dim: int, repeat_inner: int = 1) -> float:
    """
    Simulate server-side encrypted aggregation cost by summing client update vectors.
    repeat_inner allows increasing computational work per aggregation (to smooth timing).
    Returns elapsed seconds.
    """
    # Generate client updates (float vectors)
    updates = np.random.randn(num_clients, vector_dim).astype(np.float64)
    weights = np.random.randint(50, 300, size=num_clients).astype(np.float64)
    weights /= weights.sum()

    start = time.time()
    # Aggregate (weighted sum) possibly repeated to reduce timing jitter
    agg = np.zeros(vector_dim, dtype=np.float64)
    for _ in range(repeat_inner):
        # Weighted sum: sum_i w_i * update_i
        agg[:] = (updates * weights[:, None]).sum(axis=0)
        # Simulate additional server work (e.g., rescaling, noise)
        agg = np.tanh(agg)  # keeps compute bounded, avoids optimization to no-op
    elapsed = time.time() - start
    return elapsed


def run_benchmark(rounds: int, client_counts: list[int], vector_dim: int, repeat_inner: int) -> dict[int, list[float]]:
    """
    Run aggregation timing for specified client counts across rounds.
    Returns: dict of client_count -> list of per-round times.
    """
    results: dict[int, list[float]] = {}
    for c in client_counts:
        per_round = []
        for _ in range(rounds):
            t = simulate_server_aggregation(num_clients=c, vector_dim=vector_dim, repeat_inner=repeat_inner)
            per_round.append(t)
        results[c] = per_round
    return results


def plot_results(results: dict[int, list[float]], rounds: int, output_path: str):
    plt.figure(figsize=(12, 7))
    x = np.arange(1, rounds + 1)
    for c, times in results.items():
        plt.plot(x, times, marker='o', linewidth=1.5, markersize=3, label=f"{c} clients")

    plt.title("Server Aggregation Time vs Rounds (per client count)")
    plt.xlabel("Rounds")
    plt.ylabel("Aggregation time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title="Client count", loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"✅ Saved figure: {output_path}")


def save_csv(results: dict[int, list[float]], rounds: int, csv_per_round: str, csv_summary: str):
    # Per-round CSV: columns = [round, client_count, time_s]
    with open(csv_per_round, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["round", "clients", "aggregation_time_s"])
        for clients, times in results.items():
            for r_idx, t in enumerate(times, start=1):
                w.writerow([r_idx, clients, f"{t:.6f}"])

    # Summary CSV: per client count stats
    with open(csv_summary, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["clients", "mean_s", "std_s", "min_s", "p50_s", "p90_s", "max_s"])
        for clients, times in results.items():
            arr = np.array(times, dtype=float)
            w.writerow([
                clients,
                f"{arr.mean():.6f}",
                f"{arr.std(ddof=1):.6f}" if arr.size > 1 else f"{0.0:.6f}",
                f"{arr.min():.6f}",
                f"{np.median(arr):.6f}",
                f"{np.percentile(arr, 90):.6f}",
                f"{arr.max():.6f}",
            ])
    print(f"✅ Saved per-round CSV: {csv_per_round}")
    print(f"✅ Saved summary CSV: {csv_summary}")


def main():
    parser = argparse.ArgumentParser(description="Aggregation time benchmark (server-side)")
    parser.add_argument("--rounds", type=int, default=40, help="Number of rounds to plot (default: 40)")
    parser.add_argument(
        "--clients",
        type=str,
        default="10,50,100,200,400,800,1600",
        help="Comma-separated client counts (default: 10,50,100,200,400,800,1600)",
    )
    parser.add_argument("--vector-dim", type=int, default=47, help="Update vector dimension (default: 47)")
    parser.add_argument(
        "--repeat-inner",
        type=int,
        default=5,
        help="Repeat aggregation inside each round to stabilize timing (default: 5)",
    )
    parser.add_argument("--output", type=str, default="aggregation_time_scaling.png", help="Output PNG path")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV base path (will create _per_round.csv and _summary.csv)")

    args = parser.parse_args()
    client_counts = [int(x) for x in args.clients.split(",") if x.strip()]

    print("🧪 Running server aggregation benchmark...")
    print(f"  Rounds: {args.rounds}")
    print(f"  Client counts: {client_counts}")
    print(f"  Vector dim: {args.vector_dim}")
    print(f"  Repeat inner: {args.repeat_inner}")

    results = run_benchmark(
        rounds=args.rounds,
        client_counts=client_counts,
        vector_dim=args.vector_dim,
        repeat_inner=args.repeat_inner,
    )

    plot_results(results, rounds=args.rounds, output_path=args.output)
    if args.csv:
        base = args.csv
        per_round_path = f"{base}_per_round.csv"
        summary_path = f"{base}_summary.csv"
        save_csv(results, rounds=args.rounds, csv_per_round=per_round_path, csv_summary=summary_path)


if __name__ == "__main__":
    main()


