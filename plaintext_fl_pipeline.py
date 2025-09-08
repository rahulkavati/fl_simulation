"""
Plain-Text Federated Learning Pipeline (no encryption)

- Loads health_fitness_data.csv
- Uses the exact 47-feature engineering used in main pipeline
- Simulates N clients, runs R rounds
- Each round: local LR training -> send plain updates -> server FedAvg -> update global -> push back to clients
- Saves results to performance_results/plaintext_fl_results_YYYYMMDD_HHMMSS.json

Usage (examples):
  python plaintext_fl_pipeline.py --rounds 10 --clients 8 --verbose
  python plaintext_fl_pipeline.py --rounds 40 --clients 20 --verbose
"""

import os
import json
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_and_engineer_features(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    fitness_threshold = df['fitness_level'].median()
    df['health_status'] = (df['fitness_level'] >= fitness_threshold).astype(int)

    # 1) Basic
    basic_features = [
        'age', 'height_cm', 'weight_kg', 'bmi',
        'avg_heart_rate', 'resting_heart_rate',
        'blood_pressure_systolic', 'blood_pressure_diastolic',
        'hours_sleep', 'stress_level', 'daily_steps',
        'calories_burned', 'hydration_level'
    ]

    # 2) Derived
    df['steps_per_calorie'] = df['daily_steps'] / (df['calories_burned'] + 1)
    df['sleep_efficiency'] = df['hours_sleep'] / (df['stress_level'] + 1)
    df['cardio_health'] = (df['resting_heart_rate'] <= 70).astype(int)
    df['high_activity'] = (df['daily_steps'] >= 10000).astype(int)
    df['good_sleep'] = (df['hours_sleep'] >= 7.0).astype(int)
    df['low_stress'] = (df['stress_level'] <= 5.0).astype(int)
    df['heart_rate_variability'] = df['avg_heart_rate'] - df['resting_heart_rate']
    df['blood_pressure_ratio'] = df['blood_pressure_systolic'] / (df['blood_pressure_diastolic'] + 1)
    df['metabolic_efficiency'] = df['calories_burned'] / (df['weight_kg'] + 1)
    df['sleep_quality_score'] = df['hours_sleep'] * (10 - df['stress_level']) / 10
    df['activity_intensity'] = df['daily_steps'] * df['calories_burned'] / 1000
    df['health_score'] = (df['fitness_level'] + df['sleep_quality_score']) / 2

    derived_features = [
        'steps_per_calorie', 'sleep_efficiency', 'cardio_health',
        'high_activity', 'good_sleep', 'low_stress',
        'heart_rate_variability', 'blood_pressure_ratio', 'metabolic_efficiency',
        'sleep_quality_score', 'activity_intensity', 'health_score',
        'age_fitness_interaction', 'sleep_stress_interaction',
        'activity_sleep_interaction', 'heart_rate_sleep_interaction'
    ]

    # 3) Interactions
    df['age_fitness_interaction'] = df['age'] * df['fitness_level']
    df['sleep_stress_interaction'] = df['hours_sleep'] * (10 - df['stress_level'])
    df['activity_sleep_interaction'] = df['daily_steps'] * df['hours_sleep']
    df['heart_rate_sleep_interaction'] = df['avg_heart_rate'] * df['hours_sleep']

    # 4) Poly (degree 2) for selected
    poly_features = ['age', 'fitness_level', 'avg_heart_rate', 'hours_sleep', 'daily_steps']
    poly_feature_names = []
    for feat in poly_features:
        df[f'{feat}_squared'] = df[feat] ** 2
        df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
        poly_feature_names.extend([f'{feat}_squared', f'{feat}_sqrt'])

    # 5) Categorical + temporal + condition encoding
    df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
    df['intensity_encoded'] = LabelEncoder().fit_transform(df['intensity'])
    df['activity_type_encoded'] = LabelEncoder().fit_transform(df['activity_type'])
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['has_health_condition'] = (df['health_condition'] != 'None').astype(int)
    df['smoking_status_encoded'] = LabelEncoder().fit_transform(df['smoking_status'])

    categorical_features = [
        'gender_encoded', 'intensity_encoded', 'activity_type_encoded',
        'day_of_week', 'month', 'is_weekend', 'has_health_condition', 'smoking_status_encoded'
    ]

    feature_columns = basic_features + derived_features + poly_feature_names + categorical_features
    return df, feature_columns


def create_clients(df: pd.DataFrame, clients: int) -> list[pd.DataFrame]:
    participants = df['participant_id'].unique()
    # Assign each client to a participant (wrap-around if fewer participants)
    out = []
    for i in range(clients):
        pid = participants[i % len(participants)]
        dfi = df[df['participant_id'] == pid].copy()
        # ensure at least 2 classes; if not, mix-in from others
        if dfi['health_status'].nunique() < 2:
            others = df[df['participant_id'] != pid]
            pos = others[others['health_status'] == 1].sample(min(100, len(others[others['health_status'] == 1])), random_state=42)
            neg = others[others['health_status'] == 0].sample(min(100, len(others[others['health_status'] == 0])), random_state=42)
            dfi = pd.concat([dfi, pos, neg]).sample(frac=1.0, random_state=42)
        out.append(dfi)
    return out


def aggregate_plain(updates: list[np.ndarray], sample_counts: list[int]) -> np.ndarray:
    weights = np.array(sample_counts, dtype=float)
    weights /= weights.sum()
    stacked = np.vstack(updates)
    return (stacked * weights[:, None]).sum(axis=0)


def run_plaintext_fl(rounds: int, clients: int, verbose: bool) -> dict:
    df, feature_cols = load_and_engineer_features("data/health_fitness_data.csv")

    # Precompute global z-score statistics for stable convergence across all clients
    feats_df = df[feature_cols].astype(float)
    col_means = feats_df.mean(axis=0).values
    col_stds = feats_df.std(axis=0).replace(0, 1.0).values

    def zscore_matrix(x_mat: np.ndarray) -> np.ndarray:
        # Broadcast-safe z-score with zero-std guard (stds already fixed to >=1)
        return (x_mat - col_means) / col_stds

    # Initialize global model from a small, balanced sample
    pos = df[df['health_status'] == 1].sample(100, random_state=42)
    neg = df[df['health_status'] == 0].sample(100, random_state=42)
    init_df = pd.concat([pos, neg]).sample(frac=1.0, random_state=42)
    X0 = init_df[feature_cols].fillna(0).astype(float).values
    X0 = zscore_matrix(X0)
    y0 = init_df['health_status'].values
    base = LogisticRegression(
        solver='liblinear',
        max_iter=5000,
        tol=1e-3,
        class_weight='balanced',
        random_state=42,
    )
    base.fit(X0, y0)
    global_weights = base.coef_[0].copy()
    global_bias = float(base.intercept_[0])

    # Create client datasets
    client_dfs = create_clients(df, clients)

    # Build a shared test set (global holdout) for evaluation
    test_pos = df[df['health_status'] == 1].sample(200, random_state=123)
    test_neg = df[df['health_status'] == 0].sample(200, random_state=123)
    test_df = pd.concat([test_pos, test_neg]).sample(frac=1.0, random_state=123)
    X_test = test_df[feature_cols].fillna(0).astype(float).values
    X_test = zscore_matrix(X_test)
    y_test = test_df['health_status'].values

    round_results = []
    t0_all = time.time()
    for r in range(1, rounds + 1):
        # Local training
        local_updates, sample_counts = [], []
        for dfi in client_dfs:
            Xi = dfi[feature_cols].fillna(0).astype(float).values
            Xi = zscore_matrix(Xi)
            yi = dfi['health_status'].values
            if len(np.unique(yi)) < 2:
                continue
            clf = LogisticRegression(
                solver='liblinear',
                max_iter=5000,
                tol=1e-3,
                class_weight='balanced',
                random_state=42,
            )
            clf.fit(Xi, yi)
            update = np.concatenate([clf.coef_[0], np.array([float(clf.intercept_[0])])])
            local_updates.append(update)
            sample_counts.append(len(Xi))

        if not local_updates:
            continue

        # Server aggregation (plain)
        t_agg0 = time.time()
        agg = aggregate_plain(local_updates, sample_counts)
        agg_time = time.time() - t_agg0
        new_weights, new_bias = agg[:-1], float(agg[-1])

        # Update global
        global_weights, global_bias = new_weights, new_bias

        # Evaluate the global model
        global_model = LogisticRegression(max_iter=1000)
        global_model.coef_ = global_weights.reshape(1, -1)
        global_model.intercept_ = np.array([global_bias])
        global_model.classes_ = np.array([0, 1])
        y_pred = global_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        round_results.append({
            "round": r,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "aggregation_time_s": agg_time,
            "clients_participated": len(local_updates)
        })
        if verbose:
            print(f"Round {r}/{rounds}  Acc={acc:.4f}  F1={f1:.4f}  Agg={agg_time:.6f}s  Clients={len(local_updates)}")

    total_time = time.time() - t0_all
    results = {
        "config": {"rounds": rounds, "clients": clients, "features": len(feature_cols)},
        "final": round_results[-1] if round_results else {},
        "round_results": round_results,
        "total_time_s": total_time
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Plain-Text FL Pipeline (no encryption)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs("performance_results", exist_ok=True)
    results = run_plaintext_fl(args.rounds, args.clients, args.verbose)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("performance_results", f"plaintext_fl_results_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results: {out_path}")


if __name__ == "__main__":
    main()


