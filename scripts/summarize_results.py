from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "sgd",
    "svr",
    "cnn",
    "cnn_lstm",
    "bilstm",
    "bilstm_feature_attn",
    "bilstm_temporal_attn",
    "bilstm_attn",
]

MODEL_NAMES = {
    "sgd": "SGD",
    "svr": "SVR",
    "cnn": "1D CNN",
    "cnn_lstm": "CNN-LSTM",
    "bilstm": "BiLSTM",
    "bilstm_feature_attn": "BiLSTM + Feature Attention",
    "bilstm_temporal_attn": "BiLSTM + Temporal Attention",
    "bilstm_attn": "BiLSTM-Attn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize saved C-MAPSS replication runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def load_run_records(results_dir: Path, include_smoke: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metrics_path in sorted((results_dir / "runs").glob("*/metrics.json")):
        run_name = metrics_path.parent.name.lower()
        if not include_smoke and ("smoke" in run_name or run_name.startswith("test")):
            continue
        with metrics_path.open(encoding="utf-8") as f:
            row = json.load(f)
        row["run_dir"] = str(metrics_path.parent.resolve())
        records.append(row)
    if not records:
        raise SystemExit(f"No run metrics found under {results_dir / 'runs'}")
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    df = load_run_records(results_dir, args.include_smoke)
    order_map = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    df["model_order"] = df["model"].map(order_map).fillna(999).astype(int)
    df = df.sort_values(["fd", "model_order", "run_id"]).drop_duplicates(
        subset=["fd", "model"],
        keep="last",
    )
    df["paper_model_name"] = df["model"].map(MODEL_NAMES)

    metric_cols = [
        "run_id",
        "fd",
        "model",
        "window_size",
        "clip_rul",
        "include_settings",
        "seed",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "patience",
        "val_fraction",
        "hidden_size",
        "dropout",
        "rmse",
        "nasa_score",
        "r2",
        "rrmse",
        "kge",
        "paper_rmse",
        "rmse_delta_vs_paper",
        "run_dir",
    ]
    comparison_cols = [
        "fd",
        "paper_model_name",
        "model",
        "epochs",
        "batch_size",
        "rmse",
        "paper_rmse",
        "rmse_delta_vs_paper",
        "r2",
        "rrmse",
        "kge",
        "nasa_score",
        "run_id",
        "run_dir",
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    df[metric_cols].to_csv(results_dir / "metrics_clean_updated.csv", index=False)
    comparison = df[comparison_cols].sort_values(["fd", "model"])
    comparison.to_csv(results_dir / "comparison_full_updated.csv", index=False)

    fd001 = comparison[comparison["fd"] == "FD001"]
    if not fd001.empty:
        fd001.to_csv(results_dir / "comparison_fd001_updated.csv", index=False)
        leaderboard = df[df["fd"] == "FD001"][
            [
                "paper_model_name",
                "model",
                "rmse",
                "nasa_score",
                "r2",
                "rrmse",
                "kge",
                "paper_rmse",
                "rmse_delta_vs_paper",
            ]
        ].sort_values("rmse")
        leaderboard.to_csv(results_dir / "leaderboard_fd001.csv", index=False)

    pivot = df.pivot(index="model", columns="fd", values="rmse").reindex(MODEL_ORDER).dropna(how="all")
    pivot.to_csv(results_dir / "rmse_pivot_full_updated.csv")

    print(f"Wrote summaries to {results_dir.resolve()}")
    print(comparison[["fd", "model", "rmse", "paper_rmse", "rmse_delta_vs_paper"]].to_string(index=False))


if __name__ == "__main__":
    main()
