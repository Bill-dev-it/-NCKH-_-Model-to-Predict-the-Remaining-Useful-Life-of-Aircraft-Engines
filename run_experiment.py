from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from rul_replication.baselines import train_predict_sklearn
from rul_replication.constants import DEFAULT_DATA_DIR, PAPER_RMSE
from rul_replication.data import load_cmapss
from rul_replication.metrics import regression_metrics


SKLEARN_MODELS = {"sgd", "svr"}
TORCH_MODELS = {
    "cnn",
    "cnn_lstm",
    "bilstm",
    "bilstm_feature_attn",
    "bilstm_temporal_attn",
    "bilstm_attn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C-MAPSS RUL replication experiment.")
    parser.add_argument("--fd", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--model", required=True, choices=sorted(SKLEARN_MODELS | TORCH_MODELS))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--clip-rul", type=int, default=125, help="Use -1 to disable clipping.")
    parser.add_argument("--include-settings", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None, help="Optional name for the run artifact folder.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save torch checkpoint.")
    return parser.parse_args()


def rewrite_metrics_with_new_schema(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for old_row in reader:
                for name in fieldnames:
                    old_row.setdefault(name, "")
                rows.append(old_row)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_rewrite_metrics(path: Path, row: dict[str, object]) -> bool:
    try:
        rewrite_metrics_with_new_schema(path, row)
        return True
    except PermissionError:
        print(f"WARNING: Could not write {path}. Close the file if it is open in Excel/Jupyter.")
        return False


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_name:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.run_name)
        return cleaned.strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{args.fd}_{args.model}_seed{args.seed}"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    run_id = make_run_id(args)
    clip_rul = None if args.clip_rul < 0 else args.clip_rul
    data = load_cmapss(
        fd=args.fd,
        data_dir=args.data_dir,
        window_size=args.window_size,
        clip_rul=clip_rul,
        include_settings=args.include_settings,
    )

    if args.model in SKLEARN_MODELS:
        pred = train_predict_sklearn(args.model, data.train_windows, data.train_labels, data.test_windows)
        model = None
    else:
        from rul_replication.models import build_torch_model
        from rul_replication.training import TrainConfig, predict_torch, train_torch_regressor
        import torch

        model = build_torch_model(
            args.model,
            n_features=data.train_windows.shape[-1],
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
        model = train_torch_regressor(model, data.train_windows, data.train_labels, config)
        pred = predict_torch(model, data.test_windows, device=args.device)

    pred = np.maximum(pred, 0.0)
    metrics = regression_metrics(data.test_labels, pred)
    paper_rmse = PAPER_RMSE.get(args.model, {}).get(args.fd)
    row: dict[str, object] = {
        "run_id": run_id,
        "fd": args.fd,
        "model": args.model,
        "window_size": args.window_size,
        "clip_rul": clip_rul if clip_rul is not None else "none",
        "include_settings": args.include_settings,
        "seed": args.seed,
        "epochs": args.epochs if args.model in TORCH_MODELS else "",
        "batch_size": args.batch_size if args.model in TORCH_MODELS else "",
        "lr": args.lr if args.model in TORCH_MODELS else "",
        "weight_decay": args.weight_decay if args.model in TORCH_MODELS else "",
        "patience": args.patience if args.model in TORCH_MODELS else "",
        "val_fraction": args.val_fraction if args.model in TORCH_MODELS else "",
        "hidden_size": args.hidden_size if args.model in TORCH_MODELS else "",
        "dropout": args.dropout if args.model in TORCH_MODELS else "",
        **metrics,
        "paper_rmse": paper_rmse if paper_rmse is not None else "",
        "rmse_delta_vs_paper": metrics["rmse"] - paper_rmse if paper_rmse is not None else "",
    }
    pred_df = pd.DataFrame({"unit": data.test_units, "y_true": data.test_labels, "y_pred": pred})

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", row)
    config_payload = {
        "fd": args.fd,
        "model": args.model,
        "data_dir": str(args.data_dir),
        "window_size": args.window_size,
        "clip_rul": clip_rul if clip_rul is not None else "none",
        "include_settings": args.include_settings,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "device": args.device,
        "feature_names": data.feature_names,
    }
    write_json(run_dir / "config.json", config_payload)

    if args.model in TORCH_MODELS and model is not None:
        history = getattr(model, "training_history_", [])
        if history:
            pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        if not args.no_save_model:
            checkpoint = {
                "model_name": args.model,
                "fd": args.fd,
                "state_dict": model.cpu().state_dict(),
                "config": config_payload,
                "metrics": row,
            }
            torch.save(checkpoint, run_dir / "checkpoint.pt")

    safe_rewrite_metrics(args.results_dir / "metrics.csv", row)
    try:
        pred_df.to_csv(args.results_dir / f"predictions_{args.fd}_{args.model}.csv", index=False)
    except PermissionError:
        print(f"WARNING: Could not overwrite latest predictions CSV for {args.fd} {args.model}.")

    print(pd.Series(row).to_string())
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
