from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 10
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_torch_regressor(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: TrainConfig,
) -> nn.Module:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    val_size = max(1, int(len(dataset) * config.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.HuberLoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in tqdm(range(1, config.epochs + 1), desc="training", leave=False):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history_ = history
    model.best_val_loss_ = best_val
    model.epochs_trained_ = len(history)
    return model.to(device)


def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
    return float(np.mean(losses))


def predict_torch(model: nn.Module, x: np.ndarray, device: str = "auto", batch_size: int = 512) -> np.ndarray:
    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(dev)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float64)
