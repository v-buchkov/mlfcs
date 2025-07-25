import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss
from vol_predict.models.abstract_predictor import AbstractPredictor


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    train_preds: list[float] | None = None,
    val_preds: list[float] | None = None,
    n_days: int = 252,
):
    clear_output()
    n_cols = 2 if train_preds is not None or val_preds is not None else 1
    fig, axs = plt.subplots(1, n_cols, figsize=(13, 4))

    if n_cols == 1:
        axs = [axs]

    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label="val")
    axs[0].set_ylabel("Loss")

    if train_preds is not None:
        axs[1].plot(
            range(1, len(train_preds) + 1),
            np.array(train_preds) * np.sqrt(n_days),
            label="train",
        )
    if val_preds is not None:
        axs[1].plot(
            range(1, len(val_preds) + 1),
            np.array(val_preds) * np.sqrt(n_days),
            label="val",
        )

    if train_preds is not None or val_preds is not None:
        axs[1].set_ylabel("Annualized Volatility")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()

    plt.show()


def train_epoch(
    model: AbstractPredictor,
    optimizer: torch.optim.Optimizer,
    criterion: AbstractCustomLoss,
    loader: DataLoader,
    tqdm_desc: str = "Model",
    max_grad_norm: float = 1.0,
    hidden_size: int = 128,
    n_layers: int = 3,
) -> tuple[torch.float32, np.ndarray]:
    device = next(model.parameters()).device

    if device == torch.device("cuda"):
        has_cuda = True
    else:
        has_cuda = False

    if tqdm_desc is None:
        iterator = loader
    else:
        iterator = tqdm(loader, desc=tqdm_desc)

    if has_cuda:
        scaler = GradScaler()

    train_loss = 0.0
    pred_path = []
    model.train()
    for features, past_returns, past_vols, true_returns, true_vols in iterator:
        optimizer.zero_grad()

        features = features.to(device)
        past_returns = past_returns.to(device)
        past_vols = past_vols.to(device)
        true_returns = true_returns.to(device)
        true_vols = true_vols.to(device)

        if has_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_vol = model(
                    features=features,
                    past_returns=past_returns,
                    past_vols=past_vols,
                )

                loss = criterion(true_returns, true_vols, pred_vol)

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(optimizer)

            scaler.update()
        else:
            pred_vol = model(
                features=features, past_returns=past_returns, past_vols=past_vols
            )

            loss = criterion(true_returns, true_vols, pred_vol)

            if loss.requires_grad:
                loss.backward(retain_graph=True)
                optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        train_loss += loss.item()
        prediction_points = np.concatenate(
            (
                true_returns.cpu().numpy(),
                true_vols.cpu().numpy(),
                pred_vol.detach().cpu().numpy(),
            ),
            axis=1,
        )
        pred_path.append(prediction_points)

    train_loss /= len(loader.dataset)

    return train_loss, np.concatenate(pred_path, axis=0)


@torch.no_grad()
def validation_epoch(
    model: AbstractPredictor,
    criterion: AbstractCustomLoss,
    loader: DataLoader,
    tqdm_desc: str | None = None,
    hidden_size: int = 128,
    n_layers: int = 3,
) -> tuple[torch.float32, np.ndarray]:
    device = next(model.parameters()).device

    if tqdm_desc is None:
        iterator = loader
    else:
        iterator = tqdm(loader, desc=tqdm_desc)

    val_loss = 0.0
    model.eval()
    pred_path = []

    for features, past_returns, past_vols, true_returns, true_vols in iterator:
        features = features.to(device)
        past_returns = past_returns.to(device)
        past_vols = past_vols.to(device)
        true_returns = true_returns.to(device)
        true_vols = true_vols.to(device)

        pred_vol = model(
            features=features, past_returns=past_returns, past_vols=past_vols
        )

        loss = criterion(true_returns, true_vols, pred_vol)

        val_loss += loss.item()

        prediction_points = np.concatenate(
            (
                true_returns.cpu().numpy(),
                true_vols.cpu().numpy(),
                pred_vol.detach().cpu().numpy(),
            ),
            axis=1,
        )
        pred_path.append(prediction_points)

    val_loss /= len(loader.dataset)

    return val_loss, np.concatenate(pred_path, axis=0)


@torch.no_grad()
def bayesian_validation_epoch(
    model: AbstractPredictor,
    criterion: AbstractCustomLoss,
    loader: DataLoader,
    tqdm_desc: str | None = None,
    hidden_size: int = 128,
    n_layers: int = 3,
    n_experiments: int = 100,
) -> tuple[torch.float32, np.ndarray]:
    device = next(model.parameters()).device

    if tqdm_desc is None:
        iterator = loader
    else:
        iterator = tqdm(loader, desc=tqdm_desc)

    val_loss = 0.0
    pred_path = []
    uncerts = []
    for features, past_returns, past_vols, true_returns, true_vols in iterator:
        features = features.to(device)
        past_returns = past_returns.to(device)
        past_vols = past_vols.to(device)
        true_returns = true_returns.to(device)
        true_vols = true_vols.to(device)

        preds = []
        for _ in range(n_experiments):
            pred_vol = model(
                features=features, past_returns=past_returns, past_vols=past_vols
            )
            preds.append(pred_vol)

        preds = torch.stack(preds)
        pred_vol = preds.mean(dim=0)
        uncert = preds.std(dim=0)
        uncerts.append(uncert.detach().cpu().numpy())

        loss = criterion(true_returns, true_vols, pred_vol)

        val_loss += loss.item()

        prediction_points = np.concatenate(
            (
                true_returns.cpu().numpy(),
                true_vols.cpu().numpy(),
                pred_vol.detach().cpu().numpy(),
            ),
            axis=1,
        )
        pred_path.append(prediction_points)

    val_loss /= len(loader.dataset)

    return val_loss, np.concatenate(pred_path, axis=0), np.concatenate(uncerts, axis=0)
