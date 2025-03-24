import datetime as dt
from typing import Optional, Any
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss
from vol_predict.models.abstract_predictor import AbstractPredictor
from vol_predict.train.train import train_epoch, validation_epoch, plot_losses
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config

        self._train_loader = train_loader
        self._val_loader = val_loader

        self._train_losses, self._val_losses = None, None
        self._train_preds, self._val_preds = None, None

    def _train(
        self,
        model: AbstractPredictor,
        optimizer: torch.optim.optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        print_logs: bool = True,
    ):
        train_losses, val_losses = [], []
        train_preds, val_preds = [], []
        criterion = self.model_config.loss.to(self.experiment_config.DEVICE)

        assert isinstance(criterion, AbstractCustomLoss), (
            f"{criterion.__class__.__name__} is not a subclass of AbstractCustomLoss!"
        )

        for epoch in range(1, num_epochs + 1):
            if print_logs:
                desc_train = f"Training {epoch}/{num_epochs}"
                desc_val = f"Validation {epoch}/{num_epochs}"
            else:
                desc_train, desc_val = None, None

            train_loss, train_pred = train_epoch(
                model, optimizer, criterion, train_loader, tqdm_desc=desc_train
            )
            val_loss, val_pred = validation_epoch(
                model, criterion, val_loader, tqdm_desc=desc_val
            )

            if scheduler is not None:
                scheduler.step()

            train_losses += [train_loss]
            val_losses += [val_loss]

            train_preds += train_pred
            val_preds += val_pred

            # TODO @V: average pred into plot
            plot_losses(train_losses, val_losses, None, None)

        return train_losses, val_losses, np.stack(train_preds), np.stack(val_preds)

    def __call__(self, model: AbstractPredictor, n_epochs: int | None = None) -> None:
        self.run(model, n_epochs)

    def run(self, model: AbstractPredictor, n_epochs: int | None = None) -> None:
        model = model.to(self.experiment_config.DEVICE)
        optimizer = self.model_config.optimizer(
            model.parameters(), lr=self.model_config.lr
        )

        if n_epochs is None:
            n_epochs = self.model_config.train_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        self._train_losses, self._val_losses, self._train_preds, self._val_preds = (
            self._train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                num_epochs=n_epochs,
                print_logs=True,
            )
        )

        self.save(model, self.experiment_config.PATH_OUTPUT)

    @staticmethod
    def save(model: AbstractPredictor, path: Path) -> None:
        torch.save(model, path / f"run_{dt.datetime.now()}.pt")

    @property
    def train_losses(self):
        return self._train_losses

    @property
    def val_losses(self):
        return self._val_losses

    @property
    def train_preds(self):
        return self._train_preds

    @property
    def val_preds(self):
        return self._val_preds
