from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from config.experiment_config import ExperimentConfig
from config.model_config import ModelConfig
from vol_predict.models.abstract_predictor import AbstractPredictor
from vol_predict.train.train import validation_epoch


class MlMetrics:
    def __init__(self, ml_metrics: tuple[nn.Module]):
        self._ml_metrics = ml_metrics

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
        metrics = {}
        for metric in self._ml_metrics:
            metric_instance = metric()
            metrics[metric_instance.__class__.__name__] = metric_instance(y_true, y_pred).item()

        return metrics


@dataclass
class AssessmentResult:
    mean_model_loss: float
    mean_val_loss: float

    ml_metrics: dict[str, float]

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                for name, metric in value.items():
                    val = f"{metric:.12f}" if isinstance(metric, float) else metric
                    string += f"\n* {name} = {val}"
                continue
            val = f"{value:.6f}" if isinstance(value, float) else value
            string += f"\n* {key} = {val}"
        return string

    def __repr__(self) -> str:
        return str(self)


class Assessor:
    def __init__(
        self,
        test_loader: DataLoader,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
    ):
        self.test_loader = test_loader

        self.experiment_config = experiment_config
        self.model_config = model_config

        self.ml_metrics = MlMetrics(model_config.metrics)

    def run(self, model: AbstractPredictor) -> AssessmentResult:
        loss = self.model_config.loss.value().to(self.experiment_config.DEVICE)
        model_loss, model_preds = validation_epoch(model, loss, self.test_loader)

        model_preds_tensor = torch.tensor(model_preds[:, 1])
        model_true_tensor = torch.tensor(model_preds[:, 0])

        return AssessmentResult(
            mean_model_loss=model_loss,
            mean_val_loss=model_loss,
            ml_metrics=self.ml_metrics(
                y_true=model_true_tensor, y_pred=model_preds_tensor
            ),
        )

    def __call__(self, model: AbstractPredictor) -> AssessmentResult:
        return self.run(model)
