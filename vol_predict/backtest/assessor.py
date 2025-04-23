from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

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

    def __call__(
        self,
        true_returns: torch.Tensor,
        true_vols: torch.Tensor,
        pred_vol: torch.Tensor,
    ) -> dict[str, float]:
        metrics = {}
        for metric in self._ml_metrics:
            metric_instance = metric.value()
            metrics[metric_instance.__class__.__name__] = metric_instance(
                true_returns, true_vols, pred_vol
            ).item()

        return metrics


@dataclass
class AssessmentResult:
    mean_model_loss: float
    mean_val_loss: float

    mean_pred_vol: float
    mean_true_vol: float

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

    def get_df(self, model_name: str) -> pd.DataFrame:
        data = []
        columns = []
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                names = list(value.keys())
                point = list(value.values())
            else:
                names = [key]
                point = [value]

            data += point
            columns += names

        data = np.array(data)[np.newaxis, :]
        return pd.DataFrame(data, index=[model_name], columns=columns)


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

        self.assessment_result = None
        self.model_name = None

        self.ml_metrics = MlMetrics(model_config.metrics)

    def run(self, model: AbstractPredictor) -> AssessmentResult:
        loss = self.model_config.loss.value().to(self.experiment_config.DEVICE)
        model_loss, model_preds = validation_epoch(
            model,
            loss,
            self.test_loader,
            hidden_size=self.model_config.hidden_size,
            n_layers=self.model_config.n_layers,
        )

        true_returns = torch.tensor(model_preds[:, 0])
        true_vols = torch.tensor(model_preds[:, 1])
        model_preds_tensor = torch.tensor(model_preds[:, 2])

        self.assessment_result = AssessmentResult(
            mean_model_loss=model_loss,
            mean_val_loss=model_loss,
            mean_pred_vol=torch.sqrt(model_preds_tensor.mean()),
            mean_true_vol=torch.sqrt(true_vols.mean()),
            ml_metrics=self.ml_metrics(true_returns, true_vols, model_preds_tensor),
        )
        self.model_name = model.__class__.__name__

        return self.assessment_result

    def __call__(self, model: AbstractPredictor) -> AssessmentResult:
        return self.run(model)

    def save(self) -> None:
        if self.assessment_result is None:
            raise RuntimeError(
                "Model is not assessed yet! Please, call to Assessor()(model)"
            )

        new_results = self.assessment_result.get_df(self.model_name)

        if self.experiment_config.RESULTS_FILENAME in os.listdir(
            self.experiment_config.PATH_OUTPUT
        ):
            results = pd.read_csv(
                self.experiment_config.PATH_OUTPUT
                / self.experiment_config.RESULTS_FILENAME
            )
            results = results.set_index("Unnamed: 0")
            results = pd.concat([results, new_results], axis=0)
        else:
            results = new_results.copy()

        results = results.drop_duplicates(keep="last")
        results.to_csv(
            self.experiment_config.PATH_OUTPUT
            / Path("full_" + self.experiment_config.RESULTS_FILENAME)
        )

        trunc_results = results[~results.index.duplicated(keep="first")]
        trunc_results.to_csv(
            self.experiment_config.PATH_OUTPUT / self.experiment_config.RESULTS_FILENAME
        )
