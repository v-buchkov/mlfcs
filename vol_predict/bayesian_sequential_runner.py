from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.experiment_config import ExperimentConfig
    from vol_predict.models.abstract_predictor import AbstractPredictor

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from vol_predict.train.trainer import Trainer
from vol_predict.features.base_preprocessor import BasePreprocessor
from vol_predict.dataset.returns_dataset import ReturnsDataset
from vol_predict.backtest.assessor import AssessmentResult
from vol_predict.base.returns import Returns
from vol_predict.train.train import validation_epoch, bayesian_validation_epoch


@dataclass
class RunResult:
    model_result: AssessmentResult
    baseline_result: AssessmentResult


def weights_init(model):
    if model.device != "cpu":
        model = model.to("cpu")
    for name, param in model.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(param)  # Orthogonal initialization
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(param)  # Xavier initialization
    model = model.to(model.device)


class BayesianSequentialRunner:
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
        verbose: bool = True,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.verbose = verbose

        self._scaler = self.model_config.scaler

        self._initialize()

    def _load_df(self) -> pd.DataFrame:
        data_df = pd.read_csv(
            self.experiment_config.PATH_DATA / self.experiment_config.DATASET.value
        )
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
        data_df = data_df.sort_values(by="datetime")
        data_df = data_df.set_index("datetime")
        data_df.index = data_df.index.tz_localize(None)
        return data_df

    def _initialize(self):
        data = self._load_df()

        self.returns = data.loc[:, self.experiment_config.RETURN_COLUMN].iloc[1:]

        self.vols = data.loc[:, self.experiment_config.VOL_COLUMN].iloc[1:]

        feature_columns = data.columns.difference(
            [self.experiment_config.RETURN_COLUMN, self.experiment_config.VOL_COLUMN]
        ).tolist()
        self.features = data[feature_columns].shift(1).iloc[1:]

        if self.model_config.n_features is None:
            self.model_config.n_features = len(feature_columns)

        if self.model_config.n_unique_features is None:
            unique_columns = [
                "_".join(column.split("_")[:-1]) for column in feature_columns
            ]
            self.model_config.n_unique_features = np.unique(unique_columns).shape[0]

        if self.verbose:
            print(f"Available data from {data.index.min()} to {data.index.max()}")  # noqa: T201

    def _get_dataloader(
        self, time_start: pd.Timestamp, time_end: pd.Timestamp, is_train: bool = True
    ) -> DataLoader:
        init_features = self.features.loc[time_start:time_end]

        if init_features.ndim > 1 and init_features.shape[0] != 0:
            features = (
                self._scaler.fit_transform(init_features)
                if is_train
                else self._scaler.transform(init_features)
            )
            features = pd.DataFrame(
                features, columns=init_features.columns, index=init_features.index
            )
        else:
            features = init_features

        dataset = ReturnsDataset(
            returns=Returns(self.returns.loc[time_start:time_end]),
            vols=self.vols.loc[time_start:time_end],
            features=features,
            preprocessor=self.preprocessor,
        )

        return DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def available_features(self) -> list[str]:
        return self.features.columns.tolist()

    def run(
        self,
        model: AbstractPredictor,
        baseline: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> pd.DataFrame:
        step = self.experiment_config.ROLLING_STEP_DAYS
        loss = self.model_config.loss.value().to(self.experiment_config.DEVICE)

        train_start = self.experiment_config.TRAIN_START_DATE
        train_end = self.experiment_config.FIRST_TRAIN_END_DATE

        rolling_results = []
        uncert_history = []
        uncerts = 0
        while train_end <= self.returns.index[-1]:
            train_loader = self._get_dataloader(train_start, train_end, is_train=True)

            last_train_end = train_end
            train_start = (
                train_start + pd.Timedelta(days=step)
                if self.experiment_config.EXPANDING
                else train_start
            )
            train_end = train_end + pd.Timedelta(days=step)
            test_loader = self._get_dataloader(
                last_train_end + pd.Timedelta(milliseconds=1), train_end, is_train=False
            )

            if len(train_loader.dataset) > 0 and len(test_loader.dataset) > 0:
                if self.experiment_config.RETRAIN:
                    if np.mean(uncerts) > np.mean(uncert_history) + np.std(
                        uncert_history
                    ):
                        model = model.__class__(**self.model_config.dict())
                        baseline = baseline.__class__(**self.model_config.dict())

                model_trainer = Trainer(
                    train_loader=train_loader,
                    val_loader=train_loader,
                    model_config=self.model_config,
                    experiment_config=self.experiment_config,
                )

                model_trainer(model, n_epochs)

                baseline_trainer = Trainer(
                    train_loader=train_loader,
                    val_loader=train_loader,
                    model_config=self.model_config,
                    experiment_config=self.experiment_config,
                )

                baseline_trainer(baseline, n_epochs)

                model_loss, preds, uncerts = bayesian_validation_epoch(
                    model,
                    loss,
                    test_loader,
                    hidden_size=self.model_config.hidden_size,
                    n_layers=self.model_config.n_layers,
                )
                uncert_history.append(np.mean(uncerts))
                true_returns = preds[:, 0]
                true_vols = preds[:, 1]
                model_preds = preds[:, 2]
                baseline_loss, baseline_preds_full = validation_epoch(
                    baseline,
                    loss,
                    test_loader,
                    hidden_size=self.model_config.hidden_size,
                    n_layers=self.model_config.n_layers,
                )
                baseline_preds = baseline_preds_full[:, 2]

                rolling_results.append(
                    [
                        last_train_end,
                        model_loss,
                        baseline_loss,
                        true_returns,
                        true_vols,
                        model_preds,
                        uncerts,
                        baseline_preds,
                    ]
                )

        return pd.DataFrame(
            rolling_results,
            columns=[
                "datetime",
                "model_loss",
                "baseline_loss",
                "true_returns",
                "true_vols",
                "model_preds",
                "model_uncerts",
                "baseline_preds",
            ],
        ).set_index("datetime")

    def __call__(
        self,
        model: AbstractPredictor,
        baseline: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> pd.DataFrame:
        return self.run(model=model, baseline=baseline, n_epochs=n_epochs)
