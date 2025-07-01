from __future__ import annotations

from typing import TYPE_CHECKING

from vol_predict.train.train import bayesian_validation_epoch

if TYPE_CHECKING:
    from config.experiment_config import ExperimentConfig
    from vol_predict.models.abstract_predictor import AbstractPredictor

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.model_config import ModelConfig
from vol_predict.train.trainer import Trainer
from vol_predict.features.base_preprocessor import BasePreprocessor
from vol_predict.dataset.returns_dataset import ReturnsDataset
from vol_predict.base.returns import Returns


def weights_init(model):
    if model.device != "cpu":
        model = model.to("cpu")
    for name, param in model.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(param)  # Orthogonal initialization
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(param)  # Xavier initialization
    return model.to(model.device)


class BayesianFromTimestampsRunner:
    def __init__(
        self,
        output_df: pd.DataFrame,
        preprocessor: BasePreprocessor,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
        verbose: bool = True,
    ) -> None:
        self.output = output_df

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

    def extract_rebal_schedule(self) -> list[pd.Timestamp]:
        schedule = self.output[self.output["retraining_flag"]].index.to_list()
        schedule.append(self.output.index[-1])
        return schedule

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

        self.rebal_schedule = self.extract_rebal_schedule()

    def _get_dataloader(
        self, time_start: pd.Timestamp | None, time_end: pd.Timestamp, is_train: bool = True
    ) -> DataLoader:
        time_start = self.features.loc[:time_start].index[-2] if time_start is not None else self.features.index[0]
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
        n_epochs: int | None = None,
    ) -> pd.DataFrame:
        loss = self.model_config.loss.value().to(self.experiment_config.DEVICE)
        last_rebal_date = self.rebal_schedule[0]
        uncert_history = []
        uncerts = 0
        for rebal_date in (pbar := tqdm(self.rebal_schedule[1:])):
            # None to train on all data available
            train_loader = self._get_dataloader(None, last_rebal_date - pd.Timedelta(milliseconds=1), is_train=True)
            test_loader = self._get_dataloader(last_rebal_date, rebal_date - pd.Timedelta(milliseconds=1), is_train=False)

            pbar.set_description(f"Date: {rebal_date}")

            if len(train_loader.dataset) > 0:
                if self.experiment_config.RETRAIN:
                    if np.mean(uncerts) > np.mean(uncert_history) + np.std(
                        uncert_history
                    ):
                        model = model.__class__(**self.model_config.dict())

                model_trainer = Trainer(
                    train_loader=train_loader,
                    val_loader=train_loader,
                    model_config=self.model_config,
                    experiment_config=self.experiment_config,
                )

                model_trainer(model, n_epochs)

            _, preds, uncerts = bayesian_validation_epoch(
                model,
                loss,
                test_loader,
                hidden_size=self.model_config.hidden_size,
                n_layers=self.model_config.n_layers,
            )
            uncert_history.append(np.mean(uncerts))
            preds = preds[:, -1]

            model_name = model.__class__.__name__
            self.output.loc[last_rebal_date : rebal_date - pd.Timedelta(milliseconds=1), model_name] = preds

        torch.save(model.state_dict(), "model.pt")

        return self.output

    def __call__(
        self,
        model: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> pd.DataFrame:
        return self.run(model=model, n_epochs=n_epochs)
