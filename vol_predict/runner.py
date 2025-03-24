from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.experiment_config import ExperimentConfig
    from vol_predict.models.abstract_predictor import AbstractPredictor

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from vol_predict.train.trainer import Trainer
from vol_predict.dataset.returns_dataset import ReturnsDataset
from vol_predict.backtest.assessor import Assessor, AssessmentResult
from vol_predict.base.returns import Returns


@dataclass
class RunResult:
    model_result: AssessmentResult
    baseline_result: AssessmentResult


class Runner:
    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
        verbose: bool = True,
    ) -> None:
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.verbose = verbose

        self._initialize()

        self._model_trainer = Trainer(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model_config=self.model_config,
            experiment_config=self.experiment_config,
        )
        self._baseline_trainer = Trainer(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model_config=self.model_config,
            experiment_config=self.experiment_config,
        )
        self._assessor = Assessor(
            test_loader=self.test_loader,
            model_config=self.model_config,
            experiment_config=self.experiment_config,
        )

    def _load_df(self) -> pd.DataFrame:
        data_df = pd.read_csv(
            self.experiment_config.PATH_DATA / self.experiment_config.DATASET.value
        )
        data_df["date"] = pd.to_datetime(data_df["date"])
        return data_df.set_index("date")

    def _initialize(self):
        data_df = self._load_df()

        self.train_data = data_df.loc[
            self.experiment_config.TRAIN_START_DATE : self.experiment_config.VAL_START_DATE
        ]
        self.val_data = data_df.loc[
            self.experiment_config.VAL_START_DATE : self.experiment_config.TEST_START_DATE
        ]
        self.test_data = data_df.loc[self.experiment_config.TEST_START_DATE :]

        if self.experiment_config.ASSET_UNIVERSE is None:
            self.experiment_config.ASSET_UNIVERSE = tuple(
                set(self.train_data.columns.tolist())
            )

        self.train_returns = Returns(
            self.train_data.loc[:, self.experiment_config.ASSET_UNIVERSE].iloc[1:]
        )
        self.val_returns = Returns(
            self.val_data.loc[:, self.experiment_config.ASSET_UNIVERSE].iloc[1:]
        )
        self.test_returns = Returns(
            self.test_data.loc[:, self.experiment_config.ASSET_UNIVERSE].iloc[1:]
        )

        self.train_data = self.train_data.shift(1).iloc[1:]
        self.val_data = self.val_data.shift(1).iloc[1:]
        self.test_data = self.test_data.shift(1).iloc[1:]

        self.train_loader, self.val_loader, self.test_loader = self._get_dataloaders()

        if self.verbose:
            print(
                f"Train data on {self.train_data.index.min()} to {self.train_data.index.max()}"
            )  # noqa: T201
            if len(self.test_data) > 0:
                print(
                    f"Test data on {self.test_data.index.min()} to {self.test_data.index.max()}"
                )  # noqa: T201
            print(f"Num Train Iterations: {len(self.train_data)}")  # noqa: T201

    def _get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_set = ReturnsDataset(
            returns=self.train_returns,
            features=self.train_data,
        )
        val_set = ReturnsDataset(
            returns=self.val_returns,
            features=self.val_data,
        )
        test_set = ReturnsDataset(
            returns=self.test_returns,
            features=self.test_data,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader

    def available_features(self) -> list[str]:
        return self.train_data.columns.tolist()

    def train(
        self,
        model: AbstractPredictor,
        baseline: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> None:
        self._model_trainer(model, n_epochs)
        self._baseline_trainer(baseline, n_epochs)

    def assess(
        self, model: AbstractPredictor, baseline: AbstractPredictor
    ) -> RunResult:
        model_assessment = self._assessor(model)
        baseline_assessment = self._assessor(baseline)

        return RunResult(
            model_result=model_assessment,
            baseline_result=baseline_assessment,
        )

    def run(
        self,
        model: AbstractPredictor,
        baseline: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> RunResult:
        self.train(model=model, baseline=baseline, n_epochs=n_epochs)
        return self.assess(model=model, baseline=baseline)

    def __call__(
        self,
        model: AbstractPredictor,
        baseline: AbstractPredictor,
        n_epochs: int | None = None,
    ) -> RunResult:
        return self.run(model=model, baseline=baseline, n_epochs=n_epochs)

    @staticmethod
    def _plot_criterion_distr(
        model_criterion: np.ndarray, baseline_criterion: np.ndarray
    ) -> None:
        bins = np.linspace(-0.25, 0.25, 100)

        plt.hist(model_criterion, bins, alpha=0.5, label="model")
        plt.hist(baseline_criterion, bins, alpha=0.5, label="baseline")
        plt.legend(loc="upper right")
        plt.show()

    @staticmethod
    def _plot_preds_distr(model_preds: np.ndarray, baseline_preds: np.ndarray) -> None:
        bins = np.linspace(-0.25, 0.25, 100)

        plt.hist(model_preds, bins, alpha=0.5, label="model")
        plt.hist(baseline_preds, bins, alpha=0.5, label="baseline")
        plt.legend(loc="upper right")
        plt.show()

    @staticmethod
    def _plot_preds_ts(
        model_criterion: np.ndarray, baseline_criterion: np.ndarray
    ) -> None:
        bins = np.linspace(-0.25, 0.25, 100)

        plt.hist(model_criterion, bins, alpha=0.5, label="model")
        plt.hist(baseline_criterion, bins, alpha=0.5, label="baseline")
        plt.legend(loc="upper right")
        plt.show()

    @property
    def model_trainer(self) -> Trainer:
        return self._model_trainer

    @property
    def baseline_trainer(self) -> Trainer:
        return self._baseline_trainer

    @property
    def assessor(self) -> Assessor:
        return self._assessor
