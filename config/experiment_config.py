from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import pandas as pd
import torch


class AvailableDatasets(Enum):
    GBM = Path("gbm") / "gbm.csv"
    BITCOIN = ""
    SPX = Path("spx") / "final_df.csv"


@dataclass
class ExperimentConfig:
    # Training Settings
    DATASET: AvailableDatasets = field(
        default=AvailableDatasets.GBM, metadata={"docs": "Dataset to train on"}
    )

    RETRAIN_NUM_PERIODS: bool | None = field(
        default=None,
        metadata={
            "docs": "Number of retrain periods. If `None`, then the model is tested on the whole Test dataset without retraining"
        },
    )

    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    TRAIN_START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("1980-01-01"),
        metadata={"docs": "Date to start training"},
    )

    VAL_START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2021-01-01"),
        metadata={"docs": "Date to end train"},
    )

    TEST_START_DATE: pd.Timestamp = field(
        default=pd.to_datetime("2022-01-01"),
        metadata={"docs": "Date to end analysis"},
    )

    LAG_DAYS: int = field(
        default=1,
        metadata={"docs": "Number of days to lag for feature observation"},
    )

    N_LOOKBEHIND_PERIODS: int = field(
        default=252,
        metadata={"docs": "Number of days to take into rolling regression"},
    )

    MIN_ROLLING_PERIODS: int = field(
        default=252,
        metadata={"docs": "Number of minimum rebalance periods to run the Trainer"},
    )

    # Folders
    PATH_DATA: Path = field(
        default=Path(__file__).resolve().parents[1] / "data",
        metadata={"docs": "Relative path to data folder"},
    )

    PATH_OUTPUT: Path = field(
        default=Path(__file__).resolve().parents[1] / "output",
        metadata={"docs": "Relative path to data folder"},
    )

    # Technical Settings
    NUM_WORKERS: int = field(
        default=2, metadata={"docs": "Number of available workers"}
    )

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            string += f"\n* {key} = {value}"
        return string

    def __repr__(self) -> str:
        return self.__str__()
