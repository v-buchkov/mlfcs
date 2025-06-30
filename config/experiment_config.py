from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import pandas as pd
import torch


class AvailableDatasets(Enum):
    GBM = Path("gbm") / "gbm.csv"
    BITCOIN = Path("btc") / "data_df.csv"
    SPX = Path("spx") / "data_df.csv"


@dataclass
class ExperimentConfig:
    # Training Settings
    DATASET: AvailableDatasets = field(
        default=AvailableDatasets.BITCOIN, metadata={"docs": "Dataset to train on"}
    )

    RETURN_COLUMN: str = field(
        default="ret", metadata={"docs": "Realized Return column"}
    )

    VOL_COLUMN: str = field(
        default="vol", metadata={"docs": "Realized Variance column"}
    )

    EXPANDING: bool = field(
        default=True,
        metadata={
            "docs": "Number of retrain periods. If `None`, then the model is tested on the whole Test dataset without retraining"
        },
    )

    RETRAIN: bool = field(
        default=True,
        metadata={
            "docs": "Number of retrain periods. If `None`, then the model is tested on the whole Test dataset without retraining"
        },
    )

    N_FEATURES: int = field(default=12, metadata={"docs": "Fix random seed"})

    RANDOM_SEED: int = field(default=12, metadata={"docs": "Fix random seed"})

    TRAIN_START_DATE: pd.Timestamp = field(
        default=pd.Timestamp("2018-06-04"),
        metadata={"docs": "Date to start training"},
    )

    FIRST_TRAIN_END_DATE: pd.Timestamp = field(
        default=pd.Timestamp("2018-06-30"),
        metadata={"docs": "Date to end train"},
    )

    ROLLING_STEP_DAYS: int = field(
        default=5,
        metadata={"docs": "Number of days to take into rolling regression"},
    )

    VAL_START_DATE: pd.Timestamp = field(
        default=pd.Timestamp("2018-09-01"),
        metadata={"docs": "Date to end train"},
    )

    TEST_START_DATE: pd.Timestamp = field(
        default=pd.Timestamp("2018-09-15"),
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

    RESULTS_FILENAME: str = field(
        default="results.csv",
        metadata={"docs": "File with all experimental results"},
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
