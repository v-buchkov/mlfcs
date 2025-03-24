from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BasePreprocessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def truncation_len(self) -> int:
        raise NotImplementedError

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.transform(features)
