from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from vol_predict.features.base_preprocessor import BasePreprocessor


class Preprocessor(BasePreprocessor):
    def __init__(
        self,
        feature_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.feature_names = feature_names
        self.exclude_names = exclude_names
        self.n_lags = 0

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        columns = features.columns
        if self.feature_names is not None:
            return features[self.feature_names]
        if self.exclude_names is not None:
            return features[columns.difference(self.exclude_names)]
        return features

    @property
    def truncation_len(self) -> int:
        return self.n_lags
