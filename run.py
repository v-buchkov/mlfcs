from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vol_predict.runner import RunResult

import pandas as pd
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from vol_predict.runner import Runner
from vol_predict.features.preprocessor import Preprocessor

from vol_predict.models.baselines.naive_predictor import Na

HEDGE = False


def run_backtest(
    cov_estimator: BaseCovEstimator, verbose: bool = False, plot_progress: bool = False
) -> RunResult:
    experiment_config = ExperimentConfig()
    stocks = tuple(
        pd.read_csv(
            experiment_config.PATH_OUTPUT / experiment_config.STOCKS_LIST_FILENAME
        ).columns
    )
    experiment_config.ASSET_UNIVERSE = ("spx",)

    model_config = ModelConfig()

    runner = Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=verbose,
        plot=plot_progress,
    )

    hedger = MarketFuturesHedge()

    # Handles the features
    prices = [stock + "_Price" for stock in list(stocks)]
    preprocessor = Preprocessor(
        exclude_names=[*prices, *list(stocks), "acc_rate", "spx"]
    )

    strategy = MinVariance(
        cov_estimator=cov_estimator,
        trading_config=trading_config,
    )

    baseline_strategy = MinVariance(
        cov_estimator=CovEstimators.HISTORICAL.value(),
        trading_config=trading_config,
    )

    run_result = runner.train(
        feature_processor=preprocessor,
        strategy=strategy,
        baseline_strategy=baseline_strategy,
        hedger=hedger if HEDGE else None,
    )

    runner.plot_cumulative(include_factors=True)

    runner.plot_turnover()

    runner.plot_returns_histogram_vs_baseline()

    runner.plot_outperformance()

    return run_result


if __name__ == "__main__":
    ESTIMATOR = CovEstimators.HISTORICAL.value()
    VERBOSE = True
    PLOT_PROGRESS = False

    run_result = run_backtest(
        cov_estimator=ESTIMATOR,
        verbose=VERBOSE,
        plot_progress=PLOT_PROGRESS,
    )

    print(run_result.strategy)

    print("***")

    print(run_result.baseline)
