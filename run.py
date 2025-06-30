from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type
    from vol_predict.runner import RunResult

import torch

from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from vol_predict.runner import Runner
from vol_predict.sequential_runner import SequentialRunner
from vol_predict.bayesian_sequential_runner import BayesianSequentialRunner
from vol_predict.features.base_preprocessor import BasePreprocessor

from vol_predict.models.abstract_predictor import AbstractPredictor


def initialize_runner(
    model_config: ModelConfig,
    preprocessor: BasePreprocessor,
    experiment_config: ExperimentConfig = ExperimentConfig(),
) -> Runner:
    return Runner(
        preprocessor=preprocessor,
        model_config=model_config,
        experiment_config=experiment_config,
    )


def initialize_sequential_runner(
    model_config: ModelConfig,
    preprocessor: BasePreprocessor,
    experiment_config: ExperimentConfig = ExperimentConfig(),
) -> SequentialRunner:
    return SequentialRunner(
        preprocessor=preprocessor,
        model_config=model_config,
        experiment_config=experiment_config,
    )


def initialize_bayesian_sequential_runner(
    model_config: ModelConfig,
    preprocessor: BasePreprocessor,
    experiment_config: ExperimentConfig = ExperimentConfig(),
) -> BayesianSequentialRunner:
    return BayesianSequentialRunner(
        preprocessor=preprocessor,
        model_config=model_config,
        experiment_config=experiment_config,
    )


def run_backtest(
    model_cls: Type[AbstractPredictor],
    baseline_cls: Type[AbstractPredictor],
    runner: Runner,
    experiment_config: ExperimentConfig = ExperimentConfig(),
) -> RunResult:
    torch.manual_seed(experiment_config.RANDOM_SEED)

    model = model_cls(**runner.model_config.dict())
    baseline = baseline_cls(**runner.model_config.dict())

    run_result = runner(model=model, baseline=baseline)

    return run_result


if __name__ == "__main__":
    from config.model_config import ModelConfig
    from config.experiment_config import ExperimentConfig, AvailableDatasets
    from vol_predict.features.preprocessor import OneToOnePreprocessor

    from vol_predict.models.dl.mlp_predictor import MLPPredictor as Model
    from vol_predict.models.baselines.naive_predictor import NaivePredictor as Baseline

    print(AvailableDatasets)

    config = ExperimentConfig()
    config.DATASET = AvailableDatasets.BITCOIN

    model_params = ModelConfig()
    baseline_params = ModelConfig()

    # Handles the features
    feature_processor = OneToOnePreprocessor()

    model_runner = initialize_runner(
        model_config=model_params,
        preprocessor=feature_processor,
        experiment_config=config,
    )

    result = run_backtest(
        model_cls=Model,
        baseline_cls=Baseline,
        runner=model_runner,
    )

    print(result)
