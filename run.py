from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type
    from vol_predict.runner import RunResult

from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from vol_predict.runner import Runner
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


def run_backtest(
    model_cls: Type[AbstractPredictor],
    baseline_cls: Type[AbstractPredictor],
    model_config: ModelConfig,
    baseline_config: ModelConfig,
    runner: Runner,
) -> RunResult:
    model = model_cls(**model_config.dict())
    baseline = baseline_cls(**baseline_config.dict())

    run_result = runner(model=model, baseline=baseline)

    return run_result


if __name__ == "__main__":
    from config.model_config import ModelConfig
    from config.experiment_config import ExperimentConfig
    from vol_predict.features.preprocessor import OneToOnePreprocessor

    from vol_predict.models.dl.mlp_predictor import MLPPredictor as Model
    from vol_predict.models.baselines.naive_predictor import NaivePredictor as Baseline

    config = ExperimentConfig()
    config.ASSET_UNIVERSE = ("spx",)

    model_params = ModelConfig()
    baseline_params = ModelConfig()

    # Handles the features
    feature_processor = OneToOnePreprocessor()

    runner = initialize_runner(
        model_config=model_params,
        preprocessor=feature_processor,
        experiment_config=config,
    )

    result = run_backtest(
        model_cls=Model,
        baseline_cls=Baseline,
        model_config=model_params,
        baseline_config=baseline_params,
        runner=runner,
    )

    print(result)
