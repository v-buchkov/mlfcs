from __future__ import annotations

import pandas as pd
import torch

from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig, AvailableDatasets
from vol_predict.from_timestamps_runner import FromTimestampsRunner
from vol_predict.features.preprocessor import OneToOnePreprocessor
from vol_predict.loss.loss import Loss

from vol_predict.models.dl.mlp_predictor import MLPPredictor
from vol_predict.models.dl.sigma_lstm_predictor import SigmaLSTMPredictor
from vol_predict.models.dl.lstm_softplus_predictor import LSTMSoftplusPredictor
from vol_predict.models.dl.transformer_predictor import TransformerPredictor
from vol_predict.models.dl.vi_predictor import ViPredictor
from vol_predict.models.dl.vi_eval_predictor import ViEvalPredictor

OUTPUT_FILENAME = "output_retrain.csv"

config = ExperimentConfig()
config.DATASET = AvailableDatasets.BITCOIN

model_params = ModelConfig()
baseline_params = ModelConfig()

model_params.n_features = 1200
model_params.n_unique_features = 10

# Handles the features
feature_processor = OneToOnePreprocessor()

torch.manual_seed(config.RANDOM_SEED)

output_df = pd.read_csv(OUTPUT_FILENAME)
output_df["datetime"] = pd.to_datetime(output_df["datetime"])
output_df = output_df.sort_values(by="datetime")
output_df = output_df.set_index("datetime")

runner = FromTimestampsRunner(
    output_df=output_df,
    preprocessor=feature_processor,
    model_config=model_params,
    experiment_config=config,
)

runner.model_config.lr = 1e-2
runner.model_config.n_epochs = 50
runner.model_config.hidden_size = 64
runner.model_config.n_layers = 3
runner.model_config.batch_size = 16
runner.model_config.optimizer = torch.optim.Adam
runner.model_config.loss = Loss.NLL
runner.model_config.dropout = 0.20
runner.experiment_config.RETRAIN = True

# models = [MLPPredictor, LSTMSoftplusPredictor, TransformerPredictor, SigmaLSTMPredictor]
models = [ViEvalPredictor]

for model_cls in models:
    model = model_cls(**runner.model_config.dict())

    output_df = runner(model=model)

output_df.to_csv("output_retrain.csv")
