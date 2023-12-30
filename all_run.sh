#!/bin/bash
poetry run python training.py preprocessing=log_mel_spectrogram.yaml dataloader=default.yaml experiments.MLFLOW_RUN_NOTE="logmel"
poetry run python test.py preprocessing=log_mel_spectrogram.yaml    

poetry run python training.py preprocessing=log_mel_spectrogram.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="logmel small1000"
poetry run python test.py preprocessing=log_mel_spectrogram.yaml    

poetry run python training.py preprocessing=log_mel_and_differential_log_mel.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="logmel and diff logmel small1000"
poetry run python test.py preprocessing=log_mel_and_differential_log_mel.yaml

poetry run python training.py preprocessing=log_mel_and_phase_differential.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="logmel diff phase small1000"
poetry run python test.py preprocessing=log_mel_and_phase_differential.yaml

poetry run python training.py preprocessing=log_mel_and_differential_log_mel_and_phase.yaml dataloader=small1000.yaml experiments.MLFLOW_RUN_NOTE="logmel diff logmel diff phase small1000"
poetry run python test.py preprocessing=log_mel_and_differential_log_mel_and_phase.yaml