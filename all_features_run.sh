#!/bin/bash
poetry run python training.py preprocessing=log_mel_spectrogram.yaml experiments.MLFLOW_RUN_NOTE="logmel"
poetry run python test.py preprocessing=log_mel_spectrogram.yaml    

poetry run python training.py preprocessing=log_mel_and_differential_log_mel.yaml experiments.MLFLOW_RUN_NOTE="logmel and diff logmel"
poetry run python test.py preprocessing=log_mel_and_differential_log_mel.yaml

poetry run python training.py preprocessing=log_mel_and_phase_differential.yaml experiments.MLFLOW_RUN_NOTE="logmel diff phase"
poetry run python test.py preprocessing=log_mel_and_phase_differential.yaml

poetry run python training.py preprocessing=log_mel_and_differential_log_mel_and_phase.yaml experiments.MLFLOW_RUN_NOTE="logmel diff logmel diff phase"
poetry run python test.py preprocessing=log_mel_and_differential_log_mel_and_phase.yaml