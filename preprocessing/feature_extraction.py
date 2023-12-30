from operator import is_
import sys
from itertools import chain
from typing import Dict, List
import glob

import librosa
import librosa.core
import librosa.feature
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import torch

from augmentation.crop import random_crop, make_subseq


def load_audio(audio_file, cfg, mono=False):
    """
    load audio file.
    audio_file : str
        target audio file
    mono : boolean
        When loading a multi channels file and this param is True,
        the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(
            audio_file, sr=cfg.preprocessing.feature.sample_rate, mono=mono
        )
    except FileNotFoundError:
        print("file_broken or not exists!! : %s", audio_file)


def get_log_mel_spectrogram(sample_file: str, cfg: DictConfig):

    audio, _ = load_audio(sample_file, cfg, mono=False)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.preprocessing.feature.sample_rate,
        n_fft=cfg.preprocessing.feature.n_fft,
        hop_length=cfg.preprocessing.feature.hop_length,
        n_mels=cfg.preprocessing.feature.n_mels,
        power=cfg.preprocessing.feature.power,
    )

    log_mel_spectrogram = (
        20.0
        / cfg.preprocessing.feature.power
        * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    )

    return log_mel_spectrogram


def get_differential_log_mel_spectrogram(sample_file: str, cfg: DictConfig):

    log_mel_spectrogram = get_log_mel_spectrogram(sample_file=sample_file, cfg=cfg)

    differential_log_mel_spectrogram = np.zeros_like(log_mel_spectrogram)

    # _differential_log_mel_spectrogram shape is (#mel, #time(frame)-1)
    _differential_log_mel_spectrogram = (
        log_mel_spectrogram[:, 1:] - log_mel_spectrogram[:, :-1]
    )
    # differential_log_mel_spectrogram shape is (#mel, #time(frame))
    differential_log_mel_spectrogram[:, 1:] = _differential_log_mel_spectrogram

    return differential_log_mel_spectrogram


def get_differential_phase(sample_file: str, cfg: DictConfig):

    audio, _ = load_audio(sample_file, cfg, mono=False)
    C = librosa.stft(
        audio,
        n_fft=cfg.preprocessing.feature.n_fft,
        hop_length=cfg.preprocessing.feature.hop_length,
    )

    mag, phase = librosa.core.magphase(C)
    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    differential_phase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    differential_phase = (
        np.concatenate([phase_unwrapped[:, 0:1], differential_phase], axis=1) / np.pi
    )

    return differential_phase


class ExtractMultiChannelFeature:
    def __init__(self, is_eval: bool, cfg: DictConfig) -> None:

        self.is_eval = is_eval
        self.cfg = cfg

    def __call__(self, sample_dirs):

        query = f"{sample_dirs}/*"
        sample_files = sorted(glob.glob(query))

        multi_channels_samples = []
        for sample_file in sample_files:

            if self.cfg.preprocessing.name == "log_mel_spectrogram":
                # log mel spectrogram
                sample = get_log_mel_spectrogram(sample_file=sample_file, cfg=self.cfg)
                sample = torch.tensor(sample)  # to torch from numpy
                multi_channels_samples.append(sample)
            elif self.cfg.preprocessing.name == "log_mel_and_differential_log_mel":

                # log mel spectrogram
                log_mel_spectrogram = get_log_mel_spectrogram(
                    sample_file=sample_file, cfg=self.cfg
                )
                log_mel_spectrogram = torch.tensor(
                    log_mel_spectrogram
                )  # to torch from numpy
                # differential log mel spectrogram
                differential_log_mel_spectrogram = get_differential_log_mel_spectrogram(
                    sample_file=sample_file, cfg=self.cfg
                )
                differential_log_mel_spectrogram = torch.tensor(
                    differential_log_mel_spectrogram
                )  # to torch from numpy

                sample = torch.concat(
                    (
                        log_mel_spectrogram,
                        differential_log_mel_spectrogram,
                    ),
                    dim=0,
                )

                multi_channels_samples.append(sample)
            elif self.cfg.preprocessing.name == "log_mel_and_phase_differential":
                # log mel spectrogram
                log_mel_spectrogram = get_log_mel_spectrogram(
                    sample_file=sample_file, cfg=self.cfg
                )
                log_mel_spectrogram = torch.tensor(
                    log_mel_spectrogram
                )  # to torch from numpy

                # differential phase
                differential_phase = get_differential_phase(
                    sample_file=sample_file, cfg=self.cfg
                )
                differential_phase = torch.tensor(
                    differential_phase
                )  # to torch from numpy

                sample = torch.concat(
                    (
                        log_mel_spectrogram,
                        differential_phase,
                    ),
                    dim=0,
                )

                multi_channels_samples.append(sample)
            elif (
                self.cfg.preprocessing.name
                == "log_mel_and_differential_log_mel_and_phase"
            ):
                # log mel spectrogram
                log_mel_spectrogram = get_log_mel_spectrogram(
                    sample_file=sample_file, cfg=self.cfg
                )
                log_mel_spectrogram = torch.tensor(
                    log_mel_spectrogram
                )  # to torch from numpy

                # differential log mel spectrogram
                differential_log_mel_spectrogram = get_differential_log_mel_spectrogram(
                    sample_file=sample_file, cfg=self.cfg
                )
                differential_log_mel_spectrogram = torch.tensor(
                    differential_log_mel_spectrogram
                )  # to torch from numpy

                # differential phase
                differential_phase = get_differential_phase(
                    sample_file=sample_file, cfg=self.cfg
                )
                differential_phase = torch.tensor(
                    differential_phase
                )  # to torch from numpy

                sample = torch.concat(
                    (
                        log_mel_spectrogram,
                        differential_log_mel_spectrogram,
                        differential_phase,
                    ),
                    dim=0,
                )

                multi_channels_samples.append(sample)

        # sample shape is (#channel=4, #dim(features), #time)
        sample = torch.stack(multi_channels_samples)

        if self.is_eval:
            sample = make_subseq(
                sample,
                self.cfg.augumentation.n_crop_frames,
                self.cfg.preprocessing.feature.n_hop_frames,
            )
        else:
            sample = random_crop(sample, self.cfg.augumentation.n_crop_frames)
        # sample shape is (#channel, #dim(fft or mel), #time)
        return sample


class LoadMultiLabelFeatures:
    def __init__(self, is_eval: bool, cfg: DictConfig) -> None:

        self.is_eval = is_eval
        self.cfg = cfg

    def __call__(self, sample_dirs):

        features_file = f"{sample_dirs}/{self.cfg.preprocessing.file_name}"

        try:
            sample = torch.load(features_file)
        except FileNotFoundError:
            print(f"{features_file} is not found or broken.")

        if self.is_eval:
            sample = make_subseq(
                sample,
                self.cfg.augumentation.n_crop_frames,
                self.cfg.preprocessing.feature.n_hop_frames,
            )
        else:
            sample = random_crop(sample, self.cfg.augumentation.n_crop_frames)
        # sample shape is (#channel, #dim(fft or mel), #time)
        return sample
