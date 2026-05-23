"""
features.py
-----------
오디오 파일에서 특성을 추출하는 유틸리티 함수 모음.
GTZAN 데이터셋의 CSV(features_3_sec / features_30_sec)를 사용하는 것이
권장되지만, 직접 wav 파일에서 특성을 뽑아야 할 때 사용합니다.
"""

import numpy as np
import librosa


def load_audio(filepath: str, sr: int = 22050):
    """오디오 파일을 로드합니다."""
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    단일 오디오 신호에서 특성을 추출합니다.

    Returns
    -------
    dict  특성명 → 값(float 또는 ndarray)
    """
    features = {}

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo)

    # Zero Crossing Rate
    zcr = librosa.zero_crossings(y, pad=False)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_var"] = float(np.var(zcr))

    # Spectral Centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_var"] = float(np.var(sc))

    # Spectral Rolloff
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(sr_feat))
    features["spectral_rolloff_var"] = float(np.var(sr_feat))

    # Spectral Bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    features["spectral_bandwidth_var"] = float(np.var(sb))

    # Harmonic & Percussive RMS
    y_harm, y_perc = librosa.effects.hpss(y)
    features["harmony_mean"] = float(np.mean(y_harm))
    features["harmony_var"] = float(np.var(y_harm))
    features["perceptr_mean"] = float(np.mean(y_perc))
    features["perceptr_var"] = float(np.var(y_perc))

    # MFCCs (20개 계수)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i, coeff in enumerate(mfccs, 1):
        features[f"mfcc{i}_mean"] = float(np.mean(coeff))
        features[f"mfcc{i}_var"] = float(np.var(coeff))

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    features["chroma_stft_mean"] = float(np.mean(chroma))
    features["chroma_stft_var"] = float(np.var(chroma))

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_var"] = float(np.var(rms))

    return features


def visualize_audio(y: np.ndarray, sr: int, title: str = "Audio"):
    """오디오 시각화 (waveform, spectrogram, mel spectrogram, MFCCs)."""
    import matplotlib.pyplot as plt
    import librosa.display

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # Waveform
    librosa.display.waveshow(y=y, sr=sr, ax=axes[0])
    axes[0].set_title(f"{title} — Waveform")

    # Spectrogram
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    img1 = librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=axes[1])
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Spectrogram (dB)")

    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=axes[2])
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")
    axes[2].set_title("Mel Spectrogram (dB)")

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=axes[3])
    fig.colorbar(img3, ax=axes[3])
    axes[3].set_title("MFCCs")

    plt.tight_layout()
    plt.show()
