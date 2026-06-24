import os
import math
import numpy as np
import scipy.io.wavfile as scp
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ==============================================================================
# CONFIGURASI
# ==============================================================================

COVER_PATH = 'DATASET/Audio/data1_mono.wav'
STEGO_BASE = 'STEGOAUDIO'
SEGMENT_LENGTH = 4096       # sampel per segment (~93ms @ 44.1kHz)
SEGMENT_OVERLAP = 0.5      # 50% overlap
RANDOM_STATE = 42
TEST_SIZE = 0.3
BALANCE_CLASSES = True   # Undersample stego agar seimbang dengan cover (evaluasi lebih adil)


# ==============================================================================
# LOAD AUDIO
# ==============================================================================

def load_audio_samples(filepath):
    """Baca WAV, return (rate, samples float64)."""
    rate, data = scp.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data.astype(np.float64)


def build_interpolated_reference(cover_samples):
    n = len(cover_samples)
    ref = np.zeros(2 * n - 1)
    ref[::2] = cover_samples
    ref[1::2] = np.floor((cover_samples[:-1] + cover_samples[1:]) / 2.0)
    return ref


# ==============================================================================
# SEGMENTATION
# ==============================================================================

def get_segments(samples, seg_len, overlap_ratio=0.5):
    step = int(seg_len * (1 - overlap_ratio))
    if step < 1:
        step = 1
    segments = []
    for start in range(0, len(samples) - seg_len + 1, step):
        segments.append(samples[start:start + seg_len])
    return segments


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_statistical_features(segment):
    return [
        np.mean(segment),
        np.std(segment) if np.std(segment) > 0 else 1e-10,
        skew(segment),
        kurtosis(segment),
        np.median(segment),
        np.max(np.abs(segment)),
    ]


def extract_spectral_features(segment):
    fft_mag = np.abs(fft(segment))[:len(segment) // 2]
    if len(fft_mag) < 10:
        return [0.0] * 8

    n = len(fft_mag)
    band_size = n // 4
    bands = [
        fft_mag[:band_size],
        fft_mag[band_size:2*band_size],
        fft_mag[2*band_size:3*band_size],
        fft_mag[3*band_size:],
    ]
    features = []
    for band in bands:
        if len(band) > 0:
            features.append(np.mean(band))
            features.append(np.std(band) if np.std(band) > 0 else 1e-10)
        else:
            features.extend([0.0, 0.0])
    return features[:8]


def extract_difference_features(segment):
    diff = np.diff(segment.astype(np.int64))
    if len(diff) < 2:
        return [0.0] * 6
    return [
        np.mean(np.abs(diff)),
        np.std(diff) if np.std(diff) > 0 else 1e-10,
        skew(diff),
        np.median(np.abs(diff)),
        np.sum(diff == 0) / len(diff),  # proporsi diff=0
        np.sum(np.abs(diff) == 1) / len(diff),  # proporsi |diff|=1 (LSB-related)
    ]


def extract_histogram_moments(segment, n_bins=32):
    hist, _ = np.histogram(segment, bins=n_bins, density=True)
    hist = hist + 1e-10
    hist = hist / np.sum(hist)
    return [
        np.mean(hist),
        np.std(hist),
        skew(hist),
    ]


def extract_features_for_segment(segment):
    feats = []
    feats.extend(extract_statistical_features(segment))
    feats.extend(extract_spectral_features(segment))
    feats.extend(extract_difference_features(segment))
    feats.extend(extract_histogram_moments(segment))
    return np.array(feats, dtype=np.float64)


def extract_features_for_audio(samples, seg_len, overlap):
    segments = get_segments(samples, seg_len, overlap)
    return np.array([extract_features_for_segment(seg) for seg in segments])


# ==============================================================================
# BUILD DATASET
# ==============================================================================

def collect_cover_and_stego_files():
    cover = COVER_PATH
    stego_files = []

    # Cari folder hasil embedding, misalnya stego_audio1_payload1.
    base_dir = os.path.dirname(STEGO_BASE)
    if not os.path.exists(base_dir):
        return cover, stego_files

    for name in os.listdir(base_dir):
        if name.startswith('stego_audio'):
            folder = os.path.join(base_dir, name)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if f.endswith('.wav'):
                        stego_files.append(os.path.join(folder, f))

    return cover, stego_files


def build_dataset():
    cover_path, stego_paths = collect_cover_and_stego_files()

    if not os.path.exists(cover_path):
        raise FileNotFoundError(f"audio cover not found: {cover_path}")

    if not stego_paths:
        raise FileNotFoundError(
            f"No stego files found. Please ensure folders like "
            f"STEGOAUDIO/stego_audio1_payload1/ contain stegoaudio*.wav"
        )

    rate_c, cover_raw = load_audio_samples(cover_path)

    cover_samples = build_interpolated_reference(cover_raw)

    X_list = []
    y_list = []

    # Cover segments -> label 0
    X_cover = extract_features_for_audio(cover_samples, SEGMENT_LENGTH, SEGMENT_OVERLAP)
    X_list.append(X_cover)
    y_list.append(np.zeros(len(X_cover), dtype=int))

    # Stego segments -> label 1
    for sp in stego_paths:
        rate_s, stego_samples = load_audio_samples(sp)

        min_len = min(len(cover_samples), len(stego_samples))
        if min_len < SEGMENT_LENGTH:
            continue
        stego_trunc = stego_samples[:min_len]
        X_stego = extract_features_for_audio(stego_trunc, SEGMENT_LENGTH, SEGMENT_OVERLAP)
        X_list.append(X_stego)
        y_list.append(np.ones(len(X_stego), dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if BALANCE_CLASSES:
        idx_cover = np.where(y == 0)[0]
        idx_stego = np.where(y == 1)[0]
        n_cover = len(idx_cover)
        n_stego = len(idx_stego)
        if n_stego > n_cover:
            idx_stego_down = resample(
                idx_stego, replace=False, n_samples=n_cover, random_state=RANDOM_STATE
            )
            idx_keep = np.concatenate([idx_cover, idx_stego_down])
            np.random.RandomState(RANDOM_STATE).shuffle(idx_keep)
            X = X[idx_keep]
            y = y[idx_keep]

    return X, y, cover_path, stego_paths


def build_single_pair_dataset(cover_path, stego_path):
    """Build a balanced segment dataset from one GUI-selected audio pair."""
    if not os.path.isfile(cover_path):
        raise FileNotFoundError(f"Cover audio not found: {cover_path}")
    if not os.path.isfile(stego_path):
        raise FileNotFoundError(f"Stego audio not found: {stego_path}")

    _, cover_raw = load_audio_samples(cover_path)
    _, stego = load_audio_samples(stego_path)

    # The embedding method normally produces a 2N-1 stego signal.
    cover = (build_interpolated_reference(cover_raw)
             if len(stego) >= len(cover_raw) * 1.8 else cover_raw)
    min_len = min(len(cover), len(stego))
    if min_len < SEGMENT_LENGTH:
        raise ValueError(
            f"Audio too short for detector (minimum {SEGMENT_LENGTH} samples)."
        )

    X_cover = extract_features_for_audio(
        cover[:min_len], SEGMENT_LENGTH, SEGMENT_OVERLAP)
    X_stego = extract_features_for_audio(
        stego[:min_len], SEGMENT_LENGTH, SEGMENT_OVERLAP)
    pair_count = min(len(X_cover), len(X_stego))
    if pair_count < 3:
        raise ValueError("Not enough audio segments for training/testing the detector.")

    X = np.vstack((X_cover[:pair_count], X_stego[:pair_count]))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate((
        np.zeros(pair_count, dtype=int),
        np.ones(pair_count, dtype=int),
    ))
    return X, y


def evaluate_audio_pair(cover_path, stego_path):
    """Run the detector benchmark for one cover/stego pair and return metrics."""
    X, y = build_single_pair_dataset(cover_path, stego_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, clf.decision_function(X_test_scaled))
    except ValueError:
        auc = 0.0

    class_count = int(np.min(np.bincount(y)))
    folds = min(5, class_count)
    cv_mean = None
    cv_std = None
    if folds >= 2:
        full_scaler = StandardScaler()
        X_scaled = full_scaler.fit_transform(X)
        cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_scaled, y, cv=cv)
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'segments_per_class': class_count,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


# ==============================================================================
# TRAIN & EVALUATE
# ==============================================================================

def run_detector_experiment():
    """Run the detector experiment: train SVM, evaluate, print report."""
    print("=" * 70)
    print("   STEGANALYSIS DETECTOR-BASED EVALUATION (ML Benchmark)")
    print("   Audio: dataspeech_mono.wav")
    print("=" * 70)

    print("\n[1] Building dataset...")
    X, y, cover_path, stego_paths = build_dataset()
    n_cover = np.sum(y == 0)
    n_stego = np.sum(y == 1)
    print(f"    Cover segments : {n_cover}")
    print(f"    Stego segments : {n_stego}")
    print(f"    Total samples  : {len(y)}")
    print(f"    Feature dim    : {X.shape[1]}")
    print(f"    Stego files    : {len(stego_paths)}")

    # Handle class imbalance: stratify
    if n_cover == 0 or n_stego == 0:
        print("\n[ERROR] Not enough data to train/evaluate the detector. Please ensure both cover and stego segments are present.")
        return

    print("\n[2] Split train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[3] Training SVM (RBF kernel)...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\n[4] Evaluation...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        y_prob = clf.decision_function(X_test_scaled)
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    print("\n[5] Cross-validation (5-fold stratified)...")
    X_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # --- LAPORAN ---
    print("\n" + "=" * 70)
    print("   RESULTS OF DETECTOR-BASED STEGANALYSIS EXPERIMENT")
    print("=" * 70)
    print(f"\n  Confusion Matrix (Test Set):")
    print(f"                   Predicted")
    print(f"                 Cover  Stego")
    print(f"    Actual Cover   {cm[0,0]:>4}    {cm[0,1]:>4}")
    print(f"    Actual Stego   {cm[1,0]:>4}    {cm[1,1]:>4}")
    print(f"\n  Metrik (Test Set):")
    print(f"    Accuracy       : {acc*100:.2f}%")
    print(f"    Precision      : {prec*100:.2f}%")
    print(f"    Recall         : {rec*100:.2f}%")
    print(f"    F1-Score       : {f1*100:.2f}%")
    print(f"    AUC-ROC        : {auc*100:.2f}%")
    print(f"\n  Cross-Validation (5-fold):")
    print(f"    Mean Accuracy  : {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    print("\n" + "-" * 70)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm,
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    try:
        run_detector_experiment()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nEnsure the following files are present:")
        print("  1. Cover audio file: stegoaudioDataset/Audio/dataspeech_mono.wav")
        print("  2. Run embedding first to generate stego audio:")
        print("     STEGOAUDIO/stego_audio1_payload1/stegoaudio0.wav, ...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
