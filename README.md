# StegoShare

StegoShare is a reversible audio steganography project that combines interpolation-based embedding with secret sharing.
It provides end-to-end workflows for:
- embedding payload into audio,
- extracting payload from stego shares,
- checking extraction correctness,
- evaluating quality metrics,
- and running steganalysis.

## 1. Features

- Embedding with configurable share scheme (`n`, `k`)
- Extraction from selected stego shares
- Payload and audio comparison (original vs extracted)
- Quality evaluation (`MSE`, `SNR`, `PSNR`)
- Steganalysis:
	- ML detector benchmark (SVM)
	- Entropy + NC analysis (single and batch)
	- Excel report export

## 2. Project Structure

Key files:
- `main.py`: Main menu launcher
- `embedding.py`: Embedding module
- `extracting.py`: Extraction module
- `single_compare.py`: Compare original/extracted payload and audio
- `quality_evaluation.py`: Audio quality evaluation
- `steganalysis.py`: Steganalysis module
- `methods.py`: Core algorithm utilities

Main folders:
- `DATASET/Audio`: Cover audio files (`data1_mono.wav`, etc.)
- `DATASET/Payload`: Payload text files (`payload1.txt`, etc.)
- `STEGOAUDIO`: Generated stego shares
- `EXTRACTED`: Extracted payload/audio outputs
- `CLONING`: Temporary cloned cover audio for quality checks

## 3. Requirements

- Python 3.9+ (recommended 3.10)
- OS: Windows/Linux/macOS

Python packages used by the project:
- `numpy`
- `scipy`
- `sympy`
- `scikit-learn`
- `openpyxl`

## 4. Installation

From the project root directory:

```bash
py -m venv .venv
```

Activate virtual environment:

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
py -m pip install --upgrade pip
py -m pip install numpy scipy sympy scikit-learn openpyxl
```

## 5. How to Run

Run the main menu:

```bash
py main.py
```

Menu options in `main.py`:
1. Embedding
2. Extracting
3. Compare payload and audio
4. Check audio quality
5. Steganalysis

## 6. Usage Flow

### A. Embedding

Module: `embedding.py`

Inputs:
- Audio number (X)
- Payload number (Y)
- Shares `(n k)`

Source files:
- `DATASET/Audio/dataX_mono.wav`
- `DATASET/Payload/payloadY.txt`

Output:
- `STEGOAUDIO/stego_audioX_payloadY/stegoaudio0.wav ... stegoaudio(n-1).wav`

### B. Extracting

Module: `extracting.py`

Inputs:
- Audio number (X)
- Payload number (Y)
- Shares `(n k)`

Output:
- `EXTRACTED/stego_audioX_payloadY/payload.txt`
- `EXTRACTED/stego_audioX_payloadY/audio.wav`

### C. Compare Original vs Extracted

Module: `single_compare.py`

Checks:
- Payload equality
- Audio equality

Expected extracted files:
- `EXTRACTED/stego_audioX_payloadY/payload.txt`
- `EXTRACTED/stego_audioX_payloadY/audio.wav`

### D. Quality Evaluation

Module: `quality_evaluation.py`

Inputs:
- Audio number (X)
- Payload number (Y)
- Share number (default `0`)

Metrics:
- `MSE`
- `SNR`
- `PSNR`

### E. Steganalysis

Module: `steganalysis.py`

Modes:
1. Detector ML
2. NC and Entropy Analysis (single)
3. NC and Entropy Analysis (batch + Excel)

Batch Excel output:
- `steganalysis_results/steganalysis_auto_shares.xlsx` (or custom based on input)

## 7. Run Modules Directly (Optional)

You can run modules directly without `main.py`:

```bash
py embedding.py
py extracting.py
py single_compare.py
py quality_evaluation.py
py steganalysis.py
```

## 8. Common Issues

### 1) `Audio file not found`

Make sure file exists with exact naming format:
- `DATASET/Audio/dataX_mono.wav`

### 2) `Payload file not found`

Make sure payload file exists:
- `DATASET/Payload/payloadY.txt`

### 3) `No stego files found ...`

Make sure embedding has been run first and the folder contains WAV shares:
- `STEGOAUDIO/stego_audioX_payloadY/stegoaudio*.wav`

### 4) `Incomplete stego-audio file`

The number of generated share files must match the `n` used during extraction.

### 5) Import/module errors

Re-activate virtual environment and reinstall dependencies:

```bash
py -m pip install numpy scipy sympy scikit-learn openpyxl
```

## 9. Notes

- Use the same `(audio_no, payload_no, n, k)` configuration consistently between embedding and extracting.
- For reproducible experiments, keep dataset naming convention unchanged.
- Steganalysis batch mode expects stego files already generated inside `STEGOAUDIO`.