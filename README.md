# StegoShare

StegoShare is a terminal-based reversible audio steganography application that
combines interpolation-based embedding with Shamir Secret Sharing. In the
current project version, the implementation has been consolidated into one main
Python file: `main.py`.

## Features

- Embed binary payloads into cover audio using an `(n, k)` secret-sharing scheme.
- Generate multiple stego-audio shares: `stegoaudio0.wav`, `stegoaudio1.wav`, and so on.
- Extract the payload and reconstructed cover audio from at least `k` stego-audio shares.
- Compare the original payload/audio with the extracted payload/audio.
- Evaluate stego-audio quality using `MSE`, `SNR`, and `PSNR`.
- Report embedding runtime and peak memory usage.
- Run steganalysis:
  - SVM-based Machine Learning detector,
  - entropy and Normalized Correlation (NC) analysis for a single pair,
  - batch entropy and NC analysis with Excel export.

## Current Project Structure

Source files:

- `main.py`: all application logic, menus, embedding, extraction, comparison, quality evaluation, and steganalysis.
- `README.md`: project documentation.
- `LICENSE`: project license.

Input data:

- `DATASET/Audio`: cover audio files, for example `data1_mono.wav`.
- `DATASET/Payload`: payload files, for example `payload1.txt`.

Generated output folders:

- `results/STEGOAUDIO`: generated stego-audio shares.
- `results/EXTRACTED`: extracted payload and reconstructed audio.
- `results/CLONING`: temporary cloned cover audio for quality evaluation.
- `results/STEGANALYSIS`: Excel reports from batch steganalysis.

The `results` folders are created automatically when the related workflow is run.

## Requirements

- Python 3.9 or newer. Python 3.10 is recommended.
- Python packages:
  - `numpy`
  - `scipy`
  - `sympy`
  - `scikit-learn`
  - `openpyxl`

## Installation

From the project root:

```bash
py -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bash
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
py -m pip install --upgrade pip
py -m pip install numpy scipy sympy scikit-learn openpyxl
```

## Run

Run the application from the project root:

```bash
py -3 main.py
```

Main menu:

```text
1. Embedding
2. Extracting
3. Evaluation
4. Exit
```

Evaluation submenu:

```text
1. Compare payload and audio
2. Check audio quality
3. Detector ML
4. NC and entropy analysis (single)
5. NC and entropy analysis (batch + excel)
```

## Workflow

### 1. Embedding

Select:

```text
1. Embedding
```

Inputs:

- Audio number `X`.
- Payload number `Y`.
- Total and minimum shares in `(n k)` format.

Input files:

```text
DATASET/Audio/dataX_mono.wav
DATASET/Payload/payloadY.txt
```

Example:

```text
Enter audio number from DATASET: 1
Enter payload number from DATASET: 1
Enter total and minimum shares (n k): 5 3
```

Output:

```text
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio0.wav
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio1.wav
...
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio(n-1).wav
```

The program also prints the output folder, peak memory usage, and embedding runtime.

### 2. Extracting

Select:

```text
2. Extracting
```

Inputs:

- Audio number `X`.
- Payload number `Y`.
- Total and minimum shares in `(n k)` format.

Required input files:

```text
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio0.wav
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio1.wav
...
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudio(n-1).wav
```

Output:

```text
results/EXTRACTED/stego_audioX_payloadY/payload.txt
results/EXTRACTED/stego_audioX_payloadY/audio.wav
```

During extraction, the program randomly selects `k` shares from the available `n` shares.

### 3. Compare Payload and Audio

Select:

```text
3. Evaluation
1. Compare payload and audio
```

Compared files:

- Original payload: `DATASET/Payload/payloadY.txt`
- Extracted payload: `results/EXTRACTED/stego_audioX_payloadY/payload.txt`
- Original audio: `DATASET/Audio/dataX_mono.wav`
- Extracted audio: `results/EXTRACTED/stego_audioX_payloadY/audio.wav`

The output shows whether the payload and audio are `Exact` or `Different`.

### 4. Quality Evaluation

Select:

```text
3. Evaluation
2. Check audio quality
```

Inputs:

- Audio number `X`.
- Payload number `Y`.
- Stego-audio share number, default `0`.

Input files:

```text
DATASET/Audio/dataX_mono.wav
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudioS.wav
```

Output metrics:

- `MSE`
- `SNR`
- `PSNR`

The program also creates a cloned audio file:

```text
results/CLONING/data_clone_audioX.wav
```

### 5. Detector ML

Select:

```text
3. Evaluation
3. Detector ML
```

The detector uses an SVM model with statistical, spectral, difference, and
histogram-moment features. Run embedding first so `results/STEGOAUDIO` contains
`stegoaudio*.wav` files.

Main output:

- number of cover and stego segments,
- confusion matrix,
- accuracy,
- precision,
- recall,
- F1-score,
- AUC-ROC,
- cross-validation accuracy.

### 6. Single NC and Entropy Analysis

Select:

```text
3. Evaluation
4. NC and entropy analysis (single)
```

Inputs:

- Audio number `X`.
- Payload number `Y`.
- Stego-audio share number, default `0`.

Input stego file:

```text
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudioS.wav
```

Output:

- cover entropy,
- stego entropy,
- entropy difference,
- entropy change percentage,
- Normalized Correlation (NC).

### 7. Batch NC and Entropy Analysis + Excel

Select:

```text
3. Evaluation
5. NC and entropy analysis (batch + excel)
```

Inputs:

- total audio files, default `15`,
- total payload files, default `11`,
- total shares `n`, or leave blank for auto-detection.

The batch process scans stego files in:

```text
results/STEGOAUDIO/stego_audioX_payloadY/stegoaudioS.wav
```

Default Excel output:

```text
results/STEGANALYSIS/steganalysis_auto_shares.xlsx
```

If total shares is provided, the output filename follows that value:

```text
results/STEGANALYSIS/steganalysis_5_shares.xlsx
```

Generated Excel sheets:

- `Share Details`
- `Summary (Avg)`
- `Summary (Best)`
- `Summary (Worst)`
- pivot sheets per metric and share
- `Metric Notes`

## Notes

- Use the same `(audio_no, payload_no, n, k)` configuration for embedding, extraction, comparison, quality evaluation, and steganalysis.
- `n` is the total number of shares generated during embedding.
- `k` is the minimum number of shares required for reconstruction.
- CLI extraction randomly selects `k` shares from the available `n` shares.
- Batch steganalysis only processes audio-payload combinations whose stego-audio files already exist.
- Generated outputs can be recreated by running the workflow again.

## Troubleshooting

### `Audio file not found`

Make sure the cover audio file exists:

```text
DATASET/Audio/dataX_mono.wav
```

### `Payload file not found`

Make sure the payload file exists:

```text
DATASET/Payload/payloadY.txt
```

### `Incomplete stego-audio file`

The number of stego-audio files must match the `n` value used during extraction.

Example for `n = 5`:

```text
stegoaudio0.wav
stegoaudio1.wav
stegoaudio2.wav
stegoaudio3.wav
stegoaudio4.wav
```

### `No stego files found`

Run embedding first so this folder contains stego-audio files:

```text
results/STEGOAUDIO/
```

### Import or dependency error

Activate the virtual environment, then reinstall the dependencies:

```bash
py -m pip install numpy scipy sympy scikit-learn openpyxl
```
