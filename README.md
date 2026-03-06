# Neural Virtual Analog Modeling of Parametric Audio Effects

This repository contains the PyTorch implementation of a Temporal Convolutional Network (TCN) designed to emulate parametric nonlinear inertial audio effects (specifically, a vacuum-tube guitar preamplifier). The model learns a black-box mapping between a dry input audio signal, a continuous control parameter (e.g., a drive knob), and the saturated output signal.

## 📁 File Structure

Plaintext

```
audio_tcn/
├── dataset.py                # Custom PyTorch Dataset for dry/wet pairs
├── model.py                  # TCN Architecture, TCNBlock, and FiLM conditioning
├── train.py                  # Core training loop utilizing auraloss
├── train_all.py              # Master script for sequential training of all models
├── generate_table.py         # Utility to generate the results table from JSON logs
├── infer_file.py             # Offline CPU inference with Overlap-Discard & LUFS
├── process_test_signals.py   # Batch processing automation for A/B/X testing
├── benchmark.py              # Comprehensive latency and CPU load profiling suite
├── infer_rt.py               # Real-time simulation and buffer throughput wrapper
├── model_weights/            # Directory for saved .pth model checkpoints
├── results/                  # Directory for JSON logs and benchmark text files
├── test_signals/             # Drop your raw, dry .wav files here
├── output/                   # Processed and normalized .wav files will appear here
└── paper/                    # LaTeX source files for the IEEE research paper
```

## 🧩 Repository Structure Detailed

- **`model.py`:** Contains the PyTorch definitions for the `ParametricTCN`. Implements the dial conditioning via Feature-wise Linear Modulation (FiLM) and causal 1D convolutions to ensure zero algorithmic latency.

- **`train.py`:** The main training logic. Includes validation steps, early stopping, and loss calculation using Error-to-Signal Ratio and pre-emphasis filters via `auraloss`.

- **`train_all.py` & `generate_table.py`:** Pipeline scripts designed to sequentially train multiple model architectures and dynamically generate validation metric tables for paper benchmarking.

- **`infer_file.py`:** Offline audio processing optimized for file-to-file rendering on the CPU. It implements an Overlap-Discard chunking method to prevent boundary artifacts (clicks) and uses ITU-R BS.1770-4 LUFS normalization to ensure level-matched outputs for psychoacoustic evaluation.

- **`process_test_signals.py`:** Batch wrapper that sweeps through `test_signals/`, processing every audio file across all specified models and control parameters.

- **`benchmark.py` & `infer_rt.py`:** Experimental real-time simulation scripts. They load audio into RAM and stream it through the models in small discrete buffers (e.g., 64, 128, or 256 samples) to measure processing time ($T_{proc}$) and CPU load limits. This serves as a theoretical baseline for future C++ DSP ports.

## 🧠 Model Architectures

The repository evaluates three distinct architectural variations to balance parameter count, computational efficiency, and sonic accuracy:

| **Parameter / Feature**     | **V1 (Base)** | **V2 (Mix)** | **V3 (Full)** |
| --------------------------- | ------------- | ------------ | ------------- |
| **Residual Blocks**         | 10            | 10           | 10            |
| **1x1 Projection**          | No            | Yes          | Yes           |
| **Global Skip Connections** | No            | No           | Yes           |
| **Parameter Conditioning**  | Concat        | Concat       | FiLM          |
| **Est. Parameter Count**    | ~ 31k         | ~ 32k        | ~ 40k         |

## 🛠️ Requirements

Ensure you have the required dependencies installed. Python 3.8+ is recommended.

```bash
pip install torch torchaudio auraloss tqdm pyloudnorm numpy
```

## 🚀 Training & Benchmarking

### 1. Single Model Training (`train.py`)

Use this for focused training on a specific architecture. The script now utilizes AdamW for superior generalization and includes an Early Stopping mechanism to prevent over-fitting and save compute.

```bash
python train.py --version v3 --epochs 300 --batch_size 64 --lr 1e-4 --weight_decay 1e-3 --patience 30
```

**Key Options:**

- `--version`: Choose the TCN architecture (`v1`, `v2`, or `v3`).

- `--epochs`: Maximum iterations. Convergence is typically stable by epoch 100, but 300 is recommended for deep fine-tuning.

- `--lr` & `--weight_decay`: Sets the learning rate and AdamW regularization. Decoupled weight decay helps maintain harmonic accuracy during plateaus.

- `--patience`: Early stopping threshold. If validation loss doesn't improve for $N$ epochs, training terminates (default: 30).

- `--data_dir`: Path to the folder containing your `input/` and `target/` `.wav` files.

### 2. The Training Marathon (`train_all.py`)

The "Paper Mode" designed for automated benchmarking. It sequentially trains all three model versions (V1 → V2 → V3) and tracks performance metrics required for publication.

```bash
python train_all.py --epochs 300 --batch_size 32 --data_dir "../BIG_DATASET"
```

**What it automates:**

- **Full Pipeline:** Handles V1, V2, and V3 sequentially without manual restarts.

- **Metric Tracking:** Records total training time and average epoch latency for computational complexity analysis.

- **Auto-Reporting:** Upon completion, it automatically scrapes `results/*.json` and generates a comprehensive summary table including architecture details and timing stats.

- **Persistence:** Saves the final report to `results/training_all_results.txt`.

### 3. Result Visualization & Regeneration

- **`plot_results.py`:** Generates a publication-ready convergence plot comparing the validation loss of all trained versions.

- **`generate_table.py`:** Only needed if you manually modify or delete individual JSON logs and need to regenerate the summary table without retraining the entire marathon.

## 🧪 Loss Functions

The training utilizes a hybrid loss strategy specifically tuned for vacuum-tube emulation:

- **ESR (Error-to-Signal Ratio):** Time-domain accuracy focusing on the waveshaping and clipping characteristics.

- **Multi-Resolution STFT:** Frequency-domain accuracy ensuring the harmonic content (overtones) matches the original preamp.

The weights are fixed at 0.1 (ESR) and 0.9 (STFT) as established in the methodology.

## 🎧 Inference Processing

### 1. File-Based Inference

Use this for high-quality rendering of individual audio files. The chunking mechanism prevents RAM spikes on long files. Outputs are automatically duplicated to stereo and normalized to a target LUFS level.

> **Important:** If your input DI tracks have a lot of headroom and are too quiet to trigger the TCN's non-linear saturation (similar to plugging a quiet guitar into a real amplifier), use the `--pre_gain_db` argument to drive the signal harder into the model.

```bash
python infer_file.py --version v3 --input guitar_dry.wav --param 0.7 --pre_gain_db 12.0 --lufs -10.0
```

*(Note: The `--output` argument has been removed. Files are automatically saved in the `output/` directory with a dynamic naming convention, e.g., `P7.0_v3_guitar_dry_PRE+12.0dB_LUFS10.wav`)*

### 2. Batch Processing for Subjective Tests

When preparing audio samples for MUSHRA or A/B listening tests, use this script to process an entire folder of dry signals automatically. The script propagates your Pre-Gain and LUFS settings to all processed files.

```bash
python process_test_signals.py --models v1 v2 v3 --param 0.5 --pre_gain_db 12.0 --lufs -10.0
```

*Note: Ensure your dry files are placed in the `test_signals/` directory.*

## ⚡ Real-Time Simulation & Profiling

To evaluate the true real-time feasibility of the trained models on your specific hardware, run the profiling suite. The benchmark script automatically detects your system specifications (CPU, OS), forces CPU-bound inference (to accurately simulate a standard VST/AU audio plugin environment), and calculates buffer latencies to identify dropout risks.

```bash
python benchmark.py
```

*Note: The script evaluates all model versions (V1, V2, V3) across multiple buffer sizes and automatically saves a detailed system-specific report to `results/benchmark_results_[your_hostname].txt`.*

**Example Output:**

```
==============================================================================================================
🔬 COMPREHENSIVE BENCHMARK: ACCURACY VS. REAL-TIME CAPABILITIES (FS: 48.0kHz)
==============================================================================================================
💻 SYSTEM SPECIFICATIONS:
   OS:        Linux 6.19.5-3-cachyos (x86_64)
   Node Name: PC-SC6048-NiBe2
   Processor: AMD Ryzen 5 9600X 6-Core Processor
   PyTorch:   2.5.1+cu121
   Device:    CPU (Forced for audio plugin simulation)
==============================================================================================================

📦 MODEL: V3 | Parameters: 49,665 | Best Val Loss: 0.486662
--------------------------------------------------------------------------------------------------------------
Buffer     | Buff Latency    | Proc Time       | Total Latency      | CPU Load     | Status
--------------------------------------------------------------------------------------------------------------
64         |       1.33 ms |       0.68 ms |          2.02 ms |       51.4% | ✅ SAFE RT
128        |       2.67 ms |       0.74 ms |          3.40 ms |       27.6% | ✅ SAFE RT
256        |       5.33 ms |       0.81 ms |          6.15 ms |       15.3% | ✅ SAFE RT
512        |      10.67 ms |       0.95 ms |         11.62 ms |        8.9% | ✅ SAFE RT
👉 Conclusion for V3: Suitable for real-time from buffer size 64 and above.
```

Alternatively, you can measure a specific model and buffer size explicitly:

```bash
python infer_rt.py --version v3 --buffer 128
```

## 📐 Mathematical Context & Receptive Field

The TCN architecture is strictly causal. With $L=10$ layers and a kernel size of $k=5$, the model possesses a receptive field of 4093 samples.

- At **44.1 kHz**, this equates to **~92.8 ms** of temporal memory.

- At **48.0 kHz**, this equates to **~85.3 ms**.

This window is sufficient to capture both the rapid nonlinear clipping of vacuum tubes and the slower envelope characteristics of power supply sag in vintage amplifiers.

## 📊 Performance Trade-offs

| **Version** | **Conditioning** | **Reasoning**                                                                      | **Complexity** |
| ----------- | ---------------- | ---------------------------------------------------------------------------------- | -------------- |
| **V1**      | Input Concat     | Simple baseline, global bias.                                                      | Lowest         |
| **V2**      | Input Concat     | Improved non-linear reconstruction via deeper output head.                         | Medium         |
| **V3**      | FiLM             | Dynamic modulation of intermediate features; best for modeling sweepable controls. | Highest        |

## 💡 Pro-Tips for Best Results

- **Gain Staging:** If the model doesn't distort enough, your input files are likely too quiet. Use `--pre_gain_db 12` to hit the "virtual tubes" harder.

- **Phase Alignment:** Ensure your input and target files are sample-aligned. Even a 1ms offset will cause the ESR loss to fail, resulting in a dull, phasey-sounding model.

- **Training Time:** V3 takes ~20% longer to train than V1 but typically achieves a 15-20% lower Multi-Resolution STFT loss.
