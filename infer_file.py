import argparse
import torch
import torchaudio
import os
import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm
from model import ParametricTCN

def main(args):
    # Forced CPU - To avoid interference with ongoing GPU training and simulate plugin environment
    DEVICE = torch.device("cpu")
    print(f"🎸 Starting SAFE inference on: {DEVICE}")

    # --- PATH HANDLING ---
    # Check if the path is absolute or relative to 'test_signals' folder
    if not os.path.exists(args.input) and not os.path.isabs(args.input):
        input_path = os.path.join("test_signals", args.input)
    else:
        input_path = args.input

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- DYNAMIC NAMING CONVENTION ---
    # Automatically generates a filename containing all important hyper-parameters
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    pot_val = args.param * 10 # Convert 0.0-1.0 to 0-10 scale for filenames
    lufs_label = abs(int(args.lufs))

    # Include pre-gain in filename to keep track of saturation experiments
    pre_gain_str = f"PRE{args.pre_gain_db:+.1f}dB" if args.pre_gain_db != 0 else "PRE0dB"
    generated_name = f"P{pot_val:.1f}_{args.version}_{input_filename}_{pre_gain_str}_LUFS{lufs_label}.wav"
    output_path = os.path.join(output_dir, generated_name)

    # 1. Model Architecture and Weights Loading
    print(f"--- Loading model version [{args.version.upper()}] ---")
    model = ParametricTCN(version=args.version, num_layers=10, hidden_ch=32).to(DEVICE)
    weights_path = os.path.join("model_weights", f"best_model_{args.version}.pth")

    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights not found at {weights_path}")
        return

    # Load weights with map_location to ensure CPU compatibility
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval() # Set to evaluation mode (freezes dropout/batchnorm)

    # 2. Input Audio Loading
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        return

    wav, sr = torchaudio.load(input_path)
    if wav.shape[0] > 1:
        wav = wav[0:1, :] # Force Mono: Deep learning models usually expect single channel [1, Samples]

    # --- 2.5 INPUT PRE-GAIN STAGE ---
    # Simulates hitting the 'analog' circuit harder (more distortion/compression)
    if args.pre_gain_db != 0.0:
        linear_gain = 10 ** (args.pre_gain_db / 20.0) # Convert dB to linear multiplier
        wav = wav * linear_gain
        print(f"🎛️ Applied Input Pre-Gain: {args.pre_gain_db:+.1f} dB (Linear multiplier: {linear_gain:.2f})")

        # Internal Clipping Check: If signal > 1.0, it might clip if the model doesn't handle it
        max_amp = torch.max(torch.abs(wav)).item()
        if max_amp > 1.0:
            print(f"⚠️ Warning: Input signal is clipping internally before the model (Max amplitude: {max_amp:.2f})")

    # 3. Block Processing with Overlap-Discard
    # We process in chunks to manage memory and simulate real-time buffer behavior
    chunk_size = 32768
    context_size = 8192 # Receptive field context to avoid clicks/artifacts at chunk boundaries
    num_samples = wav.shape[1]
    output_wav = torch.zeros_like(wav)

    print(f"⚙️ Applying effect... (Knob: {args.param:.2f}, Target: {args.lufs} LUFS)")

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, chunk_size), desc="Processing chunks"):
            end = min(start + chunk_size, num_samples)
            # Fetch context before the current chunk to initialize the TCN state
            start_with_context = max(0, start - context_size)
            actual_context_length = start - start_with_context

            # Prepare chunk for the model: [Batch, Channels, Samples]
            chunk = wav[:, start_with_context:end].unsqueeze(0).to(DEVICE)
            p_tensor = torch.tensor([[args.param]], dtype=torch.float32, device=DEVICE)

            # Inference
            out_chunk = model(chunk, p_tensor)

            # Discard the context samples - keep only the valid output samples
            valid_out = out_chunk[:, :, actual_context_length:]
            output_wav[:, start:end] = valid_out.squeeze(0).cpu()

    # Duplicate mono result to stereo for standard playback compatibility
    output_stereo = output_wav.repeat(2, 1)

    # 4. LUFS Normalization & Anti-Clipping (Acts as Makeup Gain)
    # Loudness normalization ensures consistent volume regardless of the model's internal gain
    print(f"🔊 Normalizing to {args.lufs} LUFS with Peak Safety...")
    audio_np = output_stereo.transpose(0, 1).numpy() # pyloudnorm requires [Samples, Channels]
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio_np)

    if not np.isinf(current_loudness):
        # Apply target LUFS gain
        audio_normalized = pyln.normalize.loudness(audio_np, current_loudness, args.lufs)

        # --- ANTI-CLIPPING PROTECTION ---
        # Peak normalization if the LUFS normalization pushed samples above 0dBFS
        max_peak = np.max(np.abs(audio_normalized))
        if max_peak > 0.99:
            correction = 0.99 / max_peak
            audio_normalized *= correction
            print(f"🛡️  Safety: Anti-clipping triggered (Peak was {max_peak:.2f}). Scaled to 0.99.")
    else:
        audio_normalized = audio_np
        print("⚠️ Warning: Output is silent, skipping normalization.")

    # Convert back from Numpy to PyTorch Tensor for saving
    output_final = torch.from_numpy(audio_normalized).transpose(0, 1).float()

    # 5. Save Result
    torchaudio.save(output_path, output_final, sr)
    print(f"\n✅ Success! File saved as: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Inference with Pre-Gain and Peak Safety")
    parser.add_argument("--version", type=str, required=True, choices=['v1', 'v2', 'v3'], help="Model version to use")
    parser.add_argument("--input", type=str, required=True, help="Input .wav file name (looked up in 'test_signals/')")
    parser.add_argument("--param", type=float, default=0.5, help="Potentiometer value (0.0 = Min, 1.0 = Max)")
    parser.add_argument("--lufs", type=float, default=-25.0, help="Target Integrated LUFS for the output")

    # Pre-gain argument for experimenters
    parser.add_argument("--pre_gain_db", type=float, default=0.0, help="Pre-Gain in dB to drive the model harder (saturation)")

    args = parser.parse_args()
    main(args)
