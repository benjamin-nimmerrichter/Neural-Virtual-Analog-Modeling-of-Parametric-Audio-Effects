import os
import argparse
import torch
import torchaudio
import pyloudnorm as pyln
import numpy as np
from tqdm import tqdm

def normalize_audio(input_path, output_path, target_lufs):
    """
    Measures the integrated LUFS of an audio file and normalizes it.
    Integrated LUFS is a measurement of perceived loudness over the entire file.
    Includes an anti-clipping safety mechanism that falls back to
    peak normalization if the target LUFS pushes peaks above 0 dBFS.
    """
    try:
        # Load audio using torchaudio: [Channels, Samples]
        waveform, sr = torchaudio.load(input_path)

        # pyloudnorm expects the shape: (samples, channels)
        audio_np = waveform.numpy().T

        # Initialize the BS.1770-4 compliant loudness meter
        meter = pyln.Meter(sr)

        # 1. Measure the original integrated loudness
        current_lufs = meter.integrated_loudness(audio_np)

        # 2. Calculate the gain difference needed to reach the target
        gain_db = target_lufs - current_lufs
        # Convert dB gain to linear multiplier: power(10, dB/20)
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Apply the gain to the audio samples
        normalized_audio_np = audio_np * gain_linear

        # --- 3. ANTI-CLIPPING PROTECTION ---
        # If the loudness gain pushes the samples above digital 0 dBFS (amplitude 1.0),
        # the audio will distort (clip). We detect and prevent this.
        max_peak = np.max(np.abs(normalized_audio_np))
        warning_flag = False

        if max_peak > 0.99:
            # Scale down the entire signal so the highest peak is at exactly 0.99 (-0.1 dBFS)
            # This is "Peak Normalization" acting as a safety ceiling.
            correction_factor = 0.99 / max_peak
            normalized_audio_np = normalized_audio_np * correction_factor
            warning_flag = True
        # --------------------------------

        # Convert back to torch tensor: swap shape back to (channels, samples)
        normalized_waveform = torch.from_numpy(normalized_audio_np.T).float()

        # Save the normalized reference file
        torchaudio.save(output_path, normalized_waveform, sr)

        return True, current_lufs, warning_flag, max_peak

    except Exception as e:
        print(f"\n❌ Error processing {os.path.basename(input_path)}: {e}")
        return False, None, False, 0.0

def main():
    parser = argparse.ArgumentParser(description="Batch LUFS Normalization with Peak Safety")
    parser.add_argument("--input_dir", type=str, default="test_signals",
                        help="Directory containing raw reference .wav files")
    parser.add_argument("--output_dir", type=str, default="normalized_references",
                        help="Directory to save normalized files")
    parser.add_argument("--lufs", type=float, default=-25.0,
                        help="Target LUFS level (default: -25.0)")
    args = parser.parse_args()

    # Ensure output directory exists for organized results
    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all .wav files from the input directory
    wav_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"⚠️ No .wav files found in '{args.input_dir}'.")
        return

    # Header for the batch process
    log_width = 75
    print(f"\n{'='*log_width}")
    print(f"🌊 REFERENCE NORMALIZATION | Target: {args.lufs} LUFS | Peak Limit: 0.99")
    print(f"{'='*log_width}\n")

    success_count = 0
    clipping_prevented = 0

    # Progress bar for visual feedback
    pbar = tqdm(wav_files, desc="Normalizing")
    for filename in pbar:
        input_path = os.path.join(args.input_dir, filename)

        # DYNAMIC NAMING: e.g., 'REF_clean_guitar_LUFS25.wav'
        # This helps in ABX blind tests to identify the reference and its target loudness.
        name_only = os.path.splitext(filename)[0]
        lufs_label = abs(int(args.lufs))
        out_filename = f"REF_{name_only}_LUFS{lufs_label}.wav"
        output_path = os.path.join(args.output_dir, out_filename)

        success, original_lufs, was_clipped, peak = normalize_audio(input_path, output_path, args.lufs)

        if success:
            success_count += 1
            if was_clipped:
                clipping_prevented += 1
            # Update progress bar with real-time stats
            pbar.set_postfix(orig_lufs=f"{original_lufs:.1f}", clipped=was_clipped)

    # Final summary report
    print(f"\n{'='*log_width}")
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"✅ Successfully normalized {success_count}/{len(wav_files)} files.")
    if clipping_prevented > 0:
        print(f"🛡️  Anti-clipping triggered on {clipping_prevented} references (Peak limited to 0.99).")
    print(f"📁 Files saved to: {args.output_dir}/")
    print(f"{'='*log_width}\n")

if __name__ == "__main__":
    main()
