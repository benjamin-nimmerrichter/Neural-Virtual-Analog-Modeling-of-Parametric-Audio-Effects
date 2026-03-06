import os
import argparse
import subprocess
import sys

def main(args):
    """
    Batch processes all audio files in a directory through multiple TCN model versions.
    Acts as a wrapper around infer_file.py to automate large-scale testing.
    """
    # Define and ensure standard project directories exist
    input_dir = "test_signals"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. FILE GATHERING
    # Automatically finds all .wav files in the input folder
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"⚠️ No .wav files found in '{input_dir}'.")
        return

    # Dashboard display of the current batch configuration
    print(f"🔍 Found {len(wav_files)} files to process.")
    print(f"🚀 Starting batch processing for models: {args.models} | Knob: {args.param} | Pre-Gain: {args.pre_gain_db:+.1f} dB | LUFS: {args.lufs}")
    print("-" * 70)

    # 2. NESTED LOOP PROCESSING
    # We iterate through every file and for EACH file, we run every requested model version.
    for wav_file in wav_files:
        for model_ver in args.models:
            print(f"\n🎧 Processing: {wav_file} -> Model {model_ver.upper()}")

            # 3. SUBPROCESS COMMAND CONSTRUCTION
            # We use sys.executable to ensure we use the same Python environment/interpreter.
            # We are calling 'infer_file.py' as a separate process for each task.
            cmd = [
                sys.executable, "infer_file.py",
                "--version", model_ver,
                "--input", wav_file,
                "--param", str(args.param),
                "--lufs", str(args.lufs),
                "--pre_gain_db", str(args.pre_gain_db) # Passing saturation parameter
            ]

            try:
                # subprocess.run waits for the inference to finish before moving to the next one.
                # check=True will raise an exception if infer_file.py crashes.
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"❌ Failed processing {wav_file} with model {model_ver}.")
            except KeyboardInterrupt:
                # Graceful exit if the user presses Ctrl+C
                print("\n🛑 Batch processing interrupted by user.")
                return

    # Final wrap-up report
    print(f"\n{'='*70}")
    print(f"✅ BATCH COMPLETE! Check the '{output_dir}' folder for results.")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process test signals through TCN models")

    # Allows passing multiple models, e.g., --models v1 v3
    parser.add_argument("--models", nargs="+", default=["v1", "v2", "v3"],
                        help="List of model versions to process (default: v1 v2 v3)")

    # Global parameter for the 'knob' (e.g., gain/distortion amount)
    parser.add_argument("--param", type=float, default=0.5,
                        help="Parameter/Potentiometer value 0.0-1.0 (default: 0.5)")

    # Target volume for all generated files
    parser.add_argument("--lufs", type=float, default=-25.0,
                        help="Target LUFS for normalization (default: -25.0)")

    # Input gain control (to test how the model reacts to hot/quiet signals)
    parser.add_argument("--pre_gain_db", type=float, default=0.0,
                        help="Pre-Gain in dB to hit the model harder (default: 0.0)")

    args = parser.parse_args()
    main(args)
