import time
import torch
import argparse
import os
from model import ParametricTCN

def profile_rt(version, buffer_size=128, fs=48000, log_func=print):
    """
    Profiles the real-time performance of a specific model version.
    Simulates a single buffer processing cycle as it would occur in an audio DAW.
    Calculates processing time (T_proc), buffer latency (L_buff), and relative CPU load.
    """
    DEVICE = torch.device("cpu") # Real-time audio processing almost exclusively uses CPU

    # Initialize model based on version
    try:
        model = ParametricTCN(version=version, num_layers=10, hidden_ch=32).to(DEVICE)

        # Load weights if available to ensure the model is in a 'production-ready' state
        weights_path = os.path.join("model_weights", f"best_model_{version}.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        else:
            log_func(f"⚠️ Warning: Weights not found at {weights_path}. Profiling with random initialization.")

        model.eval() # Evaluation mode is crucial for consistent timing (no dropout)
    except Exception as e:
        log_func(f"❌ Error loading model {version}: {e}")
        return None

    # Dummy data for a single buffer slice: [Batch=1, Channels=1, Samples=buffer_size]
    input_buffer = torch.randn(1, 1, buffer_size).to(DEVICE)
    p_tensor = torch.full((1, 1), 0.5).to(DEVICE)

    # --- WARM-UP PHASE ---
    # Modern CPUs fluctuate in clock speed. We run 50 iterations to "wake up" the
    # processor and fill the instruction caches for stable measurements.
    for _ in range(50):
        with torch.no_grad():
            _ = model(input_buffer, p_tensor)

    # --- PRECISE MEASUREMENT ---
    # Measure the processing time (T_proc) over 1000 runs to get a reliable average
    num_runs = 1000
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_buffer, p_tensor)
    end_time = time.perf_counter()

    # T_proc = Average time taken to process one buffer in milliseconds
    t_proc = ((end_time - start_time) / num_runs) * 1000  # in ms

    # --- LATENCY CALCULATIONS ---
    # L_buff: Constant physical delay caused by the buffer size.
    # Formula: (Samples / Sample Rate) * 1000
    l_buff = (buffer_size / fs) * 1000  # ms

    # L_algo: Algorithmic latency. TCN with causal convolutions has 0.0ms.
    l_algo = 0.0

    # Total system latency experienced by the musician/user
    l_total = l_buff + t_proc + l_algo

    # --- CPU LOAD (DEADLINE CHECK) ---
    # For real-time audio, T_proc MUST be shorter than L_buff.
    # If load > 100%, the CPU cannot finish processing before the next buffer is needed.
    load = (t_proc / l_buff) * 100

    return {
        "ver": version.upper(),
        "t_proc": t_proc,
        "l_buff": l_buff,
        "l_total": l_total,
        "load": load
    }

def main():
    parser = argparse.ArgumentParser(description="Real-time Profiling for all model versions")
    parser.add_argument("--buffer", type=int, default=128, help="Buffer size in samples (64, 128, 256, 512...)")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz (44100, 48000)")
    args = parser.parse_args()

    versions = ['v1', 'v2', 'v3']
    results = []

    # Ensure results directory exists for logging performance metrics
    os.makedirs("results", exist_ok=True)
    output_lines = []

    def log(text):
        """Helper to output to both terminal and internal log list."""
        print(text)
        output_lines.append(text)

    # Dashboard Header
    log(f"\n{'='*85}")
    log(f"🚀 PROFILING SUITE | Buffer: {args.buffer} | Sample Rate: {args.fs/1000}kHz")
    log(f"{'='*85}")
    log(f"{'Model':<10} | {'Proc Time':<12} | {'Buff Delay':<12} | {'Total Latency':<15} | {'CPU Load':<10}")
    log(f"{'-'*85}")

    # Benchmark each model version
    for v in versions:
        res = profile_rt(v, buffer_size=args.buffer, fs=args.fs, log_func=log)
        if res:
            log(f"{res['ver']:<10} | {res['t_proc']:>8.2f} ms | {res['l_buff']:>8.2f} ms | {res['l_total']:>11.2f} ms | {res['load']:>8.1f}%")
            results.append(res)

    log(f"{'='*85}")

    # --- RESEARCH ANALYSIS ---
    # Interprets the data for academic or technical documentation
    log("\n📝 PERFORMANCE ANALYSIS FOR PAPER:")
    for r in results:
        if r['load'] > 100:
            log(f"❌ {r['ver']}: Unsuitable for RT (Dropout). Processing exceeds the buffer deadline.")
        elif r['load'] > 75:
            log(f"⚠️  {r['ver']}: Risky. High CPU load, likely to cause crackling under multi-track load.")
        else:
            log(f"✅ {r['ver']}: RT Ready. Sufficient CPU headroom for stable performance.")

    # Save to a text file for archival and comparison
    output_path = os.path.join("results", f"rt_profile_buffer_{args.buffer}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n📄 Profiling results successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
