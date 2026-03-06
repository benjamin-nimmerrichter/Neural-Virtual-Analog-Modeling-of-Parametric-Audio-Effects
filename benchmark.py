import json
import os
import torch
import time
import argparse
import platform
from model import ParametricTCN

def get_params(model):
    """
    Calculates the total number of trainable parameters in the model.
    Used to compare the complexity and size of different model versions.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_cpu_name():
    """
    Retrieves the detailed CPU model name.
    Attempts to read from /proc/cpuinfo on Linux systems, falls back to platform.processor() elsewhere.
    """
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"

def profile_iteration(model, b_size, device):
    """
    Measures the inference time for a single audio buffer.
    Includes a warm-up phase to stabilize CPU clock speeds and caches.
    """
    # Create dummy input: [Batch, Channels, Samples]
    input_tensor = torch.randn(1, 1, b_size).to(device)
    # Parameter tensor for the Parametric TCN (e.g., controlling a knob/value)
    p_tensor = torch.full((1, 1), 0.5).to(device)

    # Warm-up phase: Executes 20 runs to ensure the CPU is out of idle states
    for _ in range(20):
        _ = model(input_tensor, p_tensor)

    # Benchmark phase: Measure the average time over 1000 iterations
    num_iter = 1000
    start = time.perf_counter()
    with torch.no_grad(): # Disable gradient tracking for faster inference
        for _ in range(num_iter):
            _ = model(input_tensor, p_tensor)

    # Calculate average time per iteration in milliseconds
    t_proc = ((time.perf_counter() - start) / num_iter) * 1000 # ms
    return t_proc

def main():
    # Benchmark Configuration
    versions = ['v1', 'v2', 'v3']      # Architectures to test
    buffers = [64, 128, 256, 512]       # Buffer sizes (in samples)
    fs = 48000                          # Standard sample rate (48kHz)
    device = torch.device("cpu")        # Audio plugins usually run on CPU

    # Ensure the directory exists for saving results
    os.makedirs("results", exist_ok=True)

    # List to store log lines for simultaneous printing and file saving
    output_lines = []

    def log(text):
        """Helper to print to console and store for file export."""
        print(text)
        output_lines.append(text)

    # Header section
    log(f"\n{'='*110}")
    log(f"🔬 COMPREHENSIVE BENCHMARK: ACCURACY VS. REAL-TIME CAPABILITIES (FS: {fs/1000}kHz)")
    log(f"{'='*110}")

    # Log hardware and environment info
    log(f"💻 SYSTEM SPECIFICATIONS:")
    log(f"   OS:         {platform.system()} {platform.release()} ({platform.machine()})")
    log(f"   Node Name: {platform.node()}")
    log(f"   Processor: {get_cpu_name()}")
    log(f"   PyTorch:   {torch.__version__}")
    log(f"   Device:    CPU (Forced for audio plugin simulation)")
    log(f"{'='*110}")

    for v in versions:
        # 1. Initialize the specific model version
        try:
            model = ParametricTCN(version=v, num_layers=10, hidden_ch=32).to(device)
            model.eval() # Set to evaluation mode (disables dropout etc.)
            p_count = get_params(model)
        except Exception as e:
            log(f"\n❌ Cannot load model {v}: {e}")
            continue

        # 2. Retrieve training accuracy from existing JSON results
        loss_file = os.path.join("results", f"results_{v}.json")
        loss_val = "N/A"
        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                loss_val = f"{json.load(f).get('best_val_loss', 0):.6f}"

        # Sub-header for the specific model
        log(f"\n📦 MODEL: {v.upper()} | Parameters: {p_count:,} | Best Val Loss: {loss_val}")
        log(f"{'-'*110}")
        log(f"{'Buffer':<10} | {'Buff Latency':<15} | {'Proc Time':<15} | {'Total Latency':<18} | {'CPU Load':<12} | {'Status'}")
        log(f"{'-'*110}")

        best_rt_buffer = None

        # 3. Iterate through different buffer sizes to test real-time stability
        for b in buffers:
            t_proc = profile_iteration(model, b, device)    # Processing time
            l_buff = (b / fs) * 1000                         # Buffer duration in ms
            l_total = l_buff + t_proc                        # Round-trip estimate
            load = (t_proc / l_buff) * 100                   # CPU load relative to buffer time

            # Evaluate Real-Time (RT) performance status
            # If load > 100%, processing takes longer than the audio itself (Audio Dropout)
            if load < 70:
                status = "✅ SAFE RT"
                if best_rt_buffer is None: best_rt_buffer = b
            elif load < 100:
                status = "⚠️ RISKY RT"
                if best_rt_buffer is None: best_rt_buffer = b
            else:
                status = "❌ DROPOUT"

            log(f"{b:<10} | {l_buff:>10.2f} ms | {t_proc:>10.2f} ms | {l_total:>13.2f} ms | {load:>10.1f}% | {status}")

        # Final recommendation for the model version
        if best_rt_buffer:
            log(f"👉 Conclusion for {v.upper()}: Suitable for real-time from buffer size {best_rt_buffer} and above.")
        else:
            log(f"👉 Conclusion for {v.upper()}: This hardware CANNOT run this model in real-time.")

    log(f"\n{'='*110}")
    log("Note: 'True Real-time' (perceived zero latency) requires Total Latency < 10ms.")

    # 4. Save results to a file for later comparison
    host_name = platform.node().replace(" ", "_").replace("-", "_")
    output_path = os.path.join("results", f"benchmark_results_{host_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n📄 Complete benchmark results successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
