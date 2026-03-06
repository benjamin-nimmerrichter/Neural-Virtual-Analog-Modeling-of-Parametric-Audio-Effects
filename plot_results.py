import json
import os
import matplotlib.pyplot as plt

def plot_training_results(results_dir="results"):
    """
    Scans the results directory for training logs (JSON) and creates
    a comparative convergence plot of validation losses configured
    for IEEEtran paper standards (fonts and sizes).
    """
    # --- IEEEtran Formatting Setup ---
    # Set exact font sizes and family to match LaTeX document
    plt.rcParams.update({
        "font.family": "serif",   # Matches typical LaTeX serif fonts
        "font.size": 10,          # Standard IEEE text size
        "axes.titlesize": 10,     # Title size
        "axes.labelsize": 10,     # X/Y label size
        "xtick.labelsize": 8,     # X tick sizes
        "ytick.labelsize": 8,     # Y tick sizes
        "legend.fontsize": 8,     # Legend font size
        "figure.figsize": (3.5, 2.5), # IEEE single column width is ~3.5 inches
        "lines.linewidth": 1.2    # Slightly thinner lines for smaller plots
    })

    plt.figure()

    # Iterate through all files in the results folder
    for file in os.listdir(results_dir):
        if file.endswith(".json") and file.startswith("results_v"):
            version = file.split("_")[1].split(".")[0].upper()

            with open(os.path.join(results_dir, file), "r") as f:
                data = json.load(f)

                if "history" in data and "val_loss" in data["history"]:
                    plt.plot(data["history"]["val_loss"], label=f"{version} Val Loss")

    plt.yscale('log')

    # Note: Titles are often omitted in IEEE figures (described in caption instead),
    # but kept short here if needed.
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")

    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5, linewidth=0.5)

    # Save as EPS for LaTeX (vector graphic)
    plot_path = os.path.join(results_dir, "convergence_plot.eps")
    plt.savefig(plot_path, format='eps', bbox_inches='tight')

    print(f"📈 Plot successfully saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    if os.path.exists("results"):
        plot_training_results()
    else:
        print("❌ Error: 'results' directory not found. Run training first.")
