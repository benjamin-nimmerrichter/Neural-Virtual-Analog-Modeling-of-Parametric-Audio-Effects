import json
import os

def generate_table():
    """
    Reconstructs the comparison table from individual JSON result files.
    Useful if the main summary text was deleted or if you want to refresh the stats.
    """
    versions = ['v1', 'v2', 'v3']
    results = {}

    # 1. DIRECTORY CHECK
    # Safety first: make sure the results folder actually exists.
    if not os.path.exists("results"):
        print("❌ Directory 'results' not found. Please run the training first.")
        return

    print("📊 Generating results table from existing JSON files...")

    # 2. DATA EXTRACTION
    # We iterate through the JSONs and extract the 'best_val_loss'
    # which is the ultimate metric of how well the model mimics the target sound.
    for v in versions:
        result_file = os.path.join("results", f"results_{v}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
                results[v] = data.get("best_val_loss", "N/A")
        else:
            results[v] = "File not found"

    # 3. FORMATTING FOR THE TABLE
    # Convert floats to strings with 6 decimal places for consistent alignment.
    v1_score = f"{results['v1']:.6f}" if isinstance(results['v1'], float) else results['v1']
    v2_score = f"{results['v2']:.6f}" if isinstance(results['v2'], float) else results['v2']
    v3_score = f"{results['v3']:.6f}" if isinstance(results['v3'], float) else results['v3']

    # 4. ASSEMBLE THE REPORT
    # This table summarizes the architectural differences (Ablation Study)
    # vs. the actual performance (Best Val Loss).
    table_output = (
        "\n" + "*"*60 + "\n"
        "🎉 SUMMARY OF TRAINING RESULTS:\n"
        + "*"*60 + "\n\n"
        "| Parameter / Feature       | V1 (Base) | V2 (Mix) | V3 (Full) |\n"
        "|---------------------------|-----------|----------|-----------|\n"
        "| Residual Blocks            | 10        | 10       | 10        |\n"
        "| 1x1 Projection             | No        | Yes      | Yes       |\n"
        "| Global Skip Connections    | No        | No       | Yes       |\n"
        "| Parameter Conditioning     | Concat    | Concat   | FiLM      |\n"
        "| Est. Parameter Count       | ~ 31k     | ~ 32k    | ~ 40k     |\n"
        "|---------------------------|-----------|----------|-----------|\n"
        f"| BEST VAL LOSS (Combined)  | {v1_score:<9} | {v2_score:<8} | {v3_score:<9} |\n"
    )

    # Output to console for immediate feedback
    print(table_output)

    # 5. PERSISTENCE
    # Save the table to a text file for future reference.
    txt_path = os.path.join("results", "training_all_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_output)
        f.write("\nThe results are ready to be ported into LaTeX.\n")

    print(f"📄 The final table has been saved to: {txt_path}")

if __name__ == "__main__":
    generate_table()
