"""
Analysis script for the pTBI motif-finding project.

This script reads motif_search_results.csv and summarizes the results
from Gibbs sampling and randomized motif search.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = "motif_search_results.csv"
CONVERGENCE_FOLDER = "convergence_tables"
OUTPUT_FOLDER = "analysis_outputs"


def load_results(filename=RESULTS_FILE):
    """Read the combined motif search results file."""
    return pd.read_csv(filename)


def summarize_results(results_df):
    """Print a simple summary of motif search results."""
    print("=" * 60)
    print("Motif Search Results Summary")
    print("=" * 60)

    print("\nAlgorithms included:")
    print(results_df["algorithm"].value_counts())

    print("\nBest result by consensus score:")
    best_consensus = results_df.loc[results_df["consensus_score"].idxmin()]
    print(best_consensus[[
        "algorithm",
        "temperature",
        "scoring",
        "consensus_motif",
        "consensus_score",
        "entropy_score",
        "runtime_seconds"
    ]])

    # Only use rows where entropy_score exists
    entropy_df = results_df.dropna(subset=["entropy_score"])

    if len(entropy_df) > 0:
        print("\nBest result by entropy score:")
        best_entropy = entropy_df.loc[entropy_df["entropy_score"].idxmin()]
        print(best_entropy[[
            "algorithm",
            "temperature",
            "scoring",
            "consensus_motif",
            "consensus_score",
            "entropy_score",
            "runtime_seconds"
        ]])


def save_ranked_results(results_df):
    """Save sorted result tables."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    by_consensus = results_df.sort_values(by="consensus_score")
    by_consensus.to_csv(
        os.path.join(OUTPUT_FOLDER, "results_ranked_by_consensus_score.csv"),
        index=False
    )

    entropy_df = results_df.dropna(subset=["entropy_score"])
    by_entropy = entropy_df.sort_values(by="entropy_score")
    by_entropy.to_csv(
        os.path.join(OUTPUT_FOLDER, "results_ranked_by_entropy_score.csv"),
        index=False
    )

    print("\nSaved ranked result tables.")


def plot_consensus_scores(results_df):
    """Create a bar plot comparing consensus scores across methods."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    labels = []
    scores = []

    for _, row in results_df.iterrows():
        if row["algorithm"] == "GibbsSampler":
            label = f"Gibbs\nT={row['temperature']}\n{row['scoring']}"
        else:
            label = "Randomized\nMotif Search"

        labels.append(label)
        scores.append(row["consensus_score"])

    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores)
    plt.ylabel("Consensus Score")
    plt.title("Consensus Score by Motif Search Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "consensus_score_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved consensus score plot to {output_path}")


def plot_entropy_scores(results_df):
    """Create a bar plot comparing entropy scores across methods."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    entropy_df = results_df.dropna(subset=["entropy_score"])

    labels = []
    scores = []

    for _, row in entropy_df.iterrows():
        if row["algorithm"] == "GibbsSampler":
            label = f"Gibbs\nT={row['temperature']}\n{row['scoring']}"
        else:
            label = "Randomized\nMotif Search"

        labels.append(label)
        scores.append(row["entropy_score"])

    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores)
    plt.ylabel("Entropy Score")
    plt.title("Entropy Score by Motif Search Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "entropy_score_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved entropy score plot to {output_path}")


def plot_convergence_curves():
    """Plot all Gibbs convergence curves from the convergence_tables folder."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.isdir(CONVERGENCE_FOLDER):
        print("No convergence_tables folder found.")
        return

    plt.figure(figsize=(10, 6))

    for filename in os.listdir(CONVERGENCE_FOLDER):
        if filename.endswith(".csv"):
            file_path = os.path.join(CONVERGENCE_FOLDER, filename)
            df = pd.read_csv(file_path)

            label = filename.replace("gibbs_convergence_", "").replace(".csv", "")
            plt.plot(df["iteration"], df["best_score"], label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Best Score So Far")
    plt.title("Gibbs Sampler Convergence Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "gibbs_convergence_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved convergence plot to {output_path}")


def main():
    results_df = load_results()

    summarize_results(results_df)
    save_ranked_results(results_df)
    plot_consensus_scores(results_df)
    plot_entropy_scores(results_df)
    plot_convergence_curves()

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()