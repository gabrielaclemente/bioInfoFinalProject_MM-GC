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


def plot_convergence_curves_consensus():
    """
    Plot Gibbs convergence curves for consensus-scored runs only.

    This avoids mixing consensus scores and entropy scores, which are on
    different scales and should not be compared on the same y-axis.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.isdir(CONVERGENCE_FOLDER):
        print("No convergence_tables folder found.")
        return

    plt.figure(figsize=(8, 5))

    for filename in os.listdir(CONVERGENCE_FOLDER):
        if filename.endswith(".csv") and "consensus" in filename:
            file_path = os.path.join(CONVERGENCE_FOLDER, filename)
            df = pd.read_csv(file_path)

            # Example filename: gibbs_convergence_T0.5_consensus.csv
            temperature_label = filename.replace("gibbs_convergence_T", "")
            temperature_label = temperature_label.replace("_consensus.csv", "")

            label = f"T = {temperature_label}"

            plt.plot(df["iteration"], df["best_score"], label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Best Consensus Score So Far")
    plt.title("Gibbs Sampler Convergence by Temperature")
    plt.legend(title="Temperature", fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "gibbs_consensus_convergence_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved consensus-only convergence plot to {output_path}")


def create_motif_alignment_figure(
    results_file="motif_search_results.csv",
    promoter_file="promoter_sequences.csv",
    output_folder="analysis_outputs"
):
    """
    Create a motif alignment figure for the best motif-search result.

    The best result is selected by lowest consensus_score.
    The figure shows one motif per promoter and highlights whether each base
    matches the consensus motif.
    """

    os.makedirs(output_folder, exist_ok=True)

    results_df = pd.read_csv(results_file)
    promoters_df = pd.read_csv(promoter_file)

    # Pick best result by consensus score
    best_row = results_df.loc[results_df["consensus_score"].idxmin()]

    algorithm = best_row["algorithm"]
    consensus = best_row["consensus_motif"]
    motifs = str(best_row["motifs"]).split(";")

    # Make sure we only use matching rows
    n = min(len(motifs), len(promoters_df))
    motifs = motifs[:n]
    plot_df = promoters_df.iloc[:n].copy()

    plot_df["motif"] = motifs

    # Keep useful columns
    if "logFC" in plot_df.columns:
        plot_df = plot_df.sort_values(by="logFC", ascending=False)

    gene_symbols = plot_df["gene_symbol"].tolist()
    motifs = plot_df["motif"].tolist()

    if "logFC" in plot_df.columns:
        logfc_values = plot_df["logFC"].round(2).tolist()
    else:
        logfc_values = [""] * len(plot_df)

    k = len(consensus)

    # Create figure
    fig_height = max(6, 0.35 * len(motifs))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    # Title
    ax.set_title(
        f"Candidate Motif Instances Across Promoters\n"
        f"Best result: {algorithm} | Consensus motif: {consensus}",
        fontsize=14,
        weight="bold",
        pad=20
    )

    # Layout coordinates
    x_gene = 0
    x_logfc = 2.2
    x_motif_start = 3.2

    y_start = len(motifs) + 1

    # Headers
    ax.text(x_gene, y_start, "Gene", weight="bold", fontsize=10)
    ax.text(x_logfc, y_start, "logFC", weight="bold", fontsize=10)
    ax.text(x_motif_start, y_start, "Motif instance", weight="bold", fontsize=10)

    # Consensus row
    ax.text(x_gene, y_start - 1, "Consensus", weight="bold", fontsize=10)
    ax.text(x_logfc, y_start - 1, "", fontsize=10)

    for j, base in enumerate(consensus):
        ax.text(
            x_motif_start + j * 0.35,
            y_start - 1,
            base,
            fontsize=11,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="#dddddd", edgecolor="white")
        )

    # Motif rows
    for i, (gene, logfc, motif) in enumerate(zip(gene_symbols, logfc_values, motifs)):
        y = y_start - 2 - i

        ax.text(x_gene, y, str(gene), fontsize=9, va="center")
        ax.text(x_logfc, y, str(logfc), fontsize=9, va="center")

        for j, base in enumerate(motif):
            matches = base == consensus[j]

            facecolor = "#c7e9c0" if matches else "#fcbba1"

            ax.text(
                x_motif_start + j * 0.35,
                y,
                base,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="square,pad=0.25",
                    facecolor=facecolor,
                    edgecolor="white"
                )
            )

    # Add legend
    legend_y = -1
    ax.text(x_gene, legend_y, "Green = matches consensus", fontsize=9)
    ax.text(x_gene + 2.5, legend_y, "Red = mismatch", fontsize=9)

    ax.set_xlim(-0.2, x_motif_start + k * 0.45 + 1)
    ax.set_ylim(-2, y_start + 1)

    output_path = os.path.join(output_folder, "candidate_motif_alignment.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved motif alignment figure to {output_path}")

    # Also save the underlying table
    table_output = os.path.join(output_folder, "candidate_motif_alignment_table.csv")
    plot_df[["gene_symbol", "logFC", "adj.P.Val", "motif"]].to_csv(table_output, index=False)
    print(f"Saved motif alignment table to {table_output}")


def main():
    results_df = load_results()

    summarize_results(results_df)
    save_ranked_results(results_df)
    plot_consensus_scores(results_df)
    plot_entropy_scores(results_df)
    plot_convergence_curves()
    create_motif_alignment_figure()
    plot_convergence_curves_consensus()

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()