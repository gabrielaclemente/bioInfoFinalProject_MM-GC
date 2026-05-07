"""
Main runner for the pTBI motif-finding project.

This file:
1. Reads promoter sequences from promoter_sequences.csv.
2. Runs Gibbs sampler experiments.
3. Runs randomized motif search.
4. Saves one combined results CSV.
"""

import csv
import random
import time
import os

from gibbs_sampling_final import (
    read_promoter_sequences,
    repeated_gibbs_sampler,
    score,
    consensus_motif,
)

from randomizedMotifSearchFinalProject import (
    runMotifSearch,
    buildConsensus,
    scoreMotifs,
)


# Project settings

PROMOTER_FILE = "promoter_sequences.csv"
OUTPUT_FILE = "motif_search_results.csv"
CONVERGENCE_FOLDER = "convergence_tables"

MOTIF_LENGTH = 8
GIBBS_ITERATIONS = 100
NUM_RANDOM_STARTS = 20
RANDOM_SEED = 0
PSEUDOCOUNT = 1

GIBBS_EXPERIMENT_SETTINGS = [
    {"temperature": 0.5, "scoring": "consensus"},
    {"temperature": 1.0, "scoring": "consensus"},
    {"temperature": 2.0, "scoring": "consensus"},
    {"temperature": 5.0, "scoring": "consensus"},
    {"temperature": 1.0, "scoring": "entropy"},
    {"temperature": 2.0, "scoring": "entropy"},
    {"temperature": 5.0, "scoring": "entropy"},
]


# Main

def main():
    print("=" * 60)
    print("Reading promoter sequences")
    print("=" * 60)

    dna_sequences = read_promoter_sequences(PROMOTER_FILE)

    k = MOTIF_LENGTH
    t = len(dna_sequences)

    print(f"Loaded {t} promoter sequences.")
    print(f"Motif length k = {k}")

    if t == 0:
        print("No promoter sequences found. Stopping.")
        return

    results = []
    os.makedirs(CONVERGENCE_FOLDER, exist_ok=True)

    # Run Gibbs sampler settings

    print("\n" + "=" * 60)
    print("Running Gibbs sampler experiments")
    print("=" * 60)

    for setting in GIBBS_EXPERIMENT_SETTINGS:
        temperature = setting["temperature"]
        scoring = setting["scoring"]

        print(f"Running Gibbs: temperature={temperature}, scoring={scoring}")

        start_time = time.time()

        best_motifs, convergence = repeated_gibbs_sampler(
            dna_sequences,
            k,
            GIBBS_ITERATIONS,
            starts=NUM_RANDOM_STARTS,
            pseudo=PSEUDOCOUNT,
            temperature=temperature,
            scoring=scoring,
            seed=RANDOM_SEED,
            track_convergence=True,
        )

        runtime_seconds = time.time() - start_time

        consensus = consensus_motif(best_motifs, k)
        consensus_score = score(best_motifs, k, scoring="consensus", pseudo=PSEUDOCOUNT)
        entropy_score = score(best_motifs, k, scoring="entropy", pseudo=PSEUDOCOUNT)

        results.append({
            "algorithm": "GibbsSampler",
            "temperature": temperature,
            "scoring": scoring,
            "motif_length": k,
            "num_sequences": t,
            "iterations": GIBBS_ITERATIONS,
            "random_starts": NUM_RANDOM_STARTS,
            "consensus_motif": consensus,
            "consensus_score": consensus_score,
            "entropy_score": entropy_score,
            "final_convergence_score": convergence[-1],
            "convergence_length": len(convergence),
            "runtime_seconds": runtime_seconds,
            "motifs": ";".join(best_motifs),
        })

        # Save convergence curve for this Gibbs setting
        convergence_filename = os.path.join(CONVERGENCE_FOLDER, f"gibbs_convergence_T{temperature}_{scoring}.csv"
)
        with open(convergence_filename, "w", newline="") as convergence_file:
            writer = csv.writer(convergence_file)
            writer.writerow(["iteration", "best_score"])

            for iteration, best_score in enumerate(convergence):
                writer.writerow([iteration, best_score])


    # Run Randomized Motif Search

    print("\n" + "=" * 60)
    print("Running Randomized Motif Search")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    start_time = time.time()

    randomized_motifs, randomized_score = runMotifSearch(
        dna_sequences,
        k,
        NUM_RANDOM_STARTS,
    )

    runtime_seconds = time.time() - start_time

    randomized_consensus = buildConsensus(randomized_motifs, k)

    results.append({
        "algorithm": "RandomizedMotifSearch",
        "temperature": "",
        "scoring": "consensus",
        "motif_length": k,
        "num_sequences": t,
        "iterations": "",
        "random_starts": NUM_RANDOM_STARTS,
        "consensus_motif": randomized_consensus,
        "consensus_score": randomized_score,
        "entropy_score": "",
        "final_convergence_score": "",
        "convergence_length": "",
        "runtime_seconds": runtime_seconds,
        "motifs": ";".join(randomized_motifs),
    })


    # Write combined results

    print("\n" + "=" * 60)
    print("Writing results")
    print("=" * 60)

    fieldnames = [
        "algorithm",
        "temperature",
        "scoring",
        "motif_length",
        "num_sequences",
        "iterations",
        "random_starts",
        "consensus_motif",
        "consensus_score",
        "entropy_score",
        "final_convergence_score",
        "convergence_length",
        "runtime_seconds",
        "motifs",
    ]

    with open(OUTPUT_FILE, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved combined results to {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()