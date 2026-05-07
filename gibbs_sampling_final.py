"""
Gibbs sampler motif discovery implementation for bioinformatics project.

This script implements a baseline Gibbs sampler for DNA motif finding,
with extensions for temperature-adjusted sampling and entropy-based scoring.
It runs multiple experiment settings and saves motif results and convergence
data to CSV files.
"""

import sys
import random
import csv
import math

DNA = "ACGT"
PROMOTER_FILE = "promoter_sequences.csv"

MOTIF_LENGTH = 8
GIBBS_ITERATIONS = 100
NUM_RANDOM_STARTS = 20
RANDOM_SEED = 0
PSEUDOCOUNT = 1


def profile_matrix(motifs, k, pseudo=0):
    """
    Build a 4 x k profile matrix from motifs.
    pseudo=1 means add-one pseudocounts.
    """
    counts = [[pseudo for _ in range(k)] for _ in range(4)]

    for motif in motifs:
        for j, base in enumerate(motif):
            row = DNA.index(base)
            counts[row][j] += 1

    profile = [[0.0 for _ in range(k)] for _ in range(4)]
    for j in range(k):
        col_sum = counts[0][j] + counts[1][j] + counts[2][j] + counts[3][j]
        for row in range(4):
            profile[row][j] = counts[row][j] / col_sum

    return profile


def score_motifs(motifs, k):
    """
    Score = sum over columns of (t - max_count_in_column).
    Lower is better.
    """
    t = len(motifs)
    counts = [[0 for _ in range(k)] for _ in range(4)]

    for motif in motifs:
        for j, base in enumerate(motif):
            row = DNA.index(base)
            counts[row][j] += 1

    score = 0
    for j in range(k):
        max_count = max(counts[row][j] for row in range(4))
        score += (t - max_count)

    return score


def entropy_score(motifs, k, pseudo=1):
    """
    Entropy score = sum of entropy across motif columns.
    Lower entropy means more conserved columns, so lower is better.
    """
    profile = profile_matrix(motifs, k, pseudo=pseudo)

    total_entropy = 0.0

    for j in range(k):
        column_entropy = 0.0
        for row in range(4):
            p = profile[row][j]
            if p > 0:
                column_entropy -= p * math.log2(p)
        total_entropy += column_entropy

    return total_entropy


def score(motifs, k, scoring="consensus", pseudo=1):
    """
    Choose which scoring method to use.
    """
    if scoring == "consensus":
        return score_motifs(motifs, k)
    elif scoring == "entropy":
        return entropy_score(motifs, k, pseudo=pseudo)
    else:
        raise ValueError("scoring must be 'consensus' or 'entropy'")


def consensus_motif(motifs, k):
    """
    Return the consensus sequence for a set of motifs.
    At each position, choose the most common nucleotide.
    """
    consensus = ""

    for j in range(k):
        counts = {base: 0 for base in DNA}

        for motif in motifs:
            counts[motif[j]] += 1

        best_base = max(counts, key=counts.get)
        consensus += best_base

    return consensus


def kmer_probability(kmer, profile):
    """P(kmer | profile)"""
    p = 1.0
    for j, base in enumerate(kmer):
        row = DNA.index(base)
        p *= profile[row][j]
    return p


def random_kmer(text, k):
    """Uniform random k-mer from text."""
    start = random.randint(0, len(text) - k)
    return text[start:start + k]


def profile_randomly_generated_kmer(text, k, profile, temperature=1.0):
    """
    Choose a k-mer from text randomly, weighted by P(kmer|profile).

    temperature = 1.0 gives standard Gibbs sampling.
    temperature > 1.0 makes sampling more exploratory.
    temperature < 1.0 makes sampling greedier.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    kmers = []
    weights = []

    for i in range(len(text) - k + 1):
        kmer = text[i:i + k]
        kmers.append(kmer)

        prob = kmer_probability(kmer, profile)
        adjusted_prob = prob ** (1 / temperature)
        weights.append(adjusted_prob)

    if sum(weights) == 0:
        return random.choice(kmers)

    return random.choices(kmers, weights=weights, k=1)[0]


def gibbs_sampler(Dna, k, N, pseudo=1, temperature=1.0, scoring="consensus", track_convergence=False):
    """
    One run of GibbsSampler(Dna, t, N) using pseudocounts.
    """
    t = len(Dna)
    motifs = [random_kmer(s, k) for s in Dna]
    best_motifs = motifs[:]
    best_score = score(best_motifs, k, scoring=scoring, pseudo=pseudo)
    convergence = [best_score]

    for _ in range(N):
        i = random.randint(0, t - 1)

        motifs_except_i = motifs[:i] + motifs[i + 1:]
        profile = profile_matrix(motifs_except_i, k, pseudo=pseudo)

        motifs[i] = profile_randomly_generated_kmer(Dna[i], k, profile, temperature=temperature)

        current_score = score(motifs, k, scoring=scoring, pseudo=pseudo)

        if current_score < best_score:
            best_motifs = motifs[:]
            best_score = current_score

        convergence.append(best_score)

    if track_convergence:
        return best_motifs, convergence

    return best_motifs


def repeated_gibbs_sampler(Dna, k, N, starts=20, pseudo=1, temperature=1.0, scoring="consensus", seed=None, track_convergence=False):
    """
    Run GibbsSampler 20 times (random starts) and return best motifs found.
    """
    if seed is not None:
        random.seed(seed)

    best_overall = None
    best_score = float("inf")
    best_convergence = None

    for _ in range(starts):
        if track_convergence:
            motifs, convergence = gibbs_sampler(
                Dna, k, N,
                pseudo=pseudo,
                temperature=temperature,
                scoring=scoring,
                track_convergence=True
            )
        else:
            motifs = gibbs_sampler(
                Dna, k, N,
                pseudo=pseudo,
                temperature=temperature,
                scoring=scoring
            )
            convergence = None

        s = score(motifs, k, scoring=scoring, pseudo=pseudo)

        if s < best_score:
            best_score = s
            best_overall = motifs
            best_convergence = convergence

    if track_convergence:
        return best_overall, best_convergence

    return best_overall


def read_promoter_sequences(filename):
    """
    Read promoter sequences from the shared promoter CSV file.
    The CSV should contain a column named 'sequence'.
    """
    sequences = []

    with open(filename, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            seq = row["sequence"].strip().upper()

            # Only keep clean DNA sequences
            if seq and all(base in DNA for base in seq):
                sequences.append(seq)

    return sequences


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = PROMOTER_FILE
    

    Dna = read_promoter_sequences(filename)

    k = MOTIF_LENGTH
    N = GIBBS_ITERATIONS

    experiment_settings = [
        {"temperature": 0.5, "scoring": "consensus"},
        {"temperature": 1.0, "scoring": "consensus"},
        {"temperature": 2.0, "scoring": "consensus"},
        {"temperature": 5.0, "scoring": "consensus"},
        {"temperature": 1.0, "scoring": "entropy"},
        {"temperature": 2.0, "scoring": "entropy"},
        {"temperature": 5.0, "scoring": "entropy"},
    ]

    with open("gibbs_experiment_results.csv", "w", newline="") as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow([
            "temperature",
            "scoring",
            "consensus_motif",
            "consensus_score",
            "entropy_score",
            "final_convergence_score",
            "convergence_length",
            "motifs"
        ])

        for setting in experiment_settings:
            temperature = setting["temperature"]
            scoring = setting["scoring"]

            best_motifs, convergence = repeated_gibbs_sampler(
                Dna, k, N,
                starts=NUM_RANDOM_STARTS,
                pseudo=PSEUDOCOUNT,
                seed=RANDOM_SEED,
                temperature=temperature,
                scoring=scoring,
                track_convergence=True
            )

            consensus = consensus_motif(best_motifs, k)
            consensus_score = score(best_motifs, k, scoring="consensus", pseudo=PSEUDOCOUNT)
            entropy = score(best_motifs, k, scoring="entropy", pseudo=PSEUDOCOUNT)

            results_writer.writerow([
                temperature,
                scoring,
                consensus,
                consensus_score,
                entropy,
                convergence[-1],
                len(convergence),
                ";".join(best_motifs)
            ])

            # Save convergence data for this specific experiment setting
            convergence_filename = f"gibbs_convergence_T{temperature}_{scoring}.csv"
            with open(convergence_filename, "w", newline="") as convergence_file:
                convergence_writer = csv.writer(convergence_file)
                convergence_writer.writerow(["iteration", "best_score"])

                for iteration, best_score in enumerate(convergence):
                    convergence_writer.writerow([iteration, best_score])

            print("Temperature:", temperature)
            print("Scoring:", scoring)
            print("Consensus motif:", consensus)
            print("Consensus score:", consensus_score)
            print("Entropy score:", entropy)
            print("Final convergence score:", convergence[-1])
            print("Convergence length:", len(convergence))
            print("Motifs:")
            print("\n".join(best_motifs))
            print("-" * 40)

    print("Experiment results saved to gibbs_experiment_results.csv")