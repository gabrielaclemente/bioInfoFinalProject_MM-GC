#RESEARCH QUESTION:
#   Do genes significantly differentially expressed after pTBI in Drosophila
#   melanogaster share a common regulatory DNA motif, suggesting coordinated
#   transcriptional control?

import random

# RANDOMIZED MOTIF SEARCH FUNCTIONS
#   1. Randomly pick one k-mer from each promoter sequence (random initialization)
#   2. Build a Profile matrix — for each position, count how often each base appears
#      (Laplace pseudocounts of +1 are added to avoid zero probabilities)
#   3. For each sequence, find the k-mer that best matches the current profile
#   4. Repeat steps 2-3 until no improvement in score
#   5. Score = total mismatches from the consensus (lower = more conserved motif)
#   6. Restart many times and keep the globally best result
 
def buildProfile(motifList, k):
    """Build a position frequency matrix with Laplace pseudocounts."""
    profileMatrix = {base: [1] * k for base in "ACGT"}   # pseudocount = 1
 
    for motif in motifList:
        for pos, base in enumerate(motif):
            if base in profileMatrix:
                profileMatrix[base][pos] += 1
 
    denominator = len(motifList) + 4   # motifs + 4 pseudocounts
    for base in profileMatrix:
        profileMatrix[base] = [count / denominator for count in profileMatrix[base]]
 
    return profileMatrix
 
 
def kmerProbability(kmer, profileMatrix):
    """Probability of a k-mer under the given profile (product of column frequencies)."""
    probability = 1.0
    for pos, base in enumerate(kmer):
        if base in profileMatrix:
            probability *= profileMatrix[base][pos]
        else:
            probability *= 1e-9   # heavy penalty for ambiguous bases (N, etc.)
    return probability
 
 
def mostProbableKmer(sequence, k, profileMatrix):
    """Scan a sequence and return the k-mer with the highest profile probability."""
    bestKmer = sequence[:k]
    bestProb = -1.0
 
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if len(kmer) == k and all(base in "ACGT" for base in kmer):
            prob = kmerProbability(kmer, profileMatrix)
            if prob > bestProb:
                bestProb = prob
                bestKmer = kmer
 
    return bestKmer
 
 
def scoreMotifs(motifList, k):
    """
    Score = total mismatches from the column-wise consensus across all positions.
    Lower is better (more conserved).
    """
    totalScore = 0
    for pos in range(k):
        baseCounts = {"A": 0, "C": 0, "G": 0, "T": 0}
        for motif in motifList:
            if motif[pos] in baseCounts:
                baseCounts[motif[pos]] += 1
        totalScore += len(motifList) - max(baseCounts.values())
    return totalScore
 
 
def sampleRandomMotifs(dnaSequences, k):
    """Pick one random k-mer from each sequence."""
    sampledMotifs = []
    for seq in dnaSequences:
        maxStart = len(seq) - k
        if maxStart <= 0:
            sampledMotifs.append(seq[:k].ljust(k, "A"))
        else:
            startPos = random.randint(0, maxStart)
            sampledMotifs.append(seq[startPos:startPos + k])
    return sampledMotifs
 
 
def randomizedMotifSearch(dnaSequences, k):
    """One full run of the Randomized Motif Search. Returns best motifs found."""
    currentMotifs = sampleRandomMotifs(dnaSequences, k)
    bestMotifs    = currentMotifs[:]
 
    while True:
        currentProfile = buildProfile(currentMotifs, k)
        currentMotifs  = [mostProbableKmer(seq, k, currentProfile) for seq in dnaSequences]
 
        if scoreMotifs(currentMotifs, k) < scoreMotifs(bestMotifs, k):
            bestMotifs = currentMotifs[:]
        else:
            return bestMotifs   # converged — no further improvement
 
 
def runMotifSearch(dnaSequences, k, numRuns):
    """
    Run randomized motif search numRuns times and return the globally best result.
    Multiple restarts escape local optima.
    """
    globalBestMotifs = None
    globalBestScore  = float("inf")
 
    for _ in range(numRuns):
        candidateMotifs = randomizedMotifSearch(dnaSequences, k)
        candidateScore  = scoreMotifs(candidateMotifs, k)
 
        if candidateScore < globalBestScore:
            globalBestMotifs = candidateMotifs
            globalBestScore  = candidateScore
 
    return globalBestMotifs, globalBestScore
 
 
def buildConsensus(motifList, k):
    """Consensus string: most frequent base at each position."""
    consensusStr = ""
    for pos in range(k):
        baseCounts = {"A": 0, "C": 0, "G": 0, "T": 0}
        for motif in motifList:
            if motif[pos] in baseCounts:
                baseCounts[motif[pos]] += 1
        consensusStr += max(baseCounts, key=baseCounts.get)
    return consensusStr