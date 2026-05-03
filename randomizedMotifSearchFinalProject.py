# =============================================================================
# pTBI Randomized Motif Search Pipeline
# =============================================================================
# RESEARCH QUESTION:
#   Do genes significantly differentially expressed after pTBI in Drosophila
#   melanogaster share a common regulatory DNA motif, suggesting coordinated
#   transcriptional control?
# =============================================================================
 
import pandas as pd
import numpy as np
import requests
import random
import time
 
# =============================================================================
# CONFIGURATION — edit these values as needed
# =============================================================================
 
filePath = "/Users/gabrielacmclemente/Desktop/BioInformatics/GSE306282_4hr_PTBI_DE_Genes.xlsx"
 
# Filter settings
# NOTE: Every gene in this file has logFC > 0 (all are upregulated after injury)
#       If using a file with downregulated genes, change filterDirection to "down"
filterDirection = "up"   # "up" = logFC > cutoff  |  "down" = logFC < -cutoff
adjPvalCutoff   = 0.05   # only keep genes with adjusted p-value below this
logFcCutoff     = 1.0    # only keep genes with |logFC| above this (stronger signal)
 
# Promoter retrieval settings
promoterUpstreamBp = 500  # how many base pairs upstream of TSS to fetch
maxGenesToFetch    = 30   # keep ≤ 30 to stay within Ensembl API rate limits
 
# Motif search settings
motifLength    = 8    # k — length of motif to search for (8 bp is typical for TF sites)
numSearchRuns  = 200  # how many random restarts (more = better, but slower)
randomSeed     = 42   # for reproducibility
 
# =============================================================================
# STEP 1: LOAD EXCEL FILE
# =============================================================================
#   Row 0 → group labels: "Injured group RPKM", "Control group RPKM" (this is skipped)
#   Row 1 → real column names: Flybase ID, Symbol, N_S1, ..., logFC, adj.P.Val
# Passing header=1 tells pandas to use row 1 as the column names
 
print("=" * 60)
print("STEP 1: Loading expression data")
print("=" * 60)
 
expressionDf = pd.read_excel(filePath, header=1, engine="openpyxl")
 
print(f"Loaded {len(expressionDf)} genes.")
print(f"Columns: {expressionDf.columns.tolist()}\n")
 
# =============================================================================
# STEP 2: FILTER FOR SIGNIFICANT DE GENES
# =============================================================================
 
print("=" * 60)
print("STEP 2: Filtering significant DE genes")
print("=" * 60)
 
if filterDirection == "down":
    geneMask       = (expressionDf["logFC"] < -logFcCutoff) & (expressionDf["adj.P.Val"] < adjPvalCutoff)
    directionLabel = "downregulated"
else:
    geneMask       = (expressionDf["logFC"] > logFcCutoff) & (expressionDf["adj.P.Val"] < adjPvalCutoff)
    directionLabel = "upregulated"
 
filteredDf = expressionDf[geneMask].copy()
geneList   = filteredDf["Symbol"].dropna().unique().tolist()
 
print(f"Filter: {directionLabel}, |logFC| > {logFcCutoff}, adj.P.Val < {adjPvalCutoff}")
print(f"Genes passing filter: {len(geneList)}")
print(f"First 20 genes: {geneList[:20]}\n")
 
# =============================================================================
# STEP 3: FETCH PROMOTER SEQUENCES FROM ENSEMBL
# =============================================================================
# For each gene symbol:
#   A) Looks up the gene in Ensembl to get chromosome, start, end, strand
#   B) Calculates the promoter window (upstream of TSS):
#        Forward strand (+1): [start - upstream, start]
#        Reverse strand (-1): [end, end + upstream]
#   C) Fetches the DNA sequence for that window
 
ensemblServer = "https://rest.ensembl.org"
 
def getPromoterSequence(geneSymbol, upstreamBp=promoterUpstreamBp):
    try:
        # A: look up genomic coordinates
        lookupUrl = f"{ensemblServer}/lookup/symbol/drosophila_melanogaster/{geneSymbol}?expand=0"
        lookupResponse = requests.get(lookupUrl,
                                      headers={"Content-Type": "application/json"},
                                      timeout=10)
        if not lookupResponse.ok:
            return None
 
        geneData    = lookupResponse.json()
        chromName   = geneData["seq_region_name"]
        geneStart   = geneData["start"]
        geneEnd     = geneData["end"]
        strandDir   = geneData["strand"]   # 1 = forward, -1 = reverse
 
        # B: define promoter coordinates based on strand
        if strandDir == 1:
            promoterStart = max(1, geneStart - upstreamBp)
            promoterEnd   = geneStart
        else:
            promoterStart = geneEnd
            promoterEnd   = geneEnd + upstreamBp
 
        # C: fetch the sequence
        seqUrl      = (f"{ensemblServer}/sequence/region/drosophila_melanogaster/"
                       f"{chromName}:{promoterStart}..{promoterEnd}:{strandDir}")
        seqResponse = requests.get(seqUrl,
                                   headers={"Content-Type": "text/plain"},
                                   timeout=10)
        if not seqResponse.ok:
            return None
 
        return seqResponse.text.strip().upper()
 
    except Exception:
        return None   # skip silently on any network or parsing error
 
 
print("=" * 60)
print("STEP 3: Fetching promoter sequences from Ensembl")
print("=" * 60)
print(f"Fetching up to {maxGenesToFetch} sequences ({promoterUpstreamBp} bp upstream of TSS)...")
print("(Allow ~1 minute — Ensembl API calls are rate-limited)\n")
 
promoterSequences = []   # DNA strings for each successfully fetched gene
validGenes        = []   # gene symbols that returned a sequence
failedGenes       = []   # gene symbols that failed
 
for idx, gene in enumerate(geneList[:maxGenesToFetch]):
    seq = getPromoterSequence(gene)
    time.sleep(0.1)   # stay within Ensembl rate limit
 
    if seq and len(seq) >= 50:
        promoterSequences.append(seq)
        validGenes.append(gene)
        statusMark = "✓"
    else:
        failedGenes.append(gene)
        statusMark = "✗"
 
    print(f"  [{idx + 1:2d}/{min(maxGenesToFetch, len(geneList))}]  {gene:20s}  {statusMark}")
 
print(f"\nSuccessfully retrieved: {len(promoterSequences)} sequences")
print(f"Failed / not found:     {len(failedGenes)} genes")
if failedGenes:
    print(f"  Failed genes: {failedGenes}")
 
# =============================================================================
# STEP 4: RANDOMIZED MOTIF SEARCH FUNCTIONS
# =============================================================================
# ALGORITHM OVERVIEW:
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
 
 
# =============================================================================
# STEP 5: RUN ANALYSIS AND PRINT RESULTS
# =============================================================================
 
print("\n" + "=" * 60)
print("STEP 4: Running Randomized Motif Search")
print("=" * 60)
 
if len(promoterSequences) < 5:
    print("\n⚠  Not enough promoter sequences (need at least 5).")
    print("   Check your internet connection and that your gene symbols resolve in Ensembl.")
else:
    print(f"Running {numSearchRuns} restarts on {len(promoterSequences)} sequences, k = {motifLength}...")
    random.seed(randomSeed)
 
    bestMotifs, bestScore = runMotifSearch(promoterSequences, motifLength, numSearchRuns)
    consensusMotif        = buildConsensus(bestMotifs, motifLength)
    motifFrequency        = sum(1 for seq in promoterSequences if consensusMotif in seq) / len(promoterSequences)
 
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"  Consensus motif  :  {consensusMotif}")
    print(f"  Motif length     :  {motifLength} bp")
    print(f"  Motif score      :  {bestScore}  (lower = more conserved across sequences)")
    print(f"  Exact frequency  :  {motifFrequency:.1%} of promoters contain the exact motif")
    print(f"  Sequences used   :  {len(promoterSequences)}")
    print(f"\n  Best motif per gene:")
    for geneName, foundMotif in zip(validGenes, bestMotifs):
        print(f"    {geneName:20s}  {foundMotif}")
    print(f"\n  Genes used: {validGenes}")
 