# Excel Opener
import pandas as pd
import numpy as np
import requests
import random
import time

filePath = "/Users/gabrielacmclemente/bioInfoFinalProject_MM-GC/geneOfInterest.xlsx"
# Cutoff values and Filter settings
# Every gene in this file has logFC > 0 (all are upregulated after injury)
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

#Row 0 → group labels: "Injured group RPKM", "Control group RPKM" (this is skipped)
#   Row 1 → real column names: Flybase ID, Symbol, N_S1, ..., logFC, adj.P.Val
# Passing header=1 tells pandas to use row 1 as the column names
 
print("=" * 60)
print("STEP 1: Loading expression data")
print("=" * 60)
 
expressionDf = pd.read_excel(filePath, header=1, engine="openpyxl")
 
print(f"Loaded {len(expressionDf)} genes.")
print(f"Columns: {expressionDf.columns.tolist()}\n")
 
# STEP 2: FILTER FOR SIGNIFICANT DE GENES
 
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
 
# STEP 3: FETCH PROMOTER SEQUENCES FROM ENSEMBL
# For each gene symbol:
#   Looks up the gene in Ensembl to get chromosome, start, end, strand
#   Calculates the promoter window (upstream of TSS):
#       Forward strand (+1): [start - upstream, start]
#       Reverse strand (-1): [end, end + upstream]
#   Fetches the DNA sequence for that window
 
ensemblServer = "https://rest.ensembl.org"
 
def getPromoterSequence(geneSymbol, upstreamBp=promoterUpstreamBp):
    try:
        #look up genomic coordinates
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
 
        #define promoter coordinates based on strand
        if strandDir == 1:
            promoterStart = max(1, geneStart - upstreamBp)
            promoterEnd   = geneStart
        else:
            promoterStart = geneEnd
            promoterEnd   = geneEnd + upstreamBp
 
        #fetch the sequence
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
 
 