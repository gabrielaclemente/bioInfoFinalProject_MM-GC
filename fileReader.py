"""
File reader and promoter sequence fetcher for the pTBI motif-finding project.

This script:
1. Reads the gene expression Excel file.
2. Filters for significantly upregulated genes.
3. Fetches promoter sequences from Ensembl.
4. Writes shared CSV output for both motif algorithms.
5. Writes a Gibbs-sampler input text file.
"""

import sys
import time
import requests
import pandas as pd


# Configuration
DEFAULT_EXCEL_FILE = "geneOfInterest.xlsx"

FILTER_DIRECTION = "up"      # "up" or "down"
ADJ_PVAL_CUTOFF = 0.05
LOGFC_CUTOFF = 1.0

PROMOTER_UPSTREAM_BP = 500
MAX_GENES_TO_FETCH = 30

MOTIF_LENGTH = 8
GIBBS_ITERATIONS = 100

ENSEMBL_SERVER = "https://rest.ensembl.org"


# Excel reading / filtering
def read_gene_expression_file(file_path):
    """
    Reads the Excel file.

    The current FlyBase/expression file appears to use row 1 as the real header,
    so header=1 skips the first row of group labels.
    """
    df = pd.read_excel(file_path, header=1, engine="openpyxl")
    return df


def filter_genes(df, direction="up", logfc_cutoff=1.0, adj_pval_cutoff=0.05):
    """
    Filter genes by logFC and adjusted p-value.
    """
    if direction == "down":
        mask = (df["logFC"] < -logfc_cutoff) & (df["adj.P.Val"] < adj_pval_cutoff)
    else:
        mask = (df["logFC"] > logfc_cutoff) & (df["adj.P.Val"] < adj_pval_cutoff)

    filtered = df[mask].copy()

    # Keep only rows with usable gene symbols
    filtered = filtered.dropna(subset=["Symbol"])

    return filtered


# Promoter fetching

def get_promoter_sequence(gene_symbol, upstream_bp=500):
    """
    Look up a Drosophila gene in Ensembl and fetch its upstream promoter sequence.

    For + strand genes:
        promoter = start - upstream_bp to start

    For - strand genes:
        promoter = end to end + upstream_bp

    Returns a dictionary with metadata and sequence, or None if lookup fails.
    """
    try:
        lookup_url = (
            f"{ENSEMBL_SERVER}/lookup/symbol/"
            f"drosophila_melanogaster/{gene_symbol}?expand=0"
        )

        lookup_response = requests.get(
            lookup_url,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if not lookup_response.ok:
            return None

        gene_data = lookup_response.json()

        chrom_name = gene_data["seq_region_name"]
        gene_start = gene_data["start"]
        gene_end = gene_data["end"]
        strand = gene_data["strand"]

        if strand == 1:
            promoter_start = max(1, gene_start - upstream_bp)
            promoter_end = gene_start
        else:
            promoter_start = gene_end
            promoter_end = gene_end + upstream_bp

        seq_url = (
            f"{ENSEMBL_SERVER}/sequence/region/"
            f"drosophila_melanogaster/"
            f"{chrom_name}:{promoter_start}..{promoter_end}:{strand}"
        )

        seq_response = requests.get(
            seq_url,
            headers={"Content-Type": "text/plain"},
            timeout=10
        )

        if not seq_response.ok:
            return None

        sequence = seq_response.text.strip().upper()

        # Keep only clean DNA sequences
        if len(sequence) < 50:
            return None

        return {
            "gene_symbol": gene_symbol,
            "ensembl_id": gene_data.get("id", ""),
            "chromosome": chrom_name,
            "gene_start": gene_start,
            "gene_end": gene_end,
            "strand": strand,
            "promoter_start": promoter_start,
            "promoter_end": promoter_end,
            "sequence": sequence
        }

    except Exception:
        return None


def fetch_promoters(filtered_df, max_genes=30, upstream_bp=500):
    """
    Fetch promoter sequences for filtered genes.
    """
    promoter_records = []
    failed_genes = []

    gene_symbols = filtered_df["Symbol"].dropna().unique().tolist()

    for idx, gene_symbol in enumerate(gene_symbols[:max_genes]):
        record = get_promoter_sequence(gene_symbol, upstream_bp=upstream_bp)
        time.sleep(0.1)

        if record is None:
            failed_genes.append(gene_symbol)
            print(f"[{idx + 1:2d}] {gene_symbol:20s} failed")
        else:
            promoter_records.append(record)
            print(f"[{idx + 1:2d}] {gene_symbol:20s} success")

    return promoter_records, failed_genes


# Output files

def write_promoter_csv(promoter_records, filtered_df, output_file="promoter_sequences.csv"):
    """
    Write a shared CSV with gene metadata and promoter sequences.
    """
    promoter_df = pd.DataFrame(promoter_records)

    # Add logFC and adj.P.Val from the original filtered dataframe
    expression_cols = filtered_df[["Symbol", "logFC", "adj.P.Val"]].copy()
    expression_cols = expression_cols.rename(columns={"Symbol": "gene_symbol"})

    merged = promoter_df.merge(expression_cols, on="gene_symbol", how="left")

    merged.to_csv(output_file, index=False)
    print(f"Saved shared promoter CSV to {output_file}")


# Main

if __name__ == "__main__":
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = DEFAULT_EXCEL_FILE

    print("=" * 60)
    print("STEP 1: Reading gene expression file")
    print("=" * 60)

    expression_df = read_gene_expression_file(excel_file)
    print(f"Loaded {len(expression_df)} genes.")
    print(f"Columns: {expression_df.columns.tolist()}")

    print("\n" + "=" * 60)
    print("STEP 2: Filtering genes")
    print("=" * 60)

    filtered_df = filter_genes(
        expression_df,
        direction=FILTER_DIRECTION,
        logfc_cutoff=LOGFC_CUTOFF,
        adj_pval_cutoff=ADJ_PVAL_CUTOFF
    )

    print(f"Genes passing filter: {len(filtered_df)}")
    print(filtered_df[["Symbol", "logFC", "adj.P.Val"]].head(20))

    print("\n" + "=" * 60)
    print("STEP 3: Fetching promoter sequences")
    print("=" * 60)

    promoter_records, failed_genes = fetch_promoters(
        filtered_df,
        max_genes=MAX_GENES_TO_FETCH,
        upstream_bp=PROMOTER_UPSTREAM_BP
    )

    print(f"\nSuccessfully retrieved: {len(promoter_records)} promoter sequences")
    print(f"Failed genes: {failed_genes}")

    if len(promoter_records) == 0:
        print("No promoter sequences retrieved. Stopping.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 4: Writing output files")
    print("=" * 60)

    write_promoter_csv(promoter_records, filtered_df, "promoter_sequences.csv")

    print("\nDone.")