def compute_lipid_enrichment(local_counts, bulk_counts):
    """
    Compute lipid enrichment factor (E) for different lipid species.
    
    Parameters:
    - local_counts: Dictionary with lipid type as key and count near the protein as value.
    - bulk_counts: Dictionary with lipid type as key and count in the whole membrane as value.

    Returns:
    - Dictionary of lipid enrichment factors.
    """
    # Compute total counts
    total_local = sum(local_counts.values())
    total_bulk = sum(bulk_counts.values())
    print(f"Total lipids near protein: {total_local}")
    print(f"Total lipids in the membrane: {total_bulk}")
    # Compute enrichment factor (E) for each lipid
    enrichment_factors = {}
    for lipid in local_counts.keys():
        F_local = local_counts[lipid] / total_local # Fraction of lipid near the protein
        F_bulk = bulk_counts[lipid] / total_bulk # Fraction of lipid in the whole membrane
        enrichment_factors[lipid] = F_local / F_bulk if F_bulk != 0 else float('inf')  # Avoid division by zero
        print(f"F_local: {F_local}, F_bulk: {F_bulk}, E: {enrichment_factors[lipid]}")

    return enrichment_factors


# Example Data
local_counts = {
    "Lipid A": 50,  # Lipids near the protein
    "Lipid B": 30
}

bulk_counts = {
    "Lipid A": 200,  # Total lipids in the membrane
    "Lipid B": 300
}

# Compute enrichment factors
enrichment_results = compute_lipid_enrichment(local_counts, bulk_counts)

# Print results
print("\nLipid Enrichment Factors (E):")
for lipid, E in enrichment_results.items():
    status = "Enriched" if E > 1 else "Depleted" if E < 1 else "Neutral"
    print(f"{lipid}: E = {E:.2f} ({status})")# Lipid Fingerprint Analysis
