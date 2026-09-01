import numpy as np
import pandas as pd
import random
from pathlib import Path


def generate_test_data(n_genes=12000, n_sets=100, seed=42):
    """Generates synthetic ranked list and gene sets for benchmarking."""
    np.random.seed(seed)
    random.seed(seed)

    # Generate gene names
    genes = [f"GENE_{i:05d}" for i in range(n_genes)]

    # Generate random scores (normal distribution with signal)
    scores = np.random.randn(n_genes)
    scores[:500] += 2.0  # Top genes up
    scores[-500:] -= 2.0  # Bottom genes down

    # Create rank dataframe
    df_rnk = pd.DataFrame({"Gene": genes, "Score": scores})
    df_rnk = df_rnk.sort_values("Score", ascending=False)

    # Generate random gene sets
    gmt = {}

    # Background Null Sets
    for i in range(n_sets - 20):
        size = random.randint(15, 200)
        members = random.sample(genes, size)
        gmt[f"NULL_PATH_{i}"] = members

    # Positive Enriched Sets
    top_genes = genes[:1000]
    for i in range(10):
        size = random.randint(20, 50)
        members = random.sample(top_genes, size)
        gmt[f"POS_PATH_{i}"] = members

    # Negative Enriched Sets
    bot_genes = genes[-1000:]
    for i in range(10):
        size = random.randint(20, 50)
        members = random.sample(bot_genes, size)
        gmt[f"NEG_PATH_{i}"] = members

    return df_rnk, gmt


def save_data(df_rnk, gmt, rank_path, gmt_path):
    rank_path = Path(rank_path)
    rank_path.parent.mkdir(parents=True, exist_ok=True)

    df_rnk.to_csv(rank_path, index=False)
    with open(gmt_path, "w") as f:
        for term, members in gmt.items():
            members_str = "\t".join(members)
            f.write(f"{term}\tNA\t{members_str}\n")
