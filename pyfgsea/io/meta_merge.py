import pandas as pd


def merge_metadata_safe(adata, meta_path, allow_positional_merge=False):
    meta = pd.read_csv(meta_path)
    # Basic merge implementation
    # In real world, this should be more robust
    if allow_positional_merge and len(meta) == len(adata):
        # Assume order matches if no index provided
        adata.obs = pd.concat(
            [adata.obs.reset_index(drop=True), meta.reset_index(drop=True)], axis=1
        )
    else:
        # Merge on index if possible
        # For now just print warning
        print("Warning: Metadata merge logic is simplified.")
    return adata
