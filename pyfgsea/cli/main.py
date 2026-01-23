
import click
import os
from ..api import run as run_pipeline
from ..io.anndata_io import load_adata
from ..io.meta_merge import merge_metadata_safe

@click.group()
def cli():
    pass

@cli.command()
@click.option('--h5ad', required=True, help='Path to .h5ad file')
@click.option('--gmt', required=True, help='Path to .gmt file')
@click.option('--out', default='results', help='Output directory')
@click.option('--pseudotime-key', default='dpt_pseudotime', help='Key for pseudotime in adata.obs')
@click.option('--meta', default=None, help='Optional metadata CSV to merge')
@click.option('--allow-positional-merge', is_flag=True, help='Allow merging metadata by position (DANGEROUS)')
def run(h5ad, gmt, out, pseudotime_key, meta, allow_positional_merge):
    """Run the Universal Trajectory GSEA pipeline."""
    print(f"Loading {h5ad}...")
    adata = load_adata(h5ad)
    
    if meta:
        print(f"Merging metadata from {meta}...")
        adata = merge_metadata_safe(adata, meta, allow_positional_merge=allow_positional_merge)
        
    run_pipeline(
        adata,
        gmt_path=gmt,
        pseudotime_key=pseudotime_key,
        outdir=out
    )

if __name__ == '__main__':
    cli()