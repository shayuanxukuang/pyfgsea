import click


def load_adata(*args, **kwargs):
    from ..io.anndata_io import load_adata as implementation

    return implementation(*args, **kwargs)


def merge_metadata_safe(*args, **kwargs):
    from ..io.meta_merge import merge_metadata_safe as implementation

    return implementation(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from ..api import run as implementation

    return implementation(*args, **kwargs)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--h5ad", required=True, help="Path to .h5ad file")
@click.option("--gmt", required=True, help="Path to .gmt file")
@click.option("--out", default="results", help="Output directory")
@click.option(
    "--pseudotime-key", default="dpt_pseudotime", help="Key for pseudotime in adata.obs"
)
@click.option("--meta", default=None, help="Optional metadata CSV to merge")
@click.option(
    "--allow-positional-merge",
    is_flag=True,
    help="Allow merging metadata by position (DANGEROUS)",
)
def run(h5ad, gmt, out, pseudotime_key, meta, allow_positional_merge):
    """Run the Universal Trajectory GSEA pipeline."""
    print(f"Loading {h5ad}...")
    try:
        adata = load_adata(h5ad)
    except ImportError as error:
        raise click.ClickException(
            "trajectory support is not installed; run "
            "pip install 'pyfgsea[trajectory]'"
        ) from error

    if meta:
        print(f"Merging metadata from {meta}...")
        adata = merge_metadata_safe(
            adata, meta, allow_positional_merge=allow_positional_merge
        )

    run_pipeline(adata, gmt_path=gmt, pseudotime_key=pseudotime_key, output_dir=out)


if __name__ == "__main__":
    cli()
