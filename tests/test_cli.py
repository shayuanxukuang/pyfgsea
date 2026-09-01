from click.testing import CliRunner

from pyfgsea.cli import main as cli_module


def test_cli_help_does_not_import_trajectory_extra():
    result = CliRunner().invoke(cli_module.cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "run" in result.output


def test_cli_forwards_output_directory(monkeypatch):
    sentinel = object()
    observed = {}

    monkeypatch.setattr(cli_module, "load_adata", lambda _path: sentinel)

    def fake_run_pipeline(adata, **kwargs):
        observed["adata"] = adata
        observed.update(kwargs)

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "run",
            "--h5ad",
            "input.h5ad",
            "--gmt",
            "sets.gmt",
            "--out",
            "evidence-output",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["adata"] is sentinel
    assert observed["gmt_path"] == "sets.gmt"
    assert observed["output_dir"] == "evidence-output"
    assert observed["pseudotime_key"] == "dpt_pseudotime"


def test_cli_explains_missing_trajectory_extra(monkeypatch):
    def missing_extra(_path):
        raise ImportError("scanpy is absent")

    monkeypatch.setattr(cli_module, "load_adata", missing_extra)
    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "--h5ad", "input.h5ad", "--gmt", "sets.gmt"],
    )

    assert result.exit_code != 0
    assert "pyfgsea[trajectory]" in result.output
