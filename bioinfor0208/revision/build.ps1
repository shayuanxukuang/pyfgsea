$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

python .\generate_revision_assets.py

latexmk -pdf -interaction=nonstopmode -halt-on-error .\main_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error .\supplementary_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error .\response_to_reviewers.tex
