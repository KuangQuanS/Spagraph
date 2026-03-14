# ST_Graduation_Project Agent Guide

## Snapshot
- This repository is a research codebase centered on the `spagraph` Python package.
- The main pipeline is Stage 1 `vae` -> Stage 2 `deconv` -> Stage 3 `cellcom`.
- The repo mixes package code, experiment entrypoints, generated evaluation outputs, figures, and thesis material.
- Treat this as a live working directory, not a clean library template.

## What Is Ground Truth
- Public Python entrypoints live in `spagraph/__init__.py` and `spagraph/training/__init__.py`.
- Actual behavior is more trustworthy than some docstrings and old comments.
- `pyproject.toml` exists, but the repo has no real top-level `README.md` even though `pyproject.toml` references one.
- Some notebook outputs and comments contain encoding artifacts; verify by reading code paths and generated files instead of trusting prose.

## Repository Map
- `spagraph/`: core package code.
  - `training/vae.py`: Stage 1 wrapper.
  - `training/deconv.py`: Stage 2 wrapper and auto-k orchestration.
  - `training/cellcom.py`: Stage 3 wrapper.
  - `models/stage1.py`: Stage 1 implementation details.
  - `models/stage2.py`: Stage 2 implementation and output writing.
  - `cellcom/`: Stage 3 model, graph building, evaluation, LR scoring.
- `run_notebook/`: canonical location for dataset-specific experiment entrypoints.
- `spagraph_data/database/`: source datasets and some preprocessing artifacts.
- `spagraph_data/evaluate/`: generated experiment outputs, benchmark comparisons, plots, and many historical runs.
- `spagraph_data/thesis/`: thesis materials, not runtime code.
- `other_method/`: legacy comparison notebooks; currently already deleted in git status.
- Root scripts:
  - `run.py`: simulated dataset batch runner.
  - `evaluate_benchmark_metrics.py`, `lr_plot.py`: evaluation/plotting helpers.
  - `cellchat_human.csv`: required by Stage 3 and expected at repo root.

## Stage Semantics
- Stage 1 `spagraph.vae(...)`
  - Integrates scRNA + ST and returns `Stage1Artifacts`.
  - Current implementation is effectively memory-first.
  - Even when `output_dir` is provided, Stage 1 no longer saves `.pth` / `.npz` artifacts; stale docstrings still mention file mode.
  - What it does save when `output_dir` is set: `config_vae.txt` and modality-alignment plots.
- Stage 2 `spagraph.deconv(...)`
  - Consumes `Stage1Artifacts` and writes the main deconvolution outputs.
  - Canonical outputs include `*_composition.csv`, optional `*_reconstructed.csv`, optional `*_spot_cell_expr.csv`, training curves, and `config_deconv.txt`.
  - Stage 3 depends on Stage 2 outputs, especially `*_composition.csv` and `*_spot_cell_expr.csv`.
- Stage 3 `spagraph.cellcom(...)`
  - Requires `deconv_dir` and `st_h5ad`.
  - Searches `deconv_dir` for `*_composition.csv` and `*_spot_cell_expr.csv`.
  - If Stage 2 was not run with `save_reconstructed_genes=True`, Stage 3 will fail due to missing `*_spot_cell_expr.csv`.

## Execution Workflow For Agents
1. Start from package entrypoints, not notebooks.
2. Confirm whether the task is about pipeline logic, a dataset run, evaluation, or thesis material.
3. Only open a notebook after identifying the relevant stage wrapper or generated output naming convention.
4. Before changing anything, run `git -c safe.directory=D:/ST_Graduation_Project status --short` because the worktree is usually dirty.
5. Avoid editing anything under `spagraph_data/` unless the user explicitly asks for data/result changes.
6. When debugging Stage 3, first verify the existence of `*_composition.csv`, `*_spot_cell_expr.csv`, and root `cellchat_human.csv`.

## Environment Reality
- This repo now has a project-local `.venv` managed with `uv`.
- Preferred interpreter: `.venv\Scripts\python.exe`
- `uv sync` is not currently the safest bootstrap path for this repo.
- Reason: the combination of `pyproject.toml` and transitive dependencies can resolve into incompatible `llvmlite` paths on Python 3.10.
- Safer bootstrap pattern for this repo is:
  - `uv pip install --python .venv -r requirements.txt --cache-dir .uv-cache`
  - then install any missing runtime extras such as `torch-geometric`
- Do not assume `python` on PATH exists in shell sessions; call the interpreter via `.venv\Scripts\python.exe` when you need certainty.

## GPU Notes
- This machine has an NVIDIA GPU available via `nvidia-smi`.
- The project should prefer a conservative CUDA-enabled PyTorch build over the newest release.
- If Torch behavior looks wrong, verify all three:
  - `torch.__version__`
  - `torch.version.cuda`
  - `torch.cuda.is_available()`

## Known Pitfalls
- Many experiment scripts and notebooks use machine-specific absolute paths such as `/home/...` or `/mnt/d/...`.
- Some historical files in git status are already deleted or relocated; do not restore them unless the user asks.
- `spagraph/cellcom/cellcom_model.py` is currently user-modified in the worktree; do not overwrite unrelated changes there.
- `spagraph_data/` contains multiple old virtual environments; they are not the authoritative project environment.
- Output naming is inconsistent between datasets and historical runs. Match by suffix patterns rather than assuming exact filenames.
- `run_notebook/` is the preferred home for dataset-specific scripts and notebooks; avoid adding new dataset entrypoints at repo root.

## Minimal Checks
- Git status:
  - `git -c safe.directory=D:/ST_Graduation_Project status --short`
- Import check:
  - `.venv\Scripts\python.exe -c "import spagraph; print('ok')"`
- Torch/CUDA check:
  - `.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- Inspect dataset entrypoints:
  - `Get-ChildItem run_notebook -File`

## Editing Rules For Future Agents
- Keep reusable logic in `spagraph/`; keep dataset-specific orchestration in `run_notebook/`.
- Prefer changing wrappers and package code over rewriting notebook cells.
- Do not assume Stage 1 file-based artifacts exist just because older comments say they do.
- If a task mentions cell communication failure, immediately check whether Stage 2 was run with `save_reconstructed_genes=True`.
- If a task is only about cleanup, avoid moving files inside `spagraph_data/` because they double as experiment records.
