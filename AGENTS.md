# Repository Guidelines

## Project Structure & Module Organization

- `sft/`: supervised fine-tuning entrypoints (e.g., `sft/sft.py`).
- `kto/`: KTO training script (e.g., `kto/kto.py`).
- `inference/`: inference utilities and model merging (e.g., `inference/inference.py`).
- `utils/`: helper modules (data extraction, solving, templates).
- `prompts/`: prompt templates and generation helpers.
- `script/`: runnable shell scripts for training (`script/run_sft.sh`, `script/run_kto.sh`).
- `config/`: DeepSpeed configs (e.g., `config/ds_config_zero3.json`).
- `data/`: datasets and examples (`data/trainset_example/`, `data/testset/*.jsonl`).
- `docs/`: figures and documentation assets.

## Build, Test, and Development Commands

- Create env: `uv venv` then `source .venv/bin/activate`.
- Install deps (note: `requirements.txt` contains `pip install ...` lines):
  - `bash requirements.txt`, or
  - `sed 's/^pip install //' requirements.txt > requirements.lock && uv pip install -r requirements.lock`
- SFT training: `bash script/run_sft.sh` (wraps `torchrun` + DeepSpeed).
- KTO training: `bash script/run_kto.sh`.
- Inference example: `uv run python inference/inference.py`.

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8, type hints where practical.
- Naming: modules/functions `snake_case`, classes `CapWords`, constants `UPPER_SNAKE_CASE`.
- Keep scripts reproducible: prefer explicit args/env vars over hard-coded paths.

## Testing Guidelines

- No dedicated test suite is included. For small changes, run a quick sanity check:
  - `uv run python -m compileall sft kto inference utils prompts`
- If you add tests, place them under `tests/` and use `pytest` with `test_*.py` naming.

## Commit & Pull Request Guidelines

- Commits in this repo are typically short and imperative (e.g., “Update README.md”, “Fix typo …”).
- PRs should include: what changed, how to run/verify (exact commands), and any config/data impacts (paths under `config/` or `data/`).
