# Copilot / AI agent instructions for this repository

This project is an exploratory quantitative strategy workspace. The primary artifact is
the Jupyter notebook `trend_pullback_strategy.ipynb` in the repository root and a local
Python virtual environment in `quant-strat/` (Python 3.13). There are no project-level
modules or tests yet. Use the guidance below when making changes or generating code.

Key facts (discovered):

- Notebook: `trend_pullback_strategy.ipynb` — this contains experiments, data loading,
  plotting and model/prototype code.
- Virtualenv: `quant-strat/` — use `source quant-strat/bin/activate` or the venv's
  `python`/`pip` directly (`quant-strat/bin/python`, `quant-strat/bin/pip`).
- Python version: 3.13 (see `quant-strat/pyvenv.cfg`).
- Installed libs visible in `quant-strat/lib/python3.13/site-packages/` include
  numpy, pandas, numba, matplotlib, jupyter, debugpy and related packages.

What the AI should prioritize:

- Preserve notebook outputs and structure unless the user asks to refactor. When
  extracting code from the notebook, move logic into a new package (suggested name
  `strategy/` or `src/strategy/`) and add a small runner script that the notebook can
  import. This keeps the notebook readable and makes code testable.
- Never modify files under `quant-strat/` (the virtualenv) or `quant-strat/lib/...`
  — treat them as environment artifacts.

Commands and workflows to assume (explicit examples):

- Activate venv in macOS zsh:
  source quant-strat/bin/activate
- Run the notebook (after activating the venv):
  jupyter notebook trend_pullback_strategy.ipynb
- Use the venv's python to run scripts or tests directly:
  quant-strat/bin/python path/to/script.py
  quant-strat/bin/pip install -r requirements.txt # create requirements.txt if needed

Project-specific conventions and recommendations (discoverable patterns):

- Single-notebook-first: the repo is exploratory. Prefer incremental refactors: extract
  a small function or class from a notebook cell into `strategy/<module>.py` and then
  import it back into the notebook to validate behavior.
- Dependency management: there is no requirements file in the repo. If you add one,
  pin versions and place it at repository root (e.g., `requirements.txt`). Use the
  venv's pip to install or freeze dependencies: `quant-strat/bin/pip freeze > requirements.txt`.
- Tests: there are no tests yet. If adding tests, use `pytest` and put tests under
  `tests/` (e.g., `tests/test_<module>.py`). Run them with the venv: `quant-strat/bin/python -m pytest`.

Integration points and external assumptions:

- Data and external APIs are expected to be loaded inside the notebook; no dedicated
  data/ directory or connectors exist yet. If adding data loaders, put them under
  `strategy/data_loader.py` and document expected input paths in the README.

AI coding constraints and tips:

- Keep changes minimal and reversible. Prefer creating new files over editing large
  notebook cells in-place unless the user asks for a notebook-only change.
- Avoid editing anything in `quant-strat/` and do not add or remove files from
  `quant-strat/lib/...`.
- When you extract code from the notebook, include a short usage example and a
  corresponding small unit test that demonstrates the expected behavior.

Files worth reading for context:

- `trend_pullback_strategy.ipynb` — main notebook with the strategy prototype.
- `quant-strat/pyvenv.cfg` and `quant-strat/bin/activate` — environment details.

If anything below is unclear or you need more repository-level details (tests, CI,
or packaging conventions), ask the repo owner before making broad changes.

Please review and tell me which sections you want expanded or any examples you'd
like added (e.g., a sample `strategy/` module and one unit test).
