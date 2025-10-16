# Lab 1 — Engineering Practices for ML Research (Refactor + Reproducibility)

This lab starts from a single, messy notebook with exploratory experiments on a regression task. Your goal is to refactor the codebase so that the notebook becomes a thin, visualization-oriented report while all logic (data loading, preprocessing, training, evaluation) moves into importable Python modules. You should also make the project reproducible, fix dependencies, and improve code style with ruff.

Use the `reference_solution/` folder as inspiration for a possible end state. It is not the only correct solution; structure the code as you see fit, as long as you achieve the goals below.

## Objectives
- Extract all logic from the notebook into `*.py` modules under `src/`.
- Reduce code duplication via functions and/or pipelines.
- Ensure reproducibility (seed control, deterministic CV, pinned deps, controlled randomness).
- Fix dependency management (`requirements.txt`) so the project installs and runs from scratch.
- Improve code style using ruff (lint and format) and make the code easier to read and collaborate on.

## Starter Layout and Target Shape
You start with:
```
initial_state/
├─ notebooks/experiment.ipynb       # contains spaghetti code
└─ (this) README.md                 # assignment instructions
```

Proposed target shape after refactor (you may adjust as appropriate):
```
initial_state/
├─ notebooks/experiment.ipynb       # thin: imports code, runs, visualizes only
├─ src/
│  ├─ __init__.py                   # make src a module
│  ├─ data.py                       # dataset loading/splitting utilities
│  ├─ train.py                      # training, CV, hyperparam search, evaluation
└── requirements.txt                # pinned dependencies for reproducibility
```

## Environment Setup (Conda)
Use an isolated environment to avoid dependency conflicts.

1) Install Conda: [Conda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2). Don't forget about the `conda init` step to set up your shell.

2) Create and activate an environment (example with Python 3.11):
```
conda create -n ml-lab1 python=3.11 -y
conda activate ml-lab1
```

3) Install dependencies once your `requirements.txt` is ready:
- Using pip inside conda:
```
pip install -r requirements.txt
```
- Or with conda-forge (when packages are available):
```
conda install -c conda-forge --file requirements.txt
```

Tip: pin versions for reproducibility (e.g., `numpy==1.26.4`). Prefer exact pins (`==`) over loose ranges, you can use `pip freeze > requirements.txt` to capture the current environment's state.

## Reproducibility Checklist
- Seed all sources of randomness used:
  - NumPy: `np.random.seed(SEED)`
  - Python `random`: `random.seed(SEED)`
  - scikit-learn: use `random_state=SEED` wherever applicable (e.g., `KFold(shuffle=True, random_state=SEED)`, `train_test_split(..., random_state=SEED)`).
- Use deterministic CV: enable `shuffle=True` with a fixed `random_state` for splitters.
- Avoid data leakage: build preprocessing inside a `Pipeline` fit on training data only.
- Record configuration: keep key parameters (seed, CV, model hyperparams) centralized in code and/or a simple config file.
- Pin dependencies: ensure `requirements.txt` contains exact versions that work together.

## Refactor Requirements
Move everything except plotting/visualization from the notebook into `src/`, decomposition can look like this:
- Data loading and splitting (e.g., California Housing) → `src/data.py`
- Pipeline definition, training loop with `GridSearchCV` or similar, including CV splitter → `src/train.py`
- Evaluation helpers (metrics computation) and utilities (seeding, paths) → `src/train.py` / `src/utils.py`

You are free to create your own code structure, but aim for clarity and reusability. The goal is to have a clean separation of concerns where the notebook serves as a high-level report that orchestrates the experiments.

The notebook should:
- Import functions from `src/`.
- Call them to run experiments.
- Produce visualizations (residual histograms, predicted vs. true plots, etc.).
- Avoid duplicating logic already implemented in `src/`.

## Importing Your Modules in the Notebook
Modify `sys.path` in the first cell of your modified notebook to include the `src/` directory, allowing you to import your modules easily. Here’s an example of how to do this:

```python
import sys, os
sys.path.append(os.path.abspath(".."))  # add repo root to Python path

from src.data import load_dataset
from src.utils import set_seed
from src.train import train_model, evaluate
```

## Running Ruff (Lint + Format)
Ruff can both lint and format your code. Install it into your environment and run it at the repo root (`initial_state/`).

- Install:
```
pip install ruff
```

- Lint the codebase:
```
ruff check .
```

- Auto-fix trivial issues:
```
ruff check --fix .
```

- Format the code:
```
ruff format .
```

## Dependency Management (`requirements.txt`)
Create or fix `requirements.txt` so a teammate can install and run the project on a fresh machine.
- Include only what you actually use (e.g., `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyterlab`, `ruff`).
- Pin exact versions (`==`).
- Verify a fresh setup by installing into a clean environment and running the notebook end-to-end.
- Add `nbstripout` to keep notebooks output-free for submission.

## Acceptance Criteria
- Notebook is clean and thin: imports code, sets a seed, runs experiments, and visualizes. It should not contain any outputs.
- All core logic lives under `src/` and is reusable across experiments.
- Reproducibility measures are in place (seeds, deterministic CV, pinned deps).
- `requirements.txt` installs successfully; repo runs from scratch in a clean environment.
- Ruff passes without major issues (`ruff check .` shows no errors after fixes; code is formatted).

## Tips
- Start by adding requirements, creating a new environment, installing dependencies and making sure notebook works in it's initial state.
- Proceed by identifying duplicated code blocks in the notebook and extract them into functions.
- Work module by module, moving logic from the notebook to `src/`.
- Finish by formatting the code and running `nbstripout` on the modified notebook.

Good luck, and keep the notebook focused on storytelling and visualization while the reusable logic lives in your modules!
