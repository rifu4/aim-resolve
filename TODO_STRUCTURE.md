# AIM-Resolve — Remaining Structural Cleanup

Status overview of the package-structure improvements discussed earlier.

## Done

- [x] **Explicit imports in `__init__.py`** — Replaced all wildcard `from .X import *` with named imports.
- [x] **Ruff + pre-commit** — Configured `ruff` linter/formatter in `pyproject.toml`, added `.pre-commit-config.yaml`, hooks installed.
- [x] **Fix all lint errors** — Auto-fixed + manually resolved every ruff finding (B006, E731, SIM115, UP038, …). 0 errors remaining.
- [x] **Docstrings** — Added/standardised docstrings across `src/` and `steering/`.
- [x] **Python version alignment** — Bumped `pyproject.toml` to `requires-python = ">=3.11"`, removed 3.10 classifier. Consistent with `pixi.toml` and `ruff` target.
- [x] **Project metadata** — Added description and keywords to `pyproject.toml`.
- [x] **Reduce pixi.toml duplication** — Shared deps in default feature, CUDA/CPU only override JAX + cuda-version. ~50% fewer lines.
- [x] **Optional dependencies** — Moved `torch`, `lightning`, `segmentation-models-pytorch`, `neuraloperator`, `wandb` to `[train]` extra; `snakemake` to `[pipeline]` extra. Added `[radio]` extra for `jubik0` + `resolve` (both not on PyPI — installed via `git+` URL). Guarded `train` imports with `try/except`. `jubik0` guarded in `spectral.py`; `resolve` is already lazily imported inside functions.
- [x] **CI / CD** — Added `.github/workflows/ci.yml`: ruff lint + format check + pytest on cpu environment via `prefix-dev/setup-pixi`. Separate `test-radio` job clones `resolve` (with submodules) and `jubik0` from GitLab; marked `continue-on-error: true` since GitLab access is not guaranteed in public CI.

- [x] **Shared Test Fixtures (`conftest.py`)** — Added `tests/conftest.py` with `jax_key`, `small_grid` (8×8), `medium_grid` (16×16), `large_grid` (32×32), `background_signal`, and `rng`. Removed duplicate local `grid` fixtures from `test_prior.py` and `test_spectral.py`; delegated `background` in `test_components.py` to the shared fixture.

---

## Still Open

### 1. Documentation Build

**Priority: medium**

`docs/` only contains a single image (`procedure.png`). No Sphinx / MkDocs config exists.

**Action:**
- Set up a documentation tool (e.g. `mkdocs-material` or `sphinx` + `autodoc`).
- Auto-generate API reference from the existing docstrings.
- Add a getting-started guide (the `demos/` notebook is a good basis).

---

### 2. PEP 561 Type Marker (`py.typed`)

**Priority: low**

Add an empty `src/aim_resolve/py.typed` file so type checkers (mypy, pyright) recognise the package as typed. This is especially useful as the codebase already has some type annotations.

---

### 3. Separate `steering/` from the Library Package

**Priority: low**

`steering/` contains experiment scripts, Snakemake workflows, and run configurations. It is not part of the installable package but lives inside the repo.

Consider:
- Excluding it from the sdist/wheel via `[tool.setuptools.packages.find] exclude = ["steering*"]` (likely already the case since it's outside `src/`).
- Moving it to a separate repo or clearly documenting that it is *not* shipped with the library.
- Adding a top-level `README` note or a `steering/README.md` explaining its purpose.

---

### 4. `__all__` in Subpackage `__init__.py` Files

**Priority: low**

The top-level `__init__.py` now uses explicit imports, but the subpackage `__init__.py` files (e.g. `model/__init__.py`, `optimize/__init__.py`) do not define `__all__`. Adding `__all__` there makes the public API of each subpackage explicit and prevents accidental re-exports.
