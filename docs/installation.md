# Installation

## Requirements

- Python 3.11 or 3.12
- JAX (CPU or CUDA 12)

## Core installation

Clone the repository and install with pip:

```bash
git clone https://github.com/rifu4/aim-resolve.git
cd aim-resolve
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/rifu4/aim-resolve.git
```

## Optional extras

### Training (`[train]`)

Required for training the U-Net segmentation model:

```bash
pip install "aim-resolve[train]"
```

Installs: `torch`, `lightning`, `segmentation-models-pytorch`, `neuraloperator`, `wandb`.

### Snakemake pipeline (`[pipeline]`)

Required for running the full automation pipeline:

```bash
pip install "aim-resolve[pipeline]"
```

Installs: `snakemake`.

### Radio imaging (`[radio]`)

Required for radio interferometric reconstruction with `resolve` and multi-frequency
modelling with `jubik`:

```bash
pip install "aim-resolve[radio]"
```

!!! note
    `ift-resolve` requires a C++17 compiler and `pybind11 >= 2.6`.  
    It must be cloned with `--recursive` to fetch the bundled `ducc` submodule:

    ```bash
    pip install "pybind11>=2.6" setuptools
    git clone --recursive https://gitlab.mpcdf.mpg.de/ift/resolve
    pip install ./resolve
    ```

### All extras

```bash
pip install "aim-resolve[train,pipeline,radio]"
```

## Using pixi

The repository ships a [`pixi.toml`](https://github.com/rifu4/aim-resolve/blob/main/pixi.toml) for fully reproducible environments.

```bash
# CPU environment (default: all extras)
pixi run -e cpu pytest tests/

# CUDA environment
pixi install  # sets up default (CUDA) environment
```
