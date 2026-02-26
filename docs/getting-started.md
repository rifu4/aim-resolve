# Getting Started

This guide walks through the two main ways to use aim-resolve:

1. **As a Python library** — build and optimise sky models programmatically.
2. **Via the Snakemake pipeline** — automated end-to-end reconstruction.

A full worked example is available in the
[`demos/yaml_nifty.ipynb`](https://github.com/rifu4/aim-resolve/blob/main/demos/yaml_nifty.ipynb)
notebook.

---

## Using aim-resolve as a library

aim-resolve is configured through YAML files.  The top-level entry point is
[`get_builders`][aim_resolve.builders.get_builders], which maps YAML section keys
to the corresponding builder functions.

### Minimal example

```python
import yaml
from aim_resolve.builders import get_builders
from aim_resolve.optimize.yml import build_from_yml

# Load a YAML config (see demos/config/base.yml for a full example)
with open("demos/config/base.yml") as f:
    cfg = yaml.safe_load(f)

# Run reconstruction
build_from_yml(cfg)
```

### Key concepts

| Concept | Class / function | Description |
|---|---|---|
| Sky grid | [`SignalGrid`][aim_resolve.model.grid.SignalGrid] | Pixel grid with physical distances |
| Sky component | [`SignalModel`][aim_resolve.model.signal.SignalModel] | Single-component sky model |
| Multi-component | [`ComponentModel`][aim_resolve.model.components.ComponentModel] | Background + point sources + objects |
| Noise model | [`NoiseModel`][aim_resolve.model.noise.NoiseModel] | Scaling / variance-covariance noise |
| Clustering | [`dbscan_clustering`][aim_resolve.clustering.dbscan_clustering] | Identifies objects in a reconstruction |

---

## Running the Snakemake pipeline

The pipeline lives in `steering/`. There are three modes:

| Mode | Response | Use case |
|---|---|---|
| `image` | Unit | Synthetic image data |
| `radio` | Radio interferometric | Real visibility data |
| `fast` | Fast-resolve | Large-scale radio data |

Set the mode and parameters in `steering/config/snake.yml`, then run:

```bash
cd steering
snakemake --cores 1
# with GPU:
snakemake --cores 1 --config cuda_device=0
```

### Extending a pipeline result

After a completed run, extend the model (e.g. to higher resolution or more frequencies):

```bash
python3 nifty/extend_rec.py --config nifty/config/zoom_ext.yml --cuda_device 0
```

### Training the U-Net

```bash
python3 train/train_model.py --config train/config/unet.yml
```
