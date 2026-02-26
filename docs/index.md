# aim-resolve

**Automatic Identification and Modeling for Bayesian Radio Interferometric Imaging**

aim-resolve is a Python package and [Snakemake](https://snakemake.github.io) pipeline for automatic Bayesian sky model selection and reconstruction of radio interferometric data. It combines [NIFTy](http://ift.pages.mpcdf.de/nifty/) and [resolve](http://ift.pages.mpcdf.de/resolve/) with deep learning and clustering algorithms for object recognition and separation.

![Procedure overview](procedure.png)

Initialized with a single background model capturing the whole field of view in step (0), the method produces a preliminary reconstruction of the data in step (d). It then iterates over:

- **(a) Identification** — A [U-Net](https://arxiv.org/abs/1505.04597) trained on synthetic images, together with [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) clustering, identifies point sources and extended objects in the reconstruction.
- **(b) Modeling** — A new model configuration file is created, adding the detected components to the existing background model.
- **(c) Separation and Pre-fit** — The new model is fitted to the current reconstruction. Masking the background efficiently separates point sources and extended objects.
- **(d) Reconstruction** — The pre-fitted model is further optimised on the raw data, producing a refined full-sky image for the next iteration.

## Quick links

- [Installation](installation.md)
- [Getting Started](getting-started.md)
- [API Reference](api/core.md)
- [GitHub](https://github.com/rifu4/aim-resolve)
- [Paper](https://arxiv.org/abs/2512.04840)
