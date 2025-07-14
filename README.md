# aim-resolve

**Automatic Identification and Modeling for Bayesian Radio Interferometric Imaging**

This repository contains a [snakemake](https://snakemake.github.io) pipeline to automatically improve Bayesian imaging of complex systems like radio interferometric wide-field observations. It combines [NIFTy](http://ift.pages.mpcdf.de/nifty/) and [resolve](http://ift.pages.mpcdf.de/resolve/) with deep learning and clustering algorithms for object recognition and separation, respectively. More specifically, it utilizes different model descriptions for different types of identified objects to improve the overall reconstruction of radio interferometric data.

![image](docs/procedure.png)

Initialized with a single background model capturing the whole field of view in step (0), the method produces a preliminary reconstruction of the data in step (d). Then, it iterates over the following steps:

- **(a) Identification**: By combining a [U-Net](https://arxiv.org/abs/1505.04597) trained on synthetic images and a clustering algorithm like [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan), point sources and extended objects are identified in the reconstruction.

- **(b) Modeling**: This step creates a model configuration file for the subsequent reconstruction iteration by adding the new components to the background model.

- **(c) Pre-fit and Separation**: The new model is first fitted to the previous reconstructed image. By masking the background, this step efficiently separates the point sources and extended objects from the background.

- **(d) Reconstruction**: The pre-fitted model is further optimized on the data. The individual components can be added together to compose a full sky image, allowing for object detection in the next iteration.


These individual steps are implemented as rules in the [snakefile](steering/snakefile) by building output files from input files using specific python scripts.


## Installation

Clone the repository and install aim-resolve via
```console
git clone https://github.com/rifu4/aim-resolve.git
cd aim-resolve
pip install -e .
```

To apply the method to radio interferometric data, [resolve](http://ift.pages.mpcdf.de/resolve/) needs to be installed via
```console
git clone --recursive https://gitlab.mpcdf.mpg.de/ift/resolve
cd resolve
pip install .
```


## Steering

### Run the snakemake pipeline

There are 3 different modes the pipeline can be used with:

- **exp**: lognormal model and image data

- **radio**: lognormal model and radio interferometric data

- **fast-radio**: same as radio, but using the fast-resolve algorithm

The mode needs to be specified along with the to be reconstructed data in the snakemake [config](steering/config/snake.yml) file. Moreover, one can set the resolution and field of view of the model, the output directory, the predicting U-net, the total number of iterations, and various other modelling and optimization parameters.

To run the snakemake pipeline, change directory to the `steering` folder and run
```
snakemake --cores 1
```


### Generate image data

To generate image data, specify the desired data parameters in the data [config](steering/data/config/data.yml) file. Then, change directory to the `steering/data` folder and run
```
python3 data_gen.py --config config/data.yml
```


### Train the U-Net on image data

To train the U-Net on generated image data, specify the desired data and training parameters in the train [config](steering/train/config/unet.yml) file. Then, change directory to the `steering/train` folder and run
```
python3 train_model.py --config config/unet.yml
```


### Optimize multi-component models using NIFTy

To run a NIFTy optimization with a pre-defined multi-component model, specify the model and optimization parameters in the desired NIFTy config file, e.g. the [CygnusA](steering/nifty/config/cyg.yml) config file. Then, change directory to the `steering/nifty` folder and run
```
python3 nifty_rec.py --config config/cyg.yml --mode total
```
where `--mode` has to be set to total or major for the mode (exp, radio) and fast-radio, respectively.

## References

papers
