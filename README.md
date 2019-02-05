# Table of Contents

* [Scope](#scope)
  * [Problem](#problem)
* [Setup](#setup)
  * [Versions](#versions)
  * [Virtualenv](#virtualenv)
  * [Installation](#installation)
* [Dataset](#dataset)
  * [Normalizer](#normalizer)
  * [Augmenter](#augmenter)
  * [Loader](#loader)
* [Classifier](#classifier)
* [APIs](#apis)
  * [Console](#console)
  * [CLI](#cli)
  * [HTTP](#http)



## Scope
This is a machine learning (ML) project aimed to classify SKUs basing on the catalog images for that particular brand. Custom snapshots can also be used to classify the SKU.

### Problem
The whole system is a supervised classifier: the catalog images, associated by related SKU code, compose the training set.

## Setup

### Versions
The library is compatible and it has been tested with the following python versions:
* 3.4.8
* 3.6.4
* 3.7.1

### Virtualenv
We suggest to isolate your installation via python virtualenv:
```shell
python3 -m venv .skusclf
...
source .skuclf/bin/activate
```

### Installation
Update `pip` package manager:
```shell
pip install pip --upgrade
```

Install the external dependencies by requirements file:
```shell
pip install -r requirements.txt
```

## Dataset
The system is aimed to work with images of different sizes, saved as PNG or JPG files (supporting RGBA conversion).

### Normalizer
In order to allow the classifier working properly, all of the images are normalized by:
- resizing them to the specified max size (default to 256 pixels)
- optionally applying a squared, transparent canvas and centering the image on it, thus avoiding any deformation

### Augmenter
In order to increase the accuracy of the classifier data augmentation is performed on the whole dataset.

The number of images is augmented by an order of magnitude by applying different transformations to the original one:
- rescaling and cropping
- adding random noise
- rotating
- adjusting gamma
- blurring
- flipping (horizontally and vertically)

### Loader
A loader is responsible to load images from the specified folder, normalize them and augment the dataset.
At the end of the process a compressed ZIP or [H5](https://www.h5py.org/) file is created, allowing other frameworks to consume it easily.

## Classifier
The classifier program relies on [scikit-learn](http://scikit-learn.org/stable/index.html) Stochastic Gradient Descent (SGD) model.  
The Stochastic Gradient Descent (SGD) has been selected because it works reasonably well and is pretty fast.

The ZIP dataset can also be fed by others models, such as more advanced neural network based on the [Tensorflow](https://www.tensorflow.org/) framework.

## APIs
The interface to the existing programs are exposed by CLI:

### Console
To import the created dataset within your Jupyter/ipython console, just type the following snippets:
```python
from matplotlib import pyplot as plt
from skusclf import training as tr

# you need to split manaully the dataset in test, validation and training
X, y = tr.DatasetH5.load('./dataset_MM_256x256.h5')

# images data are flattened within the dataset, in case you need to display an 
# image with its original shape, use the following flag:
X, y = tr.DatasetH5.load('./dataset_MM_256x256.h5', orig=True)
plt.imshow(X[0])
```

### CLI
The library comes with a CLI interface aimed to create a new dataset (ZIP or H5):

```shell
$ python cli_dataset.py -h
usage: cli_dataset.py [-h] [-k {zip,h5}] -f FOLDER [-s SIZE] [-m MAX]
                      [-c CUTOFF] [-b BKG] [--brand {plain,mm,gg}]
                      [-l {debug,info,warning,error,critical}]

Create a ZIP or H5 dataset on current path by normalizing and augmenting the
images fetched from specified source

optional arguments:
  -h, --help            show this help message and exit
  -k {zip,h5}, --kind {zip,h5}
                        the dataset kind, can be an uploadable ZIP or a H5
                        file to be used by a Python framework, deafult to zip
  -f FOLDER, --folder FOLDER
                        the folder containing the image files
  -s SIZE, --size SIZE  the max size in pixels used to normalize the dataset,
                        default to 32
  -m MAX, --max MAX     limit the number of images read from disk, default to
                        unlimited
  -c CUTOFF, --cutoff CUTOFF
                        a float value indicating how many transformations to
                        apply, default to 1.0 (all transformations)
  -b BKG, --bkg BKG     if specified, apply a squared canvas behind each
                        image, can be True (white for RGB, transparent for
                        RGBA), a specific RGB string (i.e. FF00FF) or a path
                        to an existing file to be used as background, default
                        to false
  --brand {plain,mm,gg}
                        specify how to fetch labels from images, default to
                        plain file basename
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```

#### Warning
H5 dataset creation can be memory angry in case you have thousands of images (which you'll augment accordingly), causing memory swapping and occasional crashes on non optimised devices.

In such scenario you have the following options:
1. reduce the size of the normalized image (`-s` option)
2. reduce the number of images fetched from disk (`-n` option)
3. reduce the number of augmentations (`-a` option)


### HTTP
The prediction can be performed via a HTTP interface (courtesy of the Tornado application server). 
The server will start at the specified port (default to `8888`) and will immediately try to load the specified dataset (default to last h5 file on cwd) by classifying versus specified model (default di SGD):

```shell
$ python http_classifier.py --port=9292 --dataset=my_dataset.h5 --model=rf
Loading and fitting my_dataset.h5
Accepting connections on 9292
```

Use the plain form to upload the SKU image and check the prediction against the dataset by clicking on the `predict` button.  
