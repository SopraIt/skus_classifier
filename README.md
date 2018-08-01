# Table of Contents

* [Scope](#scope)
  * [Problem](#problem)
* [Dataset](#dataset)
  * [Normalizer](#normalizer)
  * [Augmenter](#augmenter)
  * [Loader](#loader)
* [Classifier](#classifier)
* [APIs](#apis)
  * [Dataset creation](#dataset-creation)
  * [Classification](#classification)
  * [Warning](#warning)


## Scope
This is a machine learning (ML) project aimed to classify SKUs basing on the catalog images for that particular brand. Custom snapshots can also be used to classify the SKU.

### Problem
The whole system is a supervised classifier: the catalog images, associated by related SKU code, compose the training set.

## Dataset
The system is aimed to work with images of different sizes, saved as transparent PNG files.

### Normalizer
In order to allow the classifier working properly, all of the images are normalized by:
- resizing them to the specified max size (default to 256 pixels)
- quantizing the PNG to reduce the number of features
- applying a squared, transparent canvas and centering the image on it, thus avoiding
  any deformation

### Augmenter
The used dataset has about 5/7 images per label: this greatly limit the efficiency of the classifier, so data augmentation has been performed on the whole dataset.

A total of about two hundreds (200) images are created for each single one, by applying different transformations:
- rescaling and cropping
- adding random noise
- rotating
- adjusting contrast
- adjusting gamma colors
- blurring
- flipping (horizontally and vertically)

### Loader
A loader is responsible to load images from the specified folder, normalize them and augment the dataset.
At the end of the process a compressed [HDF5](https://www.h5py.org/) file is created, allowing other programs to consume it easily.

## Classifier
The classifier program relies on [scikit-learn](http://scikit-learn.org/stable/index.html) Stochastic Gradient Descent (SGD) model.  
The Stochastic Gradient Descent (SGD) has been selected because it works reasonably 
well and is pretty fast.

The dataset can also be fed by others models, such as more advanced neural network based on the [Tensorflow](https://www.tensorflow.org/) framework.

## APIs
The interface to the existing programs are exposed by CLI:

### Dataset creation
This program creates a brand new dataset by sequentially applying the actions previously described on the images fetched by the file system:

```shell
$ python cli_dataset.py -h
usage: cli_dataset.py [-h] [-f FOLDER] [-s SIZE] [-m MAX] [-c CUTOFF] [-b BKG]
                      [-l {debug,info,warning,error,critical}]

Create a HDF5 dataset on current path by normalizing and augmenting the images
fetched from specified source

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        the folder containing the PNG files, default to
                        ./images
  -s SIZE, --size SIZE  the max size in pixels used to normalize the dataset,
                        default to 64
  -m MAX, --max MAX     limit the number of images read from disk, default to
                        unlimited
  -c CUTOFF, --cutoff CUTOFF
                        a float value indicating the cutoff percentage of the
                        transformations to be applied, default to 1.0 (all
                        transformations, about 200 per image)
  -b BKG, --bkg BKG     an optional path to an image to be applied as a
                        background before normalization
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```

#### Warning
Dataset creation can be memory angry in case you have thousands of images (which you'll augment accordingly), causing memory swapping and occasional crashes.

In such scenario you have the following options:
1. reduce the size of the normalized image (`-s` option)
2. reduce the number of images fetched from disk (`-n` option)
3. reduce the number of augmentations (`-a` option)

### Classification
This program accepts a path to the image to classify against the previously created dataset by returning the predicted label:

```shell
$ python cli_classifier.py -h
usage: cli_classifier.py [-h] -d DATASET -i IMG
                         [-l {debug,info,warning,error,critical}]

Classify the specified image versus the previously created dataset

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        runs classification on the specified dataset
                        (previously created)
  -i IMG, --img IMG     the path to the PNG image to classify
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```
