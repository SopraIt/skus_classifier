# Table of Contents

* [Scope](#scope)
  * [Problem](#problem)
* [Dataset](#dataset)
  * [Normalizer](#normalizer)
  * [Augmenter](#augmenter)
  * [Loader](#loader)
* [Classifier](#classifier)
* [APIs](#apis)
  * [Console](#console)
  * [CLI](#cli)
    * [Dataset creation](#dataset-creation)
    * [Prediction](#prediction)
  * [HTTP](#http)



## Scope
This is a machine learning (ML) project aimed to classify SKUs basing on the catalog images for that particular brand. Custom snapshots can also be used to classify the SKU.

### Problem
The whole system is a supervised classifier: the catalog images, associated by related SKU code, compose the training set.

## Dataset
The system is aimed to work with images of different sizes, saved as PNG or JPG files (supporting RGBA conversion).

### Normalizer
In order to allow the classifier working properly, all of the images are normalized by:
- resizing them to the specified max size (default to 256 pixels)
- applying a squared, transparent canvas and centering the image on it, thus avoiding
  any deformation

### Augmenter
The used dataset has about 5/7 images per label: this greatly limit the efficiency of the classifier, so data augmentation has been performed on the whole dataset.

The number of images is augmented by an order of magnitude by applying different transformations to the original one:
- rescaling and cropping
- adding random noise
- rotating
- adjusting gamma
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

### Console
To import the created dataset within your Jupyter/ipython console, just type the following snippets:
```python
from matplotlib import pyplot as plt
from skusclf import training as tr

# you need to split manaully the dataset in test, validation and training
ds = tr.Dataset('./dataset_MM_256x256.h5')
X, y = ds.load()

# images data are flattened within the dataset. In case you need to display an 
# image with its original shape, use the original flag:
X, y = ds.load(original=True)
plt.imshow(X[0])
```

### CLI
The library comes with two CLI interfaces:

#### Dataset creation
This program creates a brand new dataset by sequentially applying the actions previously described on the images fetched by the file system:

```shell
$ python cli_dataset.py -h
usage: cli_dataset.py [-h] -f FOLDER [-s SIZE] [-m MAX] [-c CUTOFF] [-b BKG]
                      [--brand {plain,mm,gg,vg}] [-z]
                      [-l {debug,info,warning,error,critical}]

Create a HDF5 dataset on current path by normalizing and augmenting the images
fetched from specified source

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        the folder containing the image files
  -s SIZE, --size SIZE  the max size in pixels used to normalize the dataset,
                        default to 32
  -m MAX, --max MAX     limit the number of images read from disk, default to
                        unlimited
  -c CUTOFF, --cutoff CUTOFF
                        a float value indicating the cutoff percentage of the
                        transformations to be applied, default to 0 (no
                        transformations)
  -b BKG, --bkg BKG     if specified, apply a squared canvas behind each
                        image, can be True (white for RGB, transparent for
                        RGBA), a specific RGB string (i.e. FF00FF) or a path
                        to an existing file to be used as background, default
                        to false
  --brand {plain,mm,gg,vg}
                        specify how to fetch labels from images, default to
                        file basename
  -z, --zip             if specified, creates a ZIP files containing the whole
                        dataset by using the labels to organize the images
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```

##### Warning
Dataset creation can be memory angry in case you have thousands of images (which you'll augment accordingly), causing memory swapping and occasional crashes.

In such scenario you have the following options:
1. reduce the size of the normalized image (`-s` option)
2. reduce the number of images fetched from disk (`-n` option)
3. reduce the number of augmentations (`-a` option)

#### Prediction
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
  -i IMG, --img IMG     the path to the image to classify
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```

### HTTP
The prediction can be performed via a HTTP interface (courtesy of the Tornado application server). 
The server will start at the specified port (default to 8888) and will immediately try to load the specified dataset (default to last h5 file on cwd) by classifying versus specified model (default di SGD):

```shell
$ python http_classifier.py --port=9292 --dataset=my_dataset.h5 --model=rf
Loading and fitting my_dataset.h5
Accepting connections on 9292
```

Use the plain form to upload the SKU image and check the prediction against the dataset by clicking on the `predict` button.  
If you are interested on the ML model information, click on the `model` button.
