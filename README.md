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
At the end of the process a gzipped pickled file is created, to let the dataset being consumable by further classifications.

## Classifier
The classifier program relies on [scikit-learn](http://scikit-learn.org/stable/index.html) Stochastic Gradient Descent (SGD) model.  
The Stochastic Gradient Descent (SGD) has been selected because it works reasonably 
well and is pretty fast.

The dataset can also be fed by others models, such as more advanced neural network based on the [Tensorflow](https://www.tensorflow.org/) framework.

## APIs
The only interface currently available is a CLI program that can be invoked this way:
```shell
$ python classify.py -h
usage: classify.py [-h] [-d DATASET] -i IMG [-s SIZE] [-a [1-200]] [-t TEST]
                   [-l {debug,info,warning,error,critical}]

Classify the images basing on a specific supervised model

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        runs classification on the specified dataset (fetch
                        the first *.pkl.gz available)
  -i IMG, --img IMG     the path to the PNG image to classify
  -s SIZE, --size SIZE  the max size (default to 256) used to normalize the
                        dataset (if none is available)
  -a [1-200], --augment [1-200]
                        augment each image by this limit (min 1, max 200,
                        default 200)
  -t TEST, --test TEST  runs classification versus the test dataset (default
                        to False)
  -l {debug,info,warning,error,critical}, --loglevel {debug,info,warning,error,critical}
                        the loglevel, default to error
```

### Dataset creation
The program assumes a dataset is already present on disk, if it is not, it takes care to create a brand new one by sequentially applying the actions previously described.

### Warning
Dataset creation can be memory angry in case you have thousands of images, causing frequent swapping and occasional crashes.
In such case you will have two options:
1. reduce the number of images
2. reduce the number of augmentations (default to 200 per image)
