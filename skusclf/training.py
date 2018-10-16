'''
This module is related to the images dataset, it contains the logic to normalize, 
augment, create and compress the image data loaded from a source folder.
'''

from functools import reduce
from glob import glob
from math import floor
from operator import mul
from os import path
from struct import unpack
from tempfile import mkdtemp
from zipfile import ZipFile
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma
from skimage.transform import rescale, rotate
from skimage.util import random_noise
import h5py
import numpy as np
from skusclf.logger import BASE as logger


class Normalizer:
    '''
    Synopsis
    --------
    Normalizes the images that compose the training set by:
    - resizing the largest dimension to specified max size
    - optionally creating a squared canvas by max size and pasting the image into it

    Arguments
    ---------
    - size: the size of the target image
    - canvas: an object indicating if a squared canvas must be applied 
      behind the image, can be falsey (no canvas is applied), True (default to white),
      a RGB string (transparent canvas will be used for PNG with alpha)
      or a path to a background image

    Constructor
    -----------
    >>> norm = Normalizer(size=128, canvas='./office.jpg')
    '''

    SIZE = 32
    TRANSPARENT = (255, 0, 0, 0)
    WHITE = (255, 255, 255)
    CANVAS = False
    RGBA = 'RGBA'
    
    def __init__(self, size=SIZE, canvas=CANVAS):
        self.size = int(size)
        self.canvas = canvas
        self.bkg = path.isfile(str(canvas))

    def __call__(self, name):
        '''
        Normalize the source image (path or Image object) by resizing and 
        (optionally) pasting it within a squared canvas:
        >>> norm('./images/elvis.png')
        '''
        img = self._resize(name)
        if not img: return
        return self._canvas(img)

    def adjust(self, name, shape=None):
        '''
        Normalizes the provided image (path or Image object) and returns a properly 
        reshaped binary array, cropping and converting color if the provided shape 
        differs:
        >>> norm.adjust('./images/elvis.png', shape=(64, 41, 4))
        array[...]
        '''
        img = self(name)
        if not shape:
            return np.array(img)
        h, w, c = shape
        if c > 3:
            logger.info('converting to %s', self.RGBA)
            img = img.convert(self.RGBA)
        if img.size != (w, h):
            logger.info('correcting size to (%d, %d)', w, h)
            img = img.resize((w, h))
        data = np.array(img)
        logger.info('adjusted shape %r', data.shape)
        return data

    def _resize(self, name):
        img = name if hasattr(name, 'size') else Image.open(name)
        if self._png(img):
            img = img.convert(self.RGBA)
        w, h = img.size
        _max = max(w, h)
        if self._skip(_max): return
        ratio = _max / self.size
        size = (int(w // ratio), int(h // ratio))
        logger.info('resizing image to %r', size)
        return img.resize(size)

    def _skip(self, max_size):
        return not self.bkg and max_size == self.size

    def _canvas(self, img):
        if not self.canvas: return img
        size = (self.size, self.size)
        offset = self._offset(img)
        if self.bkg:
            logger.info('applying background %s', path.basename(self.canvas))
            c = Image.open(self.canvas).convert(img.mode)
            c = c.resize(size)
            c.paste(img, offset, img.convert(self.RGBA))
        else:
            logger.info('applying squared canvas %r', size)
            c = Image.new(img.mode, size, self._color(img))
            c.paste(img, offset)
        return c
    
    def _color(self, img):
        if img.mode == self.RGBA:
            return self.TRANSPARENT
        if self.canvas is True:
            return self.WHITE
        return unpack('BBB', bytes.fromhex(self.canvas))

    def _offset(self, img):
        w, h = img.size
        return ((self.size - w) // 2, (self.size - h) // 2)

    def _png(self, img):
        return img.format == 'PNG'


class Augmenter:
    '''
    Synopsis
    --------
    Performs dataset augmentation by performing the following transformations on 
    the original image:
    - blurring
    - flipping
    - adjusting gamma
    - rescaling and cropping
    - adding random noise
    - rotating

    The transofrmers methods are collected bu iterating on attributes starting with 
    the prefix '_tr': be aware of that when extending this class.

    Arguments
    ---------
    - cutoff: a float value indicating the cutoff percentage of the transformations 
      to be applied to the original image; i.e a value of 0.5 will cut transformations 
      to half (1.0=all by default); a minimum of one transformation per type is 
      guaranteed, thus at least 7+1 images (transformers + original) are yielded.

    Constructor
    -----------
    >>> aug = Augmenter(0.75)
    '''

    CUTOFF = 0
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'
    BLUR = range(2, 7, 1)
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :])
    GAMMA = np.arange(.1, 3., .05)
    NOISE = np.arange(.0005, .0255, .0005)
    SCALE = np.arange(1.05, 3.05, .01)
    ROTATE = np.arange(-60, 60, 0.5)
    RANGES = (BLUR, FLIP, GAMMA, NOISE, SCALE, ROTATE)

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = float(cutoff)
        self.transformers = sorted(t for t in dir(self) if t.startswith('_tr'))
        self.ranges = [self._cut(r) for r in self.RANGES]
        self.count = 1 if not self.cutoff else sum(len(r) for r in self.ranges) + 1

    def __call__(self, img):
        '''
        Accepts and image as binary data and yield a generator (to be consumed within a loop)
        with the original image and all of applied  transformations.
        Exit early if cutoff is zero.
        >>> transformations = aug(array[...])
        '''
        yield img
        if not self.cutoff: return
        logger.info('applying a set of %d transformations', self.count)
        for r, t in zip(self.ranges, self.transformers):
            _m = getattr(self, t)
            logger.debug(f'applying {t} {len(r)} times')
            for a in r:
                yield from _m(img, a)

    def __str__(self):
        return f'Augmenter(cutoff={self.cutoff}, count={self.count})'

    def _cut(self, r):
        if self.cutoff >= 1: return r
        cut = floor(len(r) * self.cutoff) or 1
        return r[:cut]
    
    def _tr_blur(self, img, axe):
        yield uniform_filter(img, size=(axe, axe, 1))

    def _tr_flip(self, img, sl):
        yield img[sl]

    def _tr_gamma(self, img, gm):
        yield adjust_gamma(img, gamma=gm, gain=.9)

    def _tr_noise(self, img, var):
        yield random_noise(img, mode=self.NOISE_MODE, var=var)

    def _tr_rescale(self, img, sc):
        _data = rescale(img, sc, mode=self.RESCALE_MODE, anti_aliasing=True, multichannel=True)
        h, w, _ = _data.shape
        y, x, _ = img.shape
        cx = w // 2 - (x // 2)
        cy = h // 2 - (y // 2)
        yield _data[cy:cy+y, cx:cx+x, :]

    def _tr_rotate(self, img, ang):
        cval = 1. if self._RGB(img) else 0
        yield rotate(img, ang, cval=cval)

    def _RGB(self, img):
        return img.shape[-1] == 3


class Features:
    '''
    Synopsis
    --------
    Represents the set of features fed by a file system folder containing the
    images (PNG, JPG) to be classified.

    Arguments
    ---------
    - folder: the folder containing the images
    - limit: limit the number of images read from disk, default to unlimited
    - fetcher: a callable taking as an argument the filename and returning the
      feature label
    - normalizer: a collaborator used to normalize the images within the folder,
      if falsey no normalization is performed
    - augmenter: a collaborator used to augment data of two order of magnitude,
      if falsey no augmentation is performed

    Constructor
    -----------
    >>> feat = Features('./my_images')
    '''
    
    EXTS = ('jpg', 'jpeg', 'png')
    LIMIT = 0
    BRANDS = ('plain', 'mm', 'gg')
    MAX_VAL = 255
    FETCHERS = {
        BRANDS[0]: lambda n: path.basename(n).split('.')[0],
        BRANDS[1]: lambda n: path.basename(n).split('-')[0],
        BRANDS[2]: lambda n: '_'.join(path.basename(n).split('_')[:3])
    }

    class EmptyFolderError(ValueError):
        '''
        Indicates if the specified folder contains no images
        '''

    def __init__(self, folder, limit=LIMIT, brand=BRANDS[0], 
                 augmenter=Augmenter(), normalizer=Normalizer()):
        self.folder = path.abspath(folder)
        self.images = self._images(int(limit))
        self.fetcher = self.FETCHERS[brand]
        self.augmenter = augmenter
        self.normalizer = normalizer
        self.count = len(self.images) * (augmenter.count if augmenter else 1)
        self.types = self._types()

    def _types(self):
        sku, data = self._data(self.images[0])
        return f'S{len(sku)}', data.shape

    def __iter__(self):
        '''
        Returns a lazy iterator representing the computed labels and the list of 
        related images data:
        >>> for lbl, imgs in feat:
        >>>   # do something with label
        >>>   for img in imgs:
        >>>     # do something with array of image data
        '''
        imgs_data = (self._data(name) for name in self.images)
        for label, img in imgs_data:
            imgs = (self._scale(aug) for aug in self._augmenting(img))
            yield (label, imgs)

    def _glob(self):
        return [f for ext in self.EXTS 
                  for f in glob(path.join(self.folder, f'*.{ext}'))]

    def _images(self, limit):
        images = sorted(self._glob())
        if limit:
            images = images[:limit] 
        if not len(images):
            raise self.EmptyFolderError(f'{self.folder} contains no valid images')
        return images

    def _data(self, name):
        if self.normalizer:
            data = np.array(self.normalizer(name))
        else:
            data = plt.imread(name)
        return self.fetcher(name), self._scale(data)

    def _scale(self, data):
        return data / self.MAX_VAL if np.max(data) > 1 else data

    def _augmenting(self, img):
        if not self.augmenter:
            return [img]
        return self.augmenter(img)


class DatasetH5:
    '''
    Synopsis
    --------
    Creates an H5 dataset by a list of features

    Arguments
    ---------
    - name: the name of the persisted HDF5 file
    - features: a features collaborator object that can be iterated and 
      queried for data information

    Constructor
    -----------
    >>> ds = DatasetH5('my_dataset', <Features>)
    '''

    EXT = '.h5'
    DTYPE = np.float32
    COMPRESSION = ('gzip', 9)

    class NoentError(ValueError):
        '''
        Indicates that the specified dataset does not exist
        '''

    @classmethod
    def load(cls, name, orig=False):
        '''
        Loads the stored HDF5 dataset (if any) and returns the X and y arrays.
        If the orig attribute is truthy, reshape the data before returning them:
        >>> DatasetH5.load(orig=True)
        '''
        name = path.abspath(name)
        if not path.isfile(name):
            raise cls.NoentError(f'{name} dataset does not exist')
        with h5py.File(name, 'r') as f:
            X, y = f['X'], f['y']
            shape = tuple(X.attrs['shape'].tolist())
            X, y = X[()], y[()]
            if orig:
                n, _ = X.shape
                X = X.reshape((n,) + shape)
            return X, y

    def __init__(self, name, features):
        self.name = self._name(name)
        self.features = features
        self.lbl_type, self.shape = features.types

    def __call__(self):
        '''
        Save the dataset in the HDF5 format and returns a tuple of collected (X,y) data.
        '''
        logger.info('persisting dataset %s', self.name)
        with h5py.File(self.name, 'w') as hf:
            X, y = self._collect()
            logger.info('creating X(%r), y(%r) datasets', X.shape, y.shape)
            ds = hf.create_dataset(name='X', data=X, shape=X.shape,
                                   dtype=self.DTYPE, 
                                   compression=self.COMPRESSION[0], 
                                   compression_opts=self.COMPRESSION[1])
            ds.attrs['shape'] = self.shape
            hf.create_dataset(name='y', data=y, shape=y.shape)
            self.labels_count = len(np.unique(y))
            logger.info('dataset with %d features and %d labels created successfully', self.features.count, self.labels_count)

    def _name(self, name):
        name = name if name.endswith(self.EXT) else f'{name}{self.EXT}'
        return path.abspath(name)

    def _collect(self):
        i = 0
        X, y = self._placeholders()
        for label, imgs in self.features:
            for img in imgs:
                X[i, ...] = img.flatten()
                y[i, ...] = label
                i += 1
        return self._shuffle(X, y)

    def _shuffle(self, X, y):
        indexes = np.random.permutation(self.features.count)
        return X[indexes], y[indexes]

    def _placeholders(self):
        shape = (self.features.count, reduce(mul, self.shape))
        X = np.empty(shape, dtype=self.DTYPE)
        y = np.empty((self.features.count,), dtype=self.lbl_type)
        return X, y


class DatasetZIP:
    '''
    Synopsis
    --------
    Creates a ZIP dataset by a list of features, archiving a folder per label containing
    all of the images.

    Arguments
    ---------
    - name: the name of the persisted ZIP file
    - features: a features collaborator object that can be iterated and 
      queried for data information
    - prefix: a prefix, if any, to be used to name the label directory name 
    - filename: the filename of the saved images, postfixed by a counting index

    Constructor
    -----------
    >>> comp = DatasetZIP('dataset.zip', <Features>)
    '''

    EXT = '.zip'
    PREFIX = 'LBL_'
    FILENAME = 'sample'
    EXTS = ('jpg', 'png')

    def __init__(self, name, features, prefix=PREFIX, filename=FILENAME):
        self.name = self._name(name) 
        self.zip = ZipFile(path.abspath(self.name), 'w')
        self.features = features
        self.prefix = prefix
        self.filename = filename
        self.dir = mkdtemp(prefix='images')
        self.ext = self._ext()

    def __call__(self):
        '''
        Save the zip file as the specified name.
        '''
        try:
            logger.info('creating ZIP dataset %s', self.name)
            for img, arc in self._entries():
                self.zip.write(img, arcname=arc)
        finally:
            self.zip.close()

    def _name(self, name):
        return name if name.endswith(self.EXT) else f'{name}{self.EXT}'

    def _entries(self):
        for label, imgs in self.features:
            logger.info('archiving label %s', label)
            for i, data in enumerate(imgs):
                name = self._filename(i)
                logger.debug('archiving image %s', name)
                img = self._img(data, name)
                arc = self._arc(label, name)
                yield(img, arc)

    def _filename(self, i):
        return f'{self.filename}_{i}.{self.ext}'

    def _img(self, data, name):
        try:
            _path = path.join(self.dir, name)
            plt.imsave(_path, data)
            return _path
        except ValueError as e:
            logger.error(f'invalid image data range for {name}: {data.min()} - {data.max()} / {e}')

    def _arc(self, label, name):
        return path.join(f'{self.prefix}{label}', name)

    def _ext(self):
        _, shape = self.features.types
        return self.EXTS[1] if shape[-1] > 3 else self.EXTS[0]
