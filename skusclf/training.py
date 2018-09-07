from glob import glob
from math import floor
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


class Augmenter:
    '''
    Synopsis
    --------
    Performs dataset augmentation by performing the following transformations on 
    the original image:
    - rescaling and cropping
    - adding random noise
    - rotating
    - adjusting gamma
    - blurring
    - flipping (horizontally and vertically), when the image is squared

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
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :]) #2
    GAMMA = np.arange(.1, 3., .1) #29
    NOISE = np.arange(.0005, .0255, .0005)
    SCALE = np.arange(1.05, 3.05, .05)
    ROTATE = np.arange(-60, 60, 2.5)
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
            logger.info(f'applying {t} {len(r)} times')
            for a in r:
                yield from _m(img, a)

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


class Dataset:
    '''
    Synopsis
    --------
    Creates the training dataset by file system, assuming the name of the images
    contain the target data, which is fetched by a custom routine.
    Before to be stored into Numpy arrays, the images are normalized, augmented 
    (by using external collaborators) and shuffled.

    Arguments
    ---------
    - folder: the folder containing the PNG files
    - limit: limit the number of images read from disk, default to unlimited
    - fetcher: a callable taking as an argument the filename and returning the
      formatted label
    - name: the name of the persisted HDF5 file
    - normalizer: a collaborator used to normalize the images within the folder,
      if falsey no normalization is performed
    - augmenter: a collaborator used to augment data of two order of magnitude,
      if falsey no augmentation is performed
    - shuffle: a flag indicating id data is shuffled or not

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> ds = Dataset(folder='./my_images', fetcher=plain)
    '''

    LIMIT = 0
    COMPRESSION = ('gzip', 9)
    BRANDS = ('mm', 'gg')
    FETCHERS = {
        BRANDS[0]: lambda n: path.basename(n).split('-')[0],
        BRANDS[1]: lambda n: '_'.join(path.basename(n).split('_')[:3])
    }

    class NoentError(ValueError):
        '''
        Indicates that the specified dataset does not exist
        '''

    class EmptyFolderError(ValueError):
        '''
        Indicates if the specified folder contains no images
        '''

    def __init__(self, name, folder=None, limit=LIMIT, brand=BRANDS[0], 
                 augmenter=Augmenter(), normalizer=Normalizer(), shuffle=True):
        self.folder = folder
        self.images = self._images(int(limit))
        self.count = len(self.images) * (augmenter.count if augmenter else 1)
        self.name = path.abspath(name)
        self.fetcher = self.FETCHERS[brand]
        self.augmenter = augmenter
        self.normalizer = normalizer
        self.shuffle = shuffle
        self.sample = self._sample()
        self.labels_count = 0

    @property
    def label_dtype(self):
        sku = self.fetcher(self.images[0])
        return f'S{len(sku)}'

    def __call__(self):
        '''
        Save the dataset in the HDF5 format and returns a tuple of collected (X,y) data.
        '''
        self._check()
        logger.info('persisting dataset %s', self.name)
        with h5py.File(self.name, 'w') as hf:
            X, y = self._collect()
            logger.info('creating X(%r), y(%r) datasets', X.shape, y.shape)
            X_ds = hf.create_dataset(name='X', data=X, shape=X.shape,
                                     dtype=np.float32, 
                                     compression=self.COMPRESSION[0], 
                                     compression_opts=self.COMPRESSION[1])
            X_ds.attrs['shape'] = self.sample.shape
            hf.create_dataset(name='y', data=y, shape=y.shape)
            self.labels_count = len(np.unique(y))
            logger.info('dataset with %d features and %d labels created successfully', self.count, self.labels_count)
        return self

    def load(self, orig=False):
        '''
        Loads the stored HDF5 dataset (if any) and returns the X and y arrays.
        If the orig attribute is truthy, unflatten the data before returning them:
        >>> ds.load(orig=True)
        '''
        if not path.isfile(self.name): raise self.NoentError(f'{self.name} dataset does not exist')
        with h5py.File(self.name, 'r') as f:
            X, y = f['X'], f['y']
            shape = tuple(X.attrs['shape'].tolist())
            X, y = X[()], y[()]
            if orig:
                n, _ = X.shape
                X = X.reshape((n,) + shape)
            return X, y

    def _sample(self):
        if self.images:
            name = self.images[0]
            if self.normalizer:
                return np.array(self.normalizer(name))
            return plt.imread(name)

    def _check(self):
        if not len(self.images):
            raise self.EmptyFolderError(f'{self.folder} contains no valid images')
    
    def _collect(self):
        logger.info('collecting data from %s', self.folder)
        X = np.empty((self.count,) + self.sample.flatten().shape, dtype=np.float32)
        y = np.empty((self.count,), dtype=self.label_dtype)
        i = 0
        for name, img in self._images_data():
            sku = self.fetcher(name)
            logger.info('working on image %s', path.basename(name))
            for n, aug in enumerate(self._augmenting(img)):
                X[i, ...] = aug.flatten()
                y[i, ...] = sku
                i += 1
        return self._shuffle(X, y)

    def _shuffle(self, X, y):
        if not self.shuffle:
            return X, y
        indexes = np.random.permutation(len(X))
        return X[indexes], y[indexes]

    def _images(self, limit):
        if not self.folder: return []
        images = sorted(glob(f'{path.abspath(self.folder)}/*'))
        if limit:
            images = images[:limit] 
        return images

    def _images_data(self):
        for name in self.images:
            if self.normalizer:
                yield name, np.array(self.normalizer(name))
            else:
                yield name, plt.imread(name)

    def _augmenting(self, img):
        if not self.augmenter:
            return [img]
        return self.augmenter(img)


class Compressor:
    '''
    Synopsis
    --------
    Creates a ZIP file by starting from the dataset (imagesi data and labels) and
    stores them into a folder named after the associated label and using a custom 
    name with a counting index.

    Arguments
    ---------
    - X: the numpy array containing the images data
    - y: a numpy array containing the labels data
    - name: the name of the persisted ZIP file
    - prefix: a prefix, if any, to be used to name the label directory name 
    - filename: the filename of the saved images, postfixed by a counting index

    Constructor
    -----------
    >>> comp = Compressor(array[...], array[...], 'dataset.zip')
    '''

    EXT = '.zip'
    PREFIX = 'LBL_'
    FILENAME = 'sample'
    EXTS = ('jpg', 'png')
    MAX_VAL = 255

    def __init__(self, X, y, name, prefix=PREFIX, filename=FILENAME):
        self.X = X
        self.y = y
        self.name = f'{name}{self.EXT}'
        self.zip = ZipFile(path.abspath(self.name), 'w')
        self.prefix = prefix
        self.filename = filename
        self.dir = mkdtemp(prefix='images')

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

    def _entries(self):
        for i, (label, data) in enumerate(zip(self.y, self.X)):
            name = f'{self.filename}_{i}.{self._ext(data)}'
            img = self._img(data, name)
            arc = self._arc(label, name)
            yield(img, arc)

    def _img(self, data, name):
        _path = path.join(self.dir, name)
        plt.imsave(_path, self._scale(data))
        return _path

    def _scale(self, data):
        return data / self.MAX_VAL if np.max(data) > 1 else data

    def _arc(self, label, name):
        label = label.decode('utf-8')
        return path.join(f'{self.prefix}{label}', name)

    def _ext(self, data):
        return self.EXTS[1] if data.shape[-1] > 3 else self.EXTS[0]
