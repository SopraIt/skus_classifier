from glob import glob
from math import floor
from os import path
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma, rescale_intensity
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
    - creating a squared canvas by max size and pasting the image into it
    - optimizing the image (quantize for PNG)

    Arguments
    ---------
    - size: the size of the target image
    - optimize: a flag used to optimize the PNG file
    - mode: the color mode
    - bkg: the path tp the background image to be used as canvas (default to transparent)

    Constructor
    -----------
    >>> norm = Normalizer(size=400, optimize=False, mode='RGBA', bkg='./office.jpg')
    '''

    SIZE = 64
    MODE = 'RGBA'
    COLOR = (255, 0, 0, 0)
    
    def __init__(self, size=SIZE, optimize=True, mode=MODE, bkg=None):
        self.size = int(size)
        self.optimize = optimize
        self.mode = mode
        self.bkg = bkg

    def bulk(self, images):
        '''
        Accepts a list of images paths and replace eaach one with the normalized version:
        >>> norm.bulk('./my_images')
        '''
        for name in images:
            self.persist(name)
        return len(images)

    def persist(self, src, target=None):
        '''
        Accepts the path of the source image and the target name to save the normalized version.
        If no target is provided, the source image is normalized and replaced.
        >>> norm.persist('./images/elvis.png', './images/the_king.png')
        '''
        target = target or src
        logger.info('saving file: %s', target)
        img = self(src)
        if not img: return
        img.save(target)

    def __call__(self, name):
        '''
        Creates a squared canvas, by pasting image and optimizing it
        '''
        logger.info('hopping image by creating a squared canvas')
        img = self._resize(name)
        if not img: return
        canvas = self._canvas(img.convert(self.mode))
        if self.optimize:
            logger.debug('optmizing image')
            return canvas.quantize()
        return canvas
    
    def _resize(self, name):
        img = Image.open(name)
        w, h = img.size
        if self._skip(w, h): return
        ratio = max(w, h) / self.size
        size = (int(w // ratio), int(h // ratio))
        logger.debug('resizing image to %r', size)
        return img.resize(size)

    def _skip(self, w, h):
        return not self.bkg and w == self.size and h == self.size

    def _canvas(self, img):
        size = (self.size, self.size)
        offset = self._offset(img)
        if self.bkg and path.isfile(str(self.bkg)):
            logger.info('applpying background %s', path.basename(self.bkg))
            c = Image.open(self.bkg).convert(self.mode)
            _min = min(c.size)
            c = c.crop((0, 0, _min, _min))
            c = c.resize(size)
            c.paste(img, offset, img)
        else:
            c = Image.new(self.mode, size, self.COLOR)
            c.paste(img, offset)
        return c

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
    - adjusting contrast
    - adjusting gamma colors
    - blurring
    - flipping (horizontally and vertically)

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

    CUTOFF = 1.
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'
    BLUR = range(1, 21, 1)
    GAMMA = np.arange(.2, 8.2, .2)
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :])
    NOISE = np.arange(.04, .804, .04)
    SCALE = np.arange(1.1, 4.1, .1)
    ROTATE = range(12, 360, 6)
    RANGES = (BLUR, GAMMA, FLIP, GAMMA, NOISE, SCALE, ROTATE)

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = float(cutoff)
        self.transformers = sorted(t for t in dir(self) if t.startswith('_tr'))
        self.ranges = [self._cut(r) for r in self.RANGES]
        self.count = sum(len(r) for r in self.ranges) + 1

    def __call__(self, img):
        '''
        Accepts and image as binary data and yield a generator (to be consumed within a loop)
        with the original image and all of applied  transformations.
        >>> transformations = aug(array[...])
        '''
        yield img
        logger.info('applying a set of %d transformations', self.count)
        for r, t in zip(self.ranges, self.transformers):
            _m = getattr(self, t)
            for a in r:
                yield from _m(img, a)

    def _cut(self, r):
        if self.cutoff >= 1: return r
        cut = floor(len(r) * self.cutoff) or 1
        return r[:cut]
    
    def _tr_blur(self, img, axe):
        logger.info('blurring at the center with axe %d', axe)
        yield uniform_filter(img, size=(axe, axe, 1))

    def _tr_contrast(self, img, rng):
        logger.debug('augmenting contrast by range %.2f', rng)
        yield rescale_intensity(img, in_range=(.0, rng))

    def _tr_flip(self, img, sl):
        logger.info('flipping by slice %r', sl)
        yield img[sl]

    def _tr_gamma(self, img, gm):
        logger.debug('adjusting gamma by %.2f', gm)
        yield adjust_gamma(img, gamma=gm, gain=.9)

    def _tr_noise(self, img, var):
        logger.debug('applying %s noise with variance %.2f', self.NOISE_MODE, var)
        yield random_noise(img, mode=self.NOISE_MODE, var=var)

    def _tr_rescale(self, img, sc):
        logger.debug('rescaling by %.2f', sc)
        _size = max(img.shape)
        _data = rescale(img, sc, mode=self.RESCALE_MODE, anti_aliasing=True, multichannel=True)
        _start = _data.shape[0] // 2 - (_size // 2)
        _end = _start + _size
        yield _data[_start:_end,_start:_end]

    def _tr_rotate(self, img, ang):
        logger.debug('rotating CCW by %d', ang)
        yield rotate(img, ang)


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
    - shape: the shape to store the images data with, default to the one of the 
      first image loaded, accordingly flattened
    - normalizer: a collaborator used to normalize the images within the folder,
      if falsey no normalization is performed
    - augmenter: a collaborator used to augment data of two order of magnitude,
      if falsey no augmentation is performed
    - persist: a flag indicating if augmented images must be persisted to disk

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> ds = Dataset(folder='./my_images', fetcher=plain)
    '''

    FOLDER = './images'
    EXT = 'png'
    LIMIT = 0
    FETCHER = lambda n: '_'.join(path.basename(n).split('_')[:3])
    COMPRESSION = ('gzip', 9)

    class EmptyFolderError(ValueError):
        '''
        Indicates if the specified folder contains no images to created the dataset with
        '''

    def __init__(self, name, folder=FOLDER, limit=LIMIT,
                 shape=None, fetcher=FETCHER, persist=False,
                 augmenter=Augmenter(), normalizer=Normalizer()):
        self.folder = folder
        self.images = self._images(int(limit))
        self.count = len(self.images) * (augmenter.count if augmenter else 1)
        self.name = name
        self.shape = shape
        self.fetcher = fetcher
        self.persist = persist
        self.normalizer = normalizer
        self.augmenter = augmenter

    @property
    def img(self):
        return plt.imread(self.images[0])

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
        self._normalize()
        with h5py.File(self.name, 'w') as hf:
            X, y = self._collect(self.shape or self.img.flatten().shape)
            logger.info('creating X(%r), y(%r) datasets', X.shape, y.shape)
            X_ds = hf.create_dataset(name='X', data=X, shape=X.shape,
                                     dtype=np.float32, 
                                     compression=self.COMPRESSION[0], 
                                     compression_opts=self.COMPRESSION[1])
            X_ds.attrs['size'] = max(self.img.shape)
            hf.create_dataset(name='y', data=y, shape=y.shape)
            logger.info('dataset created successfully')

    def load(self):
        '''
        Loads the stored HDF5 dataset (if any) and returns it
        '''
        if path.isfile(self.name):
            return h5py.File(self.name, 'r')

    def _check(self):
        if not len(self.images):
            raise self.EmptyFolderError(f'{self.folder} contains no valid images')
    
    def _collect(self, shape):
        logger.info('collecting data from %s', self.folder)
        X = np.empty((self.count,) + shape, dtype=np.float32)
        y = np.empty((self.count,), dtype=self.label_dtype)
        i = 0
        for name in self.images:
            sku = self.fetcher(name)
            img = plt.imread(name)
            logger.info('working on image %s', path.basename(name))
            for n, aug in enumerate(self._augmenting(img)):
                X[i, ...] = aug.reshape(self.shape) if self.shape else aug.flatten()
                y[i, ...] = sku
                self._persist(name, n, aug)
                i += 1
        return self._shuffle(X, y)

    def _shuffle(self, X, y):
        indexes = np.random.permutation(len(X))
        return X[indexes], y[indexes]

    def _images(self, limit):
        images = sorted(glob(f'{self.folder}/*.{self.EXT}'))
        if limit:
            images = images[:limit] 
        return images

    def _normalize(self):
        if self.normalizer:
            logger.info('normalizing %d images', len(self.images))
            self.normalizer.bulk(self.images)

    def _augmenting(self, img):
        if not self.augmenter:
            return [img]
        return self.augmenter(img)
    
    def _persist(self, name, n, img):
        if self.persist and n > 0:
            basename = path.basename(name)
            suffixed = basename.replace('.', f'_{n:03d}.')
            name = path.join(self.folder, suffixed)
            if not path.isfile(name):
                logger.info('perisisting %s', name)
                plt.imsave(name, img)
