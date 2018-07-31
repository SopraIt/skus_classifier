from glob import glob
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
    - limit: limit the number of transformations, default to no limit (about 200)

    Constructor
    -----------
    >>> aug = Augmenter(50)
    '''

    LIMIT = 200
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'

    def __init__(self, limit=LIMIT):
        self.limit = int(limit)
    
    def __call__(self, img):
        '''
        Synopsis
        --------
        Returns a generator with original image and all of its transformations:

        Arguments
        ---------
        - img: a numpy.ndarray representing the image to transform

        >>> output = aug.images()
        '''
        self.count = 1
        self.img = img
        self.size = img.shape[0]
        yield self.img
        transformers = (t for t in dir(self) if t.startswith('_tr'))
        for t in transformers:
            _m = getattr(self, t)
            yield from _m()
    
    def _check(self):
        if not self.limit:
            return True
        if self.count < self.limit:
            self.count += 1
            return True

    def _tr_rescale(self):
        logger.info('applying rescaling')
        for _s in np.arange(1.1, 4.1, .1):
            logger.debug('rescaling by %.2f', _s)
            _data = rescale(self.img, _s, mode=self.RESCALE_MODE, 
                            anti_aliasing=True, multichannel=True)
            _start = _data.shape[0]//2-(self.size//2)
            _end = _start+self.size
            _data = _data[_start:_end,_start:_end]
            if not self._check(): break
            yield _data

    def _tr_noise(self):
        logger.info('applying random noise')
        for _v in np.arange(.04, .8, .04):
            logger.debug('applying %s noise by %.2f', self.NOISE_MODE, _v)
            _data = random_noise(self.img, mode=self.NOISE_MODE, var=_v)
            if not self._check(): break
            yield _data

    def _tr_rotate(self):
        logger.info('applying rotation')
        for _a in range(5, 360, 5):
            logger.debug('rotating CCW by %d', _a)
            _data = rotate(self.img, _a)
            if not self._check(): break
            yield _data

    def _tr_contrast(self):
        logger.info('applying contrast')
        for _max in np.arange(.2, 6., .2):
            logger.debug('augmenting contrast by %.2f',_max)
            _data = rescale_intensity(self.img, in_range=(.0, _max))
            if not self._check(): break
            yield _data
    
    def _tr_gamma(self):
        logger.info('applying gamma adjust')
        for _g in np.arange(.2, 6., .2):
            logger.debug('adjusting gamma by %.2f', _g)
            _data = adjust_gamma(self.img, gamma=_g, gain=.9)
            if not self._check(): break
            yield _data
    
    def _tr_blur(self):
        logger.info('applying blurring')
        for _b in range(1, 20, 1):
            logger.debug('blurring at the center by %d', _b)
            _data = uniform_filter(self.img, size=(_b, _b, 1))
            if not self._check(): break
            yield _data

    def _tr_flip_h(self):
        logger.info('applying flip H')
        _data = self.img[:, ::-1]
        if self._check():
            yield _data

    def _tr_flip_v(self):
        logger.info('applying flip V')
        _data = self.img[::-1, :]
        if self._check():
            yield _data


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
        self.count = len(self.images) * (augmenter.limit if augmenter else 1)
        self.name = name
        self.shape = shape
        self.fetcher = fetcher
        self.persist = persist
        self.normalizer = normalizer
        self.augmenter = augmenter

    def load(self):
        '''
        Loads the stored HDF5 dataset (if any) and returns it
        '''
        if path.isfile(self.name):
            return h5py.File(self.name, 'r')
    
    def __call__(self):
        '''
        Save the dataset in the HDF5 format and returns a tuple of collected (X,y) data.
        '''
        self._check()
        logger.info('persisting dataset %s', self.name)
        self._normalize()
        img = plt.imread(self.images[0]) if self.images else None
        with h5py.File(self.name, 'w') as hf:
            X, y = self._collect(self.shape or img.flatten().shape)
            logger.info('creating X(%r), y(%r) datasets', X.shape, y.shape)
            X_ds = hf.create_dataset(name='X', data=X, shape=X.shape,
                                     dtype=np.float32, 
                                     compression=self.COMPRESSION[0], 
                                     compression_opts=self.COMPRESSION[1])
            X_ds.attrs['orig_shape'] = img.shape
            hf.create_dataset(name='y', data=y, shape=y.shape)
            logger.info('dataset created successfully')

    def _check(self):
        if not len(self.images):
            raise self.EmptyFolderError(f'{self.folder} contains no valid images')
    
    def _collect(self, shape):
        logger.info('collecting data from %s', self.folder)
        X = np.empty((self.count,) + shape, dtype=np.float32)
        y = np.empty((self.count,), dtype='S17')
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
