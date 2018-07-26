from glob import glob
import gzip
from os import path
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from sklearn.externals import joblib
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
    - path to the image file
    - max_size: the size of the target image
    - mode: the color mode
    - color: the background color of the squared canvas

    Constructor
    -----------
    >>> norm = Normalizer('./my_images/img.png', 
    >>>                   max_size=400, mode='RGBA', color=(255,0,0,0))
    '''

    PNG = 'PNG'
    MODE = 'RGBA'
    COLOR = (255, 0, 0, 0)
    
    def __init__(self, max_size, optimize=True, mode=MODE, color=COLOR):
        self.max_size = int(max_size)
        self.optimize = optimize
        self.mode = mode
        self.color = color

    def bulk(self, folder):
        '''
        Overwrites all images in the specified folder, returns the number of normilzed files:
        >>> norm.bulk(folder='./my_images')
        '''
        n = 0
        for name in glob(f'{folder}/*'):
            n += self.persist(name)
        return n

    def persist(self, name):
        '''
        Create/replace a normalized version of the image (if it isn't yet):
        >>> norm.persist('./my_images/new_img.png')
        '''
        logger.info('saving file: %s', name)
        img = self(name)
        if not img: return 0
        img.save(name)
        return 1

    def __call__(self, name):
        '''
        Creates a squared canvas, by pasting image and optimizing it
        '''
        logger.info('hopping image by creating a squared canvas')
        img = self._resize(name)
        if not img: return
        w, h = img.size
        c = self._canvas()
        c.paste(img, self._offset(w, h))
        if self.optimize:
            logger.info('optmizing PNG')
            return c.quantize()
        return c
    
    def _resize(self, name):
        img = Image.open(name)
        w, h = img.size
        if self._skip(w, h): return
        ratio = max(w, h) / self.max_size
        size = (int(w // ratio), int(h // ratio))
        logger.info('resizing image to %r', size)
        return img.resize(size)

    def _skip(self, w, h):
        return w == self.max_size and h == self.max_size

    def _canvas(self):
        return Image.new(self.mode, (self.max_size, self.max_size), self.color)

    def _offset(self, w, h):
        return ((self.max_size - w) // 2, (self.max_size - h) // 2)


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

    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'

    def __init__(self, limit=0):
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
        if self._check(): yield _data

    def _tr_flip_v(self):
        logger.info('applying flip V')
        _data = self.img[::-1, :]
        if self._check(): yield _data


class Loader:
    '''
    Synopsis
    --------
    Creates the training set by file system, assuming the name of the images
    contain the target data, which is fetched by a custom routine.
    Before to be stored into NumPy arrays, the images are normalized and 
    augmented by using external collaborators.

    Arguments
    ---------
    - folder: the folder containing the PNG files
    - fetcher: a callable taking as an argument the filename and returning the
      formatted label
    - normalizer: a collaborator used to normalize the images within the folder,
      if falsy no normalization is performed
    - augmenter: a collaborator used to augment data of two order of magnitude,
      if falsy no augmentation is performed
    - persist: a flag indicating if augmented images must be persisted to disk

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> loader = Loader(folder='./my_images', fetcher=plain)
    '''

    FETCHER = lambda n: '_'.join(path.basename(n).split('_')[:3])
    DATASET_NAME = f'./dataset.pkl.gz'
    TARGET_TYPE = '<U17'

    @classmethod
    def open(cls, name=DATASET_NAME):
        '''
        Open the specified dataset by unzipping and loading it
        >>> loader.open('./mystuff.pkl.gz')
        '''
        logger.info('opening dataset: %s', name)
        with gzip.open(name, 'rb') as gz:
            return joblib.load(gz)

    def __init__(self, folder, fetcher=FETCHER, persist=False,
                 augmenter=Augmenter(200), normalizer=Normalizer(256)):
        self.folder = folder
        self.images = sorted(glob(f'{folder}/*'))
        self.images_count = len(self.images)
        self.fetcher = fetcher
        self.persist = persist
        self.normalizer = normalizer
        self.augmenter = augmenter
        self._dataset = {'COL_NAMES': ('target', 'data'),
                         'DESCR': 'the SKUs dataset, normalized and augmented'}
    
    @property
    def count(self):
        limit = self.augmenter.limit if self.augmenter else 1
        return self.images_count * limit
    
    @property
    def shape(self):
        if self.images_count:
            img = plt.imread(self.images[0])
            return img.shape


    def store_dataset(self, name=DATASET_NAME):
        '''
        Save the dataset as a gzipped piclle archive for further usage.
        >>> loader.store_dataset('./mystuff.pkl.gz')
        '''
        dataset = self.dataset()
        logger.info('saving dataset: %s', name)
        with gzip.open(name, 'wb') as gz:
            joblib.dump(dataset, gz)
        return name

    def dataset(self):
        '''
        Creates the training dataset:
        >>> _dataset = loader.dataset()
        >>> _dataset['data']
        array([[[[1., 0., 0., 0.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.],
              ...]]])
        '''
        if 'data' not in self._dataset:
            logger.info('loading training set')
            self._normalize()
            self._collect()
        return self._dataset

    def _collect(self):
        logger.info('collected %d elements', self.count)
        data = np.empty((self.count,) + self.shape, dtype=np.float32)
        target = np.empty((self.count,), dtype=self.TARGET_TYPE)
        i = 0
        for name in self.images:
            sku = self.fetcher(name)
            img = plt.imread(name)
            for n, aug in enumerate(self._augment(img)):
                data[i, ...] = aug
                target[i, ...] = sku
                self._persist(name, n, aug)
                i += 1
        self._dataset['data'] = data
        self._dataset['target'] = target

    def _augment(self, img):
        if not self.augmenter:
            return [img]
        return self.augmenter(img)

    def _persist(self, name, n, data):
        if self.persist and n > 0:
            basename = path.basename(name)
            postfixed = basename.replace('.', f'_{n:03d}.')
            name = path.join(self.folder, postfixed)
            logger.debug('perisisting %s', name)
            plt.imsave(name, data)

    def _normalize(self):
        if self.normalizer:
            logger.info('normalizing images')
            self.normalizer.bulk(self.folder)
