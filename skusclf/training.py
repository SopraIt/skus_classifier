from glob import glob
from os import path
from matplotlib.pyplot import imread
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from sklearn.externals import joblib
import numpy as np


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
    
    @classmethod
    def bulk(cls, folder, max_size, optimize=False):
        '''
        Overwrites all images in the specified folder, returns the number of normilzed files:
        >>> Normalizer.bulk(folder='./my_images', max_size=400)
        '''
        n = 0
        for name in glob(f'{folder}/*'):
            if cls(name, max_size, optimize).save():
                n += 1
        return n

    def __init__(self, name, max_size, optimize, mode=MODE, color=COLOR):
        self.name = name
        self.max_size = int(max_size)
        self.optimize = optimize
        self.mode = mode
        self.color = color
        self.img = self._resize()
        self.w, self.h = self.img.size

    def save(self, name=None):
        '''
        Create a new version of the image (if not already normalized):
        >>> norm.save('./my_images/new_img.png')
        '''
        if self._normalized():
            return
        name = name or self.name
        self.hop().save(name)
        return True

    def hop(self):
        '''
        Creates a squared canvas, by pasting image and optimizing it
        '''
        c = self._canvas()
        c.paste(self.img, self._offset())
        if self.optimize:
            return c.quantize()
        return c
    
    def _resize(self):
        img = Image.open(self.name)
        self.orig_w, self.orig_h = img.size
        ratio = max(self.orig_w, self.orig_h) / self.max_size
        size = (int(self.orig_w // ratio), int(self.orig_h // ratio))
        return img.resize(size)

    def _canvas(self):
        return Image.new(self.mode, (self.max_size, self.max_size), self.color)

    def _offset(self):
        return ((self.max_size - self.w) // 2, (self.max_size - self.h) // 2)

    def _normalized(self):
        return self.orig_w == self.max_size and self.orig_h == self.max_size


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
    - mag: the order of magnitude of the augmentation, 2 or 3

    Constructor
    -----------
    >>> aug = Augmenter(2)
    '''

    MAGS = (2, 3)
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'

    def __init__(self, mag=MAGS[1]):
        self.mag = self._check_mag(mag)
    
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
        self.img = img
        self.size = img.shape[0]
        yield self.img
        transformers = (t for t in dir(self) if t.startswith('_tr'))
        for t in transformers:
            _m = getattr(self, t)
            yield from _m()

    def _check_mag(self, mag):
        if int(mag) not in self.MAGS:
            raise ValueError(f'magnitude {mag} is not within 2 and 3')
        return int(mag)

    def _hi_mag(self):
        return self.mag >= 3

    def _step(self, low, hi):
        return hi if self.mag == 2 else low

    def _tr_rescale(self):
        step = self._step(.1, 1.1)
        for _s in np.arange(1.1, 3.1, step):
            _data = rescale(self.img, _s, mode=self.RESCALE_MODE, 
                            anti_aliasing=True, multichannel=True)
            _start = _data.shape[0]//2-(self.size//2)
            _end = _start+self.size
            yield _data[_start:_end,_start:_end]

    def _tr_noise(self):
        step = self._step(.02, .6)
        for _v in np.arange(.02, .6, step):
            yield random_noise(self.img, mode=self.NOISE_MODE, var=_v)

    def _tr_rotate(self):
        step = self._step(5, 350)
        for _a in range(5, 360, step):
            yield rotate(self.img, _a)

    def _tr_contrast(self):
        step = self._step(.2, 2.1)
        for _max in np.arange(.1, 4., step):
            yield rescale_intensity(self.img, in_range=(.0, _max))
    
    def _tr_gamma(self):
        step = self._step(.2, 4.)
        for _g in np.arange(.1, 4., step):
            yield adjust_gamma(self.img, gamma=_g, gain=.9)
    
    def _tr_blur(self):
        step = self._step(1, 18)
        for _b in range(1, 18, step):
            yield uniform_filter(self.img, size=(_b, _b, 1))

    def _tr_flip_h(self):
        if self._hi_mag():
            yield self.img[:, ::-1]

    def _tr_flip_v(self):
        if self._hi_mag():
            yield self.img[::-1, :]


class Loader:
    '''
    Synopsis
    --------
    Creates the training set by file system, assuming the name of the images
    contain the label data, which is fetched by a custom routine.
    Before to be stored into NumPy arrays, the images are resized and
    normalized by using an external collaborator.

    Arguments
    ---------
    - folder: the folder containing the PNG files
    - max_size: the max size used to normalize the images
    - fetcher: a callable taking as an argument the filename and returning the
      formatted label
    - normalizer: the class used to normalize the images within the folder
      by the specified max_size, must respond to the "bulk" method
    - augmenter: a collaborator used to augment data of two order of magnitude

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> loader = Loader(folder='./my_images', fetcher=plain)
    '''

    FETCHER = lambda n: '_'.join(path.basename(n).split('_')[:3])

    def __init__(self, folder, max_size, fetcher=FETCHER, 
                 normalizer=Normalizer, augmenter=Augmenter()):
        self.folder = folder
        self._images = glob(f'{folder}/*')
        self.max_size = int(max_size)
        self.norm_size = f'{self.max_size}x{self.max_size}'
        self._fetcher = fetcher
        self.normalizer = normalizer
        self.augmenter = augmenter
        self._set = {'COL_NAMES': ('target', 'data', 'size'),
                     'DESCR': 'the SKUs dataset, normalized and augmented',
                     'size': self.max_size}
    
    def save(self, name=None):
        '''
        Save the dataset for further usage by using scikit-learn joblib.
        >>> loader.save('./mystuff.pkl')
        '''
        _set = self.set()
        name = name or f'./dataset_{self.norm_size}.pkl'
        joblib.dump(_set, name)
        return name

    def set(self):
        '''
        Creates the training set:
        >>> _set = loader.set()
        >>> _set['data']
        array([[[[1., 0., 0., 0.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.],
              ...]]])
        '''
        if 'data' not in self._set:
            self._normalize()
            self._set['data'], self._set['target'] = self._dataset()
        return self._set

    def _dataset(self):
        _data = []
        _target = []
        for img in self._images:
            _sku = self._fetcher(img)
            for _img in self.augmenter(imread(img)):
                _data.append(_img)
                _target.append(_sku)
        return tuple(np.array(x) for x in (_data, _target))

    def _augmenter(self, img):
        return self.augmenter(img)

    def _normalize(self):
        self.normalizer.bulk(self.folder, self.max_size)
