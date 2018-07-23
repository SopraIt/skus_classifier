from glob import glob
from os import path
from matplotlib.pyplot import imread
from sklearn.externals import joblib
from PIL import Image
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
    def bulk(cls, folder, max_size):
        '''
        Overwrites all images in the specified folder, returns the number of normilzed files:
        >>> Normalizer.bulk(folder='./my_images', max_size=400)
        '''
        n = 0
        for name in glob(f'{folder}/*'):
            if cls(name, max_size).save():
                n += 1
        return n

    def __init__(self, name, max_size, mode=MODE, color=COLOR, optimize=False):
        self.name = name
        self.max_size = int(max_size)
        self.mode = mode
        self.color = color
        self.optimize = optimize
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
    - normalizer: tha class used to normalize the images within the folder
      by the specified max_size, must respond to the "bulk" method

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> loader = Loader(folder='./my_images', fetcher=plain)
    '''

    FETCHER = lambda n: '_'.join(path.basename(n).split('_')[:3])

    def __init__(self, folder, max_size, fetcher=FETCHER, normalizer=Normalizer):
        self.folder = folder
        self._images = glob(f'{folder}/*')
        self.max_size = int(max_size)
        self.norm_size = f'{self.max_size}x{self.max_size}'
        self._fetcher = fetcher
        self.normalizer = normalizer
        self._set = []
    
    def save(self, name=None):
        '''
        Save the dataset for further usage by using scikit-learn joblib.
        >>> loader.save('./mystuff.pkl')
        '''
        _set = self.set()
        name = name or f'./skus_{self.norm_size}.pkl'
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
              ...
        '''
        if not self._set:
            self._normalize()
            self._set = {'COL_NAMES': ('target', 'data', 'size'),
                         'DESCR': f'the SKUs images, normalized {self.norm_size} pixels and their codes',
                         'target': self._target(),
                         'data': self._data(),
                         'size': self.max_size}
        return self._set

    def _data(self):
        return np.array([imread(img) for img in self._images])

    def _target(self):
        return np.array([self._fetcher(img) for img in self._images])

    def _normalize(self):
        self.normalizer.bulk(self.folder, self.max_size)
