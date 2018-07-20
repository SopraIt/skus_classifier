from glob import glob
from os import path
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np


class Normalizer:
    '''
    Synopsis
    --------
    Normalizes the PNG images that compose the training set by:
    - resizing the largest dimension to specified max size
    - creating a squared canvas by max size and pasting the image into it
    - quantizing the PNG

    Arguments
    ---------
    - path to the PNG file
    - max_size: the size of the target image
    - mode: the color mode
    - color: the background color of the squared canvas

    Constructor
    -----------
    >>> norm = Normalizer('./my_images/img.png', 
    >>>                   max_size=400, mode='RGBA', color=(255,0,0,0))
    '''

    MODE = 'RGBA'
    COLOR = (255, 0, 0, 0)
    
    @classmethod
    def bulk(cls, folder, max_size):
        '''
        Overwrites all PNGs in the specified folder:
        >>> Normalizer.bulk(folder='./my_images', max_size=400)
        '''
        n = 0
        for name in glob(f'{folder}/*.png'):
            if cls(name, max_size).save():
                n += 1
        return n

    def __init__(self, name, max_size, mode=MODE, color=COLOR):
        self.name = name
        self.max_size = max_size
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
        Creates a squared canvas, by pasting image and quantizing it
        '''
        c = self._canvas()
        c.paste(self.img, self._offset())
        return c.quantize()

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
    Data and labels are converted to Numpy arrays.

    Arguments
    ---------
    - folder: the folder containing the PNG files
    - fetcher: a callable taking as an argument the filename and returning the
      formatted label

    Constructor
    -----------
    >>> plain = lambda name: name
    >>> loader = Loader(folder='./my_images', fetcher=plain)
    '''

    FETCHER = lambda n: '_'.join(path.basename(n).split('_')[:3])

    def __init__(self, folder, fetcher=FETCHER):
        self._images = glob(f'{folder}/*.png')
        self._fetcher = fetcher
        self._set = []

    def max_size(self):
        '''
        Returns the trainig set images max size.
        '''
        if not self._images:
            return
        img = Image.open(self._images[0])
        return max(img.size)

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
            self._set = {'COL_NAMES': ('labels', 'data'),
                         'DESCR': 'the SKUs images training set',
                         'labels': self._labels(),
                         'data': self._data()}
        return self._set

    def _data(self):
        return np.array([imread(img)[:,:,0] for img in self._images])

    def _labels(self):
        return np.array([self._fetcher(img) for img in self._images])
