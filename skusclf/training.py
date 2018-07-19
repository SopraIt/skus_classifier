from glob import glob
from os import path
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np


class Normalizer:
    MODE = 'RGBA'
    COLOR = (255, 0, 0, 0)

    def __init__(self, name, mode=MODE, color=COLOR):
        self.name = name
        self.img = Image.open(name)
        self.w, self.h = self.img.size
        self.max = max(self.w, self.h)
        self.mode = mode
        self.color = color

    def save(self, name=None):
        name = name or self.name
        c = self._canvas().paste(self.img, self._offset())
        c = c.quantize()
        c.save(name)

    def _canvas(self):
        return Image.new(self.mode, (self.max, self.max), self.color)

    def _offset(self):
        return ((self.max - self.w) // 2, (self.max - self.h) // 2)


class Loader:
    SKU = lambda n: '_'.join(path.basename(n).split('_')[:3])

    def __init__(self, folder='./images', sku=SKU):
        self._images = glob(f'{folder}/*.png')
        self._sku = sku
        self._set = []

    def set(self):
        if not self._set:
            self._set = {'COL_NAMES': ('labels', 'data'),
                         'DESCR': 'the SKUs images training set',
                         'labels': self._labels(),
                         'data': self._data()}
        return self._set

    def _data(self):
        return np.array([imread(img) for img in self._images])

    def _labels(self):
        return np.array([self._sku(img) for img in self._images])
