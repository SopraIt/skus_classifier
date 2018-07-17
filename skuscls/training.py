from os import listdir, path
from matplotlib.pyplot import imread
import numpy as np

class Loader:
    SKU = lambda n: ''.join(path.basename(n).split('_')[:3])

    def __init__(self, data=None, folder='.', sku=SKU):
        self._data = data or []
        self._folder = path.join(path.abspath(folder), 'images')
        self._sku = sku
        self._set = []
    
    def set(self):
        if not self._set:
            self._set = {'COL_NAMES': ('labels', 'data'),
                        'DESCR': 'the SKUs images training set',
                        'labels': np.array(self._labels()),
                        'data': np.array(self._binaries())}
        return self._set

    def _binaries(self):
        return [imread(img) for img in self._images()]

    def _images(self):
        if self._data:
            return (self._abs(img) for img in self._data)
        return [self._abs(img) for img in listdir(self._folder)]

    def _labels(self):
        return [self._sku(img) for img in self._images()]

    def _abs(self, img):
        return path.join(self._folder, img)
