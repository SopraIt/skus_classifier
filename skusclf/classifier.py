from os import path
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from skusclf.logger import BASE as logger
from skusclf.training import Normalizer


class Model:
    '''
    Synopsis
    --------
    Performs a predictions by using the Stochastic Gradient Descent (SGD) 
    scikit-learn classifier.
    
    Arguments
    ---------
    - dataset: a dict like object having the 'X' and 'y' keys
    - img: the path to the image or the binary data properly reshaped, of the
      classification target
    - size: the max size used to normalize the image to classify, try to fetch it
      from dataset meta-attributes if not specified
    - rand: the random seed used by classifier

    Returns
    -------
    - the classified label

    Constructor
    -----------
    >>> clf = Classifier({'X': array[...], 'y': array[...]}, size=64, rand=666)
    '''

    RAND = 42

    def __init__(self, dataset, size=None, rand=RAND):
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.encoder = LabelEncoder()
        self.X = dataset['X']
        self.y = self._labels(dataset)
        self.size = size or max(self.X.attrs['orig_shape'])
        self.rand = rand

    def __call__(self, img):
        '''
        Classify the specified image (path or binary data) versus the dataset:
        >>> clf()
        '''
        img = self._img(img)
        logger.info('fitting on dataset')
        self.model.fit(self.X, self.y)
        res = self.model.predict([img])
        return self.encoder.inverse_transform(res)

    def _img(self, img):
        if self._valid(img): return img
        logger.info('fetching data by %s', path.basename(img))
        norm = Normalizer(self.size)
        norm.persist(img)
        return plt.imread(img).flatten()

    def _valid(self, img):
        return hasattr(img, 'shape') and img.shape == self.X[0].shape

    def _labels(self, dataset):
        self.encoder.fit(dataset['y'])
        return self.encoder.transform(dataset['y'])
