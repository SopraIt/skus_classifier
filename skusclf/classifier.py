from os import path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from skusclf.logger import BASE as logger
from skusclf.training import Normalizer


class SGD:
    '''
    Synopsis
    --------
    Performs a predictions by using the Stochastic Gradient Descent (SGD) 
    scikit-learn classifier.
    
    Arguments
    ---------
    - dataset: a dict like object having the 'X' and 'y' keys
    - shape: the shape used to normalize the image to classify, try to fetch it
      from dataset meta-attributes if not specified
    - rand: the random seed used by classifier
    - normalizer: the collaborator used to normalize the image to classify

    Returns
    -------
    - the classified label

    Constructor
    -----------
    >>> clf = Classifier({'X': array[...], 'y': array[...]}, size=64, rand=666)
    '''

    RAND = 42

    def __init__(self, dataset, shape=None, rand=RAND, normalizer=Normalizer):
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.encoder = LabelEncoder()
        self.X = dataset['X']
        self.y = self._labels(dataset)
        self.shape = shape or self.X.attrs['shape'].tolist()
        self.normalizer = normalizer(size=max(self.shape), canvas=self._canvas())

    def __call__(self, name):
        '''
        Classify the specified image (path or binary data) versus the dataset:
        >>> clf('./images/elvis.png')
        '''
        img = self._img(name)
        logger.info('fitting on dataset')
        self.model.fit(self.X, self.y)
        logger.info('making prediction via %s', self.model.__class__.__name__)
        res = self.model.predict([img])
        label = self.encoder.inverse_transform(res)[0].decode('utf-8')
        logger.info('image classified as %s', label)
        return label
    
    def _canvas(self):
        h, w, _ = self.shape
        return h == w

    def _img(self, name):
        return self.normalizer.adjust(name, self.shape).flatten()

    def _labels(self, dataset):
        logger.info('transforming labels')
        self.encoder.fit(dataset['y'])
        return self.encoder.transform(dataset['y'])
