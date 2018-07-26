from functools import reduce
from operator import mul
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from skusclf.logger import BASE as logger


class Model:
    '''
    Synopsis
    --------
    Performs a predictions by using the Stochastic Gradient Descent (SGD) scikit-learn classifier.
    
    Arguments
    ---------
    - dataset: the dict dataset, previously normalized and augmented
    - rand: the random seed used by classifier and for train/test splitting
    - test_size: the percentage of the test set

    Returns
    -------
    - the classified label/s set

    Constructor
    -----------
    >>> clf = Classifier(dataset=./skus_400x400.pkl, rand=42, test_size=0.3) 
    '''

    RAND = 42
    KEYS = ('data', 'target')
    
    def __init__(self, dataset, rand=RAND, test_size=0.2):
        self.dataset = dataset
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_set(rand, test_size)
        self.dims = self.X_train[0].shape
        self.flat_dim = reduce(mul, self.dims)

    def predict(self, img, test=False):
        '''
        Accepts an image (as binary array) and performs a prediction
        versus the training set (the test one if test=True):
        >>> clf.predict()
        '''
        logger.info('predict on %s dataset', self._env(test))
        self._fit(test)
        data = self._flat_img(img)
        return self.model.predict([data])

    def _fit(self, test):
        logger.info('fitting data on %s', self._env(test))
        X = self.X_test if test else self.X_train
        y = self.y_test if test else self.y_train
        X, y = self._flatten(X, y)
        self.model.fit(X, y)

    def _env(self, test):
        return 'test' if test else 'training'
    
    def _split_set(self, rand, test_size):
        logger.info('splitting training and test set')
        X, y = tuple(self.dataset[k] for k in self.KEYS)
        return train_test_split(X, y, random_state=rand, test_size=test_size)
        
    def _flatten(self, X, y):
        return self._flat(X), self._flat(y)

    def _flat_img(self, img):
        dims = img.shape
        if len(dims) == 1:
            return img
        return img.reshape(reduce(mul, dims))
    
    def _flat(self, _data):
       dims = _data.shape
       if len(dims) <= 2:
           return _data
       return _data.reshape(dims[0], reduce(mul, dims[1:]))
