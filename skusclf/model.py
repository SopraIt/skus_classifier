from functools import reduce
from operator import mul
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


class Classifier:
    '''
    Synopsis
    --------
    Performs a predictions by using a custom scikit-learn classifier model.
    
    Arguments
    ---------
    - data: the image binaries training set (X)
    - labels: the labels training set (y)
    - rand: the random seed used by classifier and for train/test splitting
    - test_size: the percentage of the test set

    Returns
    -------
    - the classified label/s set

    Constructor
    -----------
    >>> clf = Classifier(data=array(...), labels=array(...),
    >>>                  rand=42, test_size=0.3) 
    '''

    RAND = 42
    
    @classmethod
    def factory(cls, loader):
        '''
        Returns an instance by passing the Loader collaborator
        '''
        _set = loader.set()
        return cls(data=_set.get('data'), labels=_set.get('labels'))

    def __init__(self, data=None, labels=None, rand=RAND, test_size=0.2):
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, random_state=rand, test_size=test_size)
        self.dims = self.X_train[0].shape
        self.flat_dim = reduce(mul, self.dims)
    
    def predict(self, img, test=False):
        '''
        Accepts an image (as binary array) and performs a prediction
        versus the training set (the test one if test=True):
        >>> clf.predict()
        '''
        self._fit(test)
        data = self._flat_img(img)
        return self.model.predict([data])

    def _fit(self, test):
        X = self.X_test if test else self.X_train
        y = self.y_test if test else self.y_train
        X, y = self._flatten(X, y)
        self.model.fit(X, y)
        
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
