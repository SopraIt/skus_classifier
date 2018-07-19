from functools import reduce
from operator import mul
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

class Classifier:
    RAND = 42

    def __init__(self, clf=SGDClassifier, data=None, 
                 labels=None, rand=RAND, test_size=0.2):
        self.clf = clf(random_state=rand, max_iter=1000, tol=1e-3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, random_state=rand, test_size=test_size)
        self.dims = self.X_train[0].shape
        self.flat_dim = reduce(mul, self.dims)
    
    def predict(self, lb, img):
        self._fit(lb)
        return self.clf.predict([img])

    def _fit(self, lb):
        self._flatten()
        self._by_label(lb)
        self.clf.fit(self.X_train, self.y_train_lb)
        
    def _by_label(self, lb):
        self.y_train_lb = (self.y_train == lb)
        self.y_test_lb = (self.y_test == lb)

    def _flatten(self):
        self.X_train = self._flat(self.X_train)
        self.y_train = self._flat(self.y_train)
    
    def _flat(self, _set):
       dims = _set.shape
       if len(dims) <= 2:
           return _set
       flat = reduce(mul, dims[1:])
       return _set.reshape(dims[0], flat)
