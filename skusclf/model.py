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
    
    def predict(self, img, test=False):
        self._fit(test)
        img = self._flat_img(img)
        return self.clf.predict([img])

    def _fit(self, test):
        X = self.X_test if test else self.X_train
        y = self.y_test if test else self.y_train
        X, y = self._flatten(X, y)
        self.clf.fit(X, y)
        
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
       flat = reduce(mul, dims[1:])
       return _data.reshape(dims[0], reduce(mul, dims[1:]))
