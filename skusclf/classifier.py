from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
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
    - shape: the shape used to normalize the image to classify
    - rand: the random seed used by classifier
    - normalizer: the collaborator used to normalize the image to classify

    Constructor
    -----------
    >>> clf = Classifier({'X': array[...], 'y': array[...]}, size=64, rand=666)
    '''

    RAND = 42
    TEST_SIZE= 0.2

    def __init__(self, X, y, shape, rand=RAND, normalizer=Normalizer):
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.encoder = LabelEncoder()
        self.X = X
        self.y = self._int_labels(y)
        self.shape = shape
        self.normalizer = normalizer(size=max(self.shape), canvas=self._canvas())

    def __call__(self, name, X=None, y=None):
        '''
        Classify the specified image (path or binary data) versus the specified
        training set and labels:
        >>> clf('./images/elvis.png')
        '''
        X = self.X if X is None else X
        y = self.y if y is None else y
        img = self._img(name)
        logger.info('fitting on dataset')
        self.model.fit(X, y)
        logger.info('making prediction via %s', self.model.__class__.__name__)
        res = self.model.predict([img])
        label = self.encoder.inverse_transform(res)[0].decode('utf-8')
        logger.info('image classified as %s', label)
        return label

    def split(self, test_size=TEST_SIZE):
        '''
        Split the dataset in training and test portions basing on the float representing test size
        >>> clf.split(0.5)
        '''
        if float(test_size) > 1.: return
        count = self.y.shape[0]
        idx = int(count * (1. - test_size))
        return self.X[:idx], self.X[idx:], self.y[:idx], self.y[idx:]
    
    def _canvas(self):
        h, w, _ = self.shape
        return h == w

    def _img(self, name):
        return self.normalizer.adjust(name, self.shape).flatten()

    def _int_labels(self, y):
        logger.info('transforming labels')
        self.encoder.fit(y)
        return self.encoder.transform(y)


class Evaluator:
    '''
    Synopsis
    --------
    The class wraps different evaluation scikit-learn tools to evaluate performance
    and accuracy of the specified classification model.

    Arguments
    ---------
    - model: an estimator instance, responding to the 'fit' method
    - X: the training dataset
    - y: the labels dataset
    - kflods: an integer indicating the cross validation splitting

    Constructor
    -----------
    >>> evl = Evaluator(model=SGDClassifier(...), X=array[...], y=array[...], kfolds=3)
    '''

    KFOLDS = 3
    SCORING = 'accuracy'
    
    def __init__(self, model, X, y, kfolds=KFOLDS):
        self.model = model
        self.X= X
        self.y = y
        self.kfolds = int(kfolds)
        self.y_pred = cross_val_predict(self.model, self.X, self.y, cv=self.kfolds)

    @property
    def accuracy(self):
        return cross_val_score(self.model, self.X, self.y, cv=self.kfolds, scoring=self.SCORING)

    @property
    def confusion(self):
        return confusion_matrix(self.y, self.y_pred)

    @property
    def precision(self):
        return precision_score(self.y, self.y_pred, average=None)

    @property
    def recall(self):
        return recall_score(self.y, self.y_pred, average=None)

    @property
    def f1_score(self):
        return f1_score(self.y, self.y_pred, average=None)
