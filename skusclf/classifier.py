from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from skusclf.logger import BASE as logger
from skusclf.training import Normalizer


class Model:
    '''
    Synopsis
    --------
    Performs a predictions by using the specified sklearn model instance and the
    provided dataset.

    Arguments
    ---------
    - model: a classifier sklearn classifier instance supporting the fit and predict
      methods
    - X: the numpy array containing the images data
    - y: a numpy array containing the labels data
    - shape: the shape used to normalize the image to classify
    - normalizer: the collaborator used to normalize the image to classify

    Constructor
    -----------
    >>> model = Model(SGDClassifier(), array[...], array[...], shape=(64, 64, 4))
    '''

    def __init__(self, model, X, y, shape, normalizer=Normalizer):
        self.model = model
        self.encoder = LabelEncoder()
        self.X = X
        self.y = self._labels(y)
        self._fit()
        self.shape = shape
        self.normalizer = normalizer(size=max(self.shape), canvas=self._canvas())

    def __call__(self, name):
        '''
        Classify the specified image (path or Image object) versus the specified
        training set and labels:
        >>> sgd('./images/elvis.png')
        '''
        img = self._img(name)
        logger.info('making prediction via %s', self.model.__class__.__name__)
        result = self.model.predict([img])
        return self._decode(result)

    def _decode(self, result):
        label = self.encoder.inverse_transform(result)[0].decode('utf-8')
        logger.info('image classified as %s', label)
        return label

    def _canvas(self):
        h, w, _ = self.shape
        return h == w

    def _img(self, name):
        return self.normalizer.adjust(name, self.shape).flatten()
    
    def _labels(self, y):
        logger.info('transforming labels')
        self.encoder.fit(y)
        return self.encoder.transform(y)
    
    def _fit(self):
        logger.info('fitting dataset')
        self.model.fit(self.X, self.y)


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

    Factory
    -------
    >>> model = Model(SGDClassifier(...), X=array[...], y=array[...], shape=(64, 64, 4))
    >>> evl = Evaluator.factory(model)
    '''

    KFOLDS = 3
    SCORING = 'accuracy'

    @classmethod
    def factory(cls, model, kfolds=KFOLDS):
        return cls(model.model, model.X, model.y, kfolds)
    
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

    def __iter__(self):
        yield(f'accuracy:  {max(self.accuracy):.3}')
        yield(f'precision: {max(self.precision):.3}')
        yield(f'recall:    {max(self.recall):.3}')
        yield(f'f1 score:  {max(self.f1_score):.3}')
        yield(f'confusion: {self.confusion.trace()}')
