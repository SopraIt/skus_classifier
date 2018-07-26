from functools import reduce
from operator import mul
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from skusclf import training
from skusclf.logger import BASE as logger


class Model:
    '''
    Synopsis
    --------
    Performs a predictions by using the Stochastic Gradient Descent (SGD) scikit-learn classifier.
    
    Arguments
    ---------
    - rand: the random seed used by classifier and for train/test splitting
    - test_size: the percentage of the test set

    Returns
    -------
    - the classified label/s set

    Constructor
    -----------
    >>> clf = Classifier(rand=42, test_size=0.3) 
    '''

    RAND = 42
    KEYS = ('data', 'target')
    
    def __init__(self, rand=RAND, test_size=0.2):
        self.model = SGDClassifier(random_state=rand, max_iter=1000, tol=1e-3)
        self.rand = rand
        self.test_size = test_size

    def predict(self, img, dataset, test=False):
        '''
        Accepts an image (as binary array) the dataset to perform a prediction.
        If test flag is true predict versus the test dataset.
        >>> clf.predict(array([[[1., 0., 0., 0.],...]]), 
        >>>             {'data': array([[[1., 0., 1., 0.],...]]},
        >>>             test=True),
        '''
        self._fit(*self._X_y(dataset, test))
        logger.info('predicting on dataset')
        return self.model.predict([img.flatten()])
    
    def _fit(self, X, y):
        logger.info('fitting on dataset')
        self.model.fit(X, y)
    
    def _X_y(self, dataset, test):
        logger.info('splitting training set')
        X, X_t, y, y_t = train_test_split(dataset['data'], dataset['target'], 
                                          random_state=self.rand, 
                                          test_size=self.test_size)
        return (X_t, y_t) if test else (X, y)
