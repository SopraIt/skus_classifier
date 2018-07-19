from numpy.random import permutation
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

class Classifier:
    RAND = 42

    def __init__(self, clf=SGDClassifier, data=None, labels=None, rand=RAND):
        self.clf = clf(random_state=rand, max_iter=1000, tol=1e-3)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, random_state=rand)
    
    def fit(self, sku):
        self._shuffle()
        self._by_sku(sku)
        self.clf.fit(self.X_train, self.y_train_sku)
        
    def _by_sku(self, sku):
        self.y_train_sku = (self.y_train == sku)
        self.y_test_sku = (self.y_test == sku)

    def _shuffle(self):
        shuffle_index = permutation(len(self.X_train))
        self.X_train = self.X_train[shuffle_index]
        self.y_train = self.y_train[shuffle_index]
