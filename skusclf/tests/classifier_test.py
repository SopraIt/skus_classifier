import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True), 
                              augmenter=training.Augmenter(0.01))
        X, y = ds.load()
        self.clf = classifier.SGD(X, y, (32, 32, 4))

    def test_attributes(self):
        for i in range(0, 3):
            self.assertIn(i, list(self.clf.y))

    def test_splitting(self):
        X_train, X_test, y_train, y_test = self.clf.split(0.3)
        self.assertEqual(X_train.shape, (14, 4096))
        self.assertEqual(X_test.shape, (7, 4096))
        self.assertEqual(y_train.shape, (14,))
        self.assertEqual(y_test.shape, (7,))
        self.assertFalse(self.clf.split(1.2))

    def test_prediction(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True), 
                              augmenter=training.Augmenter(0.3))
        X, y = ds.load()
        clf = classifier.SGD(X, y, (32, 32, 4))
        res = clf(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        X_train, _, y_train, _ = self.clf.split(0.2)
        evl = classifier.Evaluator(self.clf.model, X_train, y_train)
        for k in evl.accuracy:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_confusion(self):
        X_train, _, y_train, _ = self.clf.split(0.2)
        evl = classifier.Evaluator(self.clf.model, X_train, y_train)
        for c in evl.confusion.diagonal():
            self.assertTrue(c > 3)

    def test_precision(self):
        X_train, _, y_train, _ = self.clf.split(0.2)
        evl = classifier.Evaluator(self.clf.model, X_train, y_train)
        for k in evl.precision:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_recall(self):
        X_train, _, y_train, _ = self.clf.split(0.2)
        evl = classifier.Evaluator(self.clf.model, X_train, y_train)
        for k in evl.recall:
            self.assertAlmostEqual(k, .9, delta=.5)


if __name__ == '__main__':
    unittest.main()
