import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        self.create_ds = lambda cutoff=0.2: training.Dataset(stubs.DATASET, 
                                                             folder=stubs.FOLDER, 
                                                             brand='gg', 
                                                             normalizer=training.Normalizer(canvas=True), 
                                                             augmenter=training.Augmenter(float(cutoff)))()

    def test_attributes(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        self.assertEqual(clf.shape, [32, 32, 4]) 
        for i in range(0, 3):
            self.assertIn(i, list(clf.y))

    def test_splitting(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        X_train, X_test, y_train, y_test = clf.split(0.3)
        self.assertEqual(X_train.shape, (37, 4096))
        self.assertEqual(X_test.shape, (17, 4096))
        self.assertEqual(y_train.shape, (37,))
        self.assertEqual(y_test.shape, (17,))
        self.assertFalse(clf.split(1.2))

    def test_prediction(self):
        ds = self.create_ds(0.3)
        data = ds.load()
        clf = classifier.SGD(data)
        res = clf(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        X_train, _, y_train, _ = clf.split(0.2)
        evl = classifier.Evaluator(clf.model, X_train, y_train)
        for k in evl.accuracy:
            self.assertAlmostEqual(k, .9, delta=.4)

    def test_confusion(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        X_train, _, y_train, _ = clf.split(0.2)
        evl = classifier.Evaluator(clf.model, X_train, y_train)
        for c in evl.confusion.diagonal():
            self.assertTrue(c > 3.5)

    def test_precision(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        X_train, _, y_train, _ = clf.split(0.2)
        evl = classifier.Evaluator(clf.model, X_train, y_train)
        for k in evl.precision:
            self.assertAlmostEqual(k, .9, delta=.4)

    def test_recall(self):
        ds = self.create_ds(0.1)
        data = ds.load()
        clf = classifier.SGD(data)
        X_train, _, y_train, _ = clf.split(0.2)
        evl = classifier.Evaluator(clf.model, X_train, y_train)
        for k in evl.recall:
            self.assertAlmostEqual(k, .9, delta=.4)


if __name__ == '__main__':
    unittest.main()
