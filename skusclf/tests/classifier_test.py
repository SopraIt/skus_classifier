import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        X, y = stubs.DATASET.load()
        self.sgd = classifier.Model(stubs.SGD, X, y, (32, 32, 4))
        self.rf = classifier.Model(stubs.RF, X, y, (32, 32, 4))

    def test_attributes(self):
        for i in range(0, 3):
            self.assertIn(i, list(self.sgd.y))

    def test_prediction_sgd(self):
        res = self.sgd(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_prediction_rf(self):
        res = self.rf(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        evl = classifier.Evaluator.factory(self.sgd)
        for k in evl.accuracy:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_confusion(self):
        evl = classifier.Evaluator.factory(self.rf)
        self.assertTrue(evl.confusion.trace() > 15)

    def test_precision(self):
        evl = classifier.Evaluator.factory(self.sgd)
        for k in evl.precision:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_recall(self):
        evl = classifier.Evaluator.factory(self.rf)
        for k in evl.recall:
            self.assertAlmostEqual(k, .9, delta=.5)


if __name__ == '__main__':
    unittest.main()
