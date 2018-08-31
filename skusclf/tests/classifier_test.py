import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        X, y = stubs.DATASET.load()
        self.clf = classifier.SGD(X, y, (32, 32, 4))

    def test_attributes(self):
        for i in range(0, 3):
            self.assertIn(i, list(self.clf.y))

    def test_prediction(self):
        res = self.clf(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        evl = classifier.Evaluator.factory(self.clf)
        for k in evl.accuracy:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_confusion(self):
        evl = classifier.Evaluator.factory(self.clf)
        self.assertTrue(evl.confusion.trace() > 15)

    def test_precision(self):
        evl = classifier.Evaluator.factory(self.clf)
        for k in evl.precision:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_recall(self):
        evl = classifier.Evaluator.factory(self.clf)
        for k in evl.recall:
            self.assertAlmostEqual(k, .9, delta=.5)


if __name__ == '__main__':
    unittest.main()
