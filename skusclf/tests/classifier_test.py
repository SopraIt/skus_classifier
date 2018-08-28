import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True), 
                              augmenter=training.Augmenter(0.01), shuffle=False)
        X, y = ds.load()
        self.clf = classifier.SGD(X, y, (32, 32, 4))

    def test_attributes(self):
        for i in range(0, 3):
            self.assertIn(i, list(self.clf.y))

    def test_prediction(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True), 
                              augmenter=training.Augmenter(0.5))
        X, y = ds.load()
        clf = classifier.SGD(X, y, (32, 32, 4))
        res = clf(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        evl = classifier.Evaluator.factory(self.clf)
        for k in evl.accuracy:
            self.assertAlmostEqual(k, .9, delta=.5)

    def test_confusion(self):
        evl = classifier.Evaluator.factory(self.clf)
        for c in evl.confusion.diagonal():
            self.assertTrue(c > 1)

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
