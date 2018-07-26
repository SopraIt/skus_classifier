import unittest
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = training.Loader.open(stubs.DATASET)

    def test_prediction_training(self):
        mod = classifier.Model()
        for i, img in enumerate(self.dataset['data']):
            res = mod.predict(img, self.dataset)
            self.assertEqual(res[0], self.dataset['target'][i])

    def test_prediction_test(self):
        mod = classifier.Model()
        for i, img in enumerate(self.dataset['data']):
            res = mod.predict(img, self.dataset, test=True)
            self.assertEqual(res[0], self.dataset['target'][i])


if __name__ == '__main__':
    unittest.main()
