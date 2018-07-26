import unittest
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = training.Loader.open(stubs.DATASET)

    def test_training_test_sets(self):
        mod = classifier.Model(self.dataset)
        self.assertEqual(len(mod.X_train), 7)
        self.assertEqual(len(mod.X_test), 2)
        self.assertEqual(len(mod.y_train), 7)
        self.assertEqual(len(mod.y_test), 2)

    def test_prediction(self):
        mod = classifier.Model(self.dataset)
        img = mod.dataset['data'][2]
        res = mod.predict(img)
        self.assertEqual(res[0], mod.dataset['target'][2])


if __name__ == '__main__':
    unittest.main()
