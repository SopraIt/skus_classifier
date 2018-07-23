import unittest
from skusclf import classifier, stubs


class TestClassifier(unittest.TestCase):
    def test_training_test_sets(self):
        mod = classifier.Model(dataset=stubs.DATASET)
        self.assertEqual(len(mod.X_train), 3)
        self.assertEqual(len(mod.X_test), 1)
        self.assertEqual(len(mod.y_train), 3)
        self.assertEqual(len(mod.y_test), 1)

    def test_prediction(self):
        mod = classifier.Model(dataset=stubs.DATASET)
        img = mod.dataset['data'][2]
        res = mod.predict(img)
        self.assertEqual(res[0], '543324_0YFAT_1061')


if __name__ == '__main__':
    unittest.main()
