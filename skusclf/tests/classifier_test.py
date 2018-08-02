import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        ds = training.Dataset(f'{stubs.PATH}/dataset.h5', folder=stubs.FOLDER,
                              persist=False, augmenter=training.Augmenter(.01))
        ds()
        self.data = ds.load()

    def tearDown(self):
        self.data.close()

    def test_attributes(self):
        mod = classifier.Model(self.data)
        self.assertEqual(mod.size, 64) 
        for i in range(0, 3):
            self.assertIn(i, list(mod.y))

    def test_prediction(self):
        mod = classifier.Model(self.data)
        res = mod(self.data['X'][-1])
        self.assertEqual(res, self.data['y'][-1].decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
