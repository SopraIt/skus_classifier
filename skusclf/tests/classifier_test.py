import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.ds = training.Dataset(f'{stubs.PATH}/dataset.h5').load()
        filterwarnings('ignore')

    def tearDown(self):
        self.ds.close()

    def test_attributes(self):
        mod = classifier.Model(self.ds)
        self.assertEqual(mod.size, 64) 
        for i in range(0, 3):
            self.assertIn(i, list(mod.y))

    def test_prediction(self):
        mod = classifier.Model(self.ds)
        res = mod(self.ds['X'][-1])
        self.assertEqual(res[0], self.ds['y'][-1])


if __name__ == '__main__':
    unittest.main()
