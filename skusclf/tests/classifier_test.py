import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        ds = training.Dataset(f'{stubs.PATH}/dataset.h5', folder=stubs.FOLDER,
                              persist=False, augmenter=training.Augmenter(.1))
        ds()
        self.data = ds.load()

    def tearDown(self):
        self.data.close()

    def test_attributes(self):
        mod = classifier.Model(self.data)
        self.assertEqual(mod.size, 32) 
        for i in range(0, 3):
            self.assertIn(i, list(mod.y))

    def test_png_prediction(self):
        mod = classifier.Model(self.data)
        res = mod(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_jpg_prediction(self):
        mod = classifier.Model(self.data)
        res = mod(f'{stubs.PATH}/bag.jpg')
        self.assertEqual(res, '400249_CXZFD_5278')


if __name__ == '__main__':
    unittest.main()
