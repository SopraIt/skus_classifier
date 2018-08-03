import unittest
from warnings import filterwarnings
from skusclf import classifier, stubs, training


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')
        fetcher = training.Dataset.FETCHER_G
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER,
                              fetcher=fetcher, persist=False, 
                              augmenter=training.Augmenter(.1))
        ds()
        self.data = ds.load()

    def tearDown(self):
        self.data.close()

    def test_attributes(self):
        mod = classifier.SGD(self.data)
        self.assertEqual(mod.shape, (32, 32, 4)) 
        for i in range(0, 3):
            self.assertIn(i, list(mod.y))

    def test_png_prediction(self):
        mod = classifier.SGD(self.data)
        res = mod(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')


if __name__ == '__main__':
    unittest.main()
