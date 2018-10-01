import unittest
from warnings import filterwarnings
from skusclf import stubs


class TestClassifier(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore')

    def test_attributes(self):
        for i in range(0, 3):
            self.assertIn(i, list(stubs.MODEL.y))

    def test_prediction(self):
        res = stubs.MODEL(f'{stubs.PATH}/bag.png')
        self.assertEqual(res, '400249_CXZFD_5278')

    def test_accuracy(self):
        for k in stubs.EVL.accuracy:
            self.assertAlmostEqual(k, .9, delta=.1)

    def test_confusion(self):
        self.assertTrue(stubs.EVL.confusion.trace() > 200)

    def test_precision(self):
        for k in stubs.EVL.precision:
            self.assertAlmostEqual(k, .9, delta=.1)

    def test_recall(self):
        for k in stubs.EVL.recall:
            self.assertAlmostEqual(k, .9, delta=.1)


if __name__ == '__main__':
    unittest.main()
