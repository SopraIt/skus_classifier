import unittest
from skusclf import model, training


class TestModel(unittest.TestCase):
    def setUp(self):
        _set = training.Loader(['523675_ZKD96_9433_002_100_0000_Light.png', '526908_0PLOT_1000_002_100_0000_Light.png', '519335_0LDD0_6072_001_100_0000_Light.png', '487005_0C11E_4970_010_089_0000_Light.png']).set()
        self.clf = model.Classifier(data=_set.get('data'), labels=_set.get('labels'))
    
    def test_training_test_sets(self):
        self.assertEqual(len(self.clf.X_train), 3)
        self.assertEqual(len(self.clf.X_test), 1)
        self.assertEqual(len(self.clf.y_train), 3)
        self.assertEqual(len(self.clf.y_test), 1)

    def test_fitting(self):
        self.clf.fit('523675ZKD969433')


if __name__ == '__main__':
    unittest.main()
