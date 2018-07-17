import unittest
from skuscls import training


class TestTraining(unittest.TestCase):
    def setUp(self):
        images = ['523675_ZKD96_9433_002_100_0000_Light.png', '526908_0PLOT_1000_002_100_0000_Light.png', '519335_0LDD0_6072_001_100_0000_Light.png', '487005_0C11E_4970_010_089_0000_Light.png']
        self.l = training.Loader(images)

    def test_loader_set_descr_keys(self):
        col_names = self.l.set().get('COL_NAMES')
        descr = self.l.set().get('DESCR')
        self.assertEqual(col_names, ('labels', 'data'))
        self.assertTrue(descr)

    def test_loader_set_labels(self):
        labels = self.l.set().get('labels')
        self.assertEqual(list(labels), ['523675ZKD969433', '5269080PLOT1000', '5193350LDD06072', '4870050C11E4970'])

    def test_loader_set_data(self):
        data = self.l.set().get('data')
        for img in data:
            self.assertEqual(img.__class__.__name__, 'ndarray')


if __name__ == '__main__':
    unittest.main()
