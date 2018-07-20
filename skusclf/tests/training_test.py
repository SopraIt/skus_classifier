import unittest
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalizer_attrs(self):
        norm = training.Normalizer(stubs.IMAGES[-1], max_size=200)
        self.assertEqual(norm.img.__class__.__name__, 'Image')
        self.assertEqual(norm.w, 200)
        self.assertEqual(norm.h, 200)
        self.assertEqual(norm._offset(), (0, 0))
        self.assertEqual(norm._canvas().size, (200, 200))

    def test_set_max_size(self):
        loader = training.Loader(stubs.PATH)
        self.assertEqual(loader.max_size(), 250)

    def test_set_descr_keys(self):
        loader = training.Loader(stubs.PATH)
        col_names = loader.set().get('COL_NAMES')
        descr = loader.set().get('DESCR')
        self.assertEqual(col_names, ('labels', 'data'))
        self.assertTrue(descr)

    def test_set_labels(self):
        loader = training.Loader(stubs.PATH)
        labels = loader.set().get('labels')
        self.assertEqual(list(labels), ['543508_Z317M_9511', '543060_XRC03_4048', '543324_0YFAT_1061', '400249_CXZFD_5278'])

    def test_set_data(self):
        loader = training.Loader(stubs.PATH)
        data = loader.set().get('data')
        for img in data:
            self.assertEqual(img.__class__.__name__, 'ndarray')


if __name__ == '__main__':
    unittest.main()
