import unittest
import stubs
from skusclf import training


class TestTraining(unittest.TestCase):
    def test_normalizer_attrs(self):
        norm = training.Normalizer(stubs.IMAGES[-1])
        self.assertEqual(norm.img.__class__.__name__, 'PngImageFile')
        self.assertEqual(norm.w, 250)
        self.assertEqual(norm.h, 130)
        self.assertEqual(norm._offset(), (0, 60))
        self.assertEqual(norm._canvas().size, (250, 250))

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
