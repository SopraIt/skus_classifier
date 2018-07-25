import unittest
from matplotlib.pyplot import imread
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalizer_attrs(self):
        norm = training.Normalizer(stubs.IMAGES[-1], max_size=200, optimize=False)
        self.assertEqual(norm.img.__class__.__name__, 'Image')
        self.assertEqual(norm.w, 200)
        self.assertEqual(norm.h, 200)
        self.assertEqual(norm._offset(), (0, 0))
        self.assertEqual(norm._canvas().size, (200, 200))
    
    def test_bad_magnitude(self):
        with self.assertRaises(ValueError):
            training.Augmenter(mag=1)

    def test_data_augmenting(self):
        img = imread(stubs.IMAGES[-1])
        aug = training.Augmenter(mag=2)
        output = list(aug(img))
        self.assertEqual(len(output), 10)
        self.assertTrue(all(img.shape == (250, 250, 4) for img in output))

    def test_training_set(self):
        aug = training.Augmenter(mag=2)
        loader = training.Loader(stubs.PATH, 250, augmenter=aug)
        s = loader.set()
        X, y = [s[k] for k in ('data', 'target')]
        self.assertEqual(s.get('COL_NAMES'), ('target', 'data', 'size'))
        self.assertTrue(s.get('DESCR'))
        self.assertEqual(s.get('size'), 250)
        self.assertEqual(X.shape, (40, 250, 250, 4))
        self.assertEqual(y.shape, (40,))


if __name__ == '__main__':
    unittest.main()
