import unittest
from matplotlib.pyplot import imread
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalization(self):
        norm = training.Normalizer(max_size=128, optimize=False)
        img = norm(stubs.IMAGES[-1])
        w, h = img.size
        self.assertEqual(img.__class__.__name__, 'Image')
        self.assertEqual(w, 128)
        self.assertEqual(h, 128)
        self.assertEqual(norm._offset(w, h), (0, 0))
        self.assertEqual(norm._canvas().size, (w, h))
    
    def test_data_augmenting(self):
        img = imread(stubs.IMAGES[-1])
        aug = training.Augmenter(3)
        output = list(aug(img))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(img.shape == (256, 256, 4) for img in output))

    def test_loader_attributes(self):
        loader = training.Loader(stubs.PATH)
        self.assertEqual(loader.shape, (256, 256, 4))
        self.assertEqual(loader.count, 600)
        loader = training.Loader(stubs.PATH, augmenter=None)
        self.assertEqual(loader.count, 3)

    def test_loader_augmented_dataset(self):
        loader = training.Loader(stubs.PATH, augmenter=None, normalizer=None)
        dataset = loader.dataset()
        X, y = [dataset[k] for k in ('data', 'target')]
        self.assertEqual(X.shape, (3, 256, 256, 4))
        self.assertEqual(y.shape, (3,))

    def test_loader_augmented_dataset(self):
        loader = training.Loader(stubs.PATH, augmenter=training.Augmenter(3), normalizer=None)
        dataset = loader.dataset()
        X, y = [dataset[k] for k in ('data', 'target')]
        self.assertEqual(X.shape, (9, 256, 256, 4))
        self.assertEqual(y.shape, (9,))


if __name__ == '__main__':
    unittest.main()
