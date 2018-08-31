import unittest
from matplotlib.pyplot import imread
from PIL import Image
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalization_path(self):
        norm = training.Normalizer(size=64)
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGB')

    def test_normalization_no_canvas(self):
        norm = training.Normalizer(size=64)
        img = norm(Image.open(stubs.IMG))
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGB')

    def test_normalization_canvas(self):
        norm = training.Normalizer(size=64, canvas=True)
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_normalization_colored_canvas(self):
        norm = training.Normalizer(size=64, canvas='FF0000')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_normalization_bkg_canvas(self):
        norm = training.Normalizer(size=64, canvas=f'{stubs.PATH}/office.png')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_adjust_array(self):
        norm = training.Normalizer(size=32)
        img = norm.adjust(stubs.IMG)
        self.assertEqual(img.shape, (21, 32, 3))

    def test_adjust_shaped_array(self):
        norm = training.Normalizer(size=64)
        img = norm.adjust(stubs.IMG, (41, 64, 4))
        self.assertEqual(img.shape, (41, 64, 4))

    def test_normalization_skip(self):
        norm = training.Normalizer(size=300)
        img = norm(stubs.IMG)
        self.assertIsNone(img)

    def test_augmenting_attributes(self):
        aug = training.Augmenter(cutoff=1.)
        self.assertEqual(len(aug.transformers), 6)
        self.assertEqual(aug.count, 185)
    
    def test_augmenting(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(.05)
        images = list(aug(img))
        self.assertEqual(len(images), 8)
        for a in images:
            self.assertEqual(img.shape, a.shape)

    def test_augmenting_skip(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(0)
        images = list(aug(img))
        self.assertEqual(aug.count, 1)
        self.assertEqual(len(images), 1)

    def test_dataset_attributes(self):
        ds = stubs.DATASET
        self.assertEqual(len(ds.images), 3)
        self.assertEqual(ds.count, 108)
        self.assertEqual(ds.sample.shape, (32, 32, 4))
        self.assertEqual(ds.label_dtype, 'S17')

    def test_dataset(self):
        X, y = stubs.DATASET.load()
        self.assertEqual(X.shape, (108, 4096))
        self.assertEqual(y.shape, (108,))

    def test_original_dataset(self):
        X, y = stubs.DATASET.load(original=True)
        self.assertEqual(X.shape, (108, 32, 32, 4))
        self.assertEqual(y.shape, (108,))

    def test_empty_dataset_error(self):
        with self.assertRaises(training.Dataset.EmptyFolderError):
            ds = training.Dataset(f'./{stubs.EMPTY}/dataset.h5', folder=stubs.EMPTY)
            ds()

    def test_noent_dataset(self):
        with self.assertRaises(training.Dataset.NoentError):
            ds = training.Dataset(f'./{stubs.EMPTY}/dataset.h5')
            ds.load()


if __name__ == '__main__':
    unittest.main()
