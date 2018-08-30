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

    def test_normalization(self):
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

    def test_normalization_bkg_color(self):
        norm = training.Normalizer(size=64, canvas='FF0000')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_normalization_bkg_img(self):
        norm = training.Normalizer(size=64, canvas=f'{stubs.PATH}/office.png')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_adjust_array(self):
        norm = training.Normalizer(size=32)
        img = norm.adjust(stubs.IMG)
        self.assertEqual(img.dtype, 'float64')
        self.assertEqual(img.shape, (21, 32, 3))

    def test_adjust_shaped_array(self):
        norm = training.Normalizer(size=64)
        img = norm.adjust(stubs.IMG, (41, 64, 4))
        self.assertEqual(img.dtype, 'float64')
        self.assertEqual(img.shape, (41, 64, 4))

    def test_normalization_skip(self):
        norm = training.Normalizer(size=300)
        img = norm(stubs.IMG)
        self.assertIsNone(img)

    def test_augmenting_attributes(self):
        aug = training.Augmenter(cutoff=1.)
        self.assertEqual(len(aug.transformers), 7)
        self.assertEqual(aug.count, 188)
    
    def test_augmenting(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(.05)
        images = list(aug(img))
        self.assertEqual(len(images), 9)
        for a in images:
            self.assertEqual(img.shape, a.shape)

    def test_augmenting_skip(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(0)
        images = list(aug(img))
        self.assertEqual(aug.count, 1)
        self.assertEqual(len(images), 1)

    def test_dataset_attributes(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER,
                              augmenter=training.Augmenter(.5))
        self.assertEqual(len(ds.images), 3)
        self.assertEqual(ds.count, 276)
        self.assertEqual(ds.sample.shape, (16, 32, 4))
        self.assertEqual(ds.label_dtype, 'S40')

    def test_dataset(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True),
                              augmenter=training.Augmenter(.01))
        ds()
        X, y = ds.load()
        self.assertEqual(X.dtype, 'float64')
        self.assertEqual(y.dtype, 'S17')
        self.assertEqual(X.shape, (24, 4096))
        self.assertEqual(y.shape, (24,))
        self.assertEqual(ds.labels_count, 3)

    def test_original_dataset(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER, brand='gg', 
                              normalizer=training.Normalizer(canvas=True),
                              augmenter=training.Augmenter(.01))
        ds()
        X, y = ds.load(original=True)
        self.assertEqual(X.dtype, 'float64')
        self.assertEqual(y.dtype, 'S17')
        self.assertEqual(X.shape, (24, 32, 32, 4))
        self.assertEqual(y.shape, (24,))

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
