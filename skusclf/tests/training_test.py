import unittest
from matplotlib.pyplot import imread
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalization(self):
        norm = training.Normalizer(size=32)
        img = norm(stubs.IMAGES[-1])
        w, h = img.size
        self.assertEqual(img.__class__.__name__, 'Image')
        self.assertEqual(w, 32)
        self.assertEqual(h, 32)
        self.assertEqual(norm._offset(img), (0, 0))

    def test_normalization_bkg(self):
        norm = training.Normalizer(size=64, bkg=f'{stubs.PATH}/office.png')
        img = norm(stubs.IMAGES[-1])
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(norm._offset(img), (0, 0))

    def test_normalization_skip(self):
        img = imread(stubs.IMAGES[-1])
        norm = training.Normalizer(size=max(img.shape))
        res = norm(stubs.IMAGES[-1])
        self.assertIsNone(res)

    def test_augmenting_attributes(self):
        aug = training.Augmenter()
        self.assertEqual(len(aug.transformers), 7)
        self.assertEqual(aug.count, 211)
    
    def test_augmenting(self):
        img = imread(stubs.IMAGES[-1])
        aug = training.Augmenter(.05)
        self.assertEqual(aug.count, 11)
        self.assertTrue(all(img.shape == (64, 64, 4) for img in list(aug(img))))

    def test_dataset_attributes(self):
        ds = training.Dataset(f'{stubs.PATH}/dataset.h5', folder=stubs.FOLDER,
                              persist=False)
        self.assertEqual(len(ds.images), 3)
        self.assertEqual(ds.count, 633)
        self.assertEqual(ds.img.shape, (64, 64, 4))
        self.assertEqual(ds.label_dtype, 'S17')

    def test_dataset(self):
        ds = training.Dataset(f'{stubs.PATH}/dataset.h5', folder=stubs.FOLDER,
                              persist=False, augmenter=training.Augmenter(.01))
        ds()
        dataset = ds.load()
        self.assertEqual(dataset['X'].shape, (24, 16384))
        self.assertEqual(dataset['X'].attrs['size'], 64)
        self.assertEqual(dataset['y'].shape, (24,))
        dataset.close()

    def test_empty_dataset_error(self):
        with self.assertRaises(training.Dataset.EmptyFolderError):
            ds = training.Dataset(f'./log/dataset.h5', folder='./log')
            ds()

    def test_shaped_dataset(self):
        ds = training.Dataset(f'{stubs.PATH}/shaped.h5', folder=stubs.FOLDER,
                              persist=False, shape=(4, 64, 64), 
                              augmenter=None, normalizer=None)
        ds()
        dataset = ds.load()
        self.assertEqual(dataset['X'].shape, (3, 4, 64, 64))
        self.assertEqual(dataset['y'].shape, (3,))
        dataset.close()


if __name__ == '__main__':
    unittest.main()
