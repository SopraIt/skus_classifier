import unittest
from matplotlib.pyplot import imread
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def test_normalization_plain(self):
        norm = training.Normalizer(size=64)
        img = norm(stubs.IMG)
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

    def test_normalization_bkg(self):
        norm = training.Normalizer(size=64, canvas=f'{stubs.PATH}/office.png')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGB')

    def test_normalized_array(self):
        norm = training.Normalizer(size=32)
        img = norm.to_array(stubs.IMG)
        self.assertEqual(img.shape, (21, 32, 3))

    def test_normalized_shaped_array(self):
        norm = training.Normalizer(size=64)
        img = norm.to_array(stubs.IMG, (41, 64, 4))
        self.assertEqual(img.shape, (41, 64, 4))

    def test_normalization_skip(self):
        norm = training.Normalizer(size=300)
        img = norm(stubs.IMG)
        self.assertIsNone(img)

    def test_augmenting_attributes(self):
        aug = training.Augmenter(cutoff=1.)
        self.assertEqual(len(aug.transformers), 7)
        self.assertEqual(aug.count, 164)
    
    def test_augmenting(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(.05)
        images = list(aug(img))
        self.assertEqual(aug.count, len(images))
        for a in images:
            self.assertEqual(img.shape, a.shape)

    def test_dataset_attributes(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER)
        self.assertEqual(len(ds.images), 3)
        self.assertEqual(ds.count, 246)
        self.assertEqual(ds.sample.shape, (16, 32, 4))
        self.assertEqual(ds.label_dtype, 'S40')

    def test_dataset(self):
        ds = training.Dataset(stubs.DATASET, folder=stubs.FOLDER,
                              brand='gg', 
                              normalizer=training.Normalizer(canvas=True),
                              augmenter=training.Augmenter(.01))
        ds()
        dataset = ds.load()
        shape = dataset['X'].attrs['shape']
        self.assertEqual(dataset['X'].shape, (24, 4096))
        self.assertEqual(shape.tolist(), [32, 32, 4])
        self.assertEqual(dataset['y'].shape, (24,))
        self.assertEqual(ds.labels_count, 3)
        dataset.close()

    def test_empty_dataset_error(self):
        with self.assertRaises(training.Dataset.EmptyFolderError):
            ds = training.Dataset(f'./{stubs.EMPTY}/dataset.h5', folder=stubs.EMPTY)
            ds()


if __name__ == '__main__':
    unittest.main()
