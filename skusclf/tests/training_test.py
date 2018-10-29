import unittest
from matplotlib.pyplot import imread
from PIL import Image
from skusclf import stubs, training


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.count = 501

    def test_normalization_path(self):
        norm = training.Normalizer(size=64)
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_no_canvas(self):
        norm = training.Normalizer(size=64)
        img = norm(Image.open(stubs.IMG))
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_canvas(self):
        norm = training.Normalizer(size=64, canvas=True)
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_colored_canvas(self):
        norm = training.Normalizer(size=64, canvas='FF0000')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_bkg_canvas(self):
        norm = training.Normalizer(size=64, canvas=f'{stubs.PATH}/office.png')
        img = norm(stubs.IMG)
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')

    def test_adjust_array(self):
        norm = training.Normalizer(size=32)
        img = norm.adjust(stubs.IMG)
        self.assertEqual(img.shape, (21, 32, 4))

    def test_adjust_shaped_array(self):
        norm = training.Normalizer(size=64)
        img = norm.adjust(stubs.IMG, (41, 64, 4))
        self.assertEqual(img.shape, (41, 64, 4))

    def test_augmenting_attributes(self):
        aug = training.Augmenter(cutoff=1.)
        self.assertEqual(len(aug.transformers), 6)
        self.assertEqual(aug.count, 556)
        self.assertEqual(str(aug), 'Augmenter(cutoff=1.0, count=556)')
    
    def test_augmenting(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(.05)
        images = list(aug(img))
        self.assertEqual(len(images), aug.count)
        for a in images:
            self.assertEqual(img.shape, a.shape)

    def test_augmenting_skip(self):
        img = imread(stubs.IMG)
        aug = training.Augmenter(0)
        images = list(aug(img))
        self.assertEqual(aug.count, 1)
        self.assertEqual(len(images), 1)

    def test_features_attributes(self):
        lbl_type, img_type = stubs.FEATURES.types
        self.assertEqual(stubs.FEATURES.count, self.count)
        self.assertEqual(lbl_type, 'S17')
        self.assertEqual(img_type, (32, 32, 4))

    def test_features(self):
        features = list(stubs.FEATURES)
        sku, imgs = features[0]
        self.assertEqual(sku, '400249_CXZFD_5278')
        for img in imgs:
            self.assertEqual(img.shape, (32, 32, 4))
            self.assertTrue((img <= 1.).all())

    def test_empty_features_error(self):
        with self.assertRaises(training.Features.EmptyFolderError):
            features = training.Features(stubs.EMPTY)
            list(features)

    def test_dataset_h5(self):
        self.assertEqual(stubs.X.shape, (self.count, 4096))
        self.assertEqual(stubs.y.shape, (self.count,))

    def test_dataset_h5_orig(self):
        self.assertEqual(stubs.X_orig.shape, (self.count, 32, 32, 4))
        self.assertEqual(stubs.y_orig.shape, (self.count,))

    def test_dataset_h5_noent(self):
        with self.assertRaises(training.DatasetH5.NoentError):
            training.DatasetH5.load(f'./{stubs.EMPTY}/dataset.h5')

    def test_dataset_zip(self):
        ds = training.DatasetZIP(f'./{stubs.PATH}/dataset', stubs.FEATURES)
        ds()
        names = ds.zip.namelist()
        labels = {name.split('/')[0] for name in names}
        idx = (self.count // 3) - 1
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(names), self.count)
        self.assertEqual(names[0], 'LBL_400249_CXZFD_5278/sample_0.png')
        self.assertEqual(names[idx], 'LBL_400249_CXZFD_5278/sample_166.png')
        self.assertTrue(all(name.startswith('LBL_') for name in names))
        self.assertTrue(all(name.endswith('.png') for name in names))


if __name__ == '__main__':
    unittest.main()
