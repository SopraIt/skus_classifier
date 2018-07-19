import unittest
import stubs
from skusclf import model, training


class TestModel(unittest.TestCase):
    def setUp(self):
        _set = training.Loader(f'{stubs.PATH}/squared').set()
        self.data, self.labels = [_set[k] for k in ('data', 'labels')]
    
    def test_training_test_sets(self):
        clf = model.Classifier(data=self.data, labels=self.labels)
        self.assertEqual(len(clf.X_train), 3)
        self.assertEqual(len(clf.X_test), 1)
        self.assertEqual(len(clf.y_train), 3)
        self.assertEqual(len(clf.y_test), 1)

    def test_prediction(self):
        clf = model.Classifier(data=self.data, labels=self.labels)
        flat_data = clf._flat(self.data)
        img = flat_data[0]
        res = clf.predict(img)
        self.assertEqual(res, [self.labels[0]])


if __name__ == '__main__':
    unittest.main()
