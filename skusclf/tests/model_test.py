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

    def test_true_prediction(self):
        clf = model.Classifier(data=self.data, labels=self.labels)
        lb = self.labels[0]
        flat_data = clf._flat(self.data)
        img = flat_data[0]
        p = clf.predict(lb, img)
        self.assertTrue(p[0])

    def test_false_prediction(self):
        clf = model.Classifier(data=self.data, labels=self.labels)
        lb = self.labels[-1]
        flat_data = clf._flat(self.data)
        img = flat_data[0]
        p = clf.predict(lb, img)
        self.assertFalse(p[0])


if __name__ == '__main__':
    unittest.main()
