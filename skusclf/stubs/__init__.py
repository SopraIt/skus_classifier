from glob import glob
from tempfile import mkdtemp
from sklearn.linear_model import SGDClassifier
from skusclf import classifier, training


EMPTY = mkdtemp(prefix='images')
PATH = 'skusclf/stubs'
FOLDER = f'{PATH}/images'
IMAGES = glob(f'{FOLDER}/*')
IMG = f'{PATH}/bag.png'
FEATURES = training.Features(FOLDER, brand='gg', 
                             normalizer=training.Normalizer(canvas=True), 
                             augmenter=training.Augmenter(.3))
DATASET_H5 = training.DatasetH5(f'{PATH}/dataset', FEATURES)
DATASET_H5()
X, y = training.DatasetH5.load(f'{PATH}/dataset.h5')
X_orig, y_orig = training.DatasetH5.load(f'{PATH}/dataset.h5', orig=True)
MODEL = classifier.Model(SGDClassifier(random_state=42, max_iter=1000, tol=1e-3), X, y, (32, 32, 4))
EVL = classifier.Evaluator.factory(MODEL)
