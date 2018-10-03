from glob import glob
from tempfile import mkdtemp
from sklearn.linear_model import SGDClassifier
from skusclf import classifier, training


EMPTY = mkdtemp(prefix='images')
PATH = 'skusclf/stubs'
FOLDER = f'{PATH}/images'
IMAGES = glob(f'{FOLDER}/*')
IMG = f'{PATH}/bag.png'
DATASET = training.Dataset(f'{PATH}/dataset.h5', folder=FOLDER, brand='gg', 
                           normalizer=training.Normalizer(canvas=True), 
                           augmenter=training.Augmenter(.3))
DATASET()
X, y = DATASET.load()
MODEL = classifier.Model(SGDClassifier(random_state=42, max_iter=1000, tol=1e-3), X, y, (32, 32, 4))
EVL = classifier.Evaluator.factory(MODEL)
