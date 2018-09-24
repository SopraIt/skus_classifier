from glob import glob
from tempfile import mkdtemp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from skusclf import training


EMPTY = mkdtemp(prefix='images')
PATH = 'skusclf/stubs'
FOLDER = f'{PATH}/images'
IMAGES = glob(f'{FOLDER}/*')
IMG = f'{PATH}/bag.png'
DATASET = training.Dataset(f'{PATH}/dataset.h5', folder=FOLDER, brand='gg', 
                           normalizer=training.Normalizer(canvas=True), 
                           augmenter=training.Augmenter(.1), shuffle=False)
DATASET()
SGD = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
RF = RandomForestClassifier(random_state=0, n_jobs=-1)
