from glob import glob
from tempfile import mkdtemp
from skusclf import training


EMPTY = mkdtemp(prefix='images')
PATH = 'skusclf/stubs'
FOLDER = f'{PATH}/images'
IMAGES = glob(f'{FOLDER}/*')
IMG = f'{PATH}/bag.png'
DATASET = training.Dataset(f'{PATH}/dataset.h5', folder=FOLDER, brand='gg', 
                           normalizer=training.Normalizer(canvas=True), 
                           augmenter=training.Augmenter(0.2), shuffle=False)
DATASET()
