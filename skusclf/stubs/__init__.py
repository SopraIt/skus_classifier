from glob import glob
from tempfile import mkdtemp


EMPTY = mkdtemp(prefix='images')
PATH = 'skusclf/stubs'
FOLDER = f'{PATH}/images'
IMAGES = glob(f'{FOLDER}/*')
IMG = f'{PATH}/bag.png'
DATASET = f'{PATH}/dataset.h5'
