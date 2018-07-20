from argparse import ArgumentParser
from warnings import filterwarnings
from sys import argv
import numpy as np
from matplotlib.pyplot import imread
from skusclf import model, training


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the model.Classifier class.
    '''

    DESC = 'Classify the specified PNG basing on the passed supervised model'
    RAND = 42
    IMG_FOLDER = './images'

    def __init__(self, args=argv[1:]):
        self.args = args

    def classify(self):
        filterwarnings('ignore')
        opts = self._parser().parse_args(self.args)
        print(f'Calssifying {opts.img}')
        loader = self._loader(opts.folder)
        clf = model.Classifier.factory(loader)
        img = self._img(opts.img, loader.max_size())
        res = clf.predict(img)
        print(f'Classified as: {repr(res)}')

    def _loader(self, folder):
        return training.Loader(folder)

    def _img(self, img, max_size):
        norm = training.Normalizer(img, max_size)
        img_data = norm.hop()
        return np.asarray(img_data)

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-f', '--folder',
                            default=self.IMG_FOLDER,
                            help=f'the source folder to load PNG images from, default to {self.IMG_FOLDER}')
        parser.add_argument('-i', '--img',
                            required=True,
                            help=f'the path to the PNG image to classify')
        return parser


if __name__ == '__main__':
    CLI().classify()
