from argparse import ArgumentParser
from glob import glob
from os import path
from sys import argv
import logging
from matplotlib import pyplot as plt
from skusclf import classifier, training
from skusclf.logger import BASE as logger


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the classifier.model class.
    '''

    DESC = 'Classify the images basing on a specific supervised model'
    FOLDER = './images'
    SIZE = 256
    AUGMENT = 200

    def __init__(self, args=argv[1:]):
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def classify(self):
        print(f'Calssifying {self.opts.img}')
        self._loglevel()
        mod = classifier.Model()
        img = self._img(self.opts.size)
        res = mod.predict(img, self._dataset(), test=self.opts.test)
        print(f'Classified as: {repr(res)}')

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _img(self, max_size):
        if path.isfile(self.opts.img):
            norm = training.Normalizer(max_size)
            norm.persist(self.opts.img)
            return plt.imread(self.opts.img)

    def _dataset(self):
        name = self._dataset_name()
        if name:
            return training.Loader.open(name)
        print('creating a brand new dataset...')
        loader = training.Loader(self.FOLDER, 
                                 augmenter=training.Augmenter(self.opts.augment)) 
        loader.store_dataset() 
        return loader.dataset()

    def _dataset_name(self):
        if self.opts.dataset and path.isfile(self.opts.dataset):
            return self.opts.dataset
        files = sorted(glob('./*.pkl.gz'))
        if files:
            return files[0]

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-d', '--dataset',
                            help='runs classification on the specified dataset (fetch the first *.pkl.gz available)')
        parser.add_argument('-i', '--img',
                            required=True,
                            help='the path to the PNG image to classify')
        parser.add_argument('-s', '--size',
                            default=self.SIZE,
                            help=f'the max size (default to {self.SIZE}) used to normalize the dataset (if none is available)')
        parser.add_argument('-a', '--augment',
                            default=self.AUGMENT,
                            choices=range(1, 201),
                            metavar='[1-200]',
                            help=f'augment each image by this limit (min 1, max {self.AUGMENT}, default {self.AUGMENT})')
        parser.add_argument('-t', '--test',
                            default=False,
                            help='runs classification versus the test dataset (default to False)')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    CLI().classify()
