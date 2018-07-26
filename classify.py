from argparse import ArgumentParser
from glob import glob
from os import path
from sys import argv
import logging
from matplotlib.pyplot import imread
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

    def __init__(self, args=argv[1:], loader=training.Loader):
        self.args = args
        self.loader = loader
        self.opts = self._parser().parse_args(self.args)

    def classify(self):
        print(f'Calssifying {self.opts.img}')
        self._loglevel()
        mod = classifier.Model(self._dataset())
        img = self._img(self.opts.size)
        res = mod.predict(img)
        print(f'Classified as: {repr(res)}')

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _img(self, max_size):
        if path.isfile(self.opts.img):
            norm = training.Normalizer(self.opts.img, max_size)
            norm.save()
            return imread(self.opts.img)

    def _dataset(self):
        name = self._dataset_name()
        if name:
            return self.loader.open(name)
        print('creating a brand new dataset...')
        l = self.loader(self.FOLDER, max_size=self.SIZE) 
        l.save() 
        return l.dataset()

    def _dataset_name(self):
        if self.opts.dataset and path.isfile(self.opts.dataset):
            return self.opts.dataset
        files = sorted(glob('./*.pkl.gz'))
        if files:
            return files[0]

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-d', '--dataset',
                            help=f'the pickled dataset to be used for classification')
        parser.add_argument('-s', '--size',
                            default=self.SIZE,
                            help=f'the max size (default to {self.SIZE}) used to normalize the dataset (if none is available)')
        parser.add_argument('-i', '--img',
                            required=True,
                            help=f'the path to the PNG image to classify')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    CLI().classify()
