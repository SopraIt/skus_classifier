from argparse import ArgumentParser
from os import path
from sys import argv
import logging
from warnings import filterwarnings
from skusclf import classifier, training
from skusclf.logger import BASE as logger


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the classifier.Model class.
    '''

    DESC = 'Classify the specified image versus the previously created dataset'

    def __init__(self, args=argv[1:]):
        filterwarnings('ignore')
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def classify(self):
        print(f'Calssifying {path.basename(self.opts.img)}')
        self._loglevel()
        dataset = self._dataset()
        mod = classifier.Model(dataset)
        sku = mod(self.opts.img)
        print(f'Classified as: {sku}')

    def _dataset(self):
        return training.Dataset(name=self.opts.dataset).load()

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-d', '--dataset',
                            required=True,
                            help='runs classification on the specified dataset (previously created)')
        parser.add_argument('-i', '--img',
                            required=True,
                            help='the path to the PNG image to classify')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    CLI().classify()
