from argparse import ArgumentParser
from sys import argv
import logging
from skusclf import training
from skusclf.logger import BASE as logger


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the training.Dataset class.
    '''

    DESC = 'Create a HDF5 dataset on current path by normalizing and augmenting the images fetched from specified source'
    PREFIX = './dataset'
    EXT = '.h5'

    def __init__(self, args=argv[1:]):
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def create_dataset(self):
        name = self._name()
        print(f'Creating dataset {name}')
        self._loglevel()
        ds = training.Dataset(name, folder=self.opts.folder, limit=self.opts.max,
                              augmenter=training.Augmenter(self.opts.cutoff),
                              normalizer=training.Normalizer(self.opts.size, bkg=self.opts.bkg)) 
        ds()
        print(f'Dataset created with {ds.count} features and {ds.labels_count} labels')

    def _name(self):
        size = self.opts.size
        return f'{self.PREFIX}_{size}x{size}{self.EXT}'

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-f', '--folder',
                            default=training.Dataset.FOLDER,
                            help=f'the folder containing the image files, default to {training.Dataset.FOLDER}')
        parser.add_argument('-s', '--size',
                            default=training.Normalizer.SIZE,
                            type=int,
                            help=f'the max size in pixels used to normalize the dataset, default to {training.Normalizer.SIZE}')
        parser.add_argument('-m', '--max',
                            default=training.Dataset.LIMIT,
                            type=int,
                            help='limit the number of images read from disk, default to unlimited')
        parser.add_argument('-c', '--cutoff',
                            default=training.Augmenter.CUTOFF,
                            type=float,
                            help=f'a float value indicating the cutoff percentage of the transformations to be applied, default to {training.Augmenter.CUTOFF} (about 100 transformations per image)')
        parser.add_argument('-b', '--bkg',
                            help='an optional path to an image to be applied as a background before normalization')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    CLI().create_dataset()
