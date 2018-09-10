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

    def __init__(self, args=argv[1:]):
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def create_dataset(self):
        print(f'Creating dataset {self.name}')
        self._loglevel()
        ds = training.Dataset(self.name, folder=self.opts.folder, brand=self.opts.brand, 
                              limit=self.opts.max, augmenter=training.Augmenter(self.opts.cutoff),
                              normalizer=training.Normalizer(self.opts.size, canvas=self.canvas)) 
        ds()
        self._zip(ds)
        print(f'Dataset created with {ds.count} features and {ds.labels_count} labels')
    
    @property
    def canvas(self):
        return False if self.opts.bkg == 'False' else self.opts.bkg

    @property
    def name(self):
        return f'{self.PREFIX}_{self.opts.brand.upper()}_{self.opts.size}'

    def _zip(self, ds):
        if self.opts.zip:
            X, y = ds.load(orig=True)
            comp = training.Compressor(X, y, self.name)
            comp()

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-f', '--folder',
                            required=True,
                            help=f'the folder containing the image files')
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
                            help=f'a float value indicating the cutoff percentage of the transformations to be applied, default to {training.Augmenter.CUTOFF} (no transformations)')
        parser.add_argument('-b', '--bkg',
                            default=training.Normalizer.CANVAS,
                            help='if specified, apply a squared canvas behind each image, can be True (white for RGB, transparent for RGBA), a specific RGB string (i.e. FF00FF) or a path to an existing file to be used as background, default to false')
        parser.add_argument('--brand',
                            default=training.Dataset.BRANDS[0],
                            choices=training.Dataset.BRANDS,
                            help='specify how to fetch labels from filenames, default to MaxMara')
        parser.add_argument('-z', '--zip',
                            action='store_true',
                            help='if specified, creates a ZIP files containing the whole dataset by using the labels to organize the images')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    CLI().create_dataset()
