from argparse import ArgumentParser
from datetime import datetime
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

    DESC = 'Create a ZIP or H5 dataset on current path by normalizing and augmenting the images fetched from specified source'
    PREFIX = './dataset'
    KINDS = ('zip', 'h5')

    def __init__(self, args=argv[1:]):
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def __call__(self):
        self._loglevel()
        features = self._features()
        ds = self.dataset(self.name, features)
        print(f'creating dataset {ds.name}')
        ds()
        return ds
    
    @property
    def name(self):
        return f'{self.PREFIX}_{self.opts.brand.upper()}_{self.opts.size}'

    @property
    def dataset(self):
        if self.opts.kind == self.KINDS[0]:
            return training.DatasetZIP
        return training.DatasetH5

    def _features(self):
        canvas = False if self.opts.bkg == 'False' else self.opts.bkg
        return training.Features(self.opts.folder, limit=self.opts.max, brand=self.opts.brand, 
                                 augmenter=training.Augmenter(self.opts.cutoff),
                                 normalizer=training.Normalizer(self.opts.size, canvas=canvas)) 

    def _loglevel(self):
        loglevel = getattr(logging, self.opts.loglevel.upper())
        logger.setLevel(loglevel)

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-k', '--kind',
                            default=self.KINDS[0],
                            choices=self.KINDS,
                            help=f'the dataset kind, can be an uploadable ZIP or a H5 file to be used by a Python framework, deafult to {self.KINDS[0]}')
        parser.add_argument('-f', '--folder',
                            required=True,
                            help=f'the folder containing the image files')
        parser.add_argument('-s', '--size',
                            default=training.Normalizer.SIZE,
                            type=int,
                            help=f'the max size in pixels used to normalize the dataset, default to {training.Normalizer.SIZE}')
        parser.add_argument('-m', '--max',
                            default=training.Features.LIMIT,
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
                            default=training.Features.BRANDS[0],
                            choices=training.Features.BRANDS,
                            help='specify how to fetch labels from images, default to plain file basename')
        parser.add_argument('-l', '--loglevel',
                            default='error',
                            choices=('debug', 'info', 'warning', 'error', 'critical'),
                            help='the loglevel, default to error')
        return parser


if __name__ == '__main__':
    start = datetime.now()
    ds = CLI()()
    finish = datetime.now()
    lapse = (finish - start).total_seconds()
    print(f'dataset with {ds.features.count} features created in {lapse} seconds')
