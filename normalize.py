from argparse import ArgumentParser
from sys import argv
from skusclf import training


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the training.Normalizer class.
    '''

    DESC = 'Normalize PNG images by resizing, squaring and quantizing them'
    IMG_FOLDER = './images'
    MAX_SIZE = 600

    def __init__(self, args=argv[1:]):
        self.args = args

    def normalize(self):
        opts = self._parser().parse_args(self.args)
        print(f'Normalizing PNG within {opts.folder}')
        n = training.Normalizer.bulk(opts.folder, opts.max_size)
        print(f'Normalized {n} PNGs')

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-f', '--folder',
                            default=self.IMG_FOLDER,
                            help=f'the source folder to load PNG images from, default to {self.IMG_FOLDER}')
        parser.add_argument('-m', '--max_size',
                            default=self.MAX_SIZE,
                            type=int,
                            help=f'the max size of the target image, default to {self.MAX_SIZE}')
        return parser


if __name__ == '__main__':
    CLI().normalize()
