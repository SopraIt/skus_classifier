from argparse import ArgumentParser
from glob import glob
from os import path, remove
from sys import argv
from matplotlib.pyplot import imread
from skusclf import classifier, training


class CLI:
    '''
    Synopsis
    --------
    A plain CLI wrapper over the classifier.model class.
    '''

    DESC = 'Classify the images basing on a specific supervised model'
    FOLDER = './images'
    MAX_SIZE = 250

    def __init__(self, args=argv[1:]):
        self.args = args
        self.opts = self._parser().parse_args(self.args)

    def classify(self):
        print(f'Calssifying {self.opts.img}')
        mod = classifier.Model(self._dataset())
        img = self._img(mod.dataset.get('size'))
        res = mod.predict(img)
        print(f'Classified as: {repr(res)}')

    def _img(self, max_size):
        if path.isfile(self.opts.img):
            norm = training.Normalizer(self.opts.img, max_size)
            norm.save()
            return imread(self.opts.img)

    def _dataset(self):
        if self.opts.dataset and path.isfile(self.opts.dataset):
            return self.opts.dataset
        files = glob('./*.pkl')
        if files:
            return files[0]
        print('creating a brand new dataset...')
        l = training.Loader(self.FOLDER, self.MAX_SIZE) 
        name = l.save() 
        print(f'classifying versus the {name} dataset')
        return name

    def _parser(self):
        parser = ArgumentParser(description=self.DESC)
        parser.add_argument('-d', '--dataset',
                            help=f'the pickled dataset to be used for classification')
        parser.add_argument('-s', '--size',
                            default=self.MAX_SIZE,
                            help=f'the max size used to normalize the dataset (if none is available)')
        parser.add_argument('-i', '--img',
                            required=True,
                            help=f'the path to the PNG image to classify')
        return parser


if __name__ == '__main__':
    CLI().classify()
