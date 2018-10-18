from datetime import datetime
from glob import glob
from io import BytesIO
from os import path
from warnings import filterwarnings
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application, RequestHandler
from skusclf.training import DatasetH5
from skusclf.classifier import Evaluator, Model


class Config:
    RANDOM = 42
    JOBS = -1

    def __init__(self):
        filterwarnings('ignore')
        self.models = self._models()
        define('port', default=8888, help='run on the given port', type=int)
        define('dataset', default=self._dataset(), help='load and fit the specified dataset', type=str)
        define('model', default='sgd', help=f'the model used as clasifier, valid are: {", ".join(self.models.keys())}', type=str)
        options.parse_command_line()
        self.model = self._model()
        self.kind = self.model.model.__class__.__name__

    def _model(self):
        print(f'Loading and fitting {options.dataset}')
        X, y = DatasetH5.load(options.dataset)
        X_orig, _ = DatasetH5.load(options.dataset, orig=True)
        return Model(self.models.get(options.model, 'sgd'), X, y, X_orig[0].shape)

    def _dataset(self):
        files = glob('./*.h5')
        files.sort(key=lambda f: path.getmtime(f), reverse=True)
        if files:
            return files[0]

    def _models(self):
        models = {}
        models['sgd'] = SGDClassifier(random_state=self.RANDOM, max_iter=1000, tol=1e-3)
        models['rf'] = RandomForestClassifier(random_state=self.RANDOM, n_jobs=self.JOBS)
        models['kn'] = KNeighborsClassifier(n_neighbors=5, n_jobs=self.JOBS)
        return models


class App(Application):
    def __init__(self):
        handlers = [
            (r'/', IndexHandler),
            (r'/model', ModelHandler),
            (r'/upload', UploadHandler)
        ]
        Application.__init__(self, handlers)


class IndexHandler(RequestHandler):
    def get(self):
        sku = self.get_query_argument('sku', default='')
        filename = self.get_query_argument('filename', default='')
        self.render('upload_form.html', info=[], sku=sku, filename=filename, kind=CONFIG.kind)


class ModelHandler(RequestHandler):
    def get(self):
        evl = Evaluator.factory(CONFIG.model)
        self.render('upload_form.html', info=list(evl), sku='', filename='', kind=CONFIG.kind)


class UploadHandler(RequestHandler):
    def post(self):
        f = self.request.files['sku_file'][0]
        img = Image.open(BytesIO(f['body']))
        sku = CONFIG.model(img)
        self.redirect(f'/?sku={sku}&filename={f.filename}')


if __name__ == '__main__':
    start = datetime.now()
    CONFIG = Config()
    finish = datetime.now()
    lapse = (finish - start).total_seconds()
    print(f'{CONFIG.kind} training completed in {lapse} seconds')
    print(f'Accepting connections on {options.port}')
    http_server = HTTPServer(App())
    http_server.listen(options.port)
    IOLoop.instance().start()
