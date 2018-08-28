from glob import glob
from io import BytesIO
from os import path
from warnings import filterwarnings
from PIL import Image
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application, RequestHandler
from skusclf.training import Dataset
from skusclf.classifier import SGD


class Config:
    def __init__(self):
        filterwarnings('ignore')
        define('port', default=8888, help='run on the given port', type=int)
        define('dataset', default=self._dataset(), help='load and fit the specified dataset', type=str)
        options.parse_command_line()

    def __call__(self):
        print(f'Loading and fitting {options.dataset}')
        ds = Dataset(options.dataset)
        X, y = ds.load()
        X_orig, _ = ds.load(original=True)
        return SGD(X, y, X_orig[0].shape)

    def _dataset(self):
        files = glob('./*.h5')
        files.sort(key=lambda f: path.getmtime(f), reverse=True)
        if files:
            return files[0]


class App(Application):
    def __init__(self):
        handlers = [
            (r'/', IndexHandler),
            (r'/upload', UploadHandler)
        ]
        Application.__init__(self, handlers)


class IndexHandler(RequestHandler):
    def get(self):
        sku = self.get_query_argument('sku', default='')
        filename = self.get_query_argument('filename', default='')
        self.render('upload_form.html', sku=sku, filename=filename)


class UploadHandler(RequestHandler):
    def post(self):
        f = self.request.files['sku_file'][0]
        img = Image.open(BytesIO(f['body']))
        sku = CLF(img)
        self.redirect(f'/?sku={sku}&filename={f.filename}')


if __name__ == '__main__':
    cfg = Config()
    CLF = cfg()
    print(f'Accepting connections on {options.port}')
    http_server = HTTPServer(App())
    http_server.listen(options.port)
    IOLoop.instance().start()
