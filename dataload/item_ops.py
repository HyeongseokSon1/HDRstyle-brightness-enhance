from common import Coordinate, Item

import cStringIO
import numpy as np


class URLItem(Item):
    @staticmethod
    def _load_file(path):
        item_buffer = cStringIO.StringIO()
        with open(path, 'rb') as f:
            item_buffer.write(f.read())
        return item_buffer

    def _load_internet_file(self, _):
        raise NotImplementedError()

    def __init__(self, url, url_type='file'):
        assert isinstance(url, str)
        assert isinstance(url_type, str)
        self._url = url
        self._data = getattr(self, '_load_' + url_type)(url)

    @property
    def url(self):
        return self._url

    @property
    def data(self):
        return self._data

    def __str__(self):
        return '{url: ' + self._url + '}'


class ArrayItem(Item):
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._shape = Coordinate(*data.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def data(self):
        return self._data


class ImageItem(ArrayItem):
    def __init__(self, data, color_space=None):
        assert isinstance(color_space, (str, type(None)))
        super(ImageItem, self).__init__(data)
        if color_space is None:
            color_space = 'gray' if self._is_gray else 'rgb'
        if self._data.ndim == 2:
            self._data = np.expand_dims(self._data, axis=2)
        self._color_space = color_space

    @property
    def color_space(self):
        return self._color_space

    @property
    def _is_gray(self):
        return self._shape.c is None or self._shape.c == 1
