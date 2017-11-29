import abc
import collections
import threading

import tensorflow as tf

Coordinate = collections.namedtuple('Coordinate', ['h', 'w', 'c'])
Coordinate.__new__.__defaults__ = (None,)


def _coordinate_get(v, i):
    if isinstance(v, Coordinate):
        return v[i]
    elif isinstance(v, int):
        return v if i == 0 or i == 1 else None
    else:
        raise ValueError('Value must be Coordinate or int')


def _coordinate_add(lhs, rhs):
    return Coordinate(_coordinate_get(lhs, 0) + _coordinate_get(rhs, 0),
                      _coordinate_get(lhs, 1) + _coordinate_get(rhs, 1),
                      _coordinate_get(lhs, 2))
Coordinate.__add__ = _coordinate_add


def _coordinate_sub(lhs, rhs):
    return Coordinate(_coordinate_get(lhs, 0) - _coordinate_get(rhs, 0),
                      _coordinate_get(lhs, 1) - _coordinate_get(rhs, 1),
                      _coordinate_get(lhs, 2))
Coordinate.__sub__ = _coordinate_sub


def _coordinate_mul(lhs, rhs):
    return Coordinate(_coordinate_get(lhs, 0) * _coordinate_get(rhs, 0),
                      _coordinate_get(lhs, 1) * _coordinate_get(rhs, 1),
                      _coordinate_get(lhs, 2))
Coordinate.__mul__ = _coordinate_mul


def _coordinate_div(lhs, rhs):
    return Coordinate(_coordinate_get(lhs, 0) / _coordinate_get(rhs, 0),
                      _coordinate_get(lhs, 1) / _coordinate_get(rhs, 1),
                      _coordinate_get(lhs, 2))
Coordinate.__truediv__ = _coordinate_div


class BBox(object):
    def __init__(self, top=0, left=0, **kwargs):
        self._top = top
        self._left = left
        if 'bottom' in kwargs and 'right' in kwargs:
            self._bottom = kwargs['bottom']
            self._right = kwargs['right']
        elif 'height' in kwargs and 'width' in kwargs:
            self._bottom = top + kwargs['height'] + 1
            self._right = left + kwargs['width'] + 1
        else:
            raise ValueError('Invalid parameters')

    @property
    def top(self):
        return self._top

    @property
    def left(self):
        return self._left

    @property
    def bottom(self):
        return self._bottom

    @property
    def right(self):
        return self._right

    def crop(self, img):
        return img[self._top:self._bottom, self._left:self._right, ...]


class Item(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def data(self):
        pass


class MetaItem(object):
    def __init__(self, type_, dtype, shape_):
        assert issubclass(type_, Item)
        assert isinstance(dtype, str)
        assert isinstance(shape_, Coordinate)
        self._type = type_
        self._dtype = dtype
        self._shape = shape_

    @property
    def type(self):
        return self._type

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def __str__(self):
        return '{type: ' + str(self._type) +\
               ', dtype: ' + str(self._dtype) +\
               ', shape: ' + str(self._shape) + '}'


class MetaSample(object):
    def __init__(self, key=None, meta_item=None):
        self._item_idx = []
        self._items = dict()
        if isinstance(key, type(None)) and isinstance(meta_item, type(None)):
            return
        if isinstance(key, str) and isinstance(meta_item, MetaItem):
            key = [key]
            meta_item = [meta_item]
        assert isinstance(key, collections.Iterable)
        assert isinstance(meta_item, collections.Iterable)
        for k, mitem in zip(key, meta_item):
            self.append(k, mitem)

    def append(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, MetaItem)
        self._item_idx.append(key)
        self._items[key] = value

    def __iter__(self):
        for key in self._item_idx:
            yield key, self._items[key]

    def __str__(self):
        return str(self._items)


class Streamer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 prev=None,
                 branch_name=None):
        assert isinstance(prev, (type(None), Streamer))
        assert isinstance(branch_name, (type(None), str))
        self._prev = prev
        self._branch_name = branch_name

    @property
    def meta_sample(self):
        msample = MetaSample()
        for key, prev_meta_item in self._prev.meta_sample:
            if not self.meta_item_filter(prev_meta_item):
                msample.append(key, prev_meta_item)
                continue
            meta_item = self.meta_item(prev_meta_item)
            if self._branch_name is None:
                msample.append(key, meta_item)
            else:
                msample.append(key, prev_meta_item)
                msample.append(self._branch_name, meta_item)
        return msample

    # Return true if prev_meta_item should be processed in this streamer
    @abc.abstractmethod
    def meta_item_filter(self, prev_meta_item):
        pass

    @abc.abstractmethod
    def meta_item(self, prev_meta_item):
        pass

    def __iter__(self):
        self._on_iter_begin()
        for prev_sample in self._prev:
            sample = self._on_sampling_begin(prev_sample)
            if sample is None:
                continue
            for key, prev_item in prev_sample.items():
                if not self.item_filter(prev_item):
                    sample[key] = prev_item
                    continue
                item = self.item(prev_item)
                if self._branch_name is None:
                    sample[key] = item
                else:
                    sample[key] = prev_item
                    sample[self._branch_name] = item
            sample = self._on_sampling_end(sample)
            if isinstance(sample, dict):
                yield sample
            elif isinstance(sample, collections.Iterable):
                for s in sample:
                    yield s
            else:
                continue

    def _on_iter_begin(self):
        pass

    def _on_sampling_begin(self, prev_sample):
        return dict()

    def _on_sampling_end(self, sample):
        return sample

    # Return true if prev_item should be processed in this streamer
    @abc.abstractmethod
    def item_filter(self, prev_item):
        pass

    # Return processed item if prev_item given
    @abc.abstractmethod
    def item(self, prev_item):
        pass


class ParallelDataset(object):
    def __init__(self,
                 streamer_endpoint,
                 batch_size,
                 shuffle=True):
        assert isinstance(streamer_endpoint, Streamer)
        self._streamer = streamer_endpoint
        self._keys_to_idx = dict()
        self._placeholders = list()
        queue_dtypes = list()
        queue_shape = list()
        for idx, (key, value) in enumerate(self._streamer.meta_sample):
            self._keys_to_idx[key] = idx
            dtype = getattr(tf, value.dtype)
            queue_dtypes.append(dtype)
            self._placeholders.append(tf.placeholder(dtype=dtype,
                                                     shape=value.shape,
                                                     name=key))
            queue_shape.append(value.shape)
        if shuffle:
            self._tf_queue = tf.RandomShuffleQueue(
                capacity=batch_size*6+1,
                min_after_dequeue=batch_size*3+1,
                dtypes=queue_dtypes,
                shapes=queue_shape)
        else:
            self._tf_queue = tf.FIFOQueue(
                capacity=batch_size*6+1,
                dtypes=queue_dtypes,
                shapes=queue_shape)
        self._enqueue_op = self._tf_queue.enqueue(vals=self._placeholders)
        self._dequeue_op = self._tf_queue.dequeue_many(n=batch_size)
        self._terminate_signal = False

    @property
    def ops(self):
        return self._dequeue_op

    def _feed_dict(self, keys_to_samples):
        v = dict()
        for key, idx in self._keys_to_idx.items():
            v[self._placeholders[idx]] = keys_to_samples[key]
        return v

    def start(self, sess):
        assert isinstance(sess, tf.Session)
        self._terminate_signal = False
        sess.graph.finalize()

        def _put_to_queue(sess, streamer):
            for sample_raw in streamer:
                sample = dict()
                for key, item_raw in sample_raw.items():
                    sample[key] = item_raw.data
                if self._terminate_signal:
                    return
                sess.run(self._enqueue_op, feed_dict=self._feed_dict(sample))
        thread = threading.Thread(target=_put_to_queue,
                                  args=(sess, self._streamer))
        thread.start()

    def terminate(self):
        self._terminate_signal = True
