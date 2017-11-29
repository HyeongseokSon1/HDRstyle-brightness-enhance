from common import Coordinate, Streamer, BBox, MetaItem, MetaSample
from item_ops import URLItem, ImageItem

import abc
import collections
import random
import glob
import numpy as np
import math
import itertools

import skimage.io
import skimage.transform
import skimage.color


# Seed streamers

class FileStreamer(Streamer):
    def __init__(self,
                 key_name,
                 pattern,
                 start_idx=None,
                 end_idx=None):
        super(FileStreamer, self).__init__()
        assert isinstance(key_name, str)
        assert isinstance(pattern, str)
        self._key = key_name
        self._pattern = pattern

        assert isinstance(start_idx, (int, type(None)))
        assert isinstance(end_idx, (int, type(None)))
        self._filename_list =  sorted(glob.glob(self._pattern))
        self._filename_list = self._filename_list[start_idx:end_idx]
        self._idx = 0

    @property
    def meta_sample(self):
        meta_item = MetaItem(type_=URLItem,
                             dtype='str',
                             shape_=Coordinate(None, None))
        meta_sample = MetaSample(key=self._key,
                                 meta_item=meta_item)
        return meta_sample

    def meta_item_filter(self, prev_meta_item):
        pass

    def meta_item(self, _):
        pass

    def __iter__(self):
        while True:
            for filename in self._filename_list:
                yield {self._key: URLItem(filename)}

    def item_filter(self, prev_item):
        pass

    def item(self, prev_item):
        pass


# Control streamers

class JoinStreamer(Streamer):
    def __init__(self, prev):
        super(JoinStreamer,self).__init__()
        self._prev = prev

    @property
    def meta_sample(self):
        meta_sample = MetaSample()
        for current_prev in self._prev:
            for key, prev_meta_item in current_prev.meta_sample:
                meta_sample.append(key, prev_meta_item)
        return meta_sample

    def meta_item_filter(self, prev_meta_item):
        pass

    def meta_item(self, prev_meta_item):
        pass

    def __iter__(self):
        for prev_samples in itertools.izip(*self._prev):
            sample = dict()
            for prev_sample in prev_samples:
                for key, item in prev_sample.items():
                    sample[key] = item
            yield sample

    def item_filter(self, prev_item):
        pass

    def item(self, prev_item):
        pass


# Image processing streamers

class ImageStreamer(Streamer):
    def __init__(self,
                 prev,
                 branch_name=None,
                 **kwargs):
        super(ImageStreamer, self).__init__(prev, branch_name)
        self._reshape_module =\
            ImageReshapeStreamer(prev=self, **kwargs)
        self._color_space_module =\
            ImageColorSpaceStreamer(prev=self._reshape_module, **kwargs)

    def meta_item_filter(self, prev_meta_item):
        return prev_meta_item.type == URLItem

    def meta_item(self, prev_meta_item):
        if prev_meta_item.type == URLItem:
            mitem = MetaItem(type_=ImageItem,
                             dtype='float32',
                             shape_=Coordinate(None, None))
            mitem = self._reshape_module.meta_item(prev_meta_item=mitem)
            mitem = self._color_space_module.meta_item(prev_meta_item=mitem)
        else:
            mitem = prev_meta_item
        return mitem

    @staticmethod
    def load(path):
        img = np.divide(skimage.io.imread(path).astype('float32'), 255.)
        return ImageItem(data=img)

    def item_filter(self, prev_item):
        return type(prev_item) == URLItem

    def item(self, prev_item):
        item = self.load(prev_item.data)
        item = self._reshape_module.reshape(item)
        item = self._color_space_module.convert(item)
        return item


class ImageReshapeStreamer(Streamer):
    def __init__(self,
                 prev,
                 shape=None,
                 reshape_mode='NEAREST',
                 branch_name=None,
                 **_):
        super(ImageReshapeStreamer, self).__init__(prev, branch_name)
        assert isinstance(shape, (Coordinate, list, tuple, type(None)))
        if shape is None:
            shape = Coordinate(None, None)
        elif isinstance(shape, (tuple, list)):
            shape = Coordinate(*shape)
        self._shape = shape

        assert isinstance(reshape_mode, str)
        self._reshape_mode = \
            0 if reshape_mode == 'NEAREST' else \
            1 if reshape_mode == 'BILINEAR' else \
            2 if reshape_mode == 'BICUBIC' else \
            reshape_mode if isinstance(reshape_mode, int) else \
            reshape_mode

    def meta_item_filter(self, prev_meta_item):
        return prev_meta_item.type == ImageItem

    def meta_item(self, prev_meta_item):
        if prev_meta_item.type == ImageItem:
            meta_item = MetaItem(type_=ImageItem,
                                 dtype=prev_meta_item.dtype,
                                 shape_=Coordinate(self._shape[0],
                                                   self._shape[1],
                                                   prev_meta_item.shape.c))
        else:
            meta_item = prev_meta_item
        return meta_item

    def reshape(self, prev_item):
        assert isinstance(prev_item, ImageItem)
        if self._shape.h is None and self._shape.w is None:
            return prev_item
        data = skimage.transform.resize(prev_item.data,
                                        output_shape=self._shape[:-1],
                                        order=self._reshape_mode)
        return ImageItem(data, prev_item.color_space)

    def item_filter(self, prev_item):
        return type(prev_item) == ImageItem

    def item(self, prev_item):
        return self.reshape(prev_item)


class ImageColorSpaceStreamer(Streamer):
    def __init__(self,
                 prev,
                 color_space=None,
                 branch_name=None,
                 **kwargs):
        super(ImageColorSpaceStreamer, self).__init__(prev, branch_name)
        assert isinstance(color_space, (str, type(None)))
        self._color_space = color_space
        if 'shape' in kwargs and kwargs['shape'].c is not None:
            if color_space is None:
                num_channel = kwargs['shape'].c
                self._color_space = 'gray' if num_channel == 1 else 'rgb'

    def meta_item_filter(self, prev_meta_item):
        return prev_meta_item.type == ImageItem

    def meta_item(self, prev_meta_item):
        if prev_meta_item.type == ImageItem:
            meta_item = MetaItem(type_=ImageItem,
                                 dtype=prev_meta_item.dtype,
                                 shape_=Coordinate(prev_meta_item.shape.h,
                                                   prev_meta_item.shape.w,
                                                   self.num_channels))
        else:
            meta_item = prev_meta_item
        return meta_item

    @property
    def num_channels(self):
        if self._color_space is None:
            return None
        elif self._color_space == 'gray':
            return 1
        else:
            return 3

    def convert(self, item):
        assert isinstance(item, ImageItem)
        if self._color_space is None:
            return item
        if item.color_space == self._color_space:
            return item
        conversion_str = item.color_space + '2' + self._color_space
        conversion_func = getattr(skimage.color, conversion_str)
        return ImageItem(data=conversion_func(item.data),
                         color_space=self._color_space)

    def item_filter(self, prev_item):
        return type(prev_item) == ImageItem

    def item(self, prev_item):
        return self.convert(prev_item)


class RandomCropStreamer(Streamer):
    def __init__(self,
                 prev,
                 shape,
                 num_crop=1,
                 seed=0,
                 branch_name=None):
        super(RandomCropStreamer, self).__init__(prev, branch_name)
        self._is_bundle_sample = True
        self._shape = shape
        self._num_crop = num_crop
        self._seed = seed

        # Temporary variables when sampling
        self._bbox_list = None

    def meta_item_filter(self, prev_meta_item):
        return prev_meta_item.type == ImageItem

    def meta_item(self, prev_meta_item):
        if prev_meta_item.type == ImageItem:
            meta_item = MetaItem(type_=ImageItem,
                                 dtype=prev_meta_item.dtype,
                                 shape_=Coordinate(self._shape.h,
                                                   self._shape.w,
                                                   prev_meta_item.shape.c))
        else:
            meta_item = prev_meta_item
        return meta_item

    def _get_crop_bboxes(self, img_shape):
        patch_radius = Coordinate(int(math.floor(self._shape.h / 2)),
                                         int(math.floor(self._shape.w / 2)))
        crop_limit = BBox(top=patch_radius.h+1,
                          left=patch_radius.w+1,
                          bottom=img_shape.h-patch_radius.h-1,
                          right=img_shape.w-patch_radius.w-1)
        bounding_boxes = []
        for _ in range(self._num_crop):
            bbox_origin_y = self._generator.uniform(crop_limit.top,
                                                    crop_limit.bottom)
            bbox_origin_x = self._generator.uniform(crop_limit.left,
                                                    crop_limit.right)
            bbox_origin = Coordinate(int(bbox_origin_y),
                                            int(bbox_origin_x))
            bbox_tl = bbox_origin - patch_radius
            bbox_br = bbox_tl + self._shape
            bbox = BBox(top=bbox_tl.h,
                        left=bbox_tl.w,
                        bottom=bbox_br.h,
                        right=bbox_br.w)
            bounding_boxes.append(bbox)
        return bounding_boxes

    def item_filter(self, prev_item):
        return type(prev_item) == ImageItem

    def _on_iter_begin(self):
        self._generator = random.Random(self._seed)

    def _on_sampling_begin(self, prev_sample):
        sample_invalid = False
        img_shape = Coordinate(None, None)
        for key, prev_value in prev_sample.iteritems():
            if not self.item_filter(prev_value):
                continue
            if img_shape.h is None or img_shape.w is None:
                img_shape = Coordinate(*prev_value.shape)
            if img_shape.h != prev_value.shape.h or \
               img_shape.w != prev_value.shape.w:
                sample_invalid = True
                break
        self._bbox_list = self._get_crop_bboxes(img_shape)
        if sample_invalid:
            return None
        return prev_sample

    def item(self, prev_item):
        item = []
        for bbox in self._bbox_list:
            item_new = prev_item.data
            item_new = item_new[bbox.top:bbox.bottom, bbox.left:bbox.right]
            item_new = ImageItem(data=item_new,
                                 color_space=prev_item.color_space)
            item.append(item_new)
        return item

    def _on_sampling_end(self, sample):
        samples_new = []
        for idx in range(self._num_crop):
            sample_new = {}
            for key, value in sample.iteritems():
                sample_new[key] = value[idx] if isinstance(value, list) else\
                                  value
            samples_new.append(sample_new)
        return samples_new
