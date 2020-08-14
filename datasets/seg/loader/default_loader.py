#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import os
import numpy as np
from torch.utils import data

from exts.tools.parallel.data_container import DataContainer
from tools.helper.image_helper import ImageHelper
from tools.util.logger import Logger as Log
from tools.util.zipreader import ZipReader


class DefaultLoader(data.Dataset):
    def __init__(self, root_dir, dataset=None, aug_transform=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        if self.configer.get('use_zipreader'):
            ImageHelper.use_zipreader = True
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        Log.info('extra_msrab_ratio: {}'.format(self.configer.get('extra_msrab_ratio')))
        self.img_list, self.label_list = self.__list_dirs(root_dir, dataset)
        Log.info("{}/{} img count {}".format(root_dir, dataset, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        if self.configer.get('data.label_list', default=None):
            labelmap = self._encode_label(labelmap)

        if self.configer.get('data.reduce_zero_label', default=None):
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.to_np(labelmap)

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_wh=img_size,
            border_wh=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=True),
            labelmap=DataContainer(labelmap, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True),
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        labelmap[labelmap == 0] = 255
        labelmap = labelmap - 1
        labelmap[labelmap == 254] = 255
        if self.configer.get('data', 'image_tool') == 'pil':
            labelmap = ImageHelper.to_img(labelmap.astype(np.uint8))

        return labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)
        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.to_img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')

        def _inner_list_file(path):
            if ImageHelper.is_zip_path(path):
                return ZipReader.list_files(path)
            else:
                return os.listdir(path)

        def _inner_exist_file(path):
            if ImageHelper.is_zip_path(path):
                return ZipReader.exist_file(path)
            else:
                return os.path.exists(path)

        for file_name in _inner_list_file(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            label_path = os.path.join(label_dir, file_name)
            img_path = ImageHelper.imgpath(image_dir, image_name)
            if not _inner_exist_file(label_path) or img_path is None:
                Log.warn('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')
            for file_name in _inner_list_file(dataset, label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                label_path = os.path.join(label_dir, file_name)
                img_path = ImageHelper.imgpath(image_dir, image_name)
                if not _inner_exist_file(label_path) or img_path is None:
                    Log.warn('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)

        if dataset == 'train' and self.configer.get('extra_msrab_ratio')>0.0:
            image_dir = os.path.join(root_dir, 'msra10k_split/train/image')
            label_dir = os.path.join(root_dir, 'msra10k_split/train/label')
            for file_name in _inner_list_file(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                label_path = os.path.join(label_dir, file_name)
                img_path = ImageHelper.imgpath(image_dir, image_name)
                if not _inner_exist_file(label_path) or img_path is None:
                    Log.warn('Label Path: {} not exists.'.format(label_path))
                    continue

                for _i in range(int(114646/7500*self.configer.get('extra_msrab_ratio'))+1):
                    img_list.append(img_path)
                    label_list.append(label_path)

        return img_list, label_list


if __name__ == "__main__":
    # Test cityscapes loader.
    pass
