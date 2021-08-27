import  os

import mindspore.dataset

from data_providers.base_provider import *


class ImagenetDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        train_dataset = mindspore.dataset.ImageFolderDataset(self.train_path, train_transforms)
        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
            )
            train_sampler = mindspore.dataset.SubsetRandomSampler(train_indexes)
            valid_sampler = mindspore.dataset.SubsetRandomSampler(valid_indexes)
            valid_dataset = mindspore.dataset.ImageFolderDataset(self.train_path,
                mindspore.dataset.transforms.c_transforms.Compose([
                mindspore.dataset.vision.c_transforms.Resize(self.resize_value),
                mindspore.dataset.vision.c_transforms.CenterCrop(self.image_size),
                mindspore.dataset.vision.py_transforms.ToTensor(),
                self.normalize,
            ]))

            self.train = mindspore.dataset.NumpySlicesDataset(train_dataset,sampler=train_sampler,num_parallel_workers=n_worker)
            self.train= self.train.batch(batch_size=train_batch_size)
            self.valid = mindspore.dataset.NumpySlicesDataset(valid_dataset,sampler=valid_sampler,num_parallel_workers=n_worker)
            self.valid = self.train.batch(batch_size=test_batch_size)
        else:
            self.train = mindspore.dataset.NumpySlicesDataset(train_dataset,shuffle=True,num_parallel_workers=n_worker)
            self.train = self.train.batch(batch_size=train_batch_size)
            self.valid = None

        self.test = mindspore.dataset.NumpySlicesDataset(
            mindspore.dataset.ImageFolderDataset(self.train_path, mindspore.dataset.transforms.c_transforms.Compose([
                mindspore.dataset.vision.c_transforms.Resize(self.resize_value),
                mindspore.dataset.vision.c_transforms.CenterCrop(self.image_size),
                mindspore.dataset.vision.py_transforms.ToTensor(),
                self.normalize,
            ])),shuffle=False,num_parallel_workers=n_worker
        )
        self.test= self.test.batch(batch_size=test_batch_size)

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/dataset/imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return mindspore.dataset.vision.c_transforms.Normaliz(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = mindspore.dataset.vision.c_transforms.RandomColorAdjust(brightness=(0.4,1), contrast=(0.4,1), saturation=(0.4,1), hue=(0,0.1))
        elif distort_color == 'normal':
            color_transform =  mindspore.dataset.vision.c_transforms.RandomColorAdjust(brightness=(32. / 255.,1), saturation=(0.5,1))
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = mindspore.dataset.transforms.c_transforms.Compose([
                mindspore.dataset.vision.c_transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                mindspore.dataset.vision.c_transforms.RandomHorizontalFlip(),
                mindspore.dataset.vision.py_transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = mindspore.dataset.transforms.c_transforms.Compose([
                mindspore.dataset.vision.c_transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                mindspore.dataset.vision.c_transforms.RandomHorizontalFlip(),
                color_transform,
                mindspore.dataset.vision.py_transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224
