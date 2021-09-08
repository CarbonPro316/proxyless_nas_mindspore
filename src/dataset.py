import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, device_num=1, rank=0, batch_size=100, drop_remainder=True, shuffle=True):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """

    if device_num == 1 or do_train == False:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank)
    # define map operations
    if do_train:
        trans = [
            # C.RandomCropDecodeResize(size=(224,224),scale=(0.08, 1.0), ratio=(3. / 4.,4. / 3.) ) #mindspore default
            #                                        scale=(0.08, 1.0), ratio=(0.75, 1.33 )       #pytorch

            # C.RandomCropDecodeResize(224),
            C.RandomCropDecodeResize(size=224, scale=(0.09, 1.0)),

            # add it for CutOut
            C.CutOut(length=56, num_patches=1),
            #

            C.RandomHorizontalFlip(prob=0.5),
            # C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.1),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),

            C.Rescale(1.0 / 255.0, 0),
            C.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            C.HWC2CHW(),

        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(224),
            C.Rescale(1.0 / 255.0, 0),
            C.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            C.HWC2CHW(),
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(input_columns="image", operations=trans, num_parallel_workers=8)
    data_set = data_set.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=drop_remainder)
    return data_set