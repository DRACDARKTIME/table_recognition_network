import os
import tensorflow as tf
import numpy as np
import shutil


def copy_files(path_source, path_destination, list):
    for file_name in list:
        source = path_source.joinpath(file_name)
        destination = path_destination.joinpath(file_name)
        if not destination.exists():
            shutil.copy(source, destination)


def split_dataset(list, train_percent=0.8, val_percent=0.1, test_percent=0.1):
    """train 80%
    val   10%
    test  10%"""
    total = len(list)
    train = list[: int(train_percent * total)]
    val = list[int(train_percent * total) : int((1 - val_percent) * total)]
    test = list[int((1 - test_percent) * total) :]
    return train, val, test


def decode_and_resize(image_path, IMAGE_SIZE=(56, 56)):
    """Decode images"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def make_dataset(data, buffer_size=1000, batch_size=32):
    ds = (
        tf.data.Dataset.from_tensor_slices(data)
        .map(decode_and_resize)
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    return ds
