import tensorflow as tf
import tensorflow.keras as krs
import tensorflow_datasets as tfds


def load_tensorflow_datasets(datasets_name, split=None, builder_kwargs=None, as_dataset_kwargs=None, **kwargs):
    datasets, info = tfds.load(name=datasets_name, split=split, with_info=True, builder_kwargs=builder_kwargs,
                               as_dataset_kwargs=as_dataset_kwargs)

    return datasets, info
