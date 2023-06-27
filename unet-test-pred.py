import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy
from unet_test import display, load_image


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=4):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet_model.predict(np.expand_dims(image, axis=0))
            display([image, mask, create_mask(pred_mask)])


if __name__ == "__main__":
    dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
    test_data = dataset["test"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    MODEL_VERSION = 2
    unet_model = load_model(f"./Saved Models/unet{MODEL_VERSION}.hdf5")

    unet_model.compile(
        optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics="accuracy"
    )
    show_predictions(test_data)
