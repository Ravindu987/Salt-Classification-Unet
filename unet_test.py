import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet import get_model


def resize(img, mask):
    img = tf.image.resize(img, (128, 128), "nearest")
    mask = tf.image.resize(mask, (128, 128), "nearest")

    return img, mask


def normalize(img, mask):
    img = tf.cast(img, tf.float32) / 255.0
    mask -= 1
    return img, mask


def load_image(datapoint):
    img = datapoint["image"]
    mask = datapoint["segmentation_mask"]
    img, mask = resize(img, mask)
    img, mask = normalize(img, mask)
    return img, mask


def display(display_list):
    plt.figure(figsize=(15, 15))

    titles = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    plt.show()


def display_random_train():
    sample = next(iter(train_batches))
    random_index = np.random.choice(sample[0].shape[0])
    sample_img, sample_mask = sample[0][random_index], sample[1][random_index]

    display([sample_img, sample_mask])


if __name__ == "__main__":
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    EPOCHS = 20

    dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

    train_data = dataset["train"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = dataset["test"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batches = test_data.take(3000).batch(BATCH_SIZE)
    test_batches = test_data.skip(3000).take(669).batch(BATCH_SIZE)

    # display_random_train()

    unet_model = get_model()

    unet_model.compile(
        optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics="accuracy"
    )

    TRAIN_LENGTH = info.splits["train"].num_examples
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    # STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 3000 // BATCH_SIZE

    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
    save_check = ModelCheckpoint(
        filepath="./Saved Models/unet2.hdf5",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1,
    )

    model_history = unet_model.fit(
        train_batches,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=val_batches,
        callbacks=[early_stop, save_check],
    )
