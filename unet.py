import tensorflow as tf
from keras.layers import (
    Conv2D,
    Dropout,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Input,
)
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np


def conv2d_block(input, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(
        input
    )
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(
        x
    )
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def get_model(n_filters=16, dropout=0.1, batchnorm=True):
    input = Input(shape=[128, 128, 3])

    c1 = conv2d_block(input, n_filters, batchnorm=batchnorm)
    p1 = MaxPooling2D()(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = MaxPooling2D()(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = MaxPooling2D()(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = MaxPooling2D()(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=16, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters * 8, 3, 2, "same")(c5)
    p6 = u6 + c4
    p6 = Dropout(dropout)(p6)
    c6 = conv2d_block(p6, n_filters * 8, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, 3, 2, "same")(c6)
    p7 = u7 + c3
    p7 = Dropout(dropout)(p7)
    c7 = conv2d_block(p7, n_filters * 4, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, 3, 2, "same")(c7)
    p8 = u8 + c2
    p8 = Dropout(dropout)(p8)
    c8 = conv2d_block(p8, n_filters * 2, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters, 3, 2, "same")(c8)
    p9 = u9 + c1
    p9 = Dropout(dropout)(p9)
    c9 = conv2d_block(p9, n_filters, batchnorm=batchnorm)

    outputs = Conv2D(3, 1, activation="softmax")(c9)

    return Model(inputs=[input], outputs=[outputs], name="UNET")


def binarize_image(image):
    image = image / 255
    threshold = 0.5
    binary_image = np.where(image >= threshold, 1, 0)
    return binary_image


if __name__ == "__main__":
    BATCH_SIZE = 16

    model = get_model()
    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    im_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    msk_gen = ImageDataGenerator(preprocessing_function=binarize_image)

    img_dir = "./competition_data/competition_data/train/images/"
    mask_dir = "./competition_data/competition_data/train/masks/"
    check_path = "./Saved Models/model6.hdf5"

    im_generator = im_gen.flow_from_directory(
        img_dir,
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=1,
        target_size=(128, 128),
        color_mode="rgb",
    )

    mask_generator = msk_gen.flow_from_directory(
        mask_dir,
        class_mode=None,
        batch_size=BATCH_SIZE,
        seed=1,
        target_size=(128, 128),
        color_mode="grayscale",
    )

    data_generator = zip(im_generator, mask_generator)
    total_train = len(im_generator)

    checkpoint = ModelCheckpoint(
        filepath=check_path,
        save_best_only=True,
        monitor="accuracy",
        verbose=1,
    )

    model.fit(
        data_generator,
        steps_per_epoch=total_train,
        epochs=2,
        callbacks=[checkpoint],
    )

    # model.summary()
