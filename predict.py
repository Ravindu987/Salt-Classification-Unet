import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def binarize_image(image):
    image = image / 255
    threshold = 0.5
    binary_image = np.where(image >= threshold, 1, 0)
    return binary_image


MODEL_VERSION = 6
img_dir = "./competition_data/competition_data/train/images/images/"
mask_dir = "./competition_data/competition_data/train/masks/masks/"

model = tf.keras.models.load_model(
    f"./Saved Models/model{MODEL_VERSION}.hdf5", compile=False
)
model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

img_list = os.listdir(img_dir)
# img_list.sort()

fig, axes = plt.subplots(5, 3)
axes = axes.flatten()

tests = []
for i in range(5):
    tests.append(img_list[random.randint(0, len(img_list))])

# tests = img_list[:5]

for i, t in enumerate(tests):
    img_path = os.path.join(img_dir, t)
    mask_path = os.path.join(mask_dir, t)

    ori_img = cv.imread(img_path)
    img = cv.resize(ori_img, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    predict = model.predict(img)
    predict = binarize_image(predict)

    # predict = create_mask(predict)
    predict = np.squeeze(predict, axis=0)

    axes[i * 3].imshow(ori_img)
    axes[i * 3 + 1].imshow(cv.imread(mask_path))
    axes[i * 3 + 2].imshow(predict, cmap="gray")


plt.show()
