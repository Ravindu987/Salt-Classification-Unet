import os
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU


from sklearn.model_selection import train_test_split
from unet import get_model


def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(
        image, channels=3
    )  # Adjust channels based on your images
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # Assuming grayscale masks
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return image, mask


image_dir = "./Car/train"
mask_dir = "./Car/train_masks_images"

image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
mask_paths = [
    os.path.join(mask_dir, os.path.basename(img_path).replace(".jpg", "_mask.jpg"))
    for img_path in image_paths
]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
dataset = dataset.map(load_image_and_mask)

train_ratio = 0.8
total_samples = len(dataset)
train_samples = int(train_ratio * total_samples)

dataset = dataset.shuffle(buffer_size=100, seed=42)

batch_size = 4

train_dataset = dataset.take(train_samples).batch(batch_size)
test_dataset = dataset.skip(train_samples).batch(batch_size)

# train_batches = train_dataset.cache().shuffle(500).batch(batch_size).repeat()
train_batches = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


train_steps = len(train_dataset)
val_steps = len(test_dataset)

model = get_model(input_shape=[1920, 1280, 3])

model.compile(
    optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[MeanIoU(num_classes=2)]
)

epochs = 5

hist = model.fit(
    train_batches,
    epochs=epochs,
    validation_data=test_dataset,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)

# print(list(dataset.as_numpy_iterator()))


# for image, mask in dataset.take(5):  # Display the first 5 images as an example
#     plt.figure()
#     plt.imshow(
#         image.numpy()
#     )  # Convert the TensorFlow tensor to a NumPy array for Matplotlib
#     plt.axis("off")  # Turn off axis labels
#     plt.show()
