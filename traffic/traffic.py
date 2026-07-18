import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ReLU

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = []
    labels = []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)

        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)

                img = cv2.imread(file_path)

                if img is not None:
                    rescaled_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                    images.append(rescaled_img)
                    labels.append(int(category))
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Conv2D(
                32,
                3,
                strides=(1, 1),
                padding="valid",
                dilation_rate=(1, 1),
                groups=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=None,
                padding="valid",
                data_format=None,
                name=None,
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(
                0.5,
                noise_shape=None,
                seed=None,
            ),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    main()
