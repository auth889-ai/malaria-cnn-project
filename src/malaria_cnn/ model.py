import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    BatchNormalization,
)

from .config import IMAGE_SIZE


def build_model():
    """
    Build the improved malaria CNN.
    """

    model = tf.keras.Sequential(
        [

            # -------------------------
            # Input
            # -------------------------

            Input(
                shape=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3
                )
            ),


            # -------------------------
            # Conv block 1
            # -------------------------

            Conv2D(
                filters=6,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPool2D(
                pool_size=2,
                strides=2
            ),


            # -------------------------
            # Conv block 2
            # -------------------------

            Conv2D(
                filters=16,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPool2D(
                pool_size=2,
                strides=2
            ),


            # -------------------------
            # Feature maps → vector
            # -------------------------

            Flatten(),


            # -------------------------
            # Dense block
            # -------------------------

            Dense(
                units=100,
                activation="relu"
            ),

            BatchNormalization(),

            Dense(
                units=10,
                activation="relu"
            ),

            BatchNormalization(),


            # -------------------------
            # Binary output
            # -------------------------

            Dense(
                units=1,
                activation="sigmoid"
            ),

        ],
        name="malaria_cnn"
    )

    return model