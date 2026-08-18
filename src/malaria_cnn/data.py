import tensorflow as tf
import tensorflow_datasets as tfds

from .config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    SHUFFLE_BUFFER_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    DEFAULT_DATA_DIR,
)


def resize_rescale(sample):
    """
    Resize an image to 224x224
    and normalize pixels to [0, 1].
    """

    image = sample["image"]
    label = sample["label"]

    image = tf.image.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image, label


def get_datasets(
    data_dir=DEFAULT_DATA_DIR,
    batch_size=BATCH_SIZE
):
    """
    Load and prepare the malaria dataset.

    Returns:
        train_dataset
        val_dataset
        test_dataset
        dataset_info
    """

    (
        train_dataset,
        val_dataset,
        test_dataset
    ), dataset_info = tfds.load(
        "malaria",

        split=[
            TRAIN_SPLIT,
            VAL_SPLIT,
            TEST_SPLIT,
        ],

        shuffle_files=True,
        with_info=True,
        data_dir=data_dir,
    )

    # -------------------------
    # Preprocessing
    # -------------------------

    train_processed = train_dataset.map(
        resize_rescale,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_processed = val_dataset.map(
        resize_rescale,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_processed = test_dataset.map(
        resize_rescale,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    # -------------------------
    # Train pipeline
    # -------------------------

    train_ready = (
        train_processed
        .shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            reshuffle_each_iteration=True
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    # -------------------------
    # Validation pipeline
    # -------------------------

    val_ready = (
        val_processed
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    # -------------------------
    # Test pipeline
    # -------------------------

    test_ready = (
        test_processed
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    return (
        train_ready,
        val_ready,
        test_ready,
        dataset_info,
    )