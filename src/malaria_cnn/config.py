IMAGE_SIZE = 224

BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 1000


TRAIN_SPLIT = "train[:80%]"
VAL_SPLIT = "train[80%:90%]"
TEST_SPLIT = "train[90%:]"


DEFAULT_DATA_DIR = (
    "/content/drive/MyDrive/"
    "malaria_cnn_project/tfds_data"
)