#import csv
import math
import pandas as pd
import cv2
import os
import numpy as np

from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence

ALPHA = 0.25
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 100

IMAGE_DIR = 'images'
VALIDATION_DATASET_SIZE = 100

class DataSequence(Sequence):

    def __load_images(self, dataset):
        out = []

        for file_name in dataset:
            im = cv2.resize(cv2.imread(file_name), (self.image_size, self.image_size))
            out.append(im)

        return np.array(out)

    def __init__(self, csv_file, image_size, dataset_type = 'train', batch_size=BATCH_SIZE):
        self.csv_file = csv_file
        if dataset_type=='train':
            lbl = pd.read_csv(self.csv_file)
            labels=lbl[VALIDATION_DATASET_SIZE:]
        else:
            lbl = pd.read_csv(self.csv_file)
            labels=lbl[:VALIDATION_DATASET_SIZE]
        print(labels.shape)
        self.y = np.zeros((len(labels), 4))
        self.x = []
        self.image_size = image_size
        index = 0
        for image_name, x1, y1, x2, y2 in labels.values:
            image_path=os.path.join(IMAGE_DIR, image_name)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            self.y[index][0] = int(x1 * IMAGE_SIZE)
            self.y[index][1] = int(y1 * IMAGE_SIZE)
            self.y[index][2] = int(x2 * IMAGE_SIZE)
            self.y[index][3] = int(y2 * IMAGE_SIZE)
            self.x.append(image_path)
            index+=1
        self.batch_size = batch_size
 
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x).astype('float32')

        return images, batch_y


def create_model(size, alpha):
    model_net = VGG16(weights = "imagenet", input_shape=(size, size, 3), include_top=False)
    for layer in model_net.layers[:5]:
        layer.trainable = False
    x = model_net.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation="softmax")(x)


    return Model(inputs=model_net.input, outputs=x)


def train(model, epochs, image_size):
    train_datagen = DataSequence("with_labels.csv", image_size,"train")
    validation_datagen = DataSequence("with_labels.csv", image_size,"val")

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("model-{val_acc:.2f}.h5", monitor="val_acc", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=900//BATCH_SIZE, epochs=epochs, validation_data=validation_datagen,
                        validation_steps=22, callbacks=[checkpoint, stop])


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, IMAGE_SIZE)


if __name__ == "__main__":
    main()