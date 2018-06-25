#import csv
import math
import pandas as pd
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence

ALPHA = 0.75
IMAGE_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 300
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
        lbl1 = pd.read_csv(self.csv_file)
        lbl2 = pd.read_csv('custom_labeled.csv')
        lbl=pd.concat([lbl1, lbl2[1:]], ignore_index=True)
        self.datalen=len(lbl1)
        if dataset_type=='train':
            labels=lbl1[VALIDATION_DATASET_SIZE:]
        else:
            labels=lbl1[:VALIDATION_DATASET_SIZE]
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

def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
  #  iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou    

def iou_metric_mean(y_true, y_pred):
 
    return iou_metric(y_true, y_pred)


def iou_loss(y_true, y_pred):

    # loss for the iou value
    iou_loss = -K.log(iou_metric_mean(y_true, y_pred))

    return iou_loss

def create_model(size, alpha):
    
    model_net = MobileNet(input_shape=(size, size, 3), include_top=False, weights = "imagenet", alpha=alpha)
   # for i in range(len(model_net.layers) - 13):
    #    model_net.layers[i].trainable=False
    x = _depthwise_conv_block(model_net.layers[-1].output, 1024/4, alpha, 1, block_id=14)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(4, kernel_size=(1, 1), padding="same")(x)
    x = Reshape((4, ))(x)

    return Model(inputs=model_net.input, outputs=x)


def train(model, epochs, image_size):
    train_datagen = DataSequence("with_labels.csv", image_size,"train")
    validation_datagen = DataSequence("with_labels.csv", image_size,"val")

    model.compile(loss="mse", optimizer="adam",  metrics=["accuracy",iou_metric_mean])
    checkpoint = ModelCheckpoint("model-{val_iou_metric_mean:.2f}.h5", monitor="val_iou_metric_mean", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)
    stop = EarlyStopping(monitor="val_iou_metric_mean", patience=PATIENCE, mode="auto")
 #   print(model.metrics_names)
  #  model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
  #  checkpoint = ModelCheckpoint("model-{val_acc:.2f}.h5", monitor="val_acc", verbose=1, save_best_only=True,
   #                              save_weights_only=True, mode="auto", period=1)
   # stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=train_datagen.datalen//BATCH_SIZE, epochs=epochs, validation_data=validation_datagen,
                        validation_steps=21, callbacks=[checkpoint, stop])


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    print(model.summary())

    train(model, EPOCHS, IMAGE_SIZE)


if __name__ == "__main__":
    main()