import os
import pandas as pd
import shutil

import cv2
import numpy as np

from train_vgg import create_model, IMAGE_SIZE, ALPHA, VALIDATION_DATASET_SIZE, IMAGE_DIR
DEBUG = True
WEIGHTS_FILE = "model-0.75.h5"

dstroot='C:/ML/avito/unlabeled'

def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    model.load_weights(WEIGHTS_FILE)


    lbl = pd.read_csv('without_labels.csv')
    labels=lbl[:1999]
           
    for image_name in labels.image_name:
        image_path=os.path.join(IMAGE_DIR, image_name)
        
        shutil.copy(image_path, dstroot)



if __name__ == "__main__":
    main()