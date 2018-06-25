import os
import pandas as pd

import cv2
import numpy as np

from train import create_model, IMAGE_SIZE, ALPHA, VALIDATION_DATASET_SIZE, IMAGE_DIR
DEBUG = False
#WEIGHTS_FILE = "model_alfa025_size128-0.77.h5"
#WEIGHTS_FILE = "model_alfa05_size160-0.81.h5"
WEIGHTS_FILE = "model_alfa075_size160-0.82.h5"
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_ = interArea / (boxAArea + boxBArea - interArea)

    return max(iou_, 0)


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    model.load_weights(WEIGHTS_FILE)

    ious = []
 
    lbl = pd.read_csv('with_labels.csv')
   # lbl2 = pd.read_csv('custom_labeled.csv')
   # lbl=pd.concat([lbl1, lbl2[1:]], ignore_index=True)
    labels=lbl[:VALIDATION_DATASET_SIZE]
   # labels=lbl[1000:]

    for image_name, x1, y1, x2, y2 in labels.values:

        image_path=os.path.join(IMAGE_DIR, image_name)
        image_ = cv2.imread(image_path)
        height, width = image_.shape[:2]

        xmin = int(x1 * IMAGE_SIZE)
        ymin = int(y1 * IMAGE_SIZE)
        xmax = int(x2 * IMAGE_SIZE)
        ymax = int(y2 * IMAGE_SIZE)


        box1 = [xmin , ymin , xmax , ymax ]

        image = cv2.resize(image_, (IMAGE_SIZE, IMAGE_SIZE))
        region = model.predict(x=np.array([image]))[0]
        xmin, ymin, xmax, ymax = region

        box2 = [xmin, ymin, xmax, ymax]
        iou_ = iou(box1, box2)
        ious.append(iou_)
        if DEBUG:
            if iou_<0.7:
                print("IoU for {} is {}".format(image_name, iou_))
            
                cv2.rectangle(image_, (int(xmin/IMAGE_SIZE*width), int(ymin/IMAGE_SIZE*height)), (int(xmax/IMAGE_SIZE*width), int(ymax/IMAGE_SIZE*height)), (0, 0, 255), 1)
                cv2.rectangle(image_, (int(box1[0]/IMAGE_SIZE*width), int(box1[1]/IMAGE_SIZE*height)), (int(box1[2]/IMAGE_SIZE*width), int(box1[3]/IMAGE_SIZE*height)), (0, 255, 0), 1)
            
                cv2.imshow("image", image_)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    np.set_printoptions(suppress=True)
    print("\nAvg IoU: {}".format(np.mean(ious)))
    print("Highest IoU: {}".format(np.max(ious)))
    print("Lowest IoU: {}".format(np.min(ious)))


if __name__ == "__main__":
    main()