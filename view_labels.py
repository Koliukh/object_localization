# -*- coding: utf-8 -*-
import cv2
import os
import pandas as pd

image_dir = 'images'
labels_name = 'with_labels.csv'

labels = pd.read_csv(labels_name)
for image_name, x1, y1, x2, y2 in labels.values:
    image = cv2.imread(os.path.join(image_dir, image_name))
                

    image_h, image_w = image.shape[:2]

    # Переход из относительных координат в абсолютные
    pt1 = (int(x1 * image_w), int(y1 * image_h))
    pt2 = (int(x2 * image_w), int(y2 * image_h))
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)

    cv2.imshow('image', image)
    c = cv2.waitKey()
    if c == 27:
        break
