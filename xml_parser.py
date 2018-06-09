import xml.etree.ElementTree as ET
import os
import numpy as np
import glob
path = 'C:/ML/avito/unlabeled'

with open('custom_labeled.csv','w') as outfile:
    outfile.write("{}\n".format('image_name,x1,y1,x2,y2'))
    for infile in glob.glob( os.path.join(path, '*.xml') ):
        tree = ET.parse(infile)
        root = tree.getroot()
        for elem in root.findall('filename'):
            if elem.text is not None:
                outfile.write("{},".format(elem.text))
        for elem in root.findall('size'):
            width = elem.find('width')
            # print(width.text)
            height = elem.find('height')
            #  print(height.text)
            depth = elem.find('depth')
            #   print(depth.text)
        np.set_printoptions(precision=3)

        for elem in root.findall('object'):
            bndbox = elem.find('bndbox')
            xmin = bndbox.find('xmin')
            if xmin is not None:
                outfile.write("{:.4f},".format(int(xmin.text)/int(width.text)))
            ymin = bndbox.find('ymin')
            if ymin is not None:
                outfile.write("{:.4f},".format(int(ymin.text)/int(height.text)))
            xmax = bndbox.find('xmax')
            if xmax is not None:
                outfile.write("{:.4f},".format(int(xmax.text)/int(width.text)))
            ymax = bndbox.find('ymax')
            if ymax is not None:
                outfile.write("{:.4f}\n".format(int(ymax.text)/int(height.text)))
                        