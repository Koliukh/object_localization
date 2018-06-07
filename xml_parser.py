import xml.etree.ElementTree as ET
import os
import glob
path = 'C:/ML/avito/unlabeled'

with open('custom_labeled.csv','w') as outfile:

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
            if width is not None:
                outfile.write("{},".format(width.text))
        for elem in root.findall('object'):
            bndbox = elem.find('bndbox')
            xmin = bndbox.find('xmin')
            if xmin is not None:
                outfile.write("{},".format(xmin.text))
                ymin = bndbox.find('ymin')
                if ymin is not None:
                    outfile.write("{},".format(ymin.text))
                xmax = bndbox.find('xmax')
                if xmax is not None:
                    outfile.write("{},".format(xmax.text))
                ymax = bndbox.find('ymax')
                if ymax is not None:
                    outfile.write("{}\n".format(ymax.text))
                        