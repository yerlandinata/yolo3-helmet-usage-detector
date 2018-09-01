'''
Author: Endrawan Andika Wicaksana
Modified by: Yudhistira Erlandinata

Annotation
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside

Boxable
ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
'''

import csv
import requests
from PIL import Image
from time import time
from datetime import datetime, timedelta

# Static Variable

ANNOTATION_FILE_NAME = 'annotation.csv'
BOXABLE_FILE_NAME = 'boxable.csv'
OUTPUT_FILE_NAME = input('images metadata output file: ')
OUTPUT_IMAGE_FOLDER_NAME = input('images output folder: ')

MOTORCYCLE_LABEL = '/m/04_sv'
HELMET_LABEL = '/m/0zvk5'
PERSON_LABEL = '/m/01g317'
valid_classes = [MOTORCYCLE_LABEL, HELMET_LABEL, PERSON_LABEL]

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3
LABEL = 4
CONTAIN_GROUP = set()
CONTAIN_DEPICTION = set()
CONTAIN_CLASSES = {i: set() for i in valid_classes}

def main():
    print('started at {} system time'.format((datetime.now()).strftime('%a, %d %B %Y %H:%M:%S')))
    start = int(time())
    # find image that have certain classes
    image_box = {} # ImageID : [[XMin,XMax,YMin,YMax,class_id], ..]
    class_count = [0 for i in valid_classes]
    with open(ANNOTATION_FILE_NAME, 'r') as annotation:
        reader = csv.reader(annotation)
        header = None
        for row in reader:
            if header is None:
                header = row
            else:
                label_name = row[header.index('LabelName')]
                image_id = row[header.index('ImageID')]
                is_group = row[header.index('IsGroupOf')] == '1'
                is_depiction = row[header.index('IsDepiction')] == '1'
                if label_name in valid_classes:
                    if is_group:
                        CONTAIN_GROUP.add(image_id)
                        continue
                    if is_depiction:
                        CONTAIN_DEPICTION.add(image_id)
                        continue
                    class_count[valid_classes.index(label_name)] += 1
                    CONTAIN_CLASSES[label_name].add(image_id)
                    if image_id not in image_box: # initialize list
                        image_box[image_id] = []
                    image_box[image_id].append([\
                        float(row[header.index('XMin')]), \
                        float(row[header.index('YMin')]), \
                        float(row[header.index('XMax')]), \
                        float(row[header.index('YMax')]), \
                        valid_classes.index(label_name)]
                    )

    print('class count temporarily: [MOTOR, HELMET, PERSON]')
    print(class_count)

    # download image that have certain classes and convert it to pixel size
    image_box_pixel_size = {} # ImageID : [[XMin,YMin,XMax,YMax,class_id], ..]
    with open(BOXABLE_FILE_NAME, 'r') as boxable:
        reader = csv.reader(boxable)
        header = None
        line_count = 0
        img_count = 0
        for row in reader:
            line_count += 1
            if header == None:
                header = row
            else:
                image_id = row[header.index('ImageID')]
                if image_id in image_box:
                    image_file_name = OUTPUT_IMAGE_FOLDER_NAME + '/' + image_id + '.jpg'

                    # filter group and depiction
                    if image_id in CONTAIN_DEPICTION or image_id in CONTAIN_GROUP:
                        continue

                    # only: MOTOR | MOTOR and PERSON | MOTOR and PERSON and HELMET | PERSON AND HELMET
                    has_motor = image_id in CONTAIN_CLASSES[MOTORCYCLE_LABEL]
                    has_helmet = image_id in CONTAIN_CLASSES[HELMET_LABEL]
                    has_person = image_id in CONTAIN_CLASSES[PERSON_LABEL]

                    if (has_person and (not has_motor) and (not has_helmet)) or \
                        (has_helmet and not has_person):
                        continue

                    # download image
                    idx = ''
                    if row[header.index('Thumbnail300KURL')].find('http') >= 0:
                        idx = 'Thumbnail300KURL'
                    elif row[header.index('OriginalURL')].find('http') >= 0:
                        idx = 'OriginalURL'
                    else:
                        break
                    img_count += 1
                    image_data = requests.get(row[header.index(idx)]).content
                    with open(image_file_name, 'wb') as handler:
                        handler.write(image_data)

                    # find image size
                    im = Image.open(image_file_name)
                    width, height = im.size

                    # convert to pixel size
                    image_box_pixel_size[image_id] = []
                    for image_box_data in image_box[image_id]:
                        image_box_pixel_size[image_id].append([\
                            int(image_box_data[XMIN] * width), \
                            int(image_box_data[YMIN] * height), \
                            int(image_box_data[XMAX] * width), \
                            int(image_box_data[YMAX] * height), \
                            image_box_data[LABEL]])
                    if img_count % 50 == 0:
                        print('[{} s] downloaded images: {} images'.format(int(time()) - start, img_count))
                    
            if line_count % 20000 == 0:
                print('[{} s] read csv: {} lines'.format(int(time()) - start, line_count))


    # create new csv after convert data
    with open(OUTPUT_FILE_NAME, 'w') as output:
        for image_id in image_box_pixel_size:
            image_file_name = OUTPUT_IMAGE_FOLDER_NAME + '/' + image_id + '.jpg'
            row_data = image_file_name
            for image_box_data in image_box_pixel_size[image_id]:
                row_data += ' ' + \
                str(image_box_data[XMIN]) + ',' + \
                str(image_box_data[YMIN]) + ',' + \
                str(image_box_data[XMAX]) + ',' + \
                str(image_box_data[YMAX]) + ',' + \
                str(image_box_data[LABEL])
            output.write(row_data + '\n')

    print('Whole operation took {} seconds.'.format(int(time()) - start))

if __name__ == '__main__':
    main()
