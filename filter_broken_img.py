import argparse
from os import listdir, remove
from os.path import getsize

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--metadata", required=True, help="images metadata in YOLO qqwwee format")
ap.add_argument("-i", "--imagepath", required=True, help="images folder")
args = vars(ap.parse_args())

IMAGES_PATH = args['imagepath'] + ('/' if args['imagepath'][-1] != '/' else '')
METADATA_PATH = args['metadata']
KB = 1024

INVALID_IMAGES = set()

for img in listdir(IMAGES_PATH):
    if getsize(IMAGES_PATH + img) < 5 * KB:
        invalid_image = (IMAGES_PATH + img).strip()
        print(invalid_image)
        INVALID_IMAGES.add(invalid_image)
        remove(invalid_image)

with open(METADATA_PATH, 'r') as f:
    IMAGE_DATA = set(f.readlines())
    
with open(METADATA_PATH, 'w') as f:
    f.writelines([img for img in IMAGE_DATA if img.split()[0].strip() not in INVALID_IMAGES])
