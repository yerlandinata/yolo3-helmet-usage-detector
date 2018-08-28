source = input('file to convert: ')
class_file = input('classes (model_data/coco_classes.txt): ')
class_file = class_file if class_file != '' else 'model_data/coco_classes.txt'

lines = []

with open(source, 'r') as f:
    lines = f.readlines()

classes = []

with open(class_file, 'r') as f:
    classes = f.readlines()

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3
CLASS = 4

for line in lines:
    args = line.split()
    with open('map/ground-truth/' + args[0].split('/')[-1].replace('img/', '').replace('.jpg', '.txt').replace('.png', '.txt') , 'w') as f:
        for box in args[1:]:
            box = box.split(',')
            f.write('{} {} {} {} {}\n'.format(
                classes[int(box[CLASS])].strip(), box[XMIN], box[YMIN], box[XMAX], box[YMAX],
            ))

print('Conversion done')
