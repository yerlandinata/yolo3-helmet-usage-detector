'''
Author: Yudhistira Erlandinata
'''
from PIL import Image, ImageDraw
from benchmark import BenchmarkableYolo
from detector.helmet_detector import BoundingBox, combine_motorcycle_and_person

METADATA = 'motor_indo_test.txt'
XMIN = 1
YMIN = 0
XMAX = 3
YMAX = 2

def main():
    yolo = BenchmarkableYolo()
    with open(METADATA, 'r') as f:
        data = f.readlines()
    for datum in data[:5]:
        d = datum.split()
        image_file = d[0]
        image = Image.open(image_file)
        duration, out_classes, boxes, scores = yolo.detect_image(image)
        result = {i: [] for i in yolo.class_names}
        for p in range(len(out_classes)):
            result[yolo.class_names[out_classes[p]]]\
                .append(BoundingBox(
                    int(boxes[p][XMIN]), int(boxes[p][XMAX]), int(boxes[p][YMIN]), int(boxes[p][YMAX])
                ))
        print('motor:', result['motor'])
        print('orang:', result['orang'])
        if len(result['orang']) == 0 or len(result['motor'] == 0):
            continue
        rider_boxes = combine_motorcycle_and_person(result['motor'], result['orang'])
        draw = ImageDraw.Draw(image)
        for rb in rider_boxes:
            draw.rectangle(((rb.x_min, rb.y_min), (rb.x_max, rb.y_max)), fill='blue')
        image.show()

if __name__ == '__main__':
    main()
