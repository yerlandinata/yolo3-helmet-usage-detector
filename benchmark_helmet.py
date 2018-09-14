'''
Author: Yudhistira Erlandinata
'''
from PIL import Image, ImageDraw, ImageEnhance
from benchmark import BenchmarkableYolo
from detector.helmet_detector import BoundingBox, combine_motorcycle_and_person

METADATA = 'motor_indo_test.txt'
RESULT_IMAGE_PATH = 'motor_result/'
MAP_RESULT_PATH = 'map/predicted/'
XMIN = 1
YMIN = 0
XMAX = 3
YMAX = 2
LIGHT_GREEN = '#41f444'
LIGHT_BLUE = '#4286f4'
CONTRAST_FACTOR = 2
SHARPEN_FACTOR = 1.5
COLOR_BALANCE_FACTOR = 4
BRIGHTNESS_FACTOR = 1.0
ENHANCEMENT = (
    (ImageEnhance.Color, COLOR_BALANCE_FACTOR),
    (ImageEnhance.Sharpness, SHARPEN_FACTOR),
    (ImageEnhance.Contrast, CONTRAST_FACTOR),
    (ImageEnhance.Brightness, BRIGHTNESS_FACTOR),
)

def main():
    yolo = BenchmarkableYolo()
    with open(METADATA, 'r') as f:
        data = f.readlines()
    for datum in data:
        d = datum.split()
        image_file = d[0]
        image = Image.open(image_file)
        duration, out_classes, boxes, scores = yolo.detect_image(image)
        result = {i: [] for i in yolo.class_names}
        for p in range(len(out_classes)):
            result[yolo.class_names[out_classes[p]]]\
                .append(BoundingBox(
                    int(boxes[p][XMIN]), int(boxes[p][XMAX]), int(boxes[p][YMIN]), int(boxes[p][YMAX]), confidence=scores[p]
                ))
        if not result['orang'] and not result['motor']:
            continue
        rider_boxes = combine_motorcycle_and_person(result['motor'], result['orang'])
        draw = ImageDraw.Draw(image)
        line_width = int(image.height * 0.005)
        map_result = ''
        for rb in [r for r in rider_boxes if r]:
            extended_rb = BoundingBox(rb.x_min, rb.x_max, rb.y_min - max(int(line_width * 7), 0), rb.y_max, confidence=rb.confidence)
            cropped = image.crop((extended_rb.x_min, extended_rb.y_min, extended_rb.x_max, extended_rb.y_max))
            map_result += find_helmet(yolo, cropped, extended_rb, draw, line_width)            
        # second chance
        for motor in [m for m in result['motor'] if not [r for r in rider_boxes if r and r.is_intersect(m)]]:
            extended_motor = BoundingBox(
                motor.x_min, motor.x_max, max(motor.y_min - (motor.y_max - motor.y_min), 0), motor.y_max,
                confidence=motor.confidence
            )
            cropped = image.crop((extended_motor.x_min, extended_motor.y_min, extended_motor.x_max, extended_motor.y_max))
            map_result += find_helmet(yolo, cropped, extended_motor, draw, line_width)
        image.save(RESULT_IMAGE_PATH + image_file.split('/')[1])
        with open(MAP_RESULT_PATH + image_file.split('/')[1].replace('.jpg', '.txt'), 'w') as f:
            f.write(map_result)

def find_helmet(yolo, image, rider, draw, line_width):
    helmet = find_object(yolo, image, 'helm')
    if helmet:
        draw.line((
            (helmet.x_min + rider.x_min, helmet.y_min + rider.y_min),
            (helmet.x_max + rider.x_min, helmet.y_min + rider.y_min),
            (helmet.x_max + rider.x_min, helmet.y_max + rider.y_min),
            (helmet.x_min + rider.x_min, helmet.y_max + rider.y_min),
            (helmet.x_min + rider.x_min, helmet.y_min + rider.y_min),
        ), fill=LIGHT_BLUE, width=line_width)
    draw.line((
        (rider.x_min, rider.y_min),
        (rider.x_max, rider.y_min),
        (rider.x_max, rider.y_max),
        (rider.x_min, rider.y_max),
        (rider.x_min, rider.y_min),
    ), width=line_width, fill=LIGHT_GREEN if helmet else 'red')
    return '{} {} {} {} {} {}\n'.format(
        'Pengendara_Menggunakan_Helm' if helmet else 'Pengendara_Tanpa_Helm',
        rider.confidence, int(rider.x_min), int(rider.y_min), int(rider.x_max), int(rider.y_max)
    )

def find_object(yolo, image, object_name):
    duration, out_classes, boxes, scores = yolo.detect_image(enhance(image))
    for p in range(len(out_classes)):
        if yolo.class_names[out_classes[p]] == object_name:
            return BoundingBox(boxes[p][XMIN], boxes[p][XMAX], boxes[p][YMIN], boxes[p][YMAX], confidence=scores[p])
    return None

def enhance(image):
    for Enhancer, factor in ENHANCEMENT:
        image = Enhancer(image).enhance(factor)
    return image

if __name__ == '__main__':
    main()
