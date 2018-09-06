import csv

ANNOTATION_FILE_NAME = "annotation.csv"
MOTORCYCLE_LABEL = "/m/04_sv"
HELMET_LABEL = "/m/0zvk5"
PERSON_LABEL = "/m/01g317"
valid_classes = [MOTORCYCLE_LABEL, HELMET_LABEL, PERSON_LABEL]

class BoundingBox:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    def is_intersect(self, other):
        # https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
        return not (self.x_max < other.x_min or self.x_min > other.x_max or self.y_max < other.y_min or self.y_min > other.y_max)

class DriverBox(BoundingBox):
    def __init__(self, x_min, x_max, y_min, y_max, person_count):
        super().__init__(x_min, x_max, y_min, y_max)
        self.person_count = person_count

"""
return dictionary with class_label: [BoundingBox(), ..]
"""
def read_annotations():
    annotations = {}
    for class_label in valid_classes:
        annotations[class_label] = []
    with open(ANNOTATION_FILE_NAME, 'r') as annotation_file:
        reader = csv.reader(annotation_file)
        for row in reader:
            x_min, x_max, y_min, y_max, class_label = row.split()
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
            assert class_label in valid_classes
            annotations[class_label].append(BoundingBox(x_min, x_max, y_min, y_max))
    return annotations

"""
input: list of bounding box, list of bounding box
return list of driver box
"""
def combine_motorcycle_and_person(motorcycle_boxes, person_boxes):
    driver_boxes = []
    for motorcycle_box in motorcycle_boxes:
        person_in_motorcycle_boxes = find_person_in_motorcycle(motorcycle_box, person_boxes)
        driver_box = create_driver_box(motorcycle_box, person_in_motorcycle_boxes)
        driver_boxes.append(driver_box)
    return driver_boxes

"""
input: bounding box, list of bounding box
return list of bounding box
"""
def find_person_in_motorcycle(motorcycle_box, person_boxes):
    person_in_motorcycle_boxes = []
    for person_box in person_boxes:
        if motorcycle_box.is_intersect(person_box):
            person_in_motorcycle_boxes.append(person_box)
    return person_in_motorcycle_boxes

"""
input: bounding box, list of bounding box
return driver box
"""
def create_driver_box(motorcycle_box, person_in_motorcycle_boxes):
    x_min = min(motorcycle_box.x_min, *list(map(lambda bounding_box: bounding_box.x_min, person_in_motorcycle_boxes)))
    x_max = max(motorcycle_box.x_max, *list(map(lambda bounding_box: bounding_box.x_max, person_in_motorcycle_boxes)))
    y_min = min(motorcycle_box.y_min, *list(map(lambda bounding_box: bounding_box.y_min, person_in_motorcycle_boxes)))
    y_max = max(motorcycle_box.y_max, *list(map(lambda bounding_box: bounding_box.y_max, person_in_motorcycle_boxes)))
    driver_box = DriverBox(x_min, x_max, y_min, y_max, len(person_in_motorcycle_boxes))
    return driver_box

"""
input: list of driver box, list of bounding box
return list of bounding box, list of bounding box
"""
def filter_driver_with_helmet(driver_boxes, helmet_boxes):
    driver_with_helmet_boxes = []
    driver_with_no_helmet_boxes = []
    for driver_box in driver_boxes:
        sum_of_helmet = count_sum_of_helmet_in_driver(driver_box, helmet_boxes)
        if sum_of_helmet == driver_box.person_count:
            driver_with_helmet_boxes.append(driver_box)
        else:
            driver_with_no_helmet_boxes.append(driver_box)
    return driver_with_helmet_boxes, driver_with_no_helmet_boxes

"""
input: driver box, list of bounding box
return int
"""
def count_sum_of_helmet_in_driver(driver_box, helmet_boxes):
    sum_of_helmet = 0
    for helmet_box in helmet_boxes:
        if driver_box.is_intersect(helmet_box):
            sum_of_helmet += 1
    return sum_of_helmet

"""
input: list of bounding box, list of bounding box
"""
def write_information(driver_with_helmet_boxes, driver_with_no_helmet_boxes):
    print('Banyak pengendara terdeteksi:', len(driver_with_helmet_boxes) + len(driver_with_no_helmet_boxes))
    print('Banyak pengendara tanpa helm:', len(driver_with_no_helmet_boxes))

def main():
    annotations = read_annotations()

    motorcycle_boxes = annotations[MOTORCYCLE_LABEL]
    person_boxes = annotations[PERSON_LABEL]
    driver_boxes = combine_motorcycle_and_person(motorcycle_boxes, person_boxes)

    helmet_boxes = annotations[HELMET_LABEL]
    driver_with_helmet_boxes, driver_with_no_helmet_boxes = filter_driver_with_helmet(driver_boxes, helmet_boxes)

    write_information(driver_with_helmet_boxes, driver_with_no_helmet_boxes)

if __name__ == '__main__':
    main()
