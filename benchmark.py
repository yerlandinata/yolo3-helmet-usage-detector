'''
Author: Yudhistira Erlandinata
'''
import sys
from timeit import default_timer as timer
import numpy as np
from PIL import Image
from keras import backend as K
from yolo3.utils import letterbox_image
from yolo import YOLO
from detector.helmet_detector import BoundingBox

class BenchmarkableYolo(YOLO):

    def __init__(self):
        super().__init__()

    def benchmark(self, test_metadata='test.txt', prediction_path='map/predicted/'):
        lines = []
        with open(test_metadata) as f:
            lines = f.readlines()
        test_size = len(lines)
        print('Running test on {} images'.format(test_size))
        XMIN = 1
        YMIN = 0
        XMAX = 3
        YMAX = 2
        durations = []
        res_dict = {c: [] for c in self.class_names}
        for i in range(test_size):
            print(i)
            args = lines[i].split()
            image_file = args[0]
            duration, out_classes, boxes, scores = self.detect_image(Image.open(image_file))
            durations.append(duration)
            image_file_splitted = image_file.split('/')
            image_file = image_file_splitted[-1]
            with open(prediction_path + image_file.replace('.jpg', '.txt').replace('.png', '.txt') , 'w') as f:
                for p in range(len(out_classes)):
                    res_dict[self.class_names[out_classes[p]]] \
                        .append(BoundingBox(boxes[p][XMIN], boxes[p][XMAX], boxes[p][YMIN], boxes[p][YMAX]))
                    f.write('{} {} {} {} {} {}\n'.format(
                        self.class_names[out_classes[p]],
                        scores[p],
                        boxes[p][XMIN],
                        boxes[p][YMIN],
                        boxes[p][XMAX],
                        boxes[p][YMAX]
                    ))
        print('Ran test on {} images, total duration: {}'.format(test_size, np.sum(durations)))
        print('average duration:', np.mean(durations))
        return res_dict

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        end = timer()
        return (end-start, out_classes, out_boxes, out_scores)

if __name__ == '__main__':
    test = input('test metadata (default: train.txt): ')
    if test == '': test = 'train.txt'
    byolo = BenchmarkableYolo()
    byolo.benchmark(test_metadata=test)
