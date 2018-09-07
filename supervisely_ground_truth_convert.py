from json import load
from os import listdir

ANNOTATION_PATH = 'dataset_motor/annotation/'
DATASET_PATH = 'dataset_motor/'
CLASSES_MAP = {
    'Pengendara Tanpa Helm': 0,
    'Pengendara Menggunakan Helm': 1,
}

CLASSES_YOLO = {
    'Wajah': 2,
    'Plat Nomo': 3,
}

CLASSES_YOLO.update(CLASSES_MAP)


def main():
    yolo_data_entry = ''
    anns = listdir(ANNOTATION_PATH)
    for ann in anns:
        with open(ANNOTATION_PATH + ann, 'r') as f:
            j = load(f)
        yolo_data_entry += (DATASET_PATH + ann).replace('.json', '.jpg')
        yolo_data_entry += ''.join([
            ' {},{},{},{},{}'.format(
                int(e['points']['exterior'][0][0]),
                int(e['points']['exterior'][0][1]),
                int(e['points']['exterior'][1][0]),
                int(e['points']['exterior'][1][1]),
                CLASSES_YOLO[e['classTitle']]
            ) for e in j['objects']
        ])
        yolo_data_entry += '\n'
        with open('map/ground-truth/' + ann.replace('.json', '.txt'), 'w') as f:
            f.write(''.join([
                '{} {} {} {} {}\n'.format(
                    e['classTitle'],
                    int(e['points']['exterior'][0][0]),
                    int(e['points']['exterior'][0][1]),
                    int(e['points']['exterior'][1][0]),
                    int(e['points']['exterior'][1][1])
                ) for e in j['objects'] if e['classTitle'] in CLASSES_MAP
            ]))
    with open('motor_indo_test.txt', 'w') as f:
        f.write(yolo_data_entry)

if __name__ == '__main__':
    main()
    print('data conversion complete!')
