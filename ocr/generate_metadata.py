from os import listdir

trainval_size = int(input('training/validation examples per class: '))
val_split = int(input('validation split from training %: '))
val_size = int(val_split/100 * trainval_size)
train_size = trainval_size - val_size
test_size = int(input('test examples per class: '))

train_files = []
val_files = []
test_files = []

PREFIX = 'ocr/'

def letter_index(i):
    index = 9 + i
    if index > 9 + 15: index = index - 1
    return index

# letter train
print('letter train')
for i in range(1, 26+1):
    if i == 15: continue
    print(i)
    current_dir = 'letter/training/{}/'.format(i)
    files = listdir(current_dir)[:trainval_size]
    for img in files[:train_size]:
        train_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, letter_index(i)))
    for img in files[train_size:]:
        val_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, letter_index(i)))

# letter test
print('letter test')
for i in range(1, 26+1):
    if i == 15: continue
    print(i)
    current_dir = 'letter/testing/{}/'.format(i)
    files = listdir(current_dir)[:test_size]
    for img in files:
        test_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, letter_index(i)))

# digit train
print('digit train')
for i in range(10):
    print(i)
    current_dir = 'digit/training/{}/'.format(i)
    files = listdir(current_dir)[:trainval_size]
    for img in files[:train_size]:
        train_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, i))
    for img in files[train_size:]:
        val_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, i))

# digit test
print('digit test')
for i in range(10):
    print(i)
    current_dir = 'digit/testing/{}/'.format(i)
    files = listdir(current_dir)[:test_size]
    for img in files:
        test_files.append('{} 0,0,28,28,{}\n'.format(PREFIX + current_dir + img, i))

with open('train.txt', 'w') as f:
    f.writelines(train_files)

with open('validation.txt', 'w') as f:
    f.writelines(val_files)

with open('test.txt', 'w') as f:
    f.writelines(test_files)
