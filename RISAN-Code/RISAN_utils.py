#import json
import numpy as np
import os

#import cv2

def my_generator(func, x_train, y_train,batch_size,k):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [[res[0]], [res[1],res[1]]]

def imread_resize(p):
    a = cv2.imread(p)
    b = cv2.resize(a,(64,64))
    return p

def imread_list(p):
    return p

def create_cats_vs_dogs_npz(cats_vs_dogs_path='datasets'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.jpg')
        imgs_paths = glob(glob_path)
        images = [imread_list(p) for p in imgs_paths]
        # images = [imread_resize(p) for p in imgs_paths]#resize all images to 64x64
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('Train')
    x_test, y_test = _load_from_dir('Test')

    np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)


def load_cats_vs_dogs(cats_vs_dogs_path='datasets/'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
    x_train = npz_file['x_train']
    y_train = npz_file['y_train']
    x_test = npz_file['x_test']
    y_test = npz_file['y_test']

    return (x_train, y_train), (x_test, y_test)
#create_cats_vs_dogs_npz() # To create npz file from the images in the dataset
