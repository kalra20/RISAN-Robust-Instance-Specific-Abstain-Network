import json
import numpy as np
import os
from glob import glob
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
#import cv2

def to_train(filename):
    checkpoints = os.listdir("checkpoints/")
    if filename in checkpoints:
        return False
    else:
        return True


def save_dict(filename, dict):

    with open(filename, 'w') as fp:
        json.dump(dict, fp)


def calc_selective_risk(model, regression, calibrated_coverage=None):
    prediction = model.predict()
    if calibrated_coverage is None:
        threshold = 0.5
    else:
        threshold = np.percentile(prediction[:, -1], 100 - 100 * calibrated_coverage)
    covered_idx = prediction[:, -1] > threshold

    coverage = np.mean(covered_idx)
    y_hat = np.argmax(prediction[:, :-1], 1)
#    if regression:
#        loss = np.sum(np.mean((prediction[covered_idx, :-1] - model.y_test[covered_idx, :-1]) ** 2, -1)) / np.sum(
#            covered_idx)
#    else:
    loss = np.sum(y_hat[covered_idx] != np.argmax(model.y_test[covered_idx, :], 1)) / np.sum(covered_idx)
    return loss


def train_profile(model_name, model_cls, coverages, model_baseline=None, regression=False, alpha=0.5):
    results = {}
    for coverage_rate in coverages:
        print("running {}_{}.h5".format(model_name, coverage_rate))
        model = model_cls(train=to_train("{}_{}.h5".format(model_name, coverage_rate)),
                          filename="{}_{}.h5".format(model_name, coverage_rate),
                          coverage=coverage_rate,
                          alpha=alpha)

        loss = calc_selective_risk(model, regression)
#
#        results[coverage] = {"lambda": coverage_rate, "selective_risk": loss}
#        if model_baseline is not None:
#            if regression:
#                results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))
#
#            else:
#
#                results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
#            results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]
#
#        save_dict("results/{}.json".format(model_name), results)


def post_calibration(model_name, model_cls, lamda, calibrated_coverage=None, model_baseline=None, regression=False):
    results = {}
    print("calibrating {}_{}.h5".format(model_name, lamda))
    model = model_cls(train=to_train("{}_{}.h5".format(model_name, lamda)),
                      filename="{}_{}.h5".format(model_name, lamda), coverage=lamda)
    loss, coverage = calc_selective_risk(model, regression, calibrated_coverage)

    results[coverage]={"lambda":lamda, "selective_risk":loss}
    if model_baseline is not None:
        if regression:
            results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

        else:
            results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))

        results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

    return results


def my_generator(func, x_train, y_train,batch_size,k):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [[res[0]], [res[1]]]

#def my_generator(func, x_train, y_train,batch_size,k):
#    while True:
#        res = func(x_train, y_train, batch_size
#                   ).next()
#        yield [res[0], res[1]]

def imread_resize(p):
    #print(cv2.imread(p).shape)
    #a = cv2.imread(p)
    #b = cv2.resize(a,(64,64))
    #b = b.flatten()
    return p

def create_cats_vs_dogs_npz(cats_vs_dogs_path='datasets'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.jpg')
        # print(glob_path)
        imgs_paths = glob(glob_path)
        #print(cv2.imread(imgs_paths[0]))
        images = [imread_resize(p) for p in imgs_paths]# resize_and_crop_image(p, 64)
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('Train')
    x_test, y_test = _load_from_dir('Test')
    #print(x_train[0],y_train[0])
    #print(x_test[0],y_test[0])
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
#create_cats_vs_dogs_npz()
