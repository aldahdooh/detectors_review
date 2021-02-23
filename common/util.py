from __future__ import division, absolute_import, print_function

import os
import sys
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import random
import json
import pickle
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils, to_categorical
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, concatenate, Dense, Dropout, Activation, Flatten, Input, InputLayer, Lambda, Reshape, Conv2DTranspose, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, Add, GaussianNoise
from keras.regularizers import l2
from keras.engine.topology import Layer, get_source_inputs
from keras.initializers import RandomUniform, Initializer, Constant, glorot_uniform, Ones, Zeros
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
import time
import pickle

def normalize_mean(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def normalize_linear(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test


def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)

def load_svhn_data():
    if not os.path.isfile("/media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/train_32x32.mat"):
        print('Downloading SVHN train set...')
        call(
            "curl -o /media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/train_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            shell=True
        )
    if not os.path.isfile("/media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/test_32x32.mat"):
        print('Downloading SVHN test set...')
        call(
            "curl -o /media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/test_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            shell=True
        )
    train = sio.loadmat('/media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/train_32x32.mat')
    test = sio.loadmat('/media/aaldahdo/SAMSUNG/DL_Datasets/SVHN/cropped/test_32x32.mat')
    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    # reshape (n_samples, 1) to (n_samples,) and change 1-index to 0-index
    y_train = np.reshape(train['y'], (-1,)) - 1
    y_test = np.reshape(test['y'], (-1,)) - 1

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (73257, 32, 32, 3))
    x_test = np.reshape(x_test, (26032, 32, 32, 3))

    return (x_train, y_train), (x_test, y_test)

def load_tiny_imagenet_data():
    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_xtrain.bytes', 'rb')
    data_bytes = open_file.read()
    x_train = np.frombuffer(data_bytes, dtype=np.uint8)
    x_train = x_train.reshape(100000,64,64,3)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_ytrain.bytes', 'rb')
    data_bytes = open_file.read()
    y_train = np.frombuffer(data_bytes, dtype=np.uint8)
    y_train = y_train.reshape(100000)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_xtest.bytes', 'rb')
    data_bytes = open_file.read()
    x_test = np.frombuffer(data_bytes, dtype=np.uint8)
    x_test = x_test.reshape(10000,64,64,3)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_ytest.bytes', 'rb')
    data_bytes = open_file.read()
    y_test = np.frombuffer(data_bytes, dtype=np.uint8)
    y_test = y_test.reshape(10000)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)

def toCat_onehot(y_train, y_test, numclasses):
    y_train = to_categorical(y_train, numclasses)
    y_test = to_categorical(y_test, numclasses)

    return y_train, y_test

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap, fp, an

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]

class Average_Saliency(object):
    def __init__(self, model, output_index=0):
        pass

    def get_grad(self, input_image):
        pass

    def get_average_grad(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class Single_Saliency(Average_Saliency):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.get_input_at(0)]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_grad(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

