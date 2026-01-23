import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    import os
    
    # Check if datasets directory exists
    datasets_dir = 'datasets'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    
    train_path = os.path.join(datasets_dir, 'train_happy.h5')
    test_path = os.path.join(datasets_dir, 'test_happy.h5')
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Dataset file not found: {train_path}\n"
            f"Please download the dataset files from the Coursera course website:\n"
            f"  - train_happy.h5\n"
            f"  - test_happy.h5\n"
            f"And place them in the '{datasets_dir}' directory."
        )
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset file not found: {test_path}\n"
            f"Please download the dataset files from the Coursera course website:\n"
            f"  - train_happy.h5\n"
            f"  - test_happy.h5\n"
            f"And place them in the '{datasets_dir}' directory."
        )
    
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

