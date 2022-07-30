from os.path import join
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd


def read_dataset(dataset_in_file):
    """
    Reads a dataset into a pandas.Dataframe
    :param dataset_in_file: (str) file name of the dataset
    :return: pandas.Dataframe
    """
    return pd.read_csv(dataset_in_file)


def make_path_compatible(path):
    """
    Makes a path compatible with the operating system
    :param path: (str) path to make compatible
    :return: compatible path (str)
    """
    path = path.replace("\\",'/')
    items = path.split('/')
    # os.path.split does not always work properly
    return join(*items)


def load_images_from_folder(folder):
    images = []
    imgs_filenames = []
    for filename in os.listdir(folder):
        imgs_filename = os.path.join(folder, filename)
        img = Image.open(imgs_filename).convert('RGB')
        if img is not None:
            images.append(img)
            imgs_filenames.append(imgs_filename)
    return images, imgs_filenames
    
    
def load_for_mlp(train_file, test_file, MLP=False):
    #MLP PARAM-> True=MLPClassifier, False=MLPRegressor
    train=pd.read_csv(train_file)
    train.rename({'Unnamed: 0': 'Name'}, axis='columns', inplace=True)
    x_train=train.drop(['Name','Temp'], 1)
    y_train=np.asarray(train.drop(train.columns.difference(['Temp']), 1))
    y_train=y_train.ravel()

    test=pd.read_csv(test_file)
    test.rename({'Unnamed: 0': 'Name'}, axis='columns', inplace=True)
    x_test=test.drop(['Name','Temp'], 1)
    y_test=np.asarray(test.drop(train.columns.difference(['Temp']), 1))
    y_test=y_test.ravel()
    if MLP:
        y_train=y_train*10
        y_test=y_test*10

    return x_train, y_train, x_test, y_test
   
    
def scaling(x_train, x_test):
    sc_X = StandardScaler()
    X_trainscaled=sc_X.fit_transform(x_train)
    X_testscaled=sc_X.transform(x_test)
    return X_trainscaled, X_testscaled
    
