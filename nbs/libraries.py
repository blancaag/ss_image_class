import os
from os import listdir
from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
import sys
import psutil
import time
import re
import math
import pandas as pd
import numpy as np
from glob import glob
import bcolz
import random

from imp import reload

import cv2 # python-opencv
from skimage import exposure # scikit-image
from scipy import misc

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# import seaborn as sns
# sns.set_style("whitegrid", {'axes.grid' : False})
import multiprocessing

import matplotlib.animation as mpl_animation
matplotlib.rc('animation', html='html5')

from IPython.display import display 
from PIL import Image # Pillow

import scipy
from scipy.misc.pilutil import imread, imresize

import keras
from keras import backend as K
# from keras.datasets import mnist
from keras import utils
# from keras.preprocessing import image, sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.optimizers import *
from keras.losses import *
from keras.utils.np_utils import to_categorical

#import metrics
#from metrics import * 

from sklearn.model_selection import StratifiedKFold

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def silence():
    back = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return back
def speak(back): sys.stdout = back

def upscale(im_arr, new_size=(150, 150)):
    from PIL import Image
    im = Image.fromarray(np.uint8(im_arr))
    im = im.resize(new_size, Image.NEAREST)
    return np.array(im)

def rescale_normalize(im, mean=0.5, std=0.5):
    im = im / 255. # rescale values to [0, 1]
    im = (im - mean) / std
    return im


def load_data(split=0.8, path='../data/original/sushi_or_sandwich/', target_size=(256, 256)):
    
    flabs = [i for i in listdir(path) if not isfile(join(path, i))]

    d_lab = {'sandwich': 0, 'sushi': 1}

    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in flabs:
        ims_gen = (j for j in listdir(path+i) if isfile(join(path+i, j)))
        for k in ims_gen:
            im = plt.imread(join(path + i + '/', k))
            im = cv2.resize(im, target_size).astype(np.float32)
            im = exposure.rescale_intensity(im)
            ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
            labs = np.append(labs, (d_lab[i], ), axis=0)

    return preprocess_data(ims, labs, split)

def preprocess_data(ims, labs, split=0.8):
    
    # 1. concatenation and shuffling

    x = ims
    y = labs

    shuffle_ix = np.arange(x.shape[0])
    rgen = np.random.RandomState()
    rgen.seed(0)
    rgen.shuffle(shuffle_ix)

    x = x[shuffle_ix,:,:,:]
    y = y[shuffle_ix]
    
    mean_set = x.mean().astype(np.float32)
    std_set = x.std().astype(np.float32)
    
    # 2. setting training and validation sets
    
    split_ix = x.shape[0] * int(split*10) // 10

    x_train = x[:split_ix,:,:,:]
    y_train = y[:split_ix]

    x_test = x[split_ix:,:,:,:]
    y_test = y[split_ix:]

    sys.stdout.flush()
    
    return x_train, y_train, x_test, y_test, mean_set, std_set

def take_sample(path, target_size=(512, 512)):
    ims_gen = (i for i in listdir(path) if isfile(join(path, i)))
    for _, i in enumerate(ims_gen):
        if _ == 0:
            im = plt.imread(join(path, i))
#             b,g,r = cv2.split(im)           
#             im = cv2.merge([r,g,b]) 
            im = cv2.resize(im, target_size).astype(np.float32)
            im = exposure.rescale_intensity(im)
            plt.imshow(im)
            return im
        else: break
            
def plot_images(data, ncols=3, beautiful=False):
    ncols = ncols 
    nrows = data.shape[0] // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(12, nrows*5)
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        if beautiful: im = ax.imshow(data[i]*255)
        else: im = ax.imshow(data[i])
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()
    
def plot_hist(h, key='loss'):
    fig = plt.figure(figsize=(10, 3))
    plt.plot(h.history[key])  
    plt.plot(h.history['val_{}'.format(key)])  
    # plt.plot(h.history['recall'])  
    # plt.title('train vs val loss', size=12)  
    plt.ylabel(key)  
    plt.xlabel('epoch')  
    plt.legend([key, 'val_{}'.format(key)], loc='upper left') 
    plt.tight_layout
    
def evaluate(model, x_test, y_test):
    metrics = model.evaluate(x_test, y_test, verbose=0)
    for i,j in zip(model.metrics_names, range(len(metrics))):
            print("Test %s: %.2f%%" % (i, metrics[j] * 100))
    
    
# defining the augmented data generators:

def aug_data_generators():
    
    ad_gens = []
    count = 0
    for i in range(0, 10, 3):
        for j in range(0, 10, 2):
            for k in range(0, 6, 1): 
                count += 1
                exec("""ad_gen_{} = ImageDataGenerator(
                        horizontal_flip=True,
                        vertical_flip=True,
                        rotation_range={},             # 
                        width_shift_range={}/10,       # 
                        height_shift_range={}/10,      # 
                        shear_range={}/10,             # 
                        zoom_range={}/10,              # 
                        fill_mode='nearest')
                """.format(count, i, j, j, k, k))
                exec('ad_gens.append(ad_gen_{})'.format(count))
    print("Total number of generators: ", len(ad_gens))
    
    return ad_gens


# generates and saves a sample of the augmented data generators:     

def save_aug_data_sample(ad_gens, n_plot=6, set_return=False):
    _ = silence() # filters the stdout from 'flow_from_directory'

    sample_folder = 'aug_data_sample'
    if not os.path.exists(sample_folder): os.mkdir(sample_folder)
    
    sample = np.empty(((0,) + input_shape))
    for gen in ad_gens:
        for i, j in gen.flow_from_directory('../../data/sushi_or_sandwich/', 
                                             batch_size=1, seed=0,
                                             save_to_dir=sample_folder, save_prefix='ex'):

            sample = np.append(sample, i, axis=0); 
            sample_lab = j
            break
    speak(_)
    
    s_plot = list(np.random.randint(0, ad_sample.shape[0], n_plot))
    plot_images(s_plot)
    
    if set_return: 
        return sample, sample_lab

        
# pre-calculates without-top model output ('features')

def generate_aug_data_features(model, ad_gens, data, output_folder, batch_size=16, iters = 1):
    
    x_adf_train = np.empty(((0,) + model.output_shape[1:]))
    y_adf_train = np.empty((0,))

    x_adf_test = np.empty(((0,) + model.output_shape[1:]))
    y_adf_test = np.empty((0,))

    for i in ad_gens:

        adg_train = i.flow(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=False)
        adg_test = i.flow(data['x_test'], data['y_test'], batch_size=batch_size, shuffle=False)

        # print(adg_train.n//adg_train.batch_size, adg_test.n//adg_test.batch_size)

        x_ad_feat_train = model.predict_generator(adg_train, iters*adg_train.n//adg_train.batch_size, verbose=0)
        y_ad_feat_train = np.concatenate([data['y_train']]*iters)
        x_ad_feat_test = model.predict_generator(adg_test, iters*adg_test.n//adg_test.batch_size, verbose=0)
        y_ad_feat_test = np.concatenate([data['y_test']]*iters)
 
        x_adf_train = np.append(x_adf_train, x_ad_feat_train, axis=0)
        y_adf_train = np.append(y_adf_train, y_ad_feat_train, axis=0)
        x_adf_test = np.append(x_adf_test, x_ad_feat_test, axis=0)
        y_adf_test = np.append(y_adf_test, y_ad_feat_test, axis=0)

    # saving
    save_array(output_folder + '/x_ad_feat_train', x_ad_feat_train)
    save_array(output_folder + '/y_ad_feat_train', y_ad_feat_train)
    save_array(output_folder + '/x_ad_feat_test', x_ad_feat_test)
    save_array(output_folder + '/y_ad_feat_test', y_ad_feat_test)

    print(x_adf_train.shape, y_adf_train.shape, x_adf_test.shape, y_adf_test.shape)
    
# rm train_2225.jpg
# rm train_2276.jpg 