import os
from os import listdir
from os.path import isfile, join
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
from keras import utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.optimizers import *
from keras.losses import *
from keras.utils.np_utils import to_categorical

# data preprocessing functions
from keras.applications.imagenet_utils import preprocess_input 
from keras.applications.inception_v3 import preprocess_input as preprocess_input_iv3
from keras.applications.xception import preprocess_input as preprocess_input_x
from keras.applications.mobilenet import preprocess_input as preprocess_input_mn

from tensorflow.python.client import device_lib

import metrics
from metrics import * 

from sklearn.model_selection import StratifiedKFold

## General utils:

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

def resize(im_arr, new_size=(150, 150)):
    from PIL import Image
    im = Image.fromarray(np.uint8(im_arr))
    im = im.resize(new_size, Image.NEAREST)
    return np.array(im)

def rescale_normalize(im, mean=0.5, std=0.5):
    im = im / 255. # rescale values to [0, 1]
    im = (im - mean) / std
    return im

# to do: optimize the loading functions
def load_data(split=0.8, path='../data/sushi_or_sandwich/', target_size=(256, 256), pil=False):
    
    cd_folder = '../output/compressed_data'
    if os.path.exists( cd_folder):
        
        x_train = load_array(join(cd_folder, 'x_train'))
        y_train = load_array(join(cd_folder, 'y_train'))
        x_test = load_array(join(cd_folder, 'x_test'))
        y_test = load_array(join(cd_folder, 'y_test'))
    
        return x_train, y_train, x_test, y_test, None, None
    
    flabs = (join(path, i) for i in listdir(path) if not isfile(join(path, i)))

    d_lab = {'sandwich': 0, 'sushi': 1}

    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in flabs:
        ims_gen = (join(i, j) for j in listdir(i) if isfile(join(i, j)))
        
        for j in ims_gen:
            if pil: im = img_to_array(load_img(j, target_size=target_size))
            else:
                im = cv2.imread(j).astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # b,g,r = cv2.split(im); im = cv2.merge([r,g,b]) 
                im = cv2.resize(im, target_size)

            ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
            labs = np.append(labs, (d_lab[i.split(path)[1]], ), axis=0)
    
    return preprocess_data(ims, labs, cd_folder, split)

def load_data_extended(split=0.8, path='../data/sushi_or_sandwich/', target_size=(256, 256), pil=False):
    
    cd_folder = '../output/compressed_data_extended'
    if os.path.exists( cd_folder):
        
        x_train = load_array(join(cd_folder, 'x_train'))
        y_train = load_array(join(cd_folder, 'y_train'))
        x_test = load_array(join(cd_folder, 'x_test'))
        y_test = load_array(join(cd_folder, 'y_test'))
        
        return x_train, y_train, x_test, y_test, None, None
    
    fsplit = [join(path, i) for i in listdir(path) if not isfile(join(path, i))]

    d_lab = {'sandwich': 0, 'sushi': 1}
    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in fsplit:
        flabs = [join(i, j) for j in listdir(i) if not isfile(join(i, j))]    

        for j in flabs:
            ims_gen = (join(j, k) for k in listdir(j) if isfile(join(j, k)))

            for k in ims_gen:
                if pil: im = img_to_array(load_img(k, target_size=target_size))
                else:
                    im = cv2.imread(k).astype(np.float32)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, target_size)

                ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
                labs = np.append(labs, (d_lab[j.split(i)[1].split('/')[1]], ), axis=0)
                
    return preprocess_data(ims, labs, cd_folder, split)

def preprocess_data(ims, labs, cd_folder, split=0.8):
    
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
    
    if not os.path.exists(cd_folder): 
        os.mkdir(cd_folder)

        save_array(join(cd_folder, 'x_train'), x_train)
        save_array(join(cd_folder, 'y_train'), y_train)
        save_array(join(cd_folder, 'x_test'), x_test)
        save_array(join(cd_folder, 'y_test'), y_test)
    
    return x_train, y_train, x_test, y_test, mean_set, std_set

def take_sample(path, target_size=(512, 512), pil=False):
    ims_gen = (join(path, i) for i in listdir(path) if isfile(join(path, i)))
    for _, i in enumerate(ims_gen):
        if _ == 0:
            if pil: im = img_to_array(load_img(i, target_size=target_size))
            else:
                im = cv2.imread(i).astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, target_size)
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(im.astype(np.uint8))
                return im
            
def plot(im):
    if im.ndim == 4: im = im[0]
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(im.astype(np.uint8))
            
def plot_ims(data, ncols=3, beautiful=False):

    if data.shape[0] % ncols != 0:
        raise ValueError('Expecting data to contain a number of images multiple of 3 but got an array with shape:', data.shape[0])
    
    ncols = ncols 
    nrows = data.shape[0] // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(12, nrows*5)
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data[i].astype(np.uint8))
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

## Modelling utils

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
    
## Data augmentation utils:

# cropping functions

def center_crop(x, center_crop_size = (360, 360), **kwargs):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size = (360, 360), sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

# custom preprocessing function

def preprocess_input_function(x, model, crop=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        model: pretrained model to be used
        crop: ['centered', 'random', None].
    # Returns
        Preprocessed tensor.
    """
    
    if model in ['inception_v3', 'xception', 'mobilenet']: 
        x /= 255.
        x -= 0.5
        x *= 2.
        
    elif model in ['densenet_121', 'densenet_161', 'densenet_169']:
        # Subtract mean pixel and multiple by scaling constant 
        # Reference: https://github.com/shicai/DenseNet-Caffe
        x[:,:,0] = (x[:,:,0] - 103.94) * 0.017
        x[:,:,1] = (x[:,:,1] - 116.78) * 0.017
        x[:,:,2] = (x[:,:,2] - 123.68) * 0.017
        
    elif model in ['resnet_50', 'vgg_16', 'vgg_19']:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    if crop=='centered': center_crop(x)
    if crop=='random': random_crop(x)
         
    return x

# defining the augmented data generators

def aug_data_generators(model, n_gen):
    
    if model in ['iv3', 'xcep', 'mobilenet']: preprocess_input_function = preprocess_input_iv3
    else: preprocess_input_function = preprocess_input
    
    fill_mode = ['constant', 'nearest', 'reflect', 'wrap']
    
    ad_gens = []
    count = 0
    for i in range(0, 180, 60): 
        for j in range(0, 8, 3):
            for k in range(0, 10, 3): 
                for l in range(0, 50, 10):
                    for m in [0, 1]:
                        for n in fill_mode:
                            count += 1
                            exec("""ad_gen_{} = ImageDataGenerator(
                                    rotation_range={},             # i
                                    width_shift_range={}/10,       # j
                                    height_shift_range={}/10,      # j
                                    shear_range={}/10,             # j
                                    zoom_range={}/10,              # k
                                    channel_shift_range={},        # l
                                    zca_whitening={},              # m
                                    zca_epsilon=1e-06, 
                                    fill_mode={},                  # n
                                    cval=0,                        
                                    horizontal_flip=1, 
                                    vertical_flip=1, 
                                    preprocessing_function=preprocess_input_function)
                            """.format(count, i, j, j, j, k, l, m, 'n'))
                            exec('ad_gens.append(ad_gen_{})'.format(count))
    print("Total number of available aug. data generators: ", len(ad_gens))
    
    shuffle_ix = list(np.random.randint(0, len(ad_gens), n_gen))
    ad_gens_g = (ad_gens[i] for i in shuffle_ix)
    print("Selected number of aug. data generators: ", len(shuffle_ix))
    
    return ad_gens_g

# generates and saves a sample of the augmented data generators     

def save_aug_data_sample(ad_gens, input_shape, data_path, n_plot=6, set_return=False):
    _ = silence() # filters the stdout from 'flow_from_directory'

    sample_folder = 'aug_data_sample'
    if not os.path.exists(sample_folder): os.mkdir(sample_folder)
    
    sample = np.empty(((0,) + input_shape))
    for gen in ad_gens:
        for i, j in gen.flow_from_directory(data_path, 
                                            batch_size=1, seed=0,
                                            save_to_dir=sample_folder, save_prefix='ex'):

            sample = np.append(sample, i, axis=0); 
            sample_lab = j
            break
    speak(_)
    
    s_plot_ix = list(np.random.randint(0, sample.shape[0], n_plot))
    plot_images(sample[s_plot_ix])
    
    if set_return: 
        return sample, sample_lab
        
# pre-calculates without-top model output ('features')

def generate_aug_data_features(base_model, ad_gens, data, output_folder, batch_size=16, iters = 1, verbose=False):
    
    x_adf_train = np.empty(((0,) + base_model.output_shape[1:]))
    y_adf_train = np.empty((0,))

    x_adf_test = np.empty(((0,) + base_model.output_shape[1:]))
    y_adf_test = np.empty((0,))

    for i in ad_gens:

        adg_train = i.flow(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=False)
        adg_test = i.flow(data['x_test'], data['y_test'], batch_size=batch_size, shuffle=False)

        #print(adg_train.n//adg_train.batch_size, adg_test.n//adg_test.batch_size)

        x_ad_feat_train = base_model.predict_generator(adg_train, iters*adg_train.n//adg_train.batch_size, verbose=0)
        y_ad_feat_train = np.concatenate([data['y_train']]*iters)
        x_ad_feat_test = base_model.predict_generator(adg_test, iters*adg_test.n//adg_test.batch_size, verbose=0)
        y_ad_feat_test = np.concatenate([data['y_test']]*iters)
         
        #print (x_ad_feat_train.shape, y_ad_feat_train.shape, x_ad_feat_test.shape, y_ad_feat_test.shape)
            
        x_adf_train = np.append(x_adf_train, x_ad_feat_train, axis=0)
        y_adf_train = np.append(y_adf_train, y_ad_feat_train, axis=0)
        x_adf_test = np.append(x_adf_test, x_ad_feat_test, axis=0)
        y_adf_test = np.append(y_adf_test, y_ad_feat_test, axis=0)
        
        if verbose: print("Generator ", i)

    # saving
    save_array(join(output_folder, 'x_ad_feat_train'), x_ad_feat_train)
    save_array(join(output_folder, 'y_ad_feat_train'), y_ad_feat_train)
    save_array(join(output_folder, 'x_ad_feat_test'), x_ad_feat_test)
    save_array(join(output_folder, 'y_ad_feat_test'), y_ad_feat_test)

    return x_adf_train, y_adf_train, x_adf_test, y_adf_test
    
# rm train_2225.jpg
# rm train_2276.jpg 