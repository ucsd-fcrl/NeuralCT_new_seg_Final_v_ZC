#!/usr/bin/env python

# this script defined functions used in other scripts

import numpy as np
import math
import glob as gb
import glob
import os
import math
import string
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from config import *
from scipy.ndimage import rotate
from skimage import measure
from sklearn.metrics import mean_squared_error
from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *
from tqdm import tqdm
import shutil



# function: save numpy file and movie in the experiment:
def save_data(config,sdf_data, suffix, save_folder, filename, experiment_name,  all_thetas = None, rotate_image = False, numpy_save = True, movie_save = True):
    if isinstance(sdf_data, np.ndarray) == 0:
        print('sdf is a network/function')
        movie = fetch_movie(config, sdf_data, all_thetas)


        if rotate_image == True:
            movie = rotate(movie, 90, reshape = False)
    else:
        print('sdf is a numpy array')
        movie = np.copy(sdf_data)
    
    movie[movie<0] = 0

    if numpy_save == True:
        filepath = os.path.join(save_folder,filename + '_' + suffix)
        
        np.save(filepath,movie)
    
    if movie_save == True:
        file_folder1 = os.path.join(save_folder, 'movie_' + suffix ,'pngs')
        
        make_folder([os.path.dirname(file_folder1),file_folder1]) 
        save_movie(movie,file_folder1)

        pngs = sort_timeframe(find_all_target_files(['*.png'],file_folder1),1)
        make_movies(os.path.join(os.path.dirname(file_folder1),experiment_name+'_movie_'+suffix+'.mp4'),pngs,144)
        


# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

        
# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(gb.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F


# function: find time frame of a file
def find_timeframe(file,num_of_end_signal,start_signal = '/',end_signal = '.'):
    k = list(file)
    num_of_dots = num_of_end_signal

    if num_of_dots == 1: #.png
        num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
    else:
        num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
    num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
    kk=k[num2+1:num1]
    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_end_signal,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    num_of_dots = num_of_end_signal

    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files


# function: make movies of several .png files
def make_movies(save_path,pngs,fps):
    mpr_array=[]
    i = cv2.imread(pngs[0])
    h,w,l = i.shape
    
    for j in pngs:
        img = cv2.imread(j)
        mpr_array.append(img)

    # save movies
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for j in range(len(mpr_array)):
        out.write(mpr_array[j])
    out.release()


# find connected components:
def connected_comp(img,threhsold = 0.1):
    
    img = img >= threhsold
    connected_image = measure.label(img)
    connected_class = np.unique(connected_image)
    
    return connected_image, connected_class

# only pick the largest connected component(not background)
def pick_largest_comp(img):
    img_new = img.copy()
    connected_image, connected_class = connected_comp(img_new)
    
    count_list = []
    for i in range(0,connected_class.shape[0]):
        I = np.zeros(connected_image.shape)
        I[connected_image == connected_class[i]] = 1
  
        count_list.append(np.sum(I))
    
    
    if len(count_list) == 1: # only background
        return img_new
        
    
    count_list_sort = np.asarray(count_list).copy().tolist()
    count_list_sort.sort(reverse=True)

    big_class_index = count_list.index(count_list_sort[1]) # not 0 (which is background)

    big_class= connected_class[big_class_index]

    img_new[connected_image != big_class] = 0
    
    return img_new



