#!/usr/bin/env python

# this script defined functions used in other scripts, used in jupyter notebook

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
from sklearn.metrics import mean_squared_error
from scipy.ndimage import rotate


# function: read patient list from excel file
def get_patient_list_from_excel_file(excel_file,exclude_criteria = None):
    # exclude_criteria will be written as [[column_name,column_value],[column_name.column_value]]
    data = pd.read_excel(excel_file)
    data = data.fillna('')
    patient_list = []
    for i in range(0,data.shape[0]):
        case = data.iloc[i]
        if exclude_criteria != None:
            exclude = 0
            for e in exclude_criteria:
                if case[e[0]] == e[1]:
                    exclude += 1
            if exclude == len(exclude_criteria):
                continue
        
        patient_list.append([case['Patient_Class'],case['Patient_ID']])
    return patient_list



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


# function: normalize one vector
def normalize(x):
    x_scale = np.linalg.norm(x)
    return np.asarray([i/x_scale for i in x])

# function: get length of one vector and angle between two vectors
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    rad=math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    result = rad / math.pi * 180
    return result

# function: get a vector which is with a certain angle from one known vector
def vector_with_angle(v,angle):
    return np.dot(np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]),np.array([[v[0]],[v[1]]])).reshape(2,)


# function: get pixel dimensions
def get_voxel_size(nii_file_name):
    ii = nib.load(nii_file_name)
    h = ii.header
    return h.get_zooms()

# function: turn normalized vector into pixel unit
def turn_to_pixel(vec,size=[160,160,96]):
    t=vec.reshape(3,).tolist()
    result = [t[i]*size[i]/2 for i in range(0,3)]
    return np.array(result)



# function: count pixel in the image/segmentatio that belongs to one label
def count_pixel(seg,target_val):
    if isinstance(target_val,int) == 1 or isinstance(target_val,float) == 1:
        index_list = np.where(seg == target_val)
    
    if isinstance(target_val,list) == 1 and len(target_val) == 2:
        index_list = np.where((seg >=target_val[0]) & (seg<=target_val[1]))

    count = index_list[0].shape[0]
    pixels = []
    for i in range(0,count):
        p = []
        for j in range(0,len(index_list)):
            p.append(index_list[j][i])
        pixels.append(p)
    return count,pixels

# DICE
def DICE(seg1,seg2,target_val):
    p1_n,p1 = count_pixel(seg1,target_val)
    p2_n,p2 = count_pixel(seg2,target_val)
    p1_set = set([tuple(x) for x in p1])
    p2_set = set([tuple(x) for x in p2])
    I_set = np.array([x for x in p1_set & p2_set])
    I = I_set.shape[0] 
    DSC = (2 * I)/ (p1_n+p2_n)
    return DSC

# function: root mean square error
def RMS(true,predict,squared = False,relative = False):
    # squared = True -> MSE
    if relative == False:
        result = mean_squared_error(true,predict,squared = squared)
    else:
        result = mean_squared_error(true,predict,squared = squared) / math.sqrt(np.mean(true))
    return result

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
    


    