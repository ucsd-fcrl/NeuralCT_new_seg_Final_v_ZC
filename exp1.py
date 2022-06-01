#!/usr/bin/env python

# Kunal's experiment: single dot
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import warnings
from torch import optim         
from torch.optim.lr_scheduler import StepLR
from sklearn.mixture import GaussianMixture
from copy import deepcopy
import imageio

from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *
from tqdm import tqdm
import argparse
from scipy.ndimage import rotate
import function_list as ff

num = 1
torch.manual_seed(num)
import random
random.seed(num)
import numpy as np
np.random.seed(num)

from scipy.ndimage import binary_erosion

# define the folder path
main_folder = "/Data/McVeighLabSuper/wip/NeuralCT/Test_results"

GMM_threshold = 0.15
min_loss1 = 0.5  #0.5
max_iteration1 = 4000 # 5000
min_loss2 = 0.1   # 0.2 or 0.1
sino_loss = 0.08  # 0.08
max_iteration2 = 5000   #10000
coeffsino = 1
coeffek = 0.1  #0.1
coefftvs = 0.5 #0.5
coefftvt= 0.5  #0.5
MATRIX_SAVE = True
MOVIE_SAVE = True
use_save_data_gt = True
use_save_data_fbp = True
use_spatial= False


parser = argparse.ArgumentParser()
parser.add_argument("--offset", default=60.0, type=float, help = "Gantry offset")
parser.add_argument("--rate", default=0.80, type=float, help = "Heartrate")
parser.add_argument("--exp", type=str, help = "Experiment_Name")

opt = parser.parse_args()
print('Doing for offset : {} and rate : {}'.format(opt.offset, opt.rate))
experiment_name = opt.exp
experiment = opt.exp+ '_results/'
print(experiment)


save_folder = os.path.join(main_folder,experiment)
ff.make_folder([save_folder])      
filename = 'movie_{}_{}'.format(int(opt.offset), int(opt.rate*100))


##### Define Config
config = Config(np.array([[0.5]]), TYPE=0, NUM_HEART_BEATS=opt.rate, image_resolution = 128) #Config(intensity, type, num_heart_beats), intensity here is a Nx1 array with each one in N corresponding to a intensity class
all_thetas = np.linspace(-config.THETA_MAX/2, config.THETA_MAX/2, config.TOTAL_CLICKS)
print('config parameters: ', config.NUM_HEART_BEATS, config.INTENSITIES,config.INTENSITIES.shape,config.NUM_SDFS)
print('config parameters: ', config.TOTAL_CLICKS,config.THETA_MAX, config.GANTRY2HEART_SCALE)

##### Define Body (organs)
body = Body(config, [Organ(config,[0.5,0.5],RADIUS,RADIUS,'const','circle')])  # only one organ (one intensity level) in the body, RADIUS = 0.15, 'const' defines a size change function self.const and 'circle' defines a location change function self.circle in class Motion


#### Get ground truth
print('....GET GROUND TRUTH....')
sdfgt = SDFGt(config, body)
if use_save_data_gt == False:
    ff.save_data(config,sdfgt, 'gt', save_folder, filename, experiment_name, all_thetas = all_thetas, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)
#sinogram
print('get_sinogram')
if use_save_data_gt == False:
    gt_sinogram_numpy = get_sinogram(config, SDFGt(config, body),Intensities(config, learnable = False), all_thetas,offset = opt.offset)
    np.save(os.path.join(save_folder,filename+'_sinogram'),gt_sinogram_numpy)
else:
    gt_sinogram_numpy = np.load(os.path.join(save_folder,filename+'_sinogram.npy'),allow_pickle = True)

gt_sinogram = torch.from_numpy(gt_sinogram_numpy).cuda()


#### Get sinogram using rendering and get FBP reconstruction (the initializer of the model)
print('....GET FBP....')
if use_save_data_gt == False:
    sinogram, reconstruction_fbp = fetch_fbp_movie_exp1(config, body, gantry_offset=opt.offset)
    ff.save_data(config,reconstruction_fbp, 'fbp', save_folder, filename, experiment_name, all_thetas = all_thetas, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)
    np.save(os.path.join(save_folder,'sinogram_generates_FBP'),sinogram)
else:
    reconstruction_fbp = np.load(os.path.join(save_folder,filename+'_fbp.npy'),allow_pickle = True)
    sinogram = np.load(os.path.join(save_folder,'sinogram_generates_FBP.npy'),allow_pickle = True)


#### convert FBP reconstruction into an image with several class (using Gaussian mixture model) represented by occupancy then convert occupancy to SDF
print('....GET FBP-DERIVED SDF MAP....')
if use_save_data_fbp == False:
 
    pretraining_sdfs, init = get_pretraining_sdfs(config, sdf=reconstruction_fbp, GMM_threshold = GMM_threshold, save_GMM_label_path = os.path.join(save_folder,filename+'_GMM_labels')) 
    print('init - median intensity of the object in the image is : ',init, ' pretraining_sdfs has size: ',pretraining_sdfs.shape) #[x_dim, y_dim, num_gantry_clicks, num_SDF]
    np.save(os.path.join(save_folder,'FBP_derived_sdf'),pretraining_sdfs)
    np.save(os.path.join(save_folder,'FBP_derived_init'),init)
    
else:
    pretraining_sdfs = np.load(os.path.join(save_folder,'FBP_derived_sdf.npy'),allow_pickle = True)
    init = np.load(os.path.join(save_folder,'FBP_derived_init.npy'),allow_pickle = True)


#### Initialize the SIREN by pretraining_sdfs (the result of FBP, the SIREN should correctly implicitly represent this result), 
# here sdf is the SIREN network pipeline which outputs the SDF map.
print('....INITIALIZE SIREN....')
pretrain_training_log_file = os.path.join(save_folder,experiment_name+'_training_log_pretrain.xlsx')
sdf, init = pretrain_sdf(config, pretraining_sdfs, init,  min_loss1, max_iteration1, pretrain_training_log_file, True, lr = 1e-4)  
print('init - median intensity of the object in the image is : ',init)
# get the sinogram after pre-train SDF
intensities1 = Intensities(config, learnable = False, init = init[:,0])
renderer1 = Renderer(config, sdf,intensities1, offset = opt.offset-180)
pretrain_sinogram = np.zeros([config.IMAGE_RESOLUTION , config.TOTAL_CLICKS])
for tt in range(0,config.TOTAL_CLICKS):
    ttt = tt*(config.THETA_MAX/config.TOTAL_CLICKS)
    pred_sinogram = renderer1(np.array([ttt])).detach().cpu().numpy()
    pretrain_sinogram[:,tt] = pred_sinogram.reshape(config.IMAGE_RESOLUTION)
np.save(os.path.join(save_folder,'sinogram_after_pretrain'),pretrain_sinogram)
# save
ff.save_data(config,sdf, 'initial1', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Train SIREN so the output 
print('....TRAIN SIREN....')
training_log_file = os.path.join(save_folder,experiment_name + '_training_log_train.xlsx')
sdf,intensities = train_modified(config, sdf, gt_sinogram, lr=1e-4, init=init[:,0], gantry_offset = opt.offset, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2, training_log_file = training_log_file, 
                                sdf_log_file = os.path.join(save_folder,experiment_name + '_training_sdf_train.xlsx'),stage = 'train', output_data = False)

renderer = Renderer(config, sdf,intensities, offset = opt.offset-180)
train_sinogram = np.zeros([config.IMAGE_RESOLUTION , config.TOTAL_CLICKS])
for tt in range(0,config.TOTAL_CLICKS):
    ttt = tt*(config.THETA_MAX/config.TOTAL_CLICKS)
    pred_sinogram = renderer(np.array([ttt])).detach().cpu().numpy()
    train_sinogram[:,tt] = pred_sinogram.reshape(config.IMAGE_RESOLUTION)
np.save(os.path.join(save_folder,'sinogram_after_train'),train_sinogram)
# save_
ff.save_data(config,sdf, 'train', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Refinement
print('....REFINEMENT....')
pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
print('pretraining_sdf: ',pretraining_sdfs.shape, np.max(pretraining_sdfs[:,:,50,0]),np.min(pretraining_sdfs[:,:,50,0]))
pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)
print('pretraining_sdf after rotation: ',pretraining_sdfs.shape, np.max(pretraining_sdfs[:,:,50,0]),np.min(pretraining_sdfs[:,:,50,0]))

pretrain_refine_log_file = os.path.join(save_folder, experiment_name + '_refine_log_pretrain.xlsx')
refine_log_file = os.path.join(save_folder, experiment_name+ '_refine_log_train.xlsx')
sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, min_loss1, max_iteration1, pretrain_refine_log_file, True, lr = 5e-5)
ff.save_data(config,sdf, 'initial2', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


sdf,intensities = train_modified(config, sdf, gt_sinogram, lr=1e-5, init=init[:,0], gantry_offset = opt.offset+90, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2,training_log_file = refine_log_file, 
                                sdf_log_file = os.path.join(save_folder, experiment_name+ '_refine_sdf_train.xlsx'), stage = 'refine', output_data = False)

ff.save_data(config,sdf, 'refine', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

# pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
# print('pretraining_sdf: ',pretraining_sdfs.shape, np.max(pretraining_sdfs[:,:,50,0]),np.min(pretraining_sdfs[:,:,50,0]))
# pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)
# print('pretraining_sdf after rotation: ',pretraining_sdfs.shape, np.max(pretraining_sdfs[:,:,50,:]),np.min(pretraining_sdfs[:,:,50,:]))

# pretrain_refine_log_file = os.path.join(save_folder,experiment_name+'_refine_log_pretrain.xlsx')
# refine_log_file = os.path.join(save_folder,experiment_name+'_refine_log_train.xlsx')
# sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, min_loss1, max_iteration1, pretrain_refine_log_file, True, lr = 5e-5)
# ff.save_data(config,sdf, 'initial2', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

# #sdf,intensities = train(config, sdf, gt_sinogram, lr=1e-5, init=init[:,0], gantry_offset = opt.offset, coefftvs = 0.5, coefftvt=0.5, min_loss = min_loss2, max_iteration = max_iteration2,training_log_file = refine_log_file, save_training_log = True)
# sdf,intensities = train_modified(config, sdf, gt_sinogram, lr=1e-5, init=init[:,0], gantry_offset = opt.offset, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, max_iteration = max_iteration2,training_log_file = refine_log_file, sdf_log_file = os.path.join(save_folder,experiment_name+'_refine_sdf_train.xlsx'))
# ff.save_data(config,sdf, 'refine', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

# #### Make the movie
# # prediction
# # movie = fetch_movie(config, sdf)
# movie = rotate(fetch_movie(config, sdf, None), -90, reshape=False)
# filepath = os.path.join(save_folder,filename+'_predict')
# np.save(filepath,movie)
# file_folder_predicted = os.path.join(save_folder,'movie_predicted','pngs')
# ff.make_folder([os.path.dirname(file_folder_predicted),file_folder_predicted])
# save_movie(movie, file_folder_predicted)

# pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],file_folder_predicted),1)
# ff.make_movies(os.path.join(os.path.dirname(file_folder_predicted),'exp1_movie_predicted.mp4'),pngs,144)


# # ground truth
# sdfgt = SDFGt(config, body)
# movie = fetch_movie(config, sdfgt,all_thetas)
# filepath = os.path.join(save_folder,filename+'_gt')
# np.save(filepath,movie)
# file_folder_gt = os.path.join(save_folder,'movie_gt','pngs')
# ff.make_folder([os.path.dirname(file_folder_gt),file_folder_gt])
# save_movie(movie, file_folder_gt)

# pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],file_folder_gt),1)
# ff.make_movies(os.path.join(os.path.dirname(file_folder_gt),'exp1_movie_gt.mp4'),pngs,72)

