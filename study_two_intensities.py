#!/usr/bin/env python

# In this expertiment, we applied our NeuralCT a more complicated scene where there are multiple moving objects with distinct attenuations.
# Empirically we found that the performance of NeuralCT is highly dependent on the segmentation of FBP image (used as the initialization of SIREN).
# When there is a single object, Gaussian mixture model (purely based on intensity) works
# When there are multiple objects, we need to use spatially-aware segmentation (based on both spatial and intensity info)

# Run this script by: python study_two_intensity.py --offset XX --rate XX --exp XX (XX are something you)

from import_collection import *
from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *
from noise import *
from new_model import *
import function_list as ff
num = 1
torch.manual_seed(num)


# define the folder path
print('new')
main_folder = "/Data/McVeighLabSuper/wip/NeuralCT/Test_results"

# define parameters
GMM_threshold = 0.05  # default = 0.15 in the paper
min_loss1 = 0.5  # total loss of pre-train
max_iteration1 = 5000  # maximum iteration of pre-train
min_loss2 = 0.15  # total loss of train
sino_loss = 0.065   # sinogram loss of train
max_iteration2 = 10000  # maximum iteration of train
[coeffsino, coeffek, coefftvs, coefftvt] = [1, 1, 1, 0.5]  # weights of each loss in the loss function

MATRIX_SAVE = True  # do you want to save the numpy array of recon image?
MOVIE_SAVE = True   # do you want to save the movie of recon image?
use_save_data_gt = True  # do you want to use saved ground truth data?
use_save_data_fbp_sdf = False # do you want to use saved SDF maps generated based on FBP?
use_spatial= True  # True = use spatially-aware segmentation, False = use Gaussian Mixture Model

# define parser
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


#### Define Config
config = Config(np.array([[0.7,0.2]]), TYPE=0, NUM_HEART_BEATS=opt.rate, image_resolution = 128) # two distinct intensities
all_thetas = np.linspace(-config.THETA_MAX/2, config.THETA_MAX/2, config.TOTAL_CLICKS)

#### Define Body (organs)
body = Body(config, [Organ(config,[0.5,0.2],0.1,0.1,'const','circle'),Organ(config,[0.55,0.6],0.1,0.1,'const','inverse_circle')]) # two dots


#### Get ground truth (here, we haven't added the quantum-counting noise to the sinogram)
print('....GET GROUND TRUTH....')
sdfgt = SDFGt(config, body)
if use_save_data_gt == False:
    ff.save_data(config,sdfgt, 'gt', save_folder, filename, experiment_name, all_thetas = all_thetas, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)
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


#### segment FBP into several classess and then convert into a SDF map 
# here we need to determine which segmentation method we want to use
print('....GET FBP-DERIVED SDF MAP....')
if use_save_data_fbp_sdf == False:

    if use_spatial == False: # use Gaussian model
        pretraining_sdfs, init = get_pretraining_sdfs(config, sdf=reconstruction_fbp, GMM_threshold = GMM_threshold, save_GMM_label_path = os.path.join(save_folder,filename+'_GMM_labels')) 
        print('init - median intensity of the object in the image is : ',init, ' pretraining_sdfs has size: ',pretraining_sdfs.shape) #[x_dim, y_dim, num_gantry_clicks, num_SDF]
        np.save(os.path.join(save_folder,'FBP_derived_sdf'),pretraining_sdfs)
        np.save(os.path.join(save_folder,'FBP_derived_init'),init)
    
    else:  # use spatially-aware segmentation
        # go to jupyter_notebook/spatial_segmenter_zc.ipynb to prepare the segmentation !!!
        # then come back
      
        if os.path.isfile(os.path.join(save_folder,filename+'_spatial_labels.npy')) == 0:
            ValueError('please go to jupyter_notebook/spatial_segmenter_zc.ipynb to prepare the segmentation')
        
        label_image = np.load(os.path.join(save_folder,filename+'_spatial_labels.npy'),allow_pickle = True)
        pretraining_sdfs, init = get_pretraining_sdfs_spatial(config, label_image, reconstruction_fbp)
        print('init - median intensity of the object in the image is : ',init, ' pretraining_sdfs has size: ',pretraining_sdfs.shape) #[x_dim, y_dim, num_gantry_clicks, num_SDF]
        np.save(os.path.join(save_folder,'Spatial_derived_sdf'),pretraining_sdfs)
        np.save(os.path.join(save_folder,'Spatial_derived_init'),init)
else:
    if use_spatial == False:
        pretraining_sdfs = np.load(os.path.join(save_folder,'FBP_derived_sdf.npy'),allow_pickle = True)
        init = np.load(os.path.join(save_folder,'FBP_derived_init.npy'),allow_pickle = True)
    else:
        pretraining_sdfs = np.load(os.path.join(save_folder,'Spatial_derived_sdf.npy'),allow_pickle = True)
        init = np.load(os.path.join(save_folder,'Spatial_derived_init.npy'),allow_pickle = True)


#### Initialize the SIREN by the SDF map from FBP  
# here sdf is the SIREN network pipeline which outputs the SDF map.
print('....INITIALIZE SIREN....')
pretrain_training_log_file = os.path.join(save_folder,experiment_name+'_training_log_pretrain.xlsx')
sdf, init = pretrain_sdf(config, pretraining_sdfs, init,  min_loss1, max_iteration1, pretrain_training_log_file, True, lr = 1e-4)  
print('init - median intensity of the object in the image is : ',init)
# save
ff.save_data(config,sdf, 'initial1', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Train SIREN 
print('....TRAIN SIREN....')
training_log_file = os.path.join(save_folder,experiment_name + '_training_log_train.xlsx')
sdf,intensities = train_modified(config, sdf, gt_sinogram, lr=1e-4, init=init[:,0], gantry_offset = opt.offset, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2, training_log_file = training_log_file)
# save
ff.save_data(config,sdf, 'train', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Refinement
print('....REFINEMENT....')
pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)
# initialize
pretrain_refine_log_file = os.path.join(save_folder, experiment_name + '_refine_log_pretrain.xlsx')
sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, min_loss1, max_iteration1, pretrain_refine_log_file, True, lr = 5e-5)
ff.save_data(config,sdf, 'initial2', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

# train
refine_log_file = os.path.join(save_folder, experiment_name+ '_refine_log_train.xlsx')
sdf,intensities = train_modified(config, sdf, gt_sinogram, lr=1e-5, init=init[:,0], gantry_offset = opt.offset+90, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2,training_log_file = refine_log_file)
ff.save_data(config,sdf, 'refine', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)






