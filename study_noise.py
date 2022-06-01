#!/usr/bin/env python

# In this expertiment, we add poisson-distributed quanta-counting noise to the simulated sinogram in Kunal's single dot experiment.
# We would like to evaluate the impact of different contrast-to-noise- ratios on the performance of NeuralCT reconstruction
# Run this script by: python study_noise.py --offset XX --rate XX --exp XX --CNR XX (XX are something you)

from import_collection import *
from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *
from noise import *
from new_model import *
import function_list as ff

# define seed
num = 1
torch.manual_seed(num)


# define the folder path
main_folder = "/Data/McVeighLabSuper/wip/NeuralCT/Test_results/"

# define parameters
GMM_threshold = 0.15  # default = 0.15 in the paper
min_loss1 = 0.5  # total loss of pre-train
max_iteration1 = 5000  # maximum iteration of pre-train
min_loss2 = 0.2   # total loss of train
sino_loss = 0.17  # sinogram loss of train
max_iteration2 = 10000   # maximum itertaion of train
[coeffsino, coeffek, coefftvs, coefftvt] = [1, 0.1, 0.5, 0.5]  # weights of each loss in the loss function

MATRIX_SAVE = True  # do you want to save the numpy array of recon image?
MOVIE_SAVE = True   # do you want to save the movie of recon image?
use_save_data_gt = True  # do you want to use saved ground truth data?
use_save_data_fbp_sdf = False # do you want to use saved SDF maps generated based on FBP?


# define a list of Contrast-to-noise ratio (CNR). Please see jupyter_notebook/zc_add_noise_to_sinogram.ipynb --> each CNR corresponds to a N0
CNR_list = [3,5,10,15,20,25,30,35,40,45]
N0_list = [200, 1000, 7000, 16000, 30000, 50000, 90000, 150000, 250000, 600000] # corresponding N0


# define parser
parser = argparse.ArgumentParser()
parser.add_argument("--offset", default=60.0, type=float, help = "Gantry offset")
parser.add_argument("--rate", default=0.80, type=float, help = "Heartrate")
parser.add_argument("--exp", type=str, help = "Experiment_Name")
parser.add_argument("--CNR", default = 40, type=int)
opt = parser.parse_args()
print('Doing for offset : {} , rate : {} and CNR: {}'.format(opt.offset, opt.rate, opt.CNR))

cnr = int(opt.CNR)
if cnr in CNR_list:
    N0 = N0_list[CNR_list.index(cnr)]
else:
    ValueError('SNR you defined is not in the SNR_list')

experiment_name = opt.exp +'_'+ str(int(opt.rate*100)) + '_CNR' + str(cnr)
experiment = experiment_name+ '_results/'  
print(experiment)


# define save_folder
save_folder = os.path.join(main_folder,experiment)
ff.make_folder([save_folder])      
filename = 'movie_{}_{}_CNR{}'.format(int(opt.offset), int(opt.rate*100), cnr)


#### Define Config
config = Config(np.array([[0.5]]), TYPE=0, NUM_HEART_BEATS=opt.rate, image_resolution = 128) #Config(intensity, type, num_heart_beats), intensity here is a Nx1 array with each one in N corresponding to a intensity class
all_thetas = np.linspace(-config.THETA_MAX/2, config.THETA_MAX/2, config.TOTAL_CLICKS)

#### Define Body (organs)
body = Body(config, [Organ(config,[0.5,0.5],0.15, 0.15,'const','circle')])  

#### Get ground truth
print('....GET GROUND TRUTH....')
sdfgt = SDFGt(config, body)
if use_save_data_gt == False:
    ff.save_data(config,sdfgt, 'gt', save_folder, filename, experiment_name, all_thetas = all_thetas, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

if use_save_data_gt == False:
    # get ground truth sinogram
    gt_sinogram_numpy = get_sinogram(config, SDFGt(config, body),Intensities(config, learnable = False), all_thetas,offset = opt.offset)

    #### add noise
    gt_sinogram_noise, cnr_cal =  add_noise_to_sinogram(gt_sinogram_numpy, N0)
    print('\nEmpirically calculated CNR is : ',cnr_cal)
    np.save(os.path.join(save_folder,filename+'_sinogram_gt'),gt_sinogram_noise)
    np.save(os.path.join(save_folder,filename+'_sinogram_no_noise'),gt_sinogram_numpy)

else:
    gt_sinogram_noise = np.load(os.path.join(save_folder,filename+'_sinogram_gt.npy'),allow_pickle = True)

gt_sinogram_noise = torch.from_numpy(gt_sinogram_noise).cuda()
random.seed(num)
np.random.seed(num)


#### Get sinogram using rendering and then get FBP reconstructed image(the initializer of the model)
print('....GET FBP....')
if use_save_data_gt == False:
    sinogram_fbp, reconstruction_fbp = fetch_fbp_movie_noise(config, body, gt_sinogram_noise, gantry_offset=opt.offset)
    ff.save_data(config,reconstruction_fbp, 'fbp', save_folder, filename, experiment_name, all_thetas = all_thetas, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)
    np.save(os.path.join(save_folder,'sinogram_generates_FBP'),sinogram_fbp)
else:
    reconstruction_fbp = np.load(os.path.join(save_folder,filename+'_fbp.npy'),allow_pickle = True)
    sinogram_fbp = np.load(os.path.join(save_folder,'sinogram_generates_FBP.npy'),allow_pickle = True)


#### segment FBP into several classess and then convert into a SDF map 
print('....GET FBP-DERIVED SDF MAP....')
if use_save_data_fbp_sdf == False:
    pretraining_sdfs, init = get_pretraining_sdfs(config, sdf=reconstruction_fbp, GMM_threshold = GMM_threshold, save_GMM_label_path = os.path.join(save_folder,filename+'_GMM_labels')) 
    np.save(os.path.join(save_folder,'FBP_derived_sdf'),pretraining_sdfs)
    np.save(os.path.join(save_folder,'FBP_derived_init'),init)
    
else:
    pretraining_sdfs = np.load(os.path.join(save_folder,'FBP_derived_sdf.npy'),allow_pickle = True)
    init = np.load(os.path.join(save_folder,'FBP_derived_init.npy'),allow_pickle = True)


#### Initialize the SIREN by the SDF map from FBP  
# here sdf is the SIREN network which outputs the SDF map.
print('....INITIALIZE SIREN....')
pretrain_training_log_file = os.path.join(save_folder,experiment_name+'_training_log_pretrain.xlsx')
sdf, init = pretrain_sdf(config, pretraining_sdfs, init,  min_loss1, max_iteration1, pretrain_training_log_file, True, lr = 1e-4)  
print('init - median intensity of the object in the image is : ',init)
# save
ff.save_data(config,sdf, 'initial1', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Train SIREN 
print('....TRAIN SIREN....')
training_log_file = os.path.join(save_folder,experiment_name + '_training_log_train.xlsx')
sdf,intensities = train_modified(config, sdf, gt_sinogram_noise, lr=1e-4, init=init[:,0], gantry_offset = opt.offset, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2, training_log_file = training_log_file)
# save_
ff.save_data(config,sdf, 'train', save_folder, filename, experiment_name, all_thetas = None, rotate_image = False, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)


#### Refinement
# Initialize
print('....REFINEMENT....')
pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)

pretrain_refine_log_file = os.path.join(save_folder, experiment_name + '_refine_log_pretrain.xlsx')
sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, min_loss1, max_iteration1, pretrain_refine_log_file, True, lr = 5e-5)
ff.save_data(config,sdf, 'initial2', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)

# train
refine_log_file = os.path.join(save_folder, experiment_name+ '_refine_log_train.xlsx')
sdf,intensities = train_modified(config, sdf, gt_sinogram_noise, lr=1e-5, init=init[:,0], gantry_offset = opt.offset+90, coeffsino = coeffsino, coeffek = coeffek, coefftvs = coefftvs, coefftvt=coefftvt, min_loss = min_loss2, sino_loss = sino_loss, max_iteration = max_iteration2,training_log_file = refine_log_file)
ff.save_data(config,sdf, 'refine', save_folder, filename, experiment_name, all_thetas = None, rotate_image = True, numpy_save = MATRIX_SAVE, movie_save = MOVIE_SAVE)



#### Exclude disconnectivity (optional)
ff.make_folder([os.path.join(save_folder,'movie_final'), os.path.join(save_folder,'movie_final/pngs')])
img = np.load(os.path.join(save_folder, filename + '_refine.npy' ), allow_pickle = True)
new_img = np.zeros(img.shape)
for i in range(0,img.shape[-1]):
    new_img[:,:,i] = ff.pick_largest_comp(img[:,:,i])
    plt.imsave('{}/{}.png'.format(os.path.join(save_folder,'movie_final/pngs'),i), new_img[...,i], cmap='gray')
np.save(os.path.join(save_folder,filename +'_final.npy'), new_img)
pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'movie_final/pngs')),1)
ff.make_movies(os.path.join(save_folder,'movie_final',experiment_name+'_movie_final.mp4'),pngs,144)






