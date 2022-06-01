#!/usr/bin/env python
from import_collection import *

from config import *
from anatomy import *
from renderer import *
from siren import *



def add_noise_to_sinogram(sinogram, N0):
    # define seed
    seed =int(np.random.rand()*10000)
    np.random.seed(seed)

    counts = np.exp(sinogram) * N0
    counts_noise = np.random.poisson(counts)
    sinogram_noise = np.log(counts_noise/N0)

    # this part works for Kunal's exp1 specifically
    signal = sinogram_noise[sinogram_noise > 0.05]
    background = sinogram_noise[sinogram_noise< 0.03]

    cnr_cal = (np.mean(signal)-np.mean(background))/np.std(background)
 
    return sinogram_noise, cnr_cal


def fetch_fbp_movie_noise(config, body, sinogram_noise, gantry_offset=0.0):

    ''' this function generates sinogram (projections at number_of_theta) from differentiable rendering and sdf'''
  
    intensities = Intensities(config, learnable = False)
    sdf = SDFGt(config, body)
    incr = 180 # degree
    add_click = incr*int(config.GANTRY_VIEWS_PER_ROTATION/360)

    all_thetas = np.linspace(-config.THETA_MAX/2-incr/2, config.THETA_MAX/2+incr/2, config.TOTAL_CLICKS + incr*int(config.GANTRY_VIEWS_PER_ROTATION/360))

    sinogram_noise = sinogram_noise.detach().cpu().numpy()
    sinogram = np.concatenate((sinogram_noise[:,sinogram_noise.shape[1] - int(add_click/2) :  sinogram_noise.shape[1]], sinogram_noise, sinogram_noise[:, 0 : int(add_click/2)]),axis = 1)
    sinogram = sinogram.reshape(config.IMAGE_RESOLUTION, config.TOTAL_CLICKS + incr*int(config.GANTRY_VIEWS_PER_ROTATION/360)) # 720views(total_clicks) + 360views(incr) = 1080, gantry_views_per_rotation/360 calculates how many views/clicks can be added for one additional degree
    
    reconstruction_fbp = np.zeros((config.IMAGE_RESOLUTION,config.IMAGE_RESOLUTION,config.TOTAL_CLICKS+ 0*incr*int(config.GANTRY_VIEWS_PER_ROTATION/360)))
    count = 0
    
    # filtered back projection algorithm:
    # skimage.transform.iradon - inverse radon transform, reconstruct an image from the radon transform, using the filtered back-projection algorithm
    # iradon(radon_image, theta=None, output_size = None, filter_name = 'ramp', interpolation = 'linear), radom_image = sinogram, each colume of the image corresponds to a projection along the different angle, theta = reconstruction angles

    print('increment has how many projections/clicks: ', incr*int(config.GANTRY_VIEWS_PER_ROTATION/360))
    for i in tqdm(range(0, config.TOTAL_CLICKS + 0*incr*int(config.GANTRY_VIEWS_PER_ROTATION/360))):
        # scipy.ndimage.rotate(input,angle), here angle = -gantry_offset

        reconstruction_fbp[...,count] = rotate(iradon(sinogram[...,i:i+incr*int(config.GANTRY_VIEWS_PER_ROTATION/360)],  # so each time when we do filtered-backprojection we use incr*int(config.GANTRY_VIEWS_PER_ROTATION/360) = 360 views? Yes.
                                               theta = all_thetas[i:i+incr*int(config.GANTRY_VIEWS_PER_ROTATION/360)] + gantry_offset ,circle=True).T, 
                                               0,reshape = False)
                                               #-gantry_offset, reshape=False)
        count+=1
    print('sinogram used for fbp has shape: ', sinogram.shape, ' rereconstructino fbp: ', np.max(reconstruction_fbp[:,:,300]), reconstruction_fbp.shape, ' after scale it becomes: ', np.max(reconstruction_fbp[:,:,300]) * 132)
  

    return sinogram, 132*reconstruction_fbp  # why 132? it's a scaling factor because reconstruction fbp will generate a very very small value. This 132 was empirically set, but theorectially it's deterministic and dependent on the CT scan not on the image.