#!/usr/bin/env python

from import_collection import *
from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *

# initialization of SIREN
def pretrain_sdf(config, pretraining_sdfs, init, min_loss, max_iteration, training_log_file, save_training_log, lr = 1e-4, scale = 1.5):
        
    assert len(pretraining_sdfs.shape) == 4, 'Invalid shape : {}'.format(pretraining_sdfs.shape) # pretraining_sdfs = [x_dim, y_dim, num_gantry_clicks, num_SDFs] - SDF maps for each component
    assert not np.isinf(pretraining_sdfs).any(), 'Contains infinity'
    
    sdf = SDFNCT(config, scale = scale)
    gt = torch.from_numpy(pretraining_sdfs).cuda()  # SDF based on FBP - used to intialize the model
    
    optimizer = optim.Adam(list(sdf.parameters()), lr = lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    # here is algorithm 3 in the paper: you want to initialize your network by FBP results, which means the network should correctly represent the sdf map generated by FBP.
    # in each iteration you pick a random t -> so each t (time, or you can say a projection) is a training sample.

    itr = 0
    training_log = []
    while 0 == 0:
        optimizer.zero_grad()
        t = np.random.randint(0,config.TOTAL_CLICKS,1)[0]
        theta = t*(config.THETA_MAX/config.TOTAL_CLICKS)
        
        pred = sdf(theta)  # sdf is the SDFNCT, it will output the predicted SDF map, shape = [x_dim, y_dim, num_SDF]
        target = gt[...,t,:]

        assert target.shape == pred.shape, 'target has shape : {} while prediction has shape :{}'.format(target.shape, pred.shape)
        eikonal, _, _ = sdf.grad(theta)

        loss1 = torch.abs(pred - target).mean()
        loss = loss1 + 0.1*eikonal 
        loss.backward()
        optimizer.step()

        training_log.append([itr,loss.item(),loss1.item(), eikonal.item(),scheduler.get_last_lr()[0]*10**4])

        if itr %200 == 0:
            print('itr: {}, loss: {:.4f}, lossSinogram: {:.4f}, lossE: {:.4f}, lr: {:.4f}'.format(itr, loss.item(), loss1.item(), 
                                                                                           eikonal.item(),scheduler.get_last_lr()[0]*10**4))
            scheduler.step()

        if loss.item() <= min_loss or itr > max_iteration:
            break

        itr += 1

    print('pretraining the SIREN uses ', itr,' iterations')   

    if save_training_log == True:
        df = pd.DataFrame(training_log,columns = ['iteration','total_loss','SDF_L1_loss','Ekikonal_loss','learning_rate'])
        df.to_excel(training_log_file,index=False)
    return sdf, init 



# Train SIREN
def train_modified(config, sdf, gt_sinogram, lr=1e-4, init = np.array([0.25,0.80]), gantry_offset = 0.0, coeffsino = 1, coeffek = 0.1, coefftvs = 0.5, coefftvt=0.5, min_loss = 0.1, sino_loss = 1000, max_iteration = None,training_log_file = None):
   
    assert isinstance(max_iteration, int)
    assert isinstance(sino_loss,float)

    intensities = Intensities(config, learnable = False, init = init)
    ii = intensities.forward()
    renderer = Renderer(config, sdf,intensities, offset = gantry_offset-180) 
    optimizer = optim.Adam(list(sdf.parameters()), lr = lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        

    itr = 0
    training_log = []

    while True:
        optimizer.zero_grad()
        t = np.random.randint(0,config.TOTAL_CLICKS,config.BATCH_SIZE)
        theta = t*(config.THETA_MAX/config.TOTAL_CLICKS)
        pred = renderer(theta)
                
        target = gt_sinogram[:,t]
        loss1 = torch.abs(pred - target).mean()*100
        
        eikonal, total_variation_space, total_variation_time = sdf.grad(theta[0])
        assert target.shape == pred.shape, 'target has shape : {} while prediction has shape :{}'.format(target.shape, pred.shape)
        
        loss = coeffsino*loss1 + coeffek*eikonal + coefftvs*total_variation_space + coefftvt*total_variation_time

        loss.backward()
        optimizer.step()

        training_log.append([itr,loss.item(),loss1.item(), eikonal.item(),total_variation_space.item(), total_variation_time.item(),scheduler.get_last_lr()[0]*10**4])
        
        if itr %200 == 0:
            print('itr: {}, loss: {:.4f}, lossSinogram: {:.4f}, lossE: {:.4f}, lossTVs: {:.4f}, lossTVt: {:.4f}, lr: {:.4f}'.format(itr, loss.item(), loss1.item(), eikonal.item(), 
                                     total_variation_space.item(), total_variation_time.item(), scheduler.get_last_lr()[0]*10**4))
        
        if itr % 200 == 0:
            # update learning rate
            scheduler.step()

        if loss.item() < min_loss or itr >= max_iteration:
            break
        
        if sino_loss < 99:    
            if loss1.item() < sino_loss:
                break
        
        itr += 1
    
    df = pd.DataFrame(training_log,columns = ['iteration','total_loss','SDF_L1_loss','Ekikonal_loss','TVS','TVT','learning_rate'])
    df.to_excel(training_log_file,index=False)
            
    return sdf, intensities


def get_pretraining_sdfs_spatial(config, label_image, fbp_movie): 
    '''this function is used when you use spatially-aware segmentation.'''
    '''this function returns a movie_object (Size: (x_dim, y_dim, gantry_clicks, num_SDF)) which assigns SDF values to each pixel in the fbp recon.'''
    
    print("Computing Segmentations...")

    movie = label_image.copy()
    num_components = config.NUM_SDFS
    
    movie_objects = np.zeros((movie.shape[0],movie.shape[1],movie.shape[2],num_components))
    labels = np.arange(0,np.max(movie[...,0])).astype(np.int) # labels are all the labels except the last one, e.g. image's labels = [0,1], the variable labels = [0]

    # make each component as one channel, and turn the label to boolean values.
    # if num_components = 1, then l = 1.
    for i in range(movie.shape[2]):
        for l in labels:
            movie_objects[...,i,l] = (movie[...,i] ==l+1)
           
    print("Computing SDFs...")
    init = np.zeros((1,num_components))
    for j in tqdm(range(num_components)):
        for i in range(movie.shape[2]):
            occupancy = np.round(denoise_tv_chambolle(movie_objects[...,i,j][...,np.newaxis])) # add one more dimension for occupancy, e.g (x_dim, y_dim, 720, num_components) -> (x_dim, y_dim, 720, num_compoenents, 1)
            movie_objects[...,i,j] = denoise_tv_chambolle(occ_to_sdf(occupancy), weight=2)[...,0] #the greater weight, the more denoising
    
    idx = 0
    
    while True:
        break_sign = 0
        init = np.zeros((1,num_components))
        for j in range(0,num_components):
            img = fbp_movie[...,j]
            test = np.where(movie_objects[...,idx,j]>0)#[...,0]  # all the pixels in the first gantry image with SDF > 0
            init[0,j] = np.median(img[test[0], test[1]]) # test[0] is x coordiante and test[1] is y_coordinate
        
        s = [math.isnan(init[0,k]) for k in range(0,init.shape[-1])]
        if sum(s) == 0:
            break
        idx += 1
   
    # movie_objects.shape = [x_dim, y_dim, num_gantry_clicks, num_SDF]
    pretraining_sdfs = movie_objects.copy()

    return pretraining_sdfs, init



