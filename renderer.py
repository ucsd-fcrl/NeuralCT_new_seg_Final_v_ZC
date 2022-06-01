#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import kornia # https://kornia.readthedocs.io/en/latest/
from skimage.transform import iradon
from scipy import ndimage
from scipy.ndimage import rotate


from config import Config
from anatomy import Body

def sdf_to_occ(x):
    '''
    Converts sign distance to occupancy for rendering
    '''
    assert isinstance(x, torch.Tensor) and len(x.shape) == 3, 'Input must be a 3D torch tensor'
    occ = torch.zeros(x.shape)
    for i in range(x.shape[2]):
        occ[...,i] = torch.clamp(50*(torch.sigmoid(x[...,i]) - 0.5),0,1)  # (0,1) -> (-0.5,0.5) -> (-10,10) -> (0,1)   # limit the input in the range (min,max)
        
    return occ

def occ_to_sdf(x):
    '''
    This function convets a binary occupancy image into a signed distance image.
    '''
    assert isinstance(x, np.ndarray) and len(x.shape) == 3, 'x must be a 3D array containing separate images for each organ'
    assert np.sum(x==1) + np.sum(x==0) == x.shape[0]*x.shape[1]*x.shape[2], 'x must only have values 0 and 1' 
    
    dist_img = np.zeros_like(x)  # returns an array of zeros with the same shape and type as a given array

    # use scipy.ndimage.distance_transform_bf - distance tranform function by a brute force algorithm
    # this function calculates the distance transform of the input, by replacing each background element (Zero values), with its shortest distance to the foreground (element non-zero)
    for i in range(x.shape[2]):
        # ndimage.distance_transform_bf(x[...,i]==1) assigns SDF value for all background element (pixel value = 0)
        # - ndimage.distance_transform_bf(x[...,i]==0) assigns SDF value (negative) for all non-zero element (pixel value = 1)
        dist_img[...,i] = ndimage.distance_transform_bf(x[...,i]==1) - ndimage.distance_transform_bf(x[...,i]==0) 
 
    return dist_img


class SDF(nn.Module):
    def __init__(self):
        super(SDF, self).__init__() # super() function will make the child class inherit all the methods and properties from its parent
    
    def forward(self):
        raise NotImplementedError
        pass

class SDFGt(SDF):
    # use body.is_inside to get the occupancy for a body with rotation t.
    # then turn the occupancy into sdf - final sdf is a canvas as 3D tensor
    def __init__(self, config, body):
        super(SDFGt, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(body, Body), 'body must be an instance of class Body'
        assert config.INTENSITIES.shape[1] == len(body.organs), 'Number of organs must be equal to the number of intensities' # each organ has its own unique intensity
        
        self.config = config
        self.body = body
        x,y = np.meshgrid(np.linspace(0,1,config.IMAGE_RESOLUTION),np.linspace(0,1,config.IMAGE_RESOLUTION)) # make a meshgrid with x for all x coordinates and y for all y coordinates
        self.pts = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))  # pts has every row as a point coordinate (x,y), size is (num_of_points,2)
      
    def forward(self, t):
        '''
        Calculates the ground truth SDF or Image for given time of gantry
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)

        inside = self.body.is_inside(self.pts, t)  # inside shape = [x_dim * y_dim, num_organs]
     
        # returns a one hot encoding of point by querying each organ. returns all zeros if the point is outside the body
        # the size of inside is N x n, where N is the number of the points and n is the number of organs

        inside_sdf = occ_to_sdf(inside.reshape(self.config.IMAGE_RESOLUTION,
                                               self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1])) # convert occ to sdf
        
        canvas = torch.from_numpy(inside_sdf) # shape is [x_dim, y_dim, num_of_sdf]
        assert len(canvas.shape) == 3, 'Canvas must be a 3D tensor, instead is of shape: {}'.format(canvas.shape)
        return canvas

class Intensities(nn.Module):
    # it returns a 1X1XN torch tensor with N intensity classes
    def __init__(self, config, learnable = False, init = np.array([0.3,0.5]), bandwidth = 0.05):
        super(Intensities, self).__init__()
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(learnable, bool), 'learnable must be a boolean'
        assert isinstance(init, np.ndarray) and len(init.shape) == 1, 'init must be a 1D array of intensities'
        
        assert isinstance(bandwidth, float) and bandwidth > 0 and bandwidth < 1, 'bandwidth must be a float between 0 and 1'
        
        if learnable:
            assert config.NUM_SDFS == init.shape[0], 'init must exactly the same intensities as number of sdfs'
            self.inty = torch.nn.Parameter(torch.from_numpy(init).view(1,1,-1)) 
            self.default = 0*torch.from_numpy(config.INTENSITIES).view(1,1,-1)
        else:
            self.inty = 0*torch.from_numpy(config.INTENSITIES).view(1,1,-1)
            self.default = torch.from_numpy(config.INTENSITIES).view(1,1,-1)  # turn intensity size from Nx1 to 1x1xN 
        
        self.config = config
        self.bandwidth = bandwidth
        
    def forward(self):    
        residual = torch.clamp(self.inty, -1, 1)*self.bandwidth # when learnable = False, residual = 0
        

        return self.default + residual    # in exp1, this very is tensor([[[0.5000]]])
    
class Renderer(nn.Module):
    def __init__(self, config, sdf, intensities, offset=0.0):
        super(Renderer, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(sdf, SDF), 'sdf must be an instance of class SDF'
        assert isinstance(intensities, Intensities), 'intensities must be an instance of class Intensities'
        
        self.config = config
        self.sdf = sdf
        self.intensities = intensities
        self.offset = offset    

    def snapshot(self,t):
        '''
        Rotates the canvas at a particular angle and calculates the intensity
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        # use kornia - open source differentiable computer vision library: https://kornia.readthedocs.io/en/latest/
        # kornia.get_rotation_matrix2d(center,angle,scale) - calculate the affine matrix of 2D rotation
        # fix a bug here: torch.ones(1) - > torch.ones(1,2)
        rotM = kornia.get_rotation_matrix2d(torch.Tensor([[self.config.IMAGE_RESOLUTION/2,self.config.IMAGE_RESOLUTION/2]]), torch.Tensor([t+self.offset]) , torch.ones(1,2)).cuda()


        #print('t here is: ', t)
        canvas = sdf_to_occ(self.sdf(t)) # SDFGt.forward, canvas is a 3D tensor with dimension = x_dim_image x y_dim_image x 1
        
        intensities = self.intensities().type_as(canvas)

        assert len(intensities.shape) == 3, 'intensities must be a 3D tensor'
        
        # use equation: new attenuation = intensity * occupancy
        # intensity shape = [1,1,1], canvas shape = [6,6,1]
        canvas = canvas*intensities 
  
        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION, self.config.NUM_SDFS)
        
        canvas = torch.sum(canvas, dim=2)
        #print('canvas: ',canvas.shape) # reduce the dimension
#         canvas = torch.sum(canvas*self.intensities().type_as(canvas),dim=2)
        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)

        # canvas has dimension as x_dim * y_dim, however warp_affine  needs input with dimension = (B,C,H,W), so use unsqueeze to add dimension to canvas to eventually be (1,1,x_dim,y_dim)
        # warp_affine(src,M,dsize): apply an affine transformation to a tensor
        # src: tensor, input tensor of shape (B,C,H,W)
        # M：tensor, affine transformation of shape (B,2,3)
        # dsize: tuple, size of the output image (H,W)
        canvas = kornia.warp_affine(canvas.unsqueeze(0).unsqueeze(1).cuda(), rotM, dsize=(self.config.IMAGE_RESOLUTION, self.config.IMAGE_RESOLUTION)).view(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
        #print('canvas after applying transformation: ',canvas.shape)

        result = (torch.sum(canvas, axis=1)/self.config.IMAGE_RESOLUTION) #why summation in the row??
        #print('final snapshot 1D result!: ', result.shape)
        
        assert len(result.shape) ==1, 'result has shape :{} instead should be a 1D array'.format(result.shape)
        return result
        
    def forward(self, all_thetas):
        
        assert isinstance(all_thetas, np.ndarray) and len(all_thetas.shape) ==1, 'all_thetas must be a 1D numpy array of integers'
        assert all_thetas.dtype == float, 'all_thetas must be a float, instead is : {}'.format(all_thetas.dtype)
        assert all(abs(t) <= self.config.THETA_MAX for t in all_thetas), 'all_theta is out of range'.format(all_thetas)

        self.intensity = torch.zeros((self.config.IMAGE_RESOLUTION, all_thetas.shape[0])).cuda()
        for i, theta in enumerate(all_thetas):
            self.intensity[:,i] = self.snapshot(theta)

        return self.intensity
    
    def compute_rigid_fbp(self, x, all_thetas):
        '''
        Computes the filtered back projection assuming rigid bodies
        '''
        assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x must be a 2D numpy array'
        assert isinstance(x, np.ndarray) and len(all_thetas.shape) == 1, 'all_thetas must be a 1D numpy array'
        assert all_thetas.shape[0] == x.shape[1], 'number of angles are not equal to the number of sinogram projections!'

        return iradon(x, theta=all_thetas,circle=True)
    




class Renderer(nn.Module):
    def __init__(self, config, sdf, intensities, offset=0.0):
        super(Renderer, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(sdf, SDF), 'sdf must be an instance of class SDF'
        assert isinstance(intensities, Intensities), 'intensities must be an instance of class Intensities'
        
        self.config = config
        self.sdf = sdf
        self.intensities = intensities
        self.offset = offset    

    def snapshot(self,t):
        '''
        Rotates the canvas at a particular angle and calculates the intensity
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        # use kornia - open source differentiable computer vision library: https://kornia.readthedocs.io/en/latest/
        # kornia.get_rotation_matrix2d(center,angle,scale) - calculate the affine matrix of 2D rotation
        # fix a bug here: torch.ones(1) - > torch.ones(1,2)
        rotM = kornia.get_rotation_matrix2d(torch.Tensor([[self.config.IMAGE_RESOLUTION/2,self.config.IMAGE_RESOLUTION/2]]), torch.Tensor([t+self.offset]) , torch.ones(1,2)).cuda()

        #print('t here is: ', t)
        canvas = sdf_to_occ(self.sdf(t)) # SDFGt.forward, canvas is a 3D tensor with dimension = x_dim_image x y_dim_image x 1
        
        intensities = self.intensities().type_as(canvas)

        assert len(intensities.shape) == 3, 'intensities must be a 3D tensor'
        
        # use equation: new attenuation = intensity * occupancy
        # intensity shape = [1,1,1], canvas shape = [6,6,1]
        canvas = canvas*intensities 
  
        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION, self.config.NUM_SDFS)
        
        canvas = torch.sum(canvas, dim=2)
        #print('canvas: ',canvas.shape) # reduce the dimension
#         canvas = torch.sum(canvas*self.intensities().type_as(canvas),dim=2)
        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)

        # canvas has dimension as x_dim * y_dim, however warp_affine  needs input with dimension = (B,C,H,W), so use unsqueeze to add dimension to canvas to eventually be (1,1,x_dim,y_dim)
        # warp_affine(src,M,dsize): apply an affine transformation to a tensor
        # src: tensor, input tensor of shape (B,C,H,W)
        # M：tensor, affine transformation of shape (B,2,3)
        # dsize: tuple, size of the output image (H,W)
        canvas = kornia.warp_affine(canvas.unsqueeze(0).unsqueeze(1).cuda(), rotM, dsize=(self.config.IMAGE_RESOLUTION, self.config.IMAGE_RESOLUTION)).view(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
        #print('canvas after applying transformation: ',canvas.shape)

        result = (torch.sum(canvas, axis=1)/self.config.IMAGE_RESOLUTION) #why summation in the row??
        #print('final snapshot 1D result!: ', result.shape)
        
        assert len(result.shape) ==1, 'result has shape :{} instead should be a 1D array'.format(result.shape)
        return result
        
    def forward(self, all_thetas):
        
        assert isinstance(all_thetas, np.ndarray) and len(all_thetas.shape) ==1, 'all_thetas must be a 1D numpy array of integers'
        assert all_thetas.dtype == float, 'all_thetas must be a float, instead is : {}'.format(all_thetas.dtype)
        assert all(abs(t) <= self.config.THETA_MAX for t in all_thetas), 'all_theta is out of range'.format(all_thetas)

        self.intensity = torch.zeros((self.config.IMAGE_RESOLUTION, all_thetas.shape[0])).cuda()
        for i, theta in enumerate(all_thetas):
            self.intensity[:,i] = self.snapshot(theta)

        return self.intensity
    
    def compute_rigid_fbp(self, x, all_thetas):
        '''
        Computes the filtered back projection assuming rigid bodies
        '''
        assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x must be a 2D numpy array'
        assert isinstance(x, np.ndarray) and len(all_thetas.shape) == 1, 'all_thetas must be a 1D numpy array'
        assert all_thetas.shape[0] == x.shape[1], 'number of angles are not equal to the number of sinogram projections!'

        return iradon(x, theta=all_thetas,circle=True)
