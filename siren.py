import numpy as np
import torch
import torch.nn as nn

# SIREN: Sinusoidal Representation Networks
# Implicit neural representations are created when a neural network is used to represent a continuous and differentiable signal as a function
# https://vsitzmann.github.io/siren/
# Check the paper: Implicit Neural Representation with Periodic Activation Function
class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    


    def init_weights(self): # a weight initialization specific for SIREN network (check the paper)
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    # Finally, we propose to initialize the first layer of the sine network with weights so that the
    # sine function sin(omega_0 * Wx+ b) spans multiple periods over [-1; 1]. We found omega_0 = 30 to work
    # well for all the applications in this work. The proposed initialization scheme yielded fast and robust
    # convergence using the ADAM optimizer for all experiments in this work   
    
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))   # torch.sin, returns a new tensor with the sine of the elements of input. Outi = sin(inputi)
    
class Siren(nn.Module): 
    
    # SIREN is a series of SineLayers (linear layer + torch.sin)
    # if outermost_linear = True, (range(hidden_layers) + 1) SineLayer + 1 Linear layer
    # if outermost_linear = False, (range(hidden_layers) + 2) SineLayer

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):

        # in_features = 2 or 3 since input is always a coordinate. 
        # out_features = channels of pixel value, in greyscale = 1, RGB = 3, in our case = num_SDFs

        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net).cuda()
        
    def forward(self, coords):
        output = self.net(coords)
        return output
    
    
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y) # define a one-like tensor (just an empty tensor with 1 with the same dimension as y) ready for grad_output
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0] #torch.autograd.grad(inputs(y), outputs(x))
    return grad