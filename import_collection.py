#!/usr/bin/env python

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import argparse
import warnings
from tqdm import tqdm
from copy import deepcopy
import imageio

import torch
from torch import optim         
from torch.optim.lr_scheduler import StepLR

from skimage import measure
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from scipy.ndimage import (rotate, binary_erosion)
from sklearn.mixture import GaussianMixture


