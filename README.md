# NeuralCT with Spatially-aware Segmentation
**Author: Zhennong Chen, PhD**<br />

This repo is for the paper: 
*Motion Correction Image Reconstruction using NeuralCT Improves with Spatially Aware Object Segmentation* <br />
see our Poster [here](https://drive.google.com/file/d/15bKUaYuGyG11al5EK0t9NuuGqKPfV1PB/view?usp=sharing) and our Paper [here](https://drive.google.com/file/d/1iYOESeSJIgmkF8Dq1VP6cVcdcT9KA_ZQ/view?usp=sharing)<br />
Authors: Zhennong Chen, Kunal Gupta, Francisco Contijoch<br />

**Citation**: Zhennong Chen, Kunal Gupta, Francisco Contijoch, "Motion Correction Image Reconstruction using NeuralCT Improves with Spatially Aware Object Segmentation", International Conference on Image Formation in X-Ray Computed Tomography, June 2022.


## Description
This work is based on an [arXiv paper](https://arxiv.org/abs/2201.06574)([github repo](https://github.com/ucsd-fcrl/kg-neuralct-method)). In that paper, we proposed a implicit neural representation-based framework to correct the motion artifacts in the CT images. This framework, called "NeuralCT", takes CT sinograms as the input, uses a technique called "differentiable rendering" to optimize the estimation of object motion based on projections, and returns the time-resolved image without motion artifacts. See more details in the papers.

**The main goal of this work** is to extend NeuralCT to a more complicated scene, where there are multiple moving objects with distinct attenuations. This scene is closer to what we will see in clinical CT images (e.g., a moving LV with constrast and a RV without contrast).

Empiricially, we have found that the performance of NeuralCT is highly dependent on the model initialization driven by the segmentation of FBP images. Previously in arXiv paper, we used a [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html#:~:text=A%20Gaussian%20mixture%20model%20is,Gaussian%20distributions%20with%20unknown%20parameters.) to segment different objects from FBP; in the more complicated scene, we determine to use a **spatially-aware segmentation** that leverages both spatial and intensity information of different objects. Concretely, this segmentation is done by defining ROIs then thresholding; future work may incorporate data-driven methods(i.e., deep learning methods).



## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can set up the docker image by ```start_docker_neural_ct.sh```.

### Main Experiment
We simulated a scene with two moving dots with distinct attenuations and tested our NeuralCT on it. The user can define moving speed as well as the gantry offset. 
- main script: ```study_two_intensities.py```
- spatially-aware segmentation: ```jupyter_notebook/spatial_segmenter_zc.ipynb```

#### Additional Experiment
Based on the single-dot experiment in [arXiv paper](https://arxiv.org/abs/2201.06574), we added the quanta-counting noise to the sinogram to evaluate the impact of different Contrast-to-noise ratios (CNR) on the performance of NeuralCT.
- main script: ```study_noise.py```

### Additional guidelines
see comments in the script

Please contact the author (chenzhennong@gmail.com or zhc043@eng.ucsd.edu) for any further questions.<br />
For environement setup difficulty, please contact Kunal Gupta (k5gupta@eng.ucsd.edu) or Francisco Contijoch (fcontijoch@eng.ucsd.ddu) for help.







