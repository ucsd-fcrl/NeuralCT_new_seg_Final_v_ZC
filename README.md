# NeuralCT with Spatially-aware segmentation
**Author: Zhennong Chen, PhD**<br />

This repo is for the paper: 
*Motion Correction Image Reconstruction using NeuralCT Improves with Spatially Aware Object Segmentation* <br />
see our Poster [here](https://drive.google.com/file/d/15bKUaYuGyG11al5EK0t9NuuGqKPfV1PB/view?usp=sharing)
Authors: Zhennong Chen, Kunal Gupta, Francisco Contijoch<br />

**Citation**: Zhennong Chen, Kunal Gupta, Francisco Contijoch, "Motion Correction Image Reconstruction using NeuralCT Improves with Spatially Aware Object Segmentation", International Conference on Image Formation in X-Ray Computed Tomography, June 2022.


## Description
This work is based on an [arXiV paper](https://arxiv.org/abs/2201.06574). In that paper, we proposed a implicit neural representation-based framework to correct the motion artifacts in the CT images. This framework, called "NeuralCT", takes CT sinograms as the input, uses a technique called "differentiable rendering" to optimize the estimation of object motion based on projections, and returns the time-resolved image without motion artifacts. See more details in the papers.

**The main goal of this work** is to extend NeuralCT to a more complicated scene, where there are multiple moving objects with distinct attenuations. This scene is closer to what we will see in clinical CT images (e.g., a moving LV with constrast and RV without contrast).

Empiricially, we have found that the performance of NeuralCT is highly dependent on the model initialization driven by the segmentation of FBP images. Previously in arXiV paper, we used a [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html#:~:text=A%20Gaussian%20mixture%20model%20is,Gaussian%20distributions%20with%20unknown%20parameters.) to segment different objects from FBP; in this work with more complicated scene, we determine to use a **spatially-aware segmentation** that leverages both spatial and intensity information of different objects. Concretely, this segmentation is done by defining the regions of interest followed by thresholding; future work may incorporate data-driven methods into the segmentation (i.e., deep learning methods).



