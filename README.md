# Multiple myeloma prognosis from PET images:deep survival losses and contrastive pre-training
**by Ludivine Morvan, Cristina Nanni, Anne-Victoire Michaud, Bastien Jamet, Clément Bailly, Caroline Bodet-Milin, Stephane Chauvie, Cyrille Touzeau, Philippe Moreau, Elena Zamagni, Francoise Kraeber-Bodéré, Thomas Carlier, and Diana Mateus**

This repository provides the python (Tensorflow) code from our submitted paper -> [techrxiv link](https://www.techrxiv.org/articles/preprint/Multiple_myeloma_prognosis_from_PET_images_deep_survival_losses_and_contrastive_pre-training/20438604) 
## Abstract
**Objective:** Diagnosis and follow-up of multiple myeloma patients involve analysing full-body Positron Emission Tomography (PET) images. Towards assisting the analysis, there has been an increased interest in machine learning methods linking PET radiomics with survival analysis. Despite deep learning’s success in other fields, its adaptation to survival analysis faces several challenges. Our goal is to design a deep-learning approach to predict the progression-free survival of multiple myeloma patients from PET lesion images. 
**Methods:** We study three aspects of such deep learning approach: i) the loss function: we review existing and propose new losses for survival analysis based on contrastive triplet learning, ii) Pre-training: We conceive two pre-training strategies to cope with the relatively small datasets, based on patch classification and triplet loss embedding. iii) the architecture: we study the contribution of spatial and channel attention modules. 
**Results:** Our approach is validated on data from two prospective clinical studies, improving the c-index (0.66) over baseline methods (Lasso-Cox:0.51, basic 3D CNN: 0.6), notably thanks to the channel attention module (0.61 to 0.63) and the introduced pre-training methods (0.61 to 0.66). 
**Conclusion:** We propose an end-to-end deep-learning approach, M2P2, to predict the progression-free survival of multiple myeloma patients from PET lesion images and outperform state-of-the-art methods.
**Significance:** This work investigates for the first time deep learning in the context of the survival analysis of multiple myeloma patients from PET images. We show the feasibility of existing losses for this task and introduce two contrastive learning approaches, never used before for survival analysis.


## Requirements


## Overview


## Citation
Ludivine Morvan, et al. (2022). Multiple myeloma prognosis from PET images: deep survival losses and contrastive pre-training.
