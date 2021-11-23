# Multiple myeloma prognosis from PET images:deep survival losses and contrastive pretraining
**by Ludivine Morvan, Cristina Nanni, Anne-Victoire Michaud, Bastien Jamet, Cl´ement Bailly, Caroline Bodet-Milin, Stephane Chauvie, Cyrille Touzeau, Philippe Moreau, Elena Zamagni, Francoise Kraeber-Bodéré, Thomas Carlier, and Diana Mateus**

This repository provides the python (Tensorflow) code from our submitted paper ->

## Abstract
**Objective**: Diagnosis and follow-up of multiple myeloma (MM) patients involve analyzing full-body 3D Positron Emission Tomography (PET) images. Towards assisting the analysis, there has been an increased interest in linking radiomics features extracted from PET images with machine learning methods for survival analysis. Despite the success of deep learning in other fields, its adaptation to survival analysis faces several challenges. Our goal is to design a deep-learning approach to predict the progression-free survival (PFS) of MM patients from PET lesion images. 
**Methods**: We review existing losses for survival analysis and propose new ranking and contrastive losses that deal with censorship and are novel to survival analysis. We also conceive two pretraining strategies to cope with the relatively small datasets, based on binary patch lesion classification and triplet loss embeddings. Finally, we design an architecture that handles the lesions’ small and variable scale and includes channel attention contributing to the performance and interpretability of the model. 
**Results**: Our approach is validated on data from two prospective clinical studies, improving the c-index (0.66) over baseline methods (Lasso-Cox:0.51, basic 3D CNN: 0.6).
**Conclusion**: We propose an end-to-end deep-learning approach, M2P2, to predict the progression-free survival (PFS) of MM patients from PET lesion images and outperform state-of-the-art methods. 
**Significance**: This work investigates for the first time deep learning in the context of the survival analysis of MM patients from PET images. We show the feasibility of existing losses for this task and introduce two contrastive learning approaches, never used before for survival analysis.


## Requirements


## Overview


## Citation

