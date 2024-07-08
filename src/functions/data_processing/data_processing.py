# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io
from skimage.external import tifffile
from skimage.io import imsave
import SimpleITK as sitk
import nibabel as nib
import scipy

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPool3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16

from Patient import Patient, correction
from pretraining import random_patch


def get_augmentation_config(path:Path, num_augm = 30)-> dict:
    """Get the augmentation config file if it already exists and num_augm is correct, 
    otherwise create a new config file.

    Args:
        path (Path): path to data augmentation config file
        num_augm (int, optional): Number of augmentation par image. Defaults to 30.

    Returns:
        dict: augmentation config file
    """
    
    if os.path.exists(path):
        augmentation_config = np.load(path, allow_pickle=True)
        
        if np.shape(augmentation_config)[1] != num_augm:
            augmentation_config = create_augm_config(num_augm = num_augm)

    else:
        augmentation_config = create_augm_config(num_augm = num_augm)
    
    np.save(path, augmentation_config)
    
    return augmentation_config


def create_augm_config(num_augm = 30)-> dict:
    """Define a num_augm configuration for data augmentation
    Choices for configurations are the following
        - translation: [0,0,1,2,3,4,5,6]
        - zoom_scale: [0.75,0.8,.9,1,1,1.1,1.2,1.3,1.4],
        - rotation: [-30,-20,-10,0,0,10,20,30]
        - rotation_axes: [(1,0)]
        - Horizontal_flip: [True,False] 
        - vertical_flip: [True,False]
    Args:
        num_augm (int, optional):  Number of augmentation par image. Defaults to 30.

    Returns:
        dict: dictionary containing num_augm values for each configuration key.
    
    Example of output :
        If num_augm = 3
        dict_augm_config_selected = {'translation': [2, 6, 3],
                                    'zoom_scale': [1.1, 0.9, 0.8],
                                    'rotation': [-10, 0, -10],
                                    'rotation_axes': [(1, 0), (1, 0), (1, 0)],
                                    'Horizontal_flip': [False, False, False],
                                    'vertical_flip': [True, True, False]}
    """
    
    dict_augm_config_choices = {'translation':[0,0,1,2,3,4,5,6], 
                                'zoom_scale':[0.75,0.8,.9,1,1,1.1,1.2,1.3,1.4], 
                                'rotation':[-30,-20,-10,0,0,10,20,30],
                                'rotation_axes':[(1,0)],
                                'Horizontal_flip':[True,False], 
                                'vertical_flip':[True,False]}
    
    dict_augm_config_selected = {key: [] for key in dict_augm_config_choices.keys()}
    
    for key in dict_augm_config_selected.keys():
        choices = dict_augm_config_choices[key]
        dict_augm_config_selected[key] = [random.choice(choices) for i in range(num_augm)]
        
    return dict_augm_config_selected


    
def apply_data_augmentation(augm_config:dict, images:np.ndarray , y:np.ndarray, masks:np.ndarray, ref:list, D3:str='True', num_augm:int = 30)->dict:
    """Apply data augmentation

    Args:
        augm_config (dict): vector of predifined augmentation sparameters
        images (np.ndarray): matrice of individuals' images. Size : (B,36,36,1)
        y (np.ndarray): matrice of individuals' target values. Size : (B, 2)
        masks (np.ndarray): matrice of individuals' masks. Size : (B,36,36,1)
        ref (list): vector of individuals' names
        D3 (str, optional): dimension. Defaults to 'True'.
        num_augm (int, optional): number of augmentation par image. Defaults to 30.

    Returns:
        dict: dictionary with augmented dataset
            images_transformed: shape (B*num_augm, 1, 36, 36, 1)
            y_transformed: shape (B*num_augm, 2)
            masks_transformed: shape (B*num_augm, 1, 36, 36, 1)
            ref_transformed: shape (B*num_augm,1)
    """
    
    def translateit(image, offset:tuple, order:int = 0):
        return scipy.ndimage.interpolation.shift(image, offset, order = order, mode='nearest')
    
    def scaleit(image, D3:str, factor=1.2, isseg=False):
        order                = 0 if isseg  else 3
        height, width, depth = image.shape
        zheight              = int(np.round(factor * height))
        zwidth               = int(np.round(factor * width))
        if D3 == 'True':
            zdepth           = int(np.round(factor * depth))
        
        if factor < 1.0:
            newimg  = np.zeros_like(image)
            row     = (height - zheight) // 2
            col     = (width - zwidth) // 2
            if D3 == 'True':
                layer   = (depth - zdepth) // 2
                newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = scipy.ndimage.interpolation.zoom(image, (float(factor), float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
            else:
                newimg[row:row+zheight, col:col+zwidth] = scipy.ndimage.interpolation.zoom(image, (float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth]
            return newimg
        
        elif factor > 1.0:
            row     = (zheight - height) // 2
            col     = (zwidth - width) // 2
            if D3 == 'True':
                layer   = (zdepth - depth) // 2
                newimg =scipy.ndimage.interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), float(factor)), order=order, mode='nearest')          
            else:            
                newimg =scipy.ndimage.interpolation.zoom(image[row:row+zheight, col:col+zwidth], (float(factor), float(factor)), order=order, mode='nearest')          
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            if D3 == 'True':
                extrad = (newimg.shape[2] - depth) // 2
                newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
            else:
                newimg = newimg[extrah:extrah+height, extraw:extraw+width]
            return newimg
        else:
            return image
        
    def rotateit(image, theta:int, isseg:bool = False, axis:tuple = (1,0)):
        order = 0 if isseg  else 5
        return scipy.ndimage.rotate(image, float(theta), reshape=False,axes=axis ,order=order, mode='nearest')
    
    def flipit(image, axes):
        if axes[0]:
            image = np.fliplr(image)
        if axes[1]:
            image = np.flipud(image)  
        return image

    def get_transformation(image:np.ndarray, augm_config:dict,config_k:int, D3:str)->np.ndarray:
        """Get transformed image according to augm_config

        Args:
            image (np.ndarray): image to transform
            augm_config (dict): chosen augmentation configuration
            config_k (int): num of the config
            D3 (str): dimension

        Returns:
            np.ndarray: transformed image
        """
        size_d = 3 if D3 == 'True' else 2

        image = translateit(image, offset=(augm_config['translation'][config_k],)*size_d, order = 0)
        image = scaleit(image, D3 = D3, factor=augm_config['zoom_scale'][config_k], isseg = False)
        image = rotateit(image, augm_config['rotation'][config_k], isseg = False, axis = augm_config['rotation_axes'][config_k])
        image = flipit(image, axes = (augm_config['Horizontal_flip'][config_k],augm_config['vertical_flip'][config_k]))
        
        return image
    
    def transform_and_append(data, transformation_dict, key, patient_index, augm_config, config_k, D3):
        """Apply transformation to the data and append to the specified key in the transformation dictionary.
        
        Args:
            data (numpy.ndarray): The data to be transformed.
            transformation_dict (dict): The dictionary to store the transformed data.
            key (str): The key under which to store the transformed data.
            patient_index (int): The index of the patient data to be transformed.
            augm_config (dict): The augmentation configuration.
            config_k (str): The configuration key.
            D3 (bool): A flag indicating if the transformation is 3D.
        """
        transformed_data = get_transformation(data[patient_index, :, :, 0], augm_config, config_k, D3)
        transformation_dict[key].append(transformed_data)
    
    dict_transformation = { 'images_transformed':[],
                            'y_transformed':[],
                            'masks_transformed':[],
                            'ref_transformed':[],
                            }

    for patient in tqdm(range(len(images)))  :
        for config_k in range(num_augm):
            if D3 != '25':
                
                for data, key in zip([images, masks], ['images_transformed', 'masks_transformed']):
                    transform_and_append(data, dict_transformation, key, patient, augm_config, config_k, D3)

                dict_transformation['y_transformed'].append(y[patient])
                
            else:
                
                transformed_images = [
                    get_transformation(images[patient, :, :, i, 0], augm_config, config_k, D3) for i in range(3)
                ]
                dict_transformation['images_transformed'].append(np.array(transformed_images).transpose(1, 2, 0))

                transformed_masks = [
                    get_transformation(masks[patient, :, :, i, 0], augm_config, config_k, D3) for i in range(3)
                ]
                dict_transformation['masks_transformed'].append(np.array(transformed_masks).transpose(1, 2, 0))
                
            dict_transformation['y_transformed'].append(y[patient])
            dict_transformation['ref_transformed'].append(ref[patient])
                
    for key in dict_transformation.keys():
        dict_transformation[key] = np.array(dict_transformation[key]) 
        
    dict_transformation['masks_transformed'] = 1 - (dict_transformation['masks_transformed'] > 0.5).astype(int)
       
    if D3=='True':
        reshape_to_4d(dict_transformation, ['images_transformed', 'masks_transformed'])
    else:        
        reshape_to_5d(dict_transformation, ['images_transformed', 'masks_transformed'])

    return dict_transformation

def reshape_to_4d(dict_transformation:dict, keys:list):
    for key in keys:
        dict_transformation[key] = np.reshape(
            dict_transformation[key],
            (dict_transformation[key].shape[0], dict_transformation[key].shape[1], dict_transformation[key].shape[2], 1)
        )

def reshape_to_5d(dict_transformation:dict, keys:list):
    for key in keys:
        dict_transformation[key] = np.reshape(
            dict_transformation[key],
            (dict_transformation[key].shape[0], dict_transformation[key].shape[1], dict_transformation[key].shape[2], dict_transformation[key].shape[3], 1)
        )

def create_train(patient_train:list,k:int, pourcent_ratio:float, nb_patch:int,path:Path):    
    """ 
    Args:
        patient_train (list) : list of patients
        k (int): number of patches per lesion
        pourcent_ratio (float): ratio of lesion to be considered as positive
        nb_patch (int) : number of patches per lesion
        path (Path) : path to the individual's folder
    Returns:
        liste_patch : list of patches for all individuals
        liste_y : list of associated target values
    """        
    list_patch=[]
    list_y=[]
    for patient in patient_train:
        print(f"Processing patient: {patient} .......................................................")
        
        # Load and preprocess the images and masks
        image3D = io.imread(os.path.join(path, patient,'image.tif')).T
        pd_image = padding(image3D, k)
        mask = io.imread(os.path.join(path, patient, 'max.tif')).T
        pd_mask = padding(mask, k)
        
        # Generate patches and target values
        l_value, l_patch = random_patch(mask, 
                                        k, 
                                        pourcent_ratio, 
                                        nb_patch, image3D, 
                                        pd_image, 
                                        pd_mask,
                                        os.path.join(path,patient))
        
        # Append results to the lists
        list_patch.extend(l_patch)
        list_y.extend(l_value)
        
    return list_patch, list_y

def normalize_image(image):
    """Normalize an image to have pixel values between 0 and 1 using (x - min) / (max - min)."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def normalize_batch(images):
    """Normalize a batch of images."""
    return np.array([normalize_image(image) for image in images])
    
def change_name(list_files:list):
    """Rename files to ensure they can be opened in the correct order.
    
    The file name should end with _XX or _XXX where X is a number.

    Args:
        list_files (list): List of file names to be renamed.
    """
    for file_name in list_files:
        if file_name[-2] == '_':
            new_name = f"{file_name[:-1]}00{file_name[-1]}"
            os.rename(file_name, new_name)

        elif file_name[-3] == '_':
            new_name = f"{file_name[:-2]}0{file_name[-2:]}"
            os.rename(file_name, new_name)

def image_mask_ref(path_data: Path) -> dict:
    """ Extract images and masks for all patients in path_data

    Args:
        path_data (Path): path to the folder containing all patients

    Returns:
        dict: dictionary with list of images, list of masks and list of patients
    """           
    data = pd.read_csv("output.csv", sep=';', names=['patient','event','time'])
    data.index = list(data.loc[:,'patient'])
    data.drop(['patient'], axis = 1, inplace=True)
    
    # Initialize dictionary to hold images, masks, and references
    dict_input = {'list_images': [], 'list_masks': [], 'list_refs': []}

    # Process each patient's data
    for ref_patient in tqdm(data.index): # each folder contains images for the lesion of the patient
        list_files = os.listdir(os.listdir(os.path.join(path_data, ref_patient)))
        
        if 'dcm'  in list_files and 'max' in list_files:
            try:
                if 'image.tif' not in list_files:
                    patients_list = Patient(ref_patient, path_data)
                    correct = patients_list.correct
                    np_image = sitk.GetArrayFromImage(patients_list.image)
                    tifffile.imsave(os.path.join(path_data,ref_patient, "image.tif"), np_image.T)
                    
                else:
                    correct = correction(ref_patient)
                    np_image = io.imread(os.path.join(path_data, ref_patient, 'image.tif')).T                    
                
                if 'max.tif' not in list_files:
                    path_all = makeTifFromPile(os.path.join(path_data,ref_patient, "max"),correct)
                    mask = io.imread(path_all).T
                
                else:
                    mask_path = os.path.join(path_data,ref_patient, "max.tif")
                    mask = io.imread(mask_path).T
                    
                dict_input['list_refs'].append(ref_patient)                    
                dict_input['list_images'].append(np_image)                  
                dict_input['list_masks'].append(mask)
                
            except Exception as e:
                print(f"Error processing {ref_patient}: {e}")
                continue
            
    return dict_input
    
def padding(image: np.ndarray, k: int) -> np.ndarray:
    """Apply padding to the image to handle edges.

    Args:
        image (np.ndarray): The image to be padded.
        k (int): Size of the padding is 2*k.

    Returns:
        np.ndarray: The padded image.
    """

    padded_image = np.pad(image, ((0, 2*k), (0, 2*k), (0, 2*k)), mode='symmetric')
    return padded_image

def read_output():
    """Read 'output.csv' and return a dataframe with event and time.

    Returns:
        pd.DataFrame: DataFrame containing event and time data.
        list: List of patient identifiers.
    """
    data = pd.read_csv("output.csv", sep=';', names=['patient', 'event', 'time'])
    patient = data['patient'].tolist()
    data.index = patient
    data.drop(columns=['patient'], inplace=True)
    return data, patient

def interpolation3D(liste_patch:list, interpolation_size:int, third_dim:int=3, sess=None):
    """Resize images in liste_patch to interpolation_size and interpolate them.

    Args:
        liste_patch (list): List of image patches to be interpolated.
        interpolation_size (int): Size to interpolate the images to.
        third_dim (str or int): Dimension to handle the interpolation ('3D' for 3D unpooling).
        sess (tf.Session, optional): TensorFlow session for 3D unpooling. Defaults to None.

    Returns:
        list: List of interpolated image patches.
    """
    if interpolation_size is None:
        return liste_patch

    liste_interpolate = []
    print('Interpolation started')

    if third_dim != '3D':
        for patch in liste_patch:
            image = sitk.GetImageFromArray(np.array(patch))
            
            resampleImageFilter = sitk.ExpandImageFilter()
            resampleImageFilter.SetExpandFactor(4 / np.array(patch)[1])
            resampleImageFilter.SetInterpolator(sitk.sitkBSplineResamplerOrder2)
            resultedImage = resampleImageFilter.Execute(image)
            
            arr_image = sitk.GetArrayFromImage(resultedImage)
            
            if third_dim == 3:
                arr_image = np.array(arr_image)[:, :, interpolation_size // 2 - 1:interpolation_size // 2 + 2]
            
            liste_interpolate.append(arr_image)
    else:
        def unpool_3D(input_layer, x, y, z):
            return tf.keras.layers.UpSampling3D(size=(x, y, z))(input_layer)
        
        y = interpolation_size / np.shape(liste_patch[0])[0]
        
        liste_interpolate = unpool_3D(liste_patch, int(y), int(y), int(y)).eval(session=sess)

    return liste_interpolate

def shuffle_and_interpolate(list_patch:list, 
                            list_y:list, 
                            list_refs: list = None,
                            interpolation_size:int = 36, 
                            apply_interpolation:bool=True, 
                            list_masks:list = [],
                            third_dim:int = 3,
                            sess = None):
    """
    Interpolate a list of images and shuffle them along with their associated true values and references.

    Args:
        list_patch (list): List of image patches.
        list_y (list): List of true values associated with the patches.
        list_refs (list, optional): List of reference values.
        interpolation_size (int, optional): The size to interpolate the image patches to Default to 36.
        apply_interpolation (bool, optional): Flag to apply interpolation. Defaults to True.
        list_masks (list, optional): List of masks. Defaults to an empty list.
        third_dim (int, optional): Third dimension for the interpolation. Defaults to 3.
        sess (optional): TensorFlow session for interpolation. Defaults to None.

    Returns:
        tuple: Shuffled and interpolated patches, true values, references  and masks as applicable.
    """

    list_patch = np.array(list_patch)
    list_refs = np.array(list_refs)
    list_y = np.array(list_y)

    if list_masks is not None and len(np.shape(list_masks)) != 1:
        list_masks = np.array(list_masks)
        
    if apply_interpolation:
        list_patch = interpolation3D(list_patch, interpolation_size, third_dim, sess=sess)

    indices = np.arange(len(list_y))
    np.random.shuffle(indices)

    shuffled_ref = list_refs[indices]
    shuffled_y = list_y[indices]
    shuffled_patches = list_patch[indices]

    if list_masks is not None and len(list_masks.shape) != 1:
        shuffled_masks = list_masks[indices]
        return shuffled_patches, shuffled_y, shuffled_masks, shuffled_ref

    return shuffled_patches, shuffled_y, shuffled_ref
    

def ground_truth_processing(data, list_patients:list = None, classes = None):
    # Filter the dataframe to include only the patients in list_patients
    y = data[data.index.isin(list_patients)].copy()
    
    # Create a copy of the filtered dataframe and prepare the group column
    y_grp = y.drop(columns='time').rename(columns={'event': 'group'})
    
    for pat in y_grp.index:
        match classes:
            case ['TWO', 'TWOWC']:
                if y.loc[pat, 'time'] <= 1095:
                    y_grp.loc[pat, 'group'] = 0 if y.loc[pat, 'event'] == 1 else 2
                else:
                    y_grp.loc[pat, 'group'] = 1
            case 'THREE':
                if y.loc[pat, 'time'] <= 803:
                    y_grp.loc[pat, 'group'] = 0
                elif y.loc[pat, 'time'] > 1606:
                    y_grp.loc[pat, 'group'] = 2
                else:
                    y_grp.loc[pat, 'group'] = 1
            case 'YEAR' | 'YEARWC':  # group 1: less than 1 year, 5: between 4 and 5 years, 10: censorship
                if y.loc[pat, 'event'] == 1 and y.loc[pat, 'time'] // 365 <= 5:
                    y_grp.loc[pat, 'group'] = y.loc[pat, 'time'] // 365
                elif y.loc[pat, 'time'] // 365 >= 5:
                    y_grp.loc[pat, 'group'] = 5
                else:
                    y_grp.loc[pat, 'group'] = 6
            case _:
                y_grp = data
        
    y_grp.to_csv('y_grp.csv')
    
    return y_grp


def ground_truth_processing3(y,classe): #binaire entre CLASSE 0 et autres
    y= pd.DataFrame(y) 
    y.columns = ('event','time')
    y_grp = y.copy()
    y_grp = y_grp.drop('time',axis=1)
    y_grp=y_grp.rename(columns={'event': 'group'})
    
    for pat in y_grp.index:
        if y.loc[pat,'time']<= 1095 :
            y_grp.loc[pat,'group'] = 0
        else :
             y_grp.loc[pat,'group'] = 1
    y_grp.to_csv('y_grp.csv')
    y_grp = to_categorical(y_grp.values)
    return y_grp



def ground_truth_processing4(y,classes):
    y= pd.DataFrame(y) 
    y.columns = ('event','time')
    y_grp = y.copy()
    y_grp = y_grp.drop('time',axis=1)
    y_grp=y_grp.rename(columns={'event': 'group'})
    
    for pat in y_grp.index:
        if classes == 'TWO' or classes == 'TWOWC' :
            if y.loc[pat,'time']<= 1095 :
                if  classes == 'TWO':
                    y_grp.loc[pat,'group'] = 0
                else:
                    if y.loc[pat,'event'] == 1:
                        y_grp.loc[pat,'group'] = 0
                    else :
                        y_grp.loc[pat,'group'] = 2  #♠suppr la ou c'est 2 quand wc
            else :
                 y_grp.loc[pat,'group'] = 1
        elif classes == 'THREE':
            if y.loc[pat,'time']<= 730 :
                y_grp.loc[pat,'group'] = 0
            elif y.loc[pat,'time'] >= 1460  :
                y_grp.loc[pat,'group'] = 2 
            else :
                 y_grp.loc[pat,'group'] = 1  
        elif classes == 'THREEWC':
            if y.loc[pat,'time']<= 730 :
                if y.loc[pat,'event'] == 1:
                    y_grp.loc[pat,'group'] = 0
                else:
                    y_grp.loc[pat,'group'] = 10
            elif y.loc[pat,'time'] >= 1460  :
                y_grp.loc[pat,'group'] = 2 
            else :
                if y.loc[pat,'event'] == 1:
                    y_grp.loc[pat,'group'] = 1
                else:
                    y_grp.loc[pat,'group'] = 10
        elif classes == 'YEAR' or classes == 'YEARWC' : # group 0 : less than 1 year, 4 : between 4 and 5 years, 
            if y.loc[pat,'time']//365 <= 5:
                if y.loc[pat,'event'] == 1 and classes == 'YEARWC' :
                    y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
                elif classes == 'YEAR':
                    y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
                    
                else:
                    y_grp.loc[pat,'group'] = 10
            else:
                y_grp.loc[pat,'group'] = 6
        else:
            y_grp = y
    y_grp.to_csv('y_grp.csv')
    y_grp = to_categorical(y_grp.values)
    return y_grp




def ground_truth_processing2(y,classe):
    if classe == 3:
        classes = 'THREE'
    elif classe == 7 or classe == 5:
        classes = 'YEAR'
    y= pd.DataFrame(y) 
    y.columns = ('event','time')
    
    y_grp = y.copy()
    
    y_grp = y_grp.drop('time',axis=1)
    y_grp=y_grp.rename(columns={'event': 'group'})
    
    for pat in y_grp.index:
        if classes == 'TWO' or classes == 'TWOWC' :
            if y.loc[pat,'time']<= 1095 :
                if y.loc[pat,'event'] == 1:
                    y_grp.loc[pat,'group'] = 0
                else :
                    y_grp.loc[pat,'group'] = 2  
            else :
                 y_grp.loc[pat,'group'] = 1
        elif classes == 'THREE':
            if y.loc[pat,'time']<= 730 :
                y_grp.loc[pat,'group'] = 0
            elif y.loc[pat,'time'] >= 1460  :
                y_grp.loc[pat,'group'] = 2 
            else :
                 y_grp.loc[pat,'group'] = 1            
        elif classes == 'YEAR' or classes == 'YEARWC' : # group 0 : less than 1 year, 4 : between 4 and 5 years, 
            if y.loc[pat,'time']//365 <= 5:
                y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
            else:
                y_grp.loc[pat,'group'] = 6
        else:
            y_grp = y
    y_grp.to_csv('y_grp.csv')
    y_grp = to_categorical(y_grp.values)
    return y_grp

def separation(folder, list_patients, rates=[0.5,0.25,0.25], outnames = "./output.csv", sep=';', classes_method='all'): 
    """
    Separate patients into groups with balanced classes and save the results as .npy files.
    
    Args:
    - folder (str): Folder path where the output files will be saved.
    - list_patients (list): List of patient names/IDs.
    - rates (list): List of rates for splitting the patients. Default is [0.5, 0.25, 0.25].
    - outnames (str): Output filename for classified patients data. Default is "./output.csv".
    - sep (str): Separator for the CSV file. Default is ';'.
    - classes_method (str): Class category for filtering patients. Default is 'all'.
    
    Returns:
    - list_references (list): List of lists with patient IDs separated into folds.
    """
    
    def fill_classes(to_remove:int=None):
        return [str(i) for i in range(7) if i is not to_remove] + [f"{i}Censorship" for i in range(7) if i is not to_remove]


    #random.shuffle(list_patients)
    list_names = [os.path.join(folder, f"r{i}.npy") for i in range(len(rates))]
        
    if all(os.path.exists(name) for name in list_names) :   
        # Load existing files
        list_references = [np.load(name) for name in list_names]

    else:  
        class_map = {'all':fill_classes()}
        for c in range(7):
            class_map[f"without_{c}"] = fill_classes(c)
            class_map[f"class_{c}"] = [str(c), f"{c}_censorship"]
        
        name_classes = class_map[classes_method]

        y_grp = classify_patients(list_patients, outnames= outnames, sep=sep)

        list_classes=[y_grp.loc[y_grp['group'] == classes].index for classes in name_classes]

        list_references=[[] for _ in range(len(rates))]
        fold_index = 0

        # distribute patients from each classes in each fold 
        for classes in range(len(list_classes)):
            patient_index = 0
            # Continue while there are patients in the current class
            while patient_index < len(list_classes[classes]):
                # Add the patient to the current fold
                list_references[fold_index].append(list_classes[classes][patient_index])
                patient_index+=1
                fold_index+=1
                # Reset fold_index if it exceeds the number of folds
                if fold_index >=len(rates):
                    fold_index=0

        list_references=[random.sample(list_references[i],len(list_references[i])) for i in range(len(list_references))]
   
        # Save the groups to .npy files
        for i, group in enumerate(list_references):
            np.save(os.path.join(folder, f"r{i}.npy"), np.array(group))

    return list_references



def classify_patients(list_patients, outnames = "output.csv", sep=';') :
   # Load the data from the CSV file
    data = pd.read_csv(outnames, sep=sep, names=['patient', 'event', 'time'])
    
    # Rename the index with patient names
    patient_names = data['patient'].tolist()
    data.index = patient_names
    data.drop(columns=['patient'], inplace=True)
    
    # Filter the data to include only the patients in list_patients
    y = data.loc[data.index.isin(list_patients)].copy()
    
    # Create a new dataframe for the groups
    y_grp = y.drop(columns='time').rename(columns={'event': 'group'})
    
    # Classify patients into groups
    for pat in y_grp.index:
        event = y.loc[pat, 'event']
        time = int(y.loc[pat, 'time']) // 365
        y_grp.loc[pat, 'group'] = str(time) if event == '1' or event == 1 else f"{time}Censorship"
    
    return y_grp


def balanced_separation(list_patients, file_images, rates=[0.5,0.25,0.25], classe = 'YEAR', rerun_separation = True, outnames = "./output.csv"): 
    # Shuffle the list of patients
    random.shuffle(list_patients)
    
    # Load survival data
    data = pd.read_csv(outnames, sep=';', names=['patient', 'event', 'time'])
    data.index = data['patient']
    data.drop(columns=['patient'], inplace=True)
    
    # correspondance between classe and number of group
    number_of_classes = {'YEAR': 7, 'YEARWC': 6, 'TWO': 3, 'surv': 7}.get(classe, 2)
    
    # Process survival data for classification
    if classe == 'surv':
        y = data.loc[data.index.isin(list_patients)].copy()
        y_grp = y.drop(columns='time').rename(columns={'event': 'group'})
        y_grp['group'] = y['time'] // 365
        
    else:
        y_grp = ground_truth_processing(data, list_patients, classe)
    
    # Create lists to store patients for each class
    C = [[] for _ in range(noC)]
    for patient in list_patients:
        C[int(y_grp.loc[patient, 'group'])].append(patient)
    
    # Define filenames for saving/loading splits
    filenames = [os.path.join(file_images, f"{name}.npy") for name in ["Balance", "Ate", "Aval"]]

    # Load existing splits if available and rerun_separation is False
    if all(os.path.exists(name) for name in filenames) and not rerun_separation:
        fold_train, fold_test, fold_validation = (np.load(name) for name in filenames)
    else:
        # Create new splits
        fold_train, fold_test, fold_validation = [], [], []
        for group in C:
            n = len(group)
            split1 = int(0.5 * n)
            split2 = (n - split1) // 2
            fold_train.extend(group[:split1])
            fold_test.extend(group[split1:split1 + split2])
            fold_validation.extend(group[split1 + split2:])
        
        # Shuffle the splits
        fold_train = np.random.permutation(fold_train)
        fold_test = np.random.permutation(fold_test)
        fold_validation = np.random.permutation(fold_validation)

        # Save the splits
        for name, data in zip(filenames, [fold_train, fold_test, fold_validation]):
            np.save(name, data)
    
    return [fold_train, fold_test, fold_validation]


def save_training_plots(fitting, title, eval_train, eval_val, resu, file_images, metrics):
    if fitting is not None:
        # Plot training And validation accuracy values
        plt.figure(figsize=(15,5), dpi=70)
        
        title_map = {
            'accuracy': '_accuracy',
            'rankAndMSE': ['_mse loss', '_rank loss', '_cindex'],
            'coxAndrank': ['_rank loss', '_cox loss', '_cindex'],
            'coxAndclassif': ['_tf_cindex', '_accuracy', '_classif_loss', '_cox_loss'],
            'coxAndmse': ['_mse loss', '_cox loss', '_cindex'],
            'tf_cindex': '_cindex',
            'tf_cindexTD': '_cindex',
            'triplet': ['_cindex', '_accuracy'],
            'RankAnddiscret': ['_rank loss', '_cindex', '_surv_likelihood']
        }
        metric_map = {
            'accuracy': ['acc', 'val_acc'],
            'rankAndMSE': ['mse', 'val_mse', 'rank', 'val_rank', 'tf_cindexT', 'val_tf_cindexT'],
            'coxAndrank': ['rank', 'val_rank', 'cox', 'val_cox', 'tf_cindexR', 'val_tf_cindexR'],
            'coxAndclassif': ['Partial_likelihood_tf_cindexR', 'val_Partial_likelihood_tf_cindexR',
                              'Cross_entropy_loss_acc', 'val_Cross_entropy_loss_acc',
                              'Cross_entropy_loss_loss', 'val_Cross_entropy_loss_loss',
                              'Partial_likelihood_loss', 'val_Partial_likelihood_loss'],
            'coxAndmse': ['mse', 'val_mse', 'cox', 'val_cox', 'tf_cindexR', 'val_tf_cindexR'],
            'tf_cindex': ['tf_cindexR', 'val_tf_cindexR'],
            'tf_cindexTD': ['tf_cindexTD', 'val_tf_cindexTD'],
            'triplet': ['tf_cindexT', 'val_tf_cindexT', 'acc', 'val_acc'],
            'RankAnddiscret': ['rankL', 'val_rankL', 'tf_cindexTD', 'val_tf_cindexTD', 'discretcox', 'val_discretcox']
        }
        
        plots = metric_map.get(metrics, [])
        titles = title_map.get(metrics, '')
        for idx, plot in enumerate(plots, start=2):
            plt.subplot(2, 2, idx)
            plt.plot(fitting.history[plot], label="train")
            plt.plot(fitting.history[f'val_{plot}'], label="valid")
            plt.title(title + titles[idx - 2])
            plt.legend(loc='upper left')
            plt.ylabel(plot.replace('_', ' ').title())
            plt.xlabel('Epoch')

        if metrics in ['cox', 'accuracy', 'tf_cindexTD']:
            plt.subplot(1, 2, 1)
        elif metrics == 'coxAndclassif':
            plt.subplot(3, 2, 1)
        else:
            plt.subplot(2, 2, 1)

        plt.plot(fitting.history['loss'], label="train")
        plt.plot(fitting.history['val_loss'], label="valid")
        plt.title(title + '_loss')
        plt.legend(loc='upper left')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')

        plt.savefig(os.path.join(file_images, f'{title}_curves.png'))
        plt.close('all')

    results = pd.DataFrame(index=['train', 'val'], columns=['Loss', metrics])
    results.loc['train', 'Loss'] = eval_train[0]
    results.loc['train', metrics] = eval_train[1]
    results.loc['val', 'Loss'] = eval_val[0]
    results.loc['val', metrics] = eval_val[1]

    if resu is not None:
        results.loc['test', metrics] = resu

    return results

def makeTifFromPile(path_to_pile:Path, correct):
    """Takes an absolute path containing a pile of masks, compute the resulting .tif mask and
     output the path to the created .tif mask

    Args:
        path_to_pile (Path): path to pile of masks
        correct (_type_): _description_

    Raises:
        ValueError: Unknown mask type in path: must contain 'max' or 'tous'.

    Returns:
        Path: path to tif mask
    """

    # Collect all mask files from the given directory
    pile_files = [
        os.path.join(dirpath, file)
        for dirpath, _, filenames in os.walk(path_to_pile)
        for file in filenames if not file.startswith('.')
    ]
    
    change_name(pile_files)
    
    # Recollect files after renaming
    pile_files = [
        os.path.join(dirpath, file)
        for dirpath, _, filenames in os.walk(path_to_pile)
        for file in filenames if not file.startswith('.')
    ]
    
    pile_files.sort()
    pile_files = pile_files[correct['avant']:len(pile_files)-correct['apres']]
    
    # Determine the shape of the images
    first_file = pile_files[0]
    # get the shape of the image
    with open(first_file, mode='r', encoding='utf-8') as tif_file:
        tif_file.readline()  # first line is junk data
        x_shape, y_shape = map(int, get_words(tif_file.readline()))  

    num_files = len(pile_files)
    mask_array = np.zeros((num_files, x_shape, y_shape), dtype=np.uint8)
    
    # Process each file and fill the mask array
    for file_index, pile_file in enumerate(pile_files):
        with open(pile_file, mode='r', encoding='utf-8') as tif_file:
            for _ in range(3):  # Skip the first three lines
                tif_file.readline()
                # first line is junk line, second line is shape of the dcm pile
                # third line is junk data
            for row_index in range(x_shape):
                col_index = 0
                while col_index < y_shape:
                    val = tif_file.read(1)
                    if val in '01':  # Only consider '0' and '1'
                        mask_array[file_index, row_index, col_index] = int(val)
                        col_index += 1

    # Determine the output file path
    path_to_lesion = os.path.abspath(os.path.join(path_to_pile, os.pardir))
    if 'max' in path_to_pile:
        path_to_tif_mask = os.path.join(path_to_lesion, 'max.tif')
    elif 'tous' in path_to_pile:
        path_to_tif_mask = os.path.join(path_to_lesion, 'all.tif')
    else:
        raise ValueError("Unknown mask type in path: must contain 'max' or 'tous'.")

    # Save the mask array as a .tif file
    imsave(path_to_tif_mask, mask_array)
    
    return path_to_tif_mask


def display_predictions(name, y, prediction, writer, sess, file_images, titleP, log_dir, number=1000):
    """
    Displays and logs the predictions, comparing them with the true values.
    
    Args:
        name (str): Name of the prediction model.
        y (array-like): True values.
        prediction (array-like): Predicted values.
        writer (tf.summary.FileWriter): TensorBoard writer.
        sess (tf.Session): TensorFlow session.
        file_images (str): Directory to save plots.
        titleP (str): Title prefix for plots.
        log_dir (str): Directory for TensorBoard logs.
        number (int, optional): Number of samples to display. Default is 1000.
    
    Returns:
        Matplotlib axis with the plot.
    """
    
    number = min(len(y), number)
    prediction = np.reshape(np.array(prediction), (-1, 1)) if len(prediction.shape) == 1 else prediction

    rank = prediction[:number, 0].argsort()
    ranked_array = np.array([y[rank, 0], y[rank, 1], prediction[rank, 0]])

    evalu = [
        tf.convert_to_tensor([str(ranked_array[0, i]) for i in range(number)]),
        tf.convert_to_tensor([str(ranked_array[1, i]) for i in range(number)]),
        tf.convert_to_tensor([str(ranked_array[2, i]) for i in range(number)])
    ]
    summary_text = sess.run(tf.summary.text(f'Ranked{name}Prediction', tf.stack(evalu)))
    writer.add_summary(summary_text)
    
    plot_path = os.path.join(file_images, f'{titleP}{name}Pred.png')
    plt.plot(ranked_array[2, np.where(ranked_array[0, :] == 0)], ranked_array[1, np.where(ranked_array[0, :] == 0)], 'ro', label='with censorship')
    plt.plot(ranked_array[2, np.where(ranked_array[0, :] == 1)], ranked_array[1, np.where(ranked_array[0, :] == 1)], 'bo', label='without censorship')
    plt.xlabel(f'{name} prediction')
    plt.ylabel('real time')
    plt.title(f'Time according to {name} prediction (red: censorship)')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    curves = plt.imread(plot_path)
    reshaped_curves = np.reshape(curves, (1, *curves.shape))
    summary_image = sess.run(tf.summary.image(f'{name}predictions', tf.convert_to_tensor(reshaped_curves)))
    writer.add_summary(summary_image)
    
    return plt.gca()


def get_optimal_separation(ytrain, predictions_train) -> dict:
    """
    Determines the optimal separation threshold for the given training data and predictions by
    performing log-rank tests to identify the threshold with the best p-value.

    Args:
        ytrain (array-like): Training data with columns [event, time].
        predictions_train (array-like): Predictions corresponding to the training data.

    Returns:
        dict: Contains the best p-value, test statistic, best separation threshold, 
               number of samples in group 0, and number of samples in group 1.
    """
    predictions = predictions_train[:, 0] if predictions_train.ndim == 2 else predictions_train
    df = pd.DataFrame({
        'event': ytrain[:, 0],
        'time': ytrain[:, 1],
        'prediction': predictions,
        'group': np.zeros(len(ytrain))
    })

    best = {
        'pval': float('inf'),
        'stat': 0,
        'separation': float('inf'),
        'num_g0': 0,
        'num_g1': 0
    }

    best_alt = {
        'pval': float('inf'),
        'stat': 0,
        'separation': float('inf'),
        'num_g0': 0,
        'num_g1': 0
    }

    sep_step = (np.max(predictions) - np.min(predictions)) / 20

    for i in range(1, 20):
        sep_threshold = np.min(predictions) + sep_step * i
        group1 = df[df['prediction'] <= sep_threshold]
        group2 = df[df['prediction'] > sep_threshold]

        if len(group1) >= 0.1 * len(df) and len(group2) >= 0.1 * len(df):
            T1, E1 = group1['time'], group1['event']
            T2, E2 = group2['time'], group2['event']
            results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
            if results.p_value < best['pval']:
                best.update({
                    'pval': results.p_value,
                    'stat': results.test_statistic,
                    'separation': sep_threshold,
                    'num_g0': len(group1),
                    'num_g1': len(group2)
                })
                
        elif len(group1) >= 2 and len(group2) >= 2:
            T1, E1 = group1['time'], group1['event']
            T2, E2 = group2['time'], group2['event']
            results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
            if results.p_value < best_alt['pval']:
                best_alt.update({
                    'pval': results.p_value,
                    'stat': results.test_statistic,
                    'separation': sep_threshold,
                    'num_g0': len(group1),
                    'num_g1': len(group2)
                })

    if best['pval'] == float('inf'):
        best = best_alt
            
    return best



def group_by_optimal_separation(ytrain, predictions_train, optimal_separation, file_images, titleP, name, sess, typee='risk'):
    """
    Groups data based on the optimal separation threshold and computes log-rank test statistics.
    Optionally plots Kaplan-Meier curves and saves them as images.

    Args:
        ytrain (array-like): Training data with columns [event, time].
        predictions_train (array-like): Predictions corresponding to the training data.
        optimal_separation (float): Optimal separation threshold.
        file_images (str): Directory to save images and logs.
        titleP (str): Prefix for saved image files.
        name (str): Name identifier for the plot.
        sess (tf.Session): TensorFlow session for summary writing.
        typee (str, optional): Type of analysis ('risk' or 'time'). Defaults to 'risk'.

    Returns:
        tuple: Contains p-value from the log-rank test, best separation threshold,
               number of samples in group 0, and number of samples in group 1.
    """
    if len(np.shape(predictions_train)) == 2:
        predd = predictions_train[:, 0]
    else:
        predd = predictions_train

    df = pd.DataFrame({
        'event': ytrain[:, 0],
        'time': ytrain[:, 1],
        'prediction': predd,
        'group': np.zeros(len(ytrain))
    })

    groupe1 = df[df['prediction'] <= optimal_separation]
    groupe2 = df[df['prediction'] > optimal_separation]

    results = logrank_test(groupe1['time'], groupe2['time'], event_observed_A=groupe1['event'], event_observed_B=groupe2['event'])
    pval = results.p_value
    numG0 = len(groupe1)
    numG1 = len(groupe2)

    kmf = KaplanMeierFitter()

    if typee == 'time':
        kmf.fit(groupe1['time'], groupe1['event'], label='bad prognosis')
        axa = kmf.plot()
        kmf.fit(groupe2['time'], groupe2['event'], label='good prognosis')
        axa = kmf.plot(ax=axa)
    else:
        kmf.fit(groupe1['time'], groupe1['event'], label='good prognosis')
        axa = kmf.plot()
        kmf.fit(groupe2['time'], groupe2['event'], label='bad prognosis')
        axa = kmf.plot(ax=axa)

    kaplan_figure = os.path.join(file_images, f"{titleP}{name}Kaplan.png")
    axa.get_figure().savefig(kaplan_figure)
    plt.close()
    curves2 = plt.imread(kaplan_figure)

    log_dir = os.path.join(file_images, "logs", "fit", titleP)
    writer = tf.summary.FileWriter(log_dir)
    fi2 = tf.convert_to_tensor(np.reshape(curves2, (1, *np.shape(curves2))))
    s = sess.run(tf.summary.image(f'{name}kaplanMeier', fi2))
    writer.add_summary(s)
    plt.close()

    return pval, optimal_separation, numG0, numG1