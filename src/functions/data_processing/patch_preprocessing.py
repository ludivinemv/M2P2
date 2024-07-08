# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

Extract patches to pass from 3D to 2D
"""
# Image processing libraries
from PIL import Image
from skimage import io
from skimage.external import tifffile
import SimpleITK as sitk

# Data handling libraries
import pandas as pd
import numpy as np

# File and OS management
import os

# Custom modules
from functions.Patient import Patient
from Patient import correction
from data_processing import *

class PatchPreprocessing:
    
    def __init__(self, path, inter_s, method='box', reload_patches=True, d3=False, npatch=9, name_i='image.tif', sess=None):
        """
        Initializes the PatchPreprocessing class.
        
        Args:
            path (str): The path to the data.
            inter_s (str): The interaction strategy.
            method (str, optional): The method to use for patch extraction. Defaults to 'box'.
            reload_patches (bool, optional): Whether to reload patches. Defaults to True.
            d3 (bool, optional): Flag for 3D processing. Defaults to False.
            npatch (int, optional): Number of patches. Defaults to 9.
            name_i (str, optional): The name of the image file. Defaults to 'image.tif'.
            sess (optional): Session information, if any. Defaults to None.
        """
        self.inter_s = inter_s
        self.method = method
        self.reload_patches = reload_patches
        self.d3 = d3
        self.npatch = npatch
        self.name_i = name_i
        self.sess = sess
        
        # Determine the processing function based on the image name
        if name_i == 'image.tif':
            self.data, self.liste_patches, self.liste_ref, self.liste_patients, self.errors, self.liste_mask = processing_patches(
                path, self.inter_s, self.method, self.reload_patches, self.d3, self.npatch, self.name_i, self.sess
            )
        else:
            self.data, self.liste_patches, self.liste_ref, self.liste_patients, self.errors, self.liste_mask = processing_patchesHN(
                sess, path, self.inter_s, self.method, self.reload_patches, self.d3, self.npatch, self.name_i
            )

"""###############################################################################
######################### Interpolation       ####################################
###############################################################################"""

def interpolate_patches(patch_list, ref_list, show_image, interpolation_size, is_3d=False):
    """
    Interpolates the patches to the specified size using bicubic interpolation.

    Args:
        patch_list (list): List of patches to be interpolated.
        ref_list (list): List of reference values (not used in this function).
        show_image (bool): Flag to determine if the image should be shown (not used in this function).
        interpolation_size (int): The size to interpolate the patches to. If None, no interpolation is performed.
        is_3d (bool, optional): Flag for 3D processing (not used in this function). Defaults to False.

    Returns:
        list: List of interpolated patches.
    """
    # Check if interpolation is needed
    if interpolation_size is not None:
        interpolated_list = []

        # Iterate through the list of patches
        for patch in patch_list:
            # Convert the patch to an image
            img = Image.fromarray(patch)

            # Resize the image using bicubic interpolation
            resized_img = img.resize((interpolation_size, interpolation_size), Image.BICUBIC)

            # Convert the resized image back to an array and append to the interpolated list
            interpolated_list.append(np.asarray(resized_img))

        return interpolated_list

    # If no interpolation is needed, return the original list of patches
    return patch_list

    
def all_size_patches(np_image, mask, patch_size, is_3d='False', num_patches=9):
    """
    Extract patches of the specified size from the given image and mask.

    Args:
        np_image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask indicating the region of interest.
        patch_size (int): The size of the patch.
        D3 (str, optional): Flag for 3D processing. Defaults to 'False'.
        num_patches (int, optional): Number of patches to extract. Defaults to 9.

    Returns:
        list: List of image patches.
        list: List of mask patches.
    """        
    patches_list, masks_list = [], []
    
    # Find the coordinates of non-zero regions in the mask
    coords = np.where(mask != 0)
    center_box = [min(coords[i]) + int((max(coords[i]) - min(coords[i])) / 2) for i in range(3)]
    half_size = int(patch_size / 2)
    left, right = [0] * 3, [0] * 3
    
    for i in range(3):
        if center_box[i] < half_size:
            left[i] = 0
            right[i] = patch_size
        elif (np.shape(np_image)[i] - center_box[i]) < half_size:
            right[i] = np.shape(np_image)[i] - 1
            left[i] = np.shape(np_image)[i] - 1 - patch_size
        else:
            left[i] = center_box[i] - half_size
            right[i] = center_box[i] + half_size

    new_image = np_image[left[0]:right[0], left[1]:right[1], left[2]:right[2]]
    new_mask = mask[left[0]:right[0], left[1]:right[1], left[2]:right[2]]

    if D3=='False': 
        coords = np.where(new_mask != 0)
        for d in range(3):
            if num_patches != 1 or (num_patches == 1 and d == 0):
                c = int((max(coords[d]) - min(coords[d])) / 4)
                p1, p2, p3 = int(min(coords[d]) + c), int(min(coords[d]) + 2 * c), int(min(coords[d]) + 3 * c)

                if d == 0:
                    patch1 = new_image[p1,:,:]
                    patch2 = new_image[p2,:,:]
                    patch3 = new_image[p3,:,:]
                    patch4 = new_mask[p1,:,:]
                    patch5 = new_mask[p2,:,:]
                    patch6 = new_mask[p3,:,:]
                if d==1 :
                    patch1 = new_mask[:,p1,:]
                    patch2 = new_mask[:,p2,:]
                    patch3 = new_mask[:,p3,:]
                    patch4 = new_mask[:,p1,:]
                    patch5 = new_mask[:,p2,:]
                    patch6 = new_mask[:,p3,:]
                if d==2 :
                    patch1 = new_image[:,:,p1]
                    patch2 = new_image[:,:,p2]
                    patch3 = new_image[:,:,p3]
                    patch4 = new_mask[:,:,p1]
                    patch5 = new_mask[:,:,p2]
                    patch6 = new_mask[:,:,p3]
                if num_patches == 9:
                    patches_list.append(patch1)
                    masks_list.append(patch4)
                    patches_list.append(patch3)
                    masks_list.append(patch6)
                
                patches_list.append(patch2)
                masks_list.append(patch5)
    else: 
        patches_list.append(new_image)
        masks_list.append(new_mask)
        
    return patches_list, masks_list

def patches(np_image,mask,D3='False',npatch=9): #k the size/2 of the patches 
        l_patch=[]  
        l_mask=[]
        #récupérer 3 patchs par lésions
        a=np.where(mask != 0)
        #for each direction have 5 patchs with the index pi
        New_im = np_image[min(a[0]):max(a[0]),min(a[1]):max(a[1]),min(a[2]):max(a[2])]
        shap = np.shape(New_im)
        k = np.max(shap)
        if D3 == False:
            for d in range(3):
                if npatch !=1 or (npatch==1 and d==0):
                    c=int((max(a[d])-min(a[d]))/4)
                    p1=int(min(a[d])+c)
                    p2=int(min(a[d])+2*c)
                    p3=int(min(a[d])+3*c)
                    # to have the bounders of the pixel 
                    if d==0 :
                        dir1=2
                        dir2=1
                    if d==1 :
                        dir1=0
                        dir2=2
                    if d==2 :
                        dir1=0
                        dir2=1
                    esp = k - (max(a[dir1])-min(a[dir1]))
                    #print(d,' non error',ess, np_image.shape[dir1])
                    x1 = min(a[dir1]) - int(esp/2)
                    x2 = max(a[dir1]) + (esp-int(esp/2))
                    
                    esp2 = k - (max(a[dir2])-min(a[dir2]))
                    y1 = min(a[dir2]) - int(esp2/2)
                    y2 = max(a[dir2]) + (esp2-int(esp2/2))
                    #print(x1,x2,y1,y2,p2)
                    if d==0 :
                        dir1=2
                        dir2=1
                        patch1 = np_image[p1,y1:y2,x1:x2]
                        patch2 = np_image[p2,y1:y2,x1:x2]
                        patch3 = np_image[p3,y1:y2,x1:x2]
                        patch4 = mask[p1,y1:y2,x1:x2]
                        patch5 = mask[p2,y1:y2,x1:x2]
                        patch6 = mask[p3,y1:y2,x1:x2]
                    if d==1 :
                        dir1=0
                        dir2=2
                        patch1 = np_image[x1:x2,p1,y1:y2]
                        patch2 = np_image[x1:x2,p2,y1:y2]
                        patch3 = np_image[x1:x2,p3,y1:y2]
                        patch4 = mask[x1:x2,p1,y1:y2]
                        patch5 = mask[x1:x2,p2,y1:y2]
                        patch6 = mask[x1:x2,p3,y1:y2]
                    if d==2 :
                        patch1 = np_image[x1:x2,y1:y2,p1]
                        patch2 = np_image[x1:x2,y1:y2,p2]
                        patch3 = np_image[x1:x2,y1:y2,p3]
                        patch4 = mask[x1:x2,y1:y2,p1]
                        patch5 = mask[x1:x2,y1:y2,p2]
                        patch6 = mask[x1:x2,y1:y2,p3]
                    if npatch ==9:
                        l_patch.append(patch1)
                        l_mask.append(patch4)
                        l_patch.append(patch3)
                        l_mask.append(patch6)
                    l_patch.append(patch2)
                    l_mask.append(patch5)
        else: 
            l_patch.append(New_im)
            l_mask.append(mask[min(a[0]):max(a[0]),min(a[1]):max(a[1]),min(a[2]):max(a[2])])

        return l_patch,l_mask

def processing_patches(PATH,interS,method,reload_patches,D3='False',npatch=9,nameI='image.tif',sess=None):
    ''' for each patient '''
    filename = "output.csv"
    data = pd.read_csv(filename,sep=';',names=['patient','event','time'])
    patient = list(data.loc[:,'patient'])
    for i in range(len(data.loc[:,'patient'])):
        data=data.rename({i : patient[i]})
    data=data.drop(['patient'],axis = 1)
    liste_patches, liste_ref, liste_patients, errors,liste_mask=[],[],[],[],[]
    print('******************', PATH)
    for refPatient in os.listdir(PATH):
        print('ref patient',refPatient)
        if refPatient not in liste_patients:
            list_files = [file for file in os.listdir(os.path.join(PATH,refPatient))]
            if 'dcm'  in list_files and 'max' in list_files:
                try:
                    if nameI not in list_files or reload_patches == False:
                        patient = Patient(refPatient, PATH)
                        correct = patient.correct
                        image = patient.image
                        np_image = sitk.GetArrayFromImage(image)
                        im_Path = os.path.join(PATH,refPatient, nameI)
                        tifffile.imsave(im_Path, np_image.T)
                    else:
                        correct = correction(refPatient)
                        np_image = io.imread(os.path.join(PATH,refPatient,nameI)).T
                    
                    if 'max.tif' not in list_files or reload_patches == False:
                        pathToAll=uf.makeTifFromPile(os.path.join(PATH,refPatient, "max"),correct)
                        mask = io.imread(pathToAll).T
                    else:
                        m_Path = os.path.join(PATH,refPatient, "max.tif")
                        mask = io.imread(m_Path).T
                    a=np.where(mask != 0)
                    if D3== 'True': 
                        p_Path = os.path.join(PATH,refPatient, "patch3D",method,str(npatch))
                    else:
                        p_Path = os.path.join(PATH,refPatient, "patch",method,str(npatch))
                    if os.path.exists(p_Path)  and reload_patches :
                        lp=os.listdir(p_Path)
                        for i in lp:
                            if i[0]=='I':
                                patchh=io.imread(os.path.join(p_Path,i)).T
                                liste_patches.append(patchh)
                                liste_ref.append(refPatient)
                                liste_patients.append(refPatient)
                                patchh=io.imread(os.path.join(p_Path,'M'+i[1:])).T
                                liste_mask.append(patchh)
                    else:
                        try :
                            if len(a[0]) != 0:
                                s=mask.shape
                                s2=np_image.shape
                                if s[2] == s2[2] : # if mask and image have the same size
                                    if method == 'box':
                                        l_patch, l_mask = patches(np_image,mask,D3,npatch)
                                    else:
                                        l_patch, l_mask = all_size_patches(np_image,mask,interS,D3,npatch)
                                    for i in range(len(l_patch)):
                                        liste_patches.append(l_patch[i])
                                        liste_mask.append(l_mask[i])
                                        if D3=='False':
                                            image = Image.fromarray(l_patch[i])
                                            mask = Image.fromarray(l_mask[i])
                                            os.makedirs(p_Path, exist_ok=True) 
                                            impa ="I"+refPatient + "n"+str(i) + ".tif"
                                            mapa ="M"+refPatient + "n"+str(i) + ".tif"
                                            try:
                                                tifffile.imsave(os.path.join(p_Path,impa),l_patch[i].T)
                                                tifffile.imsave(os.path.join(p_Path,mapa),l_mask[i].T)
                                            except:
                                                print("problem patch image saving")
                         
                                        liste_ref.append(refPatient)
                                    liste_patients.append(refPatient)
                                else :
                                    errors.append([refPatient,'wrong number of slices'])
                            else :
                                errors.append([refPatient,'empty mask'])
                        except:
                            errors.append([refPatient,'unknow error'])
                            pass
                except:
                    errors.append([refPatient,'image error'])
                    pass
            else:
                if 'dcm'  in list_files :
                    errors.append([refPatient,'no mask'])
                elif 'max' in list_files:
                    errors.append([refPatient,'no dcm'])
                else:
                    errors.append([refPatient,'no mask and dcm'])
    liste_patients=np.array(liste_patients)
    liste_ref=np.array(liste_ref)

    if method == 'box': 
        if D3=='True':
            liste_interpolate = uf.interpolation3D_MS(liste_patches,interS,'3D',sess=sess)
            liste_maskinterpolate = uf.interpolation3D_MS(liste_mask,interS,'3D',sess=sess)

        else:
            liste_interpolate = interpolation_patches(liste_patches,liste_ref,True,interS)
            liste_maskinterpolate = interpolation_patches(liste_mask,liste_ref,True,interS)

        liste_interpolate=np.array(liste_interpolate)
        liste_maskinterpolate=np.array(liste_maskinterpolate)
        return data,liste_interpolate,liste_ref,liste_patients,errors,liste_maskinterpolate
    else: 
        liste_patches=np.array(liste_patches)
        liste_mask=np.array(liste_mask)
        return data,liste_patches,liste_ref,liste_patients,errors,liste_mask
    
 