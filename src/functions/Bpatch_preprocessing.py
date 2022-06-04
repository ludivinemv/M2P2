# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

Extract patches to pass from 3D to 2D
"""
import functions.usefull_functions as uf
from PIL import Image
import pandas as pd
#import re
#from Lesion import Lesion
from functions.Patient import Patient
import numpy as np
from skimage import io
import os
import SimpleITK as sitk
from skimage.external import tifffile
#from skimage.io import imsave
from functions.Patient import correction
#import cv2 as cv

#import matplotlib.pyplot as plt
class patch_preprocessing:

    def __init__(self, PATH,interS,method='box',reload=True,D3=False,npatch=9,nameI='image.tif',sess=None):
        self.interS = interS
        self.method = method
        self.reload = reload
        self.D3 = D3
        self.npatch=npatch
        self.nameI=nameI
        self.sess=sess
        if nameI=='image.tif':
            self.data, self.liste_patches, self.liste_ref, self.liste_patients, self.errors,self.liste_mask = processing_patches(PATH,self.interS, self.method,self.reload,self.D3,self.npatch,self.nameI,self.sess)
        else:
            self.data, self.liste_patches, self.liste_ref, self.liste_patients, self.errors,self.liste_mask = processing_patchesHN(sess,PATH,self.interS, self.method,self.reload,self.D3,self.npatch,self.nameI)

"""###############################################################################
######################### Interpolation       ####################################
###############################################################################"""

def interpolation(liste_patches,liste_ref,imageShow,interpolSize,D3=False):
    if interpolSize != None:
        liste_interpolate = []
        for i in range(len(liste_patches)):
            im=Image.fromarray(liste_patches[i])
            new_img = im.resize((interpolSize,interpolSize),Image.BICUBIC)
            im= np.asarray(new_img)
            liste_interpolate.append(im)
        return liste_interpolate
    else :
        return liste_patches
    
def AllSize_patches(np_image,mask,interS, D3=False,npatch=9): #k the size/2 of the patches 
        """ crop des patchs de 36x36"""
        l_patch, l_mask=[], []
        #récupérer 3 patchs par lésions
        a=np.where(mask != 0)
        #for each direction have 3 patchs with the index pi
        centerBox = [min(a[0]) + int((max(a[0])-min(a[0]))/2),
                    min(a[1]) + int((max(a[1])-min(a[1]))/2),
                    min(a[2]) + int((max(a[2])-min(a[2]))/2)]
        middle = int(interS/2)
        L=[0,0,0]
        R=[0,0,0]
        for i in range(3): 
            if centerBox[i] < middle: 
                L[i]=0
                R[i]=interS
            if (np.shape(np_image)[i] - centerBox[i]) < middle: 
                R[i]= np.shape(np_image)[i] - 1
                L[i] = np.shape(np_image)[i] - 1 - interS
            else: 
                L[i]= centerBox[i] - middle
                R[i]= centerBox[i] + middle

        New_im = np_image[L[0]:R[0],L[1]:R[1],L[2]:R[2]]
        New_mask = mask[L[0]:R[0],L[1]:R[1],L[2]:R[2]]
        if D3==False: 
            aa=np.where(New_mask != 0)
            for d in range(3):                
                if npatch !=1 or (npatch==1 and d==0):

                    c=int((max(aa[d])-min(aa[d]))/4)
                    p1=int(min(aa[d])+c)
                    p2=int(min(aa[d])+2*c)
                    p3=int(min(aa[d])+3*c)
                    # to have the bounders of the pixel 
                    if d==0 :
                        patch1 = New_im[p1,:,:]
                        patch2 = New_im[p2,:,:]
                        patch3 = New_im[p3,:,:]
                        patch4 = New_mask[p1,:,:]
                        patch5 = New_mask[p2,:,:]
                        patch6 = New_mask[p3,:,:]
                    if d==1 :
                        patch1 = New_im[:,p1,:]
                        patch2 = New_im[:,p2,:]
                        patch3 = New_im[:,p3,:]
                        patch4 = New_mask[:,p1,:]
                        patch5 = New_mask[:,p2,:]
                        patch6 = New_mask[:,p3,:]
                    if d==2 :
                        patch1 = New_im[:,:,p1]
                        patch2 = New_im[:,:,p2]
                        patch3 = New_im[:,:,p3]
                        patch4 = New_mask[:,:,p1]
                        patch5 = New_mask[:,:,p2]
                        patch6 = New_mask[:,:,p3]
                    if npatch ==9:
                        l_patch.append(patch1)
                        l_mask.append(patch4)
                        l_patch.append(patch3)
                        l_mask.append(patch6)
                    l_patch.append(patch2)
                    l_mask.append(patch5)
        else: 
            l_patch.append(New_im)
            l_mask.append(New_mask)
        return l_patch, l_mask
 
def patches(np_image,mask,D3=False,npatch=9): #k the size/2 of the patches 
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

def processing_patches(PATH,interS,method,reload,D3=False,npatch=9,nameI='image.tif',sess=None):
    ''' for each patient '''
    filename = "./output.csv"
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
                    if nameI not in list_files or reload == False:
                        patient = Patient(refPatient, PATH)
                        correct = patient.correct
                        image = patient.image
                        np_image = sitk.GetArrayFromImage(image)
                        im_Path = os.path.join(PATH,refPatient, nameI)
                        tifffile.imsave(im_Path, np_image.T)
                    else:
                        correct = correction(refPatient)
                        np_image = io.imread(os.path.join(PATH,refPatient,nameI)).T
                    
                    if 'max.tif' not in list_files or reload == False:
                        pathToAll=uf.makeTifFromPile(os.path.join(PATH,refPatient, "max"),correct)
                        mask = io.imread(pathToAll).T
                    else:
                        m_Path = os.path.join(PATH,refPatient, "max.tif")
                        mask = io.imread(m_Path).T
                    a=np.where(mask != 0)
                    if D3== True: 
                        p_Path = os.path.join(PATH,refPatient, "patch3D",method,str(npatch))
                    else:
                        p_Path = os.path.join(PATH,refPatient, "patch",method,str(npatch))
                    if os.path.exists(p_Path)  and reload == True:
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
                                        l_patch, l_mask = AllSize_patches(np_image,mask,interS,D3,npatch)
                                    for i in range(len(l_patch)):
                                        liste_patches.append(l_patch[i])
                                        liste_mask.append(l_mask[i])
                                        if D3==False:
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
        if D3==True:
            liste_interpolate = uf.interpolation3D_MS(liste_patches,interS,'3D',sess=sess)
            liste_maskinterpolate = uf.interpolation3D_MS(liste_mask,interS,'3D',sess=sess)

        else:
            liste_interpolate = interpolation(liste_patches,liste_ref,True,interS)
            liste_maskinterpolate = interpolation(liste_mask,liste_ref,True,interS)

        liste_interpolate=np.array(liste_interpolate)
        liste_maskinterpolate=np.array(liste_maskinterpolate)
        return data,liste_interpolate,liste_ref,liste_patients,errors,liste_maskinterpolate
    else: 
        liste_patches=np.array(liste_patches)
        liste_mask=np.array(liste_mask)
        return data,liste_patches,liste_ref,liste_patients,errors,liste_mask
    
 
def processing_patchesHN(PATH,interS,method,reload,D3=False,npatch=9,nameI='image.tif',sess=None):
    ''' 
    To use with the Head and Neck dataset
    Extract patches for each patient 
    
    '''
    filename = "../data/HeadAndNeck/finalData.csv"
    data = pd.read_csv(filename,sep='\t',names=['patient','event','time'])
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
            if 'PET.tif'  in list_files and 'max.tif' in list_files:
                try:
                    if nameI not in list_files or reload == False:
                        patient = Patient(refPatient, PATH)
                        correct = patient.correct
                        image = patient.image
                        np_image = sitk.GetArrayFromImage(image)
                        im_Path = os.path.join(PATH,refPatient, nameI)
                        tifffile.imsave(im_Path, np_image.T)
                    else:
                        np_image = io.imread(os.path.join(PATH,refPatient,nameI)).T
                    
                    if 'max.tif' not in list_files or reload == False:
                        pathToAll=uf.makeTifFromPile(os.path.join(PATH,refPatient, "max"),correct)
                        mask = io.imread(pathToAll).T
                    else:
                        m_Path = os.path.join(PATH,refPatient, "max.tif")
                        mask = io.imread(m_Path).T
                    a=np.where(mask != 0)
                    if D3== True: 
                        p_Path = os.path.join(PATH,refPatient, "patch3D",method,str(npatch))
                    else:
                        p_Path = os.path.join(PATH,refPatient, "patch",method,str(npatch))
                    if os.path.exists(p_Path)  and reload == True:
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
                                        l_patch, l_mask = AllSize_patches(np_image,mask,interS,D3,npatch)
                                    for i in range(len(l_patch)):
                                        liste_patches.append(l_patch[i])
                                        liste_mask.append(l_mask[i])
                                        if D3==False:
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
        if D3==True:
            liste_interpolate = uf.interpolation3D_MS(liste_patches,interS,'3D',sess=sess)
            liste_maskinterpolate = uf.interpolation3D_MS(liste_mask,interS,'3D',sess=sess)

        else:
            liste_interpolate = interpolation(liste_patches,liste_ref,True,interS)
            liste_maskinterpolate = interpolation(liste_mask,liste_ref,True,interS)

        liste_interpolate=np.array(liste_interpolate)
        liste_maskinterpolate=np.array(liste_maskinterpolate)
        return data,liste_interpolate,liste_ref,liste_patients,errors,liste_maskinterpolate
    else: 
        liste_patches=np.array(liste_patches)
        liste_mask=np.array(liste_mask)
        return data,liste_patches,liste_ref,liste_patients,errors,liste_mask
    
       