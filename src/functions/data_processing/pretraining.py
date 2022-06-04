# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

Functions used for the pretraining methods
"""
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import SimpleITK as sitk
import pandas as pd
import re
from keras.utils import to_categorical
from functions.Patient import Patient
import numpy as np
from skimage import io
from skimage.external import tifffile
from skimage.io import imsave
from functions.Patient import correction
from keras.layers import Input, Dense, Conv2D, Flatten,  Dropout,MaxPool3D #,# BatchNormalization#,GlobalAveragePooling2D
from keras.models import Sequential, Model #, load_model
import functions.survival.nnet_survival as ns
import keras
import scipy
from keras.applications.vgg16 import VGG16
from tensorflow.python.ops import gen_nn_ops
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder
import data_processing as dpr
import models as mds

def calcul_ratio(patch): 
    """Calculate the ratio of positive (!= 0) elements in a binary array"""
    nb_pos=len(np.where(patch!=0))
    nb_neg=len(np.where(patch==0))
    ratio=nb_pos/(nb_pos+nb_neg)
    return ratio

def patch_creation(x,y,z,k,image): 
    """Create a patch from the 3D image.
    
    x, y, z are the coordonates of the middle pixel of the patch with size (2*k+1,2*k+1,2*k+1) """
    patch = np.ones((2*k+1,2*k+1,2*k+1))
    for xp in range(patch.shape[0]):
        for yp in range(patch.shape[1]):
            for zp in range(patch.shape[2]):
                patch[xp][yp][zp]=image[xp+x][yp+y][zp+z]  
    return patch

def makeTifFromPile(pathToPile,correct):
    '''Takes an absolute path containing a pile of masks, compute the resulting .tif mask and
     output the path to the created .tif mask
     Input
         pathToPile : path of the folder containing a pile od masks (Dicom)
         correct : dataFrame containing the correction to apply per lesion
    Return
        pathToTifMask : path to the tiff mask       
    '''
    list_pileFiles = []
    for dirpath, dirnames, fileNames in os.walk(pathToPile):
        for fileName in fileNames:
            if not fileName.startswith('.'):
                list_pileFiles.append(os.path.join(dirpath, fileName))
    dpr.change_name(list_pileFiles)

    list_pileFiles = []
    for dirpath, dirnames, fileNames in os.walk(pathToPile):
        for fileName in fileNames:
            if not fileName.startswith('.'):
                    list_pileFiles.append(os.path.join(dirpath, fileName))
                    
    list_pileFiles.sort()   
    size=len(list_pileFiles)
    list_pileFiles=list_pileFiles[correct['avant']:size-correct['apres']]
    first_file = list_pileFiles[0]
    # get the shape of the image
    with open(first_file, mode='r', encoding='utf-8') as tifFile:
        tifFile.readline()  # first line is junk data
        shapeLocalMask = mds.getWords(tifFile.readline())  
        xShape = int(shapeLocalMask[0])
        yShape = int(shapeLocalMask[1])

    num_file = len(list_pileFiles)
    
    mask_array = np.zeros((num_file, xShape, yShape))
    fileIndex = 0
    # Run through the files of the pile
    for pileFile in list_pileFiles:
        with open(pileFile, mode='r', encoding='utf-8') as tifFile:
            tifFile.readline()  # junk line
            tifFile.readline()  # second line is shape of the dcm pile
            tifFile.readline()  # third line is junk data
            # Run through rows and columns of the file
            for rowIndex in range(xShape):
                for colIndex in range(yShape):
                    val = tifFile.read(1)
                    # Takes only 0 and 1 values (removes spaces)
                    while val != '0' and val != '1':
                        val = tifFile.read(1)
                    mask_array[fileIndex, rowIndex, colIndex] = int(val)
            fileIndex = fileIndex + 1    
    pathToLesion = os.path.abspath(os.path.join(pathToPile, os.pardir))
    if 'max' in pathToPile:
        pathToTifMask = os.path.join(pathToLesion, 'max.tif')
    if 'tous' in pathToPile:
        pathToTifMask = os.path.join(pathToLesion, 'all.tif')

    mask_array=mask_array.astype(np.uint8)
    imsave(pathToTifMask, mask_array)
    return pathToTifMask

def positive_patch(mask,k, pd_mask,pourcent_ratio=0.4,ThirdDim=3,num=5): #taille patch = 2k+1, 2k+1, 3
    ''' 
    Input
        mask : array with one mask per lesion
        k : half of the size of the image
        pd_mask : mask with padding
        pourcent_ratio : ratio of pixels belonging to the lesion to considere the patch as positive
        ThirdDim : if equal to 3, patches with size (2k+1,2k+1,3)
                   if equal to "3D"", patches with size (2k+1,2k+1,2k+1)
    Return 
        liste_pos : list of positive patches from a 3D mask
        liste_coord_pos : list of the coordinates of the positive patches
    
    Donne liste des patchs positifs du mask 3D. Pourcent_ratio: pourcentage of positive pixels to be positive.
    
    if ThirdDim = 3:  ou (2k+1,2k+1,2k+1) si ThirdDim="3D"'''
    print('     Looking for positives patchs ...')
    indice=np.where(mask!=0)
    min_x, max_x, min_y, max_y,min_z, max_z =indice[0].min(),indice[0].max(),indice[1].min(),indice[1].max(),indice[2].min(),indice[2].max()
    sup=int(pourcent_ratio*(2*k+1)+1)
    liste_pos=[]
    liste_coord_pos=[]
    allx = [random.randrange(min_x-sup,max_x+sup) for i in range(num)]
    ally = [random.randrange(min_y-sup,max_y+sup) for i in range(num)]
    allz = [random.randrange(min_z-sup,max_z+sup) for i in range(num)]

    for x in allx :
        for y in ally:
            for z in allz :
                patch = patch_creation(x,y,z,k,pd_mask)
                if calcul_ratio(patch)>= pourcent_ratio : 
                    liste_coord_pos.append((x,y,z))
                    liste_pos.append(patch)
    return liste_pos, liste_coord_pos            


def random_patch(mask,k, pourcent_ratio, nb_patch,image,pd_image,pd_mask,PATHpat,ThirdDim=3):
    ''' 
    Input
        mask : array with one mask per lesion
        k : half of the size of the image
        pourcent_ratio : ratio of pixels belonging to the lesion to considere the patch as positive
        nb_patch : number of patches
        image : array with the lesions' images
        pd_mask : mask with padding
        pd_mask : images with padding
        PATHpat : path to the lesion folder
        ThirdDim : if equal to 3, patches with size (2k+1,2k+1,3)
                   if equal to "3D"", patches with size (2k+1,2k+1,2k+1)
    Return
        liste_value : Binary value of the patch (0 if it belongs to the background (negative) and 1 if it belongs to the lesion (positive))
        liste_patch : list of patches (50% of positive and 50% of negative). Positive patches are choosed randomly.
    
    Pour une lesion, avoir une liste de patchs (equivalents pos/neg) aléatoires avec leur valeur pos/neg associée'''
    print('creation of random patches list ...')
    nb_pos=nb_patch//2
    nb_neg = nb_patch - nb_pos
    liste_pos, liste_coord_pos = positive_patch(mask,k,pd_mask,pourcent_ratio,ThirdDim=3)
    #liste_pos : list of positive patches, liste_coord-pos: list of the coordonates (x,y,z) of the patches
    liste_patch_coord,liste_indice, liste_value,liste_patch,listI=[],[],[],[],[]
    while len(liste_indice) <= nb_pos and len(listI) <= len(liste_pos):
        i = random.randint(0,len(liste_pos)-1)
        if i not in liste_indice :
            (x,y,z)=liste_coord_pos[i]
            ppa = patch_creation(x,y,z,k,pd_image)
            liste_patch.append(ppa)
            listI.append(i)
            liste_indice.append(i) # x
            liste_patch_coord.append(liste_coord_pos[i])
            liste_value.append(1)   # y 
    c=0
    #create negative patch
    while c < nb_neg :
        x=random.randint(0,len(image)-(2*k))
        y=random.randint(0,len(image[0])-2*k)
        z=random.randint(0,len(image[0][0])-2*k)
        if (x,y,z) not in liste_coord_pos:
            patch = patch_creation(x,y,z,k,pd_image)
            liste_patch_coord.append((x,y,z))
            liste_patch.append(patch)
            liste_value.append(0)
            c=c+1
    return liste_value, liste_patch




def all_patch(mask,k, pourcent_ratio, nb_patch,nim,pd_image,pd_mask):
    '''
    Avoir tous les patchs d'une image
    '''
    liste_patch, liste_patch_value, liste_patch_coord=[],[],[]
    for x in range(0,len(nim),(2*k+1)):
        for y in range(0,len(nim[0]),(2*k+1)):
            for z in range(0,len(nim[0][0]),(2*k+1)):
                patch=patch_creation(x,y,z,k,pd_image)
                liste_patch.append(patch)
                liste_patch_coord.append((x,y,z))
                if calcul_ratio(patch)>= pourcent_ratio :
                    liste_patch_value.append(1)
                else :
                    liste_patch_value.append(0)
    return liste_patch, liste_patch_value, liste_patch_coord


def binary_extraction(RR,CrossVal,PATH,sess,TEST=False, interpolSize=36):
    if CrossVal== False:
        print("import binary")
        rtrainB = np.load('./Results/PATIENTS_SEP_TO_KEEP_154/Balance.npy')
        rvalB = np.load('./Results/PATIENTS_SEP_TO_KEEP_154/Aval.npy')
        rtestB = np.load('./Results/PATIENTS_SEP_TO_KEEP_154/Ate.npy')
        print(' .............. TRAIN LIST CREATION ...............................')     
        xbitrain3D, ybitrain3D=mds.create_train(rtrainB,4, 0.4, 10,PATH)
        xbitrain3D, ybitrain3D = dpr.shuffle_and_interpolate(xbitrain3D, ybitrain3D,interpolSize,ThirdDim = '3D',sess=sess)
        print(' .............. VALIDATION LIST CREATION .................................')
        xbival3D, ybival3D = mds.create_train(rvalB,4,0.4,10,PATH)
        xbival3D, ybival3D = dpr.shuffle_and_interpolate(xbival3D, ybival3D,interpolSize,ThirdDim = '3D',sess=sess)
        print(' .............. TEST LIST CREATION .................................')
        # xbitest, ybitest = uf.create_train(rtest,4,0.4,50,PATH)
        # xbitest, ybitest = dpr.shuffle_and_interpolate(xbitest, ybitest,interpolSize,ThirdDim = '3D')
        
        encoder = LabelEncoder()
        ybitrain3D = to_categorical(encoder.fit_transform(ybitrain3D))
        ybival3D = to_categorical(encoder.fit_transform(ybival3D))
        # ybitest = to_categorical(encoder.fit_transform(ybitest))
        print('expand')
        xbitrain3D = dpr.normalize(np.expand_dims(np.array(xbitrain3D),axis = 4)) 
        xbival3D = dpr.normalize(np.expand_dims(np.array(xbival3D),axis = 4)) 
        
        xbival3D=np.reshape(xbival3D, (np.shape(xbival3D)[0],np.shape(xbival3D)[1],np.shape(xbival3D)[2],np.shape(xbival3D)[3],np.shape(xbival3D)[4]))
        xbitrain3D=np.reshape(xbitrain3D, (np.shape(xbitrain3D)[0],np.shape(xbitrain3D)[1],np.shape(xbitrain3D)[2],np.shape(xbitrain3D)[3],np.shape(xbitrain3D)[4]))
        
        xbitrain2D=[]
        ybitrain2D=[]
        for j in range(len(xbitrain3D)):
            for i in range(len(xbitrain3D[j])):
                if np.max(xbitrain3D[j,:,:,i]) ==1 and np.argmax(ybitrain3D[j,:])==1 :
                    xbitrain2D.append(xbitrain3D[j,:,:,i])
                    ybitrain2D.append(ybitrain3D[j,:])
        
                if  np.argmax(ybitrain3D[j,:])==0:
                    xbitrain2D.append(xbitrain3D[j,:,:,i])
                    ybitrain2D.append(ybitrain3D[j,:])
        xbitrain2D, ybitrain2D = shuffle(xbitrain2D,ybitrain2D)
        
        xbival2D=[]
        ybival2D=[]
        for j in range(len(xbival3D)):
            for i in range(len(xbival3D[j])):
                if np.max(xbival3D[j,:,:,i]) ==1 and np.argmax(ybival3D[j,:])==1 :
                    xbival2D.append(xbival3D[j,:,:,i])
                    ybival2D.append(ybival3D[j,:])
        
                if  np.argmax(ybival3D[j,:])==0:
                    xbival2D.append(xbival3D[j,:,:,i])
                    ybival2D.append(ybival3D[j,:])
        
        xbival2D, ybival2D = shuffle(xbival2D,ybival2D)
        return xbitrain2D, ybitrain2D, xbival2D,ybival2D
    else:
        # rrB = np.delete(RR,f)
        Xb2D, Yb2D= [[] for i in range(len(RR))],[[] for i in range(len(RR))]
        Xb3D, Yb3D = [[] for i in range(len(RR))],[[] for i in range(len(RR))]
        print('calculate binary patches')
        for k in range(len(RR)):
            print(k)
            rvalB = RR[k]
            if TEST==True:
                if k==0:
                    rtrainB=[RR[1],RR[2],RR[3],RR[4]]
                elif k==1:
                    rtrainB=[RR[0],RR[2],RR[3],RR[4]]
                elif k==2:
                    rtrainB=[RR[1],RR[0],RR[3],RR[4]]
                elif k==3:
                    rtrainB=[RR[1],RR[0],RR[2],RR[4]]
                else:
                    rtrainB=[RR[1],RR[2],RR[0],RR[3]]
                rtrainB=np.concatenate((rtrainB[0],rtrainB[1],rtrainB[2],rtrainB[3]))
            else:
                if k==0:
                    rtrainB=[RR[1],RR[2],RR[3]]
                elif k==1:
                    rtrainB=[RR[0],RR[2],RR[3]]
                elif k==2:
                    rtrainB=[RR[1],RR[0],RR[3]]
                else:
                    rtrainB=[RR[1],RR[0],RR[2]]
                rtrainB=np.concatenate((rtrainB[0],rtrainB[1],rtrainB[2]))
            
            
            print(' .............. TRAIN LIST CREATION ...............................')     
            xbitrain3D, ybitrain3D=mds.create_train(rtrainB,4, 0.5, 10,PATH)
            xbitrain3D, ybitrain3D = dpr.shuffle_and_interpolate2(xbitrain3D, ybitrain3D,interpolSize,ThirdDim = '3D',sess = sess)
            print(' .............. VALIDATION LIST CREATION .................................')
            xbival3D, ybival3D = mds.create_train(rvalB,4,0.5,10,PATH)
            xbival3D, ybival3D = dpr.shuffle_and_interpolate2(xbival3D, ybival3D,interpolSize,ThirdDim = '3D',sess=sess)
            encoder = LabelEncoder()
            ybitrain3D = to_categorical(encoder.fit_transform(ybitrain3D))
            ybival3D = to_categorical(encoder.fit_transform(ybival3D))
            print('expand')
            xbitrain3D = dpr.normalize(np.expand_dims(np.array(xbitrain3D,dtype = np.float32),axis = 4)) 
            xbival3D = dpr.normalize(np.expand_dims(np.array(xbival3D),axis = 4)) 
            
            xbival3D=np.reshape(xbival3D, (np.shape(xbival3D)[0],np.shape(xbival3D)[1],np.shape(xbival3D)[2],np.shape(xbival3D)[3],np.shape(xbival3D)[4]))
            xbitrain3D=np.reshape(xbitrain3D, (np.shape(xbitrain3D)[0],np.shape(xbitrain3D)[1],np.shape(xbitrain3D)[2],np.shape(xbitrain3D)[3],np.shape(xbitrain3D)[4]))
            
            xbitrain2D=[]
            ybitrain2D=[]
            for j in range(len(xbitrain3D)):
                for i in range(len(xbitrain3D[j])):
                    if np.max(xbitrain3D[j,:,:,i]) ==1 and np.argmax(ybitrain3D[j,:])==1 :
                        xbitrain2D.append(xbitrain3D[j,:,:,i])
                        ybitrain2D.append(ybitrain3D[j,:])
            
                    if  np.argmax(ybitrain3D[j,:])==0:
                        xbitrain2D.append(xbitrain3D[j,:,:,i])
                        ybitrain2D.append(ybitrain3D[j,:])
    
            xbitrain2D, ybitrain2D = shuffle(xbitrain2D,ybitrain2D)
            xbival2D=[]
            ybival2D=[]
            for j in range(len(xbival3D)):
                for i in range(len(xbival3D[j])):
                    if np.max(xbival3D[j,:,:,i]) ==1 and np.argmax(ybival3D[j,:])==1 :
                        xbival2D.append(xbival3D[j,:,:,i])
                        ybival2D.append(ybival3D[j,:])
            
                    if  np.argmax(ybival3D[j,:])==0:
                        xbival2D.append(xbival3D[j,:,:,i])
                        ybival2D.append(ybival3D[j,:])
            
            xbival2D, ybival2D = shuffle(xbival2D,ybival2D)
            Xb2D[k], Yb2D[k]=xbival2D,ybival2D
            Xb3D[k], Yb3D[k]=xbival3D,ybival3D
            return Xb2D, Yb2D, Xb3D, Yb3D