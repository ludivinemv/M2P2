# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
"""


import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt

import re
from keras.utils import to_categorical
from Patient import Patient
import numpy as np
from skimage import io
from skimage.external import tifffile
from skimage.io import imsave
from Patient import correction
from keras.layers import Input, Dense, Conv2D, Flatten,  Dropout,MaxPool3D #,# BatchNormalization#,GlobalAveragePooling2D
from keras.models import Sequential, Model #, load_model
#import functions.survival.nnet_survival as ns
import nibabel as nib
import keras
import scipy
from keras.applications.vgg16 import VGG16
from tensorflow.python.ops import gen_nn_ops


def createDataAug(numberA=30):
    ''' 
    Input
        numberA : number of augmentation par image
    Return
        A matrix containing numberA vectors of augmentation parameters (one vector with par augmentation ). One vector : [translation, zoom scale, rotation, rotation axes, (Horizontal flip, vertical flip)]
    '''
    tr, sr, rr, r2r, fr  = [],[],[],[],[]
    t=[0,0,1,2,3,4,5,6] # translation in the 3 axes
    s=[0.75,0.8,.9,1,1,1.1,1.2,1.3,1.4] # zoom scale
    r=[-30,-20,-10,0,0,10,20,30] # rotation
    r2=[(1,0),(1,0)] #rotation axes
    f=[(False,False ), (False,False ), (True,False ), (True,True ), (False,True )] # (Horizontal flip, vertical flip)
    for k in range(numberA):
        tr.append(t[random.randrange(1,len(t))])
        sr.append(s[random.randrange(1,len(s))])
        rr.append(r[random.randrange(1,len(r))])
        r2r.append( r2[random.randrange(1,len(r2))])
        fr.append(f[random.randrange(1,len(f))])
    da = np.array([tr,sr,rr,r2r,fr])
    return da

def DataAugmentation3D(da,x,y,mask,ref,num):
    ''' 
    Input
        da : vector of predifined augmentation parameters
        x : matrice of individuals' 3D images. Size : (B,36,36,36,1)
        y : matrice of individuals' target values. Size : (B,2)
        mask : matrice of individuals' 3D masks. Size : (B,36,36,36,1)
        ref : vector of individuals' names. Size : (B,1)
        num : number of augmentation par image
        B : number of individuals
    Return
        Augmented matrice of individuals' images, Augmented matrice of individuals' target values, Augmented matrice of individuals' masks,Augmented matrice of individuals' names
    '''
    def translateit(image, offset=[5,5,5], order = 0):
        return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]),int(offset[2])), order=order, mode='nearest')
    
    def scaleit(image, factor=1.2, isseg=False):
        order = 0 if isseg == True else 3
        height, width, depth= image.shape
        zheight             = int(np.round(factor * height))
        zwidth              = int(np.round(factor * width))
        zdepth              = int(np.round(factor * depth))
        if factor < 1.0:
            newimg  = np.zeros_like(image)
            row     = (height - zheight) // 2
            col     = (width - zwidth) // 2
            layer   = (depth - zdepth) // 2
            newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = scipy.ndimage.interpolation.zoom(image, (float(factor), float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
            return newimg
        elif factor > 1.0:
            row     = (zheight - height) // 2
            col     = (zwidth - width) // 2
            layer   = (zdepth - depth) // 2
            newimg =scipy.ndimage.interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), float(factor)), order=order, mode='nearest')          
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            extrad = (newimg.shape[2] - depth) // 2
            newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
            return newimg
        else:
            return image

    def rotateit(image, theta, isseg=False,axis=(1,0)):
        order = 0 if isseg == True else 5
        return scipy.ndimage.rotate(image, float(theta), reshape=False,axes=axis ,order=order, mode='nearest')
    
    def flipit(image, axes):
        if axes[0]:
            image = np.fliplr(image)
        if axes[1]:
            image = np.flipud(image)  
        return image

    def transf(image,tr,sr,rr,r2r,fr):
        image = translateit(image, offset=[tr,tr,tr], order = 0)
#        image = scaleit(image, factor=sr, isseg=False)
        image = rotateit(image, rr, isseg=False,axis=r2r)
        image=flipit(image,axes = fr)
        return image
    
    xt,yt,mt,rt = [],[],[],[]
    for b in range(len(x))  :
        if b%5==0:
            print(b)
        for k in range(num):
            xt.append(transf(x[b,:,:,:,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k]))
            mt.append(transf(mask[b,:,:,:,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k]))
            yt.append(y[b])
            rt.append(ref[b])
        
    xt=np.array(xt)    
    mt=np.array(mt)  
    rt=np.array(rt)
    mt=np.where(mt <0.5,0,1)
    xt=np.reshape(xt,(np.shape(xt)[0],np.shape(xt)[1],np.shape(xt)[2],np.shape(xt)[3],1))
    mt=np.reshape(mt,(np.shape(mt)[0],np.shape(mt)[1],np.shape(mt)[2],np.shape(mt)[3],1))

    return xt,yt,mt,rt



def dataAugmentation(da,x,y,m,ref,D3=True,num = 30,fold=5):
    """ 
    Input
        da : vector of predifined augmentation parameters
        x : matrice of individuals' images. Size : (B,36,36,1)
        y : matrice of individuals' target values. Size : (B, 2)
        m : matrice of individuals' masks. Size : (B,36,36,1)
        ref : vector of individuals' names
        num : number of augmentation par image
        B : number of individuals
    Return
        xt shape (B*num, 1, 36, 36, 1)
        yt shape (B*num, 2)
        maskt shape (B*num, 1, 36, 36, 1)
    """
    def translateit(image, offset=[5,5], order = 0):
        return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1])), order=order, mode='nearest')
    
    def scaleit(image, factor=1.2, isseg=False):
        order = 0 if isseg == True else 3
        height, width, depth= image.shape
        zheight             = int(np.round(factor * height))
        zwidth              = int(np.round(factor * width))
        if factor < 1.0:
            newimg  = np.zeros_like(image)
            row     = (height - zheight) // 2
            col     = (width - zwidth) // 2
            newimg[row:row+zheight, col:col+zwidth] = scipy.ndimage.interpolation.zoom(image, (float(factor), float(factor)), order=order, mode='nearest')[0:zheight, 0:zwidth]
            return newimg
        elif factor > 1.0:
            row     = (zheight - height) // 2
            col     = (zwidth - width) // 2
            newimg =scipy.ndimage.interpolation.zoom(image[row:row+zheight, col:col+zwidth], (float(factor), float(factor)), order=order, mode='nearest')          
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            newimg = newimg[extrah:extrah+height, extraw:extraw+width]
            return newimg
        else:
            return image

    def rotateit(image, theta, isseg=False,axis=(1,0)):
        order = 0 if isseg == True else 5
        return scipy.ndimage.rotate(image, float(theta), reshape=False,axes=axis ,order=order, mode='nearest')
    
    def flipit(image, axes):
        if axes[0]:
            image = np.fliplr(image)
        if axes[1]:
            image = np.flipud(image)  
        return image

    def transf(image,tr,sr,rr,r2r,fr):
        image = translateit(image, offset=[tr,tr], order = 0)
#        image = scaleit(image, factor=sr, isseg=False)
        image = rotateit(image, rr, isseg=False,axis=r2r)
        image=flipit(image,axes = fr)
        return image
    
    xt=[]
    yt=[]
    mt=[]
    rt=[]
    for b in range(len(x))  :
        if b%5==0:
            print(b)
        for k in range(num):
            if D3==False:
                xt.append(transf(x[b,:,:,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k]))
                mt.append(transf(m[b,:,:,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k]))
                yt.append(y[b])
                rt.append(ref[b])
            elif D3=='25':
                a1 = transf(x[b,:,:,0,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                a2 = transf(x[b,:,:,1,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                a3 = transf(x[b,:,:,2,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                aT = np.array([a1,a2,a3]).transpose(1,2,0)
                xt.append(aT)
                m1 = transf(m[b,:,:,0,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                m2 = transf(m[b,:,:,1,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                m3 = transf(m[b,:,:,2,0],da[0,k],da[1,k],da[2,k],da[3,k],da[4,k])
                mT = np.array([m1,m2,m3]).transpose(1,2,0)
                mt.append(mT)
                yt.append(y[b])
                
    xt=np.array(xt)    
    mt=np.array(mt)  
    rt=np.array(rt)
    mt=np.where(mt <0.5,0,1)
    if D3=='25':        
        xt=np.reshape(xt,(np.shape(xt)[0],np.shape(xt)[1],np.shape(xt)[2],np.shape(xt)[3],1))
        mt=np.reshape(mt,(np.shape(mt)[0],np.shape(mt)[1],np.shape(mt)[2],np.shape(xt)[3],1))
    else:
        xt=np.reshape(xt,(np.shape(xt)[0],np.shape(xt)[1],np.shape(xt)[2],1))
        mt=np.reshape(mt,(np.shape(mt)[0],np.shape(mt)[1],np.shape(mt)[2],1))
    return xt,yt,mt,rt



def create_train(patient_train,k, pourcent_ratio, nb_patch,PATH):    
    """ 
    Input
        patient_train : list of patients
        k : number of patches per lesion
        pourcent_ratio : ratio of lesion to be considered as positive
        nb_patch : number of patches per lesion
        PATH : path to the individual's folder
    Return
        liste_patch : list of patches for all individuals
        liste_y : list of associated target values
    """        
    liste_patch=[]
    liste_y=[]
    for patient in patient_train:
        print(patient, '.......................................................')
        image3D = io.imread(os.path.join(PATH,patient,'image.tif')).T
        pd_image = padding(image3D,k)
        mask=io.imread(os.path.join(PATH,patient,'max.tif')).T
        pd_mask=padding(mask,k)
        l_value, l_patch = random_patch(mask,k, pourcent_ratio, nb_patch,image3D,pd_image,pd_mask,os.path.join(PATH,patient))
        liste_patch.extend(l_patch)
        liste_y.extend(l_value)
    return liste_patch, liste_y

def normalize(x):
    ''' normalize the images with (x-min)/(max - min) '''
    return np.array(list(map(lambda a: normi(a),x )))

    
def change_name(list_Files):
    """Change names of files to permit to open the files in the good order. the file name should finish by _XX or _XXX with X a number"""
    for k in range(len(list_Files)):
        if list_Files[k][-2]=='_':
            new=list(list_Files[k][0:-1])
            new.append('0')
            new.append('0')
            new.append(list_Files[k][-1])
            new_name = "".join(new)
            os.rename(list_Files[k],new_name)   

        if list_Files[k][-3]=='_':
            new=list(list_Files[k][0:-2])
            new.append('0')
            new.append(list_Files[k][-2])
            new.append(list_Files[k][-1])
    
            new_name = "".join(new) 
            os.rename(list_Files[k],new_name)  

def image_mask_ref(PATH):
    """
    Extract images and masks for all patients in PATH
    Input
        PATH : path to the folder containing all patients
    Return
        list of images, list of masks and list of patients
    """
    
    filename = "./output.csv"
    data = pd.read_csv(filename,sep=';',names=['patient','event','time'])
    patient = list(data.loc[:,'patient'])
    for i in range(len(data.loc[:,'patient'])):
        data=data.rename({i : patient[i]})
    data=data.drop(['patient'],axis = 1)
    
    liste_image, liste_mask,liste_ref =[],[],[]
    print('******************', PATH)
    for refPatient in os.listdir(PATH):
        print('ref patient',refPatient)
        if refPatient in data.index:
            list_files = [file for file in os.listdir(os.path.join(PATH,refPatient))]
            if 'dcm'  in list_files and 'max' in list_files:
                try:
                    if 'image.tif' not in list_files:
                        patient = Patient(refPatient, PATH)
                        correct = patient.correct
                        image = patient.image
                        np_image = sitk.GetArrayFromImage(image)
                        im_Path = os.path.join(PATH,refPatient, "image.tif")
                        tifffile.imsave(im_Path, np_image.T)
                    else:
                        correct = correction(refPatient)
                        np_image = io.imread(os.path.join(PATH,refPatient,'image.tif')).T                    
                    if 'max.tif' not in list_files:
                        pathToAll=makeTifFromPile(os.path.join(PATH,refPatient, "max"),correct)
                        mask = io.imread(pathToAll).T
                    else:
                        m_Path = os.path.join(PATH,refPatient, "max.tif")
                        mask = io.imread(m_Path).T
                    liste_ref.append(refPatient)                    
                    liste_image.append(np_image)                  
                    liste_mask.append(mask)
                except:
                    pass
    return liste_image,liste_mask,liste_ref
    
def padding(image,k): #padding to permit to take in count edges
    """ Padding the image to take in count edges. 
    Input
        x : image which will be applied the padding 
        k : Size of the padding is 2*k

    Return
        image with the padding
    """
    pad_image=np.pad(image,((0,2*k),(0,2*k),(0,2*k)),mode = 'symmetric')
    return pad_image

def read_output():
    """ Read ./output.csv and give a dataframe with event and time."""
    filename = "./output.csv"
    data = pd.read_csv(filename,sep=';',names=['patient','event','time'])
    patient = list(data.loc[:,'patient'])
    for i in range(len(data.loc[:,'patient'])):
        data=data.rename({i : patient[i]})
    data=data.drop(['patient'],axis = 1)
    return data, patient



def interpolation3D(liste_patch,interpolSize,ThirdDim=3,sess=None):
    """  Resize images in liste_patch with the size interpolSize and interpolate"""
    if interpolSize != None:
        liste_interpolate = []
        print('interpolation')
        if ThirdDim!= '3D':
            for i in range(len(liste_patch)): 
                im=sitk.GetImageFromArray(np.array(liste_patch[i]))
                resampleImageFilter = sitk.ExpandImageFilter()
                resampleImageFilter.SetExpandFactor(4/np.array(liste_patch[i])[1])
                resampleImageFilter.SetInterpolator(sitk.sitkBSplineResamplerOrder2)
                resultedImage = resampleImageFilter.Execute(im)
                im2= sitk.GetArrayFromImage(resultedImage)
            if ThirdDim == 3:
                im2 = np.array(im2)[:,:,interpolSize//2-1:interpolSize//2+2]
            liste_interpolate.append(im2)
        else:
            def unpool_3D(input_layer,x,y,z):
                unpol_layer = tf.keras.layers.UpSampling3D(size=(x,y,z))(input_layer)
                return unpol_layer
            y=interpolSize/np.shape(liste_patch[0])[0]

            liste_interpolate= unpool_3D(liste_patch,int(y),int(y),int(y)).eval(session=sess)
        return liste_interpolate
    else :
        return liste_patch
    
def shuffle_and_interpolate(liste_patch, liste_y,liste_ref,interpolSize,interStatue=True,liste_mask=[],ThirdDim=3,sess = None):
    """ Interpolate a list of images and shuffle with the true values vector (y) associated. 
    
    If interStatue = True, interpolation"""
    liste_patch=np.array(liste_patch)
    liste_ref=np.array(liste_ref)
    if len(np.shape(liste_mask)) != 1:
        liste_mask=np.array(liste_mask)
    liste_y=np.array(liste_y)
    
    if interStatue == True:
        liste_patch = interpolation3D(liste_patch,interpolSize,ThirdDim,sess=sess)

    x = np.arange(0,len(liste_y))
    random.shuffle(x)
    
    sr = np.array(liste_ref)[x]
    sy = np.array(liste_y)[x]
    if len(np.array(liste_patch).shape) == 4:
        sp = np.array(liste_patch)[x,:,:,:]
        if len(np.shape(liste_mask)) != 1:  
            sp2 = np.array(liste_mask)[x,:,:,:]
    else: 
        sp = np.array(liste_patch)[x,:,:]
        if len(np.shape(liste_mask)) != 1:  
            sp2 = np.array(liste_mask)[x,:,:]
    if len(np.shape(liste_mask)) != 1:  
        return sp,sy, sp2,sr
    else:  
        return sp,sy,sr


def shuffle_and_interpolate2(liste_patch, liste_y,interpolSize,interStatue=True,liste_mask=[],ThirdDim=3,sess = None):
    """ Interpolate a list of images and shuffle with the y associated. 
    If interStatue = True, interpolation.
    The same than shuffle_and_interpolate() but without liste_ref as input"""
    liste_patch=np.array(liste_patch)
    if len(np.shape(liste_mask)) != 1:
        liste_mask=np.array(liste_mask)
    liste_y=np.array(liste_y)
    
    if interStatue == True:
        liste_patch = interpolation3D(liste_patch,interpolSize,ThirdDim,sess=sess)

    x = np.arange(0,len(liste_y))
    random.shuffle(x)
    
    sy = np.array(liste_y)[x]
    if len(np.array(liste_patch).shape) == 4:
        sp = np.array(liste_patch)[x,:,:,:]
        if len(np.shape(liste_mask)) != 1:  
            sp2 = np.array(liste_mask)[x,:,:,:]
    else: 
        sp = np.array(liste_patch)[x,:,:]
        if len(np.shape(liste_mask)) != 1:  
            sp2 = np.array(liste_mask)[x,:,:]
    if len(np.shape(liste_mask)) != 1:  
        return sp,sy, sp2
    else:  
        return sp,sy



def YProcessing(data,liste_patients,classes):    
    y = data.copy()
    
    for pat in data.index:
        if pat not in liste_patients :
            y=y.drop(pat)
    
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
            if y.loc[pat,'time']<= 803 :
                y_grp.loc[pat,'group'] = 0
            elif y.loc[pat,'time'] > 1606  :
                y_grp.loc[pat,'group'] = 2 
            else :
                 y_grp.loc[pat,'group'] = 1            
        elif classes == 'YEAR' or classes == 'YEARWC' : # group 1 : less than 1 year, 5 : between 4 and 5 years, 10 : censorship
            if y.loc[pat,'event'] == 1  and y.loc[pat,'time']//365 <= 5:
                y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
            elif y.loc[pat,'time']//365 >= 5:
                y_grp.loc[pat,'group'] = 5
            else:
                y_grp.loc[pat,'group'] = 6
        else:
            y_grp = data
    y_grp.to_csv('y_grp.csv')
    return y_grp


def YProcessing3(y,classe): #binaire entre CLASSE 0 et autres
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



def YProcessing4(y,classes):
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




def YProcessing2(y,classe):
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

def separation(doss,List_of_Patient,rates=[0.5,0.25,0.25],outnames = "./output.csv",sep=';',cl='all'): 
    """ Input: List with the patient names. rates = liste of rates.
    
    Output: a list of patient list with the same number of elements than the size of the rate."""
    #random.shuffle(List_of_Patient)
    liste=[]
    name1 = doss+"r0.npy"
    name2 = doss+"r1.npy"
    name3 = doss+"r2.npy"
    name4 = doss+"r3.npy"
    if len(rates)==5:
        name5=doss+ "r4.npy"

    if os.path.exists(name1) and os.path.exists(name2) and os.path.exists(name3) and os.path.exists(name4) :        
        r1= np.load(name1)
        r2 = np.load(name2)
        r3 = np.load(name3)
        r4 = np.load(name4)
        if len(rates)==5 and os.path.exists(name5):
            r5= np.load(name5)
            rcc=[r1,r2,r3,r4,r5]
        elif len(rates)==4:
            rcc=[r1,r2,r3,r4]
        else:
            listeClasses=[]
            y_grp = classifyPatients(List_of_Patient,outnames=outnames)
            if cl == 'all':
                nameC = ["0","1","2","3","4","5","6","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
            elif cl == 'sans0':
                nameC = ["1","2","3","4","5","6","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
            elif cl == 'sans1':
                nameC = ["0","2","3","4","5","6","0Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
            elif cl == 'sans2':
                nameC = ["0","1","3","4","5","6","0Censorship","1Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
            elif cl == 'sans3':
                nameC = ["0","1","2","4","5","6","0Censorship","1Censorship","2Censorship","4Censorship","5Censorship","6Censorship"]
            elif cl == 'sans4':
                nameC = ["0","1","2","3","5","6","0Censorship","1Censorship","2Censorship","3Censorship","5Censorship","6Censorship"]
            elif cl == 'sans5':
                nameC = ["0","1","2","3","4","6","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","6Censorship"]
            elif cl == 'sans6':
                nameC = ["0","1","2","3","4","5","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship"]
                        
            elif cl == 'classe0':
                nameC = ["0","0Censorship"]
            elif cl == 'classe1':
                nameC = ["1","1Censorship"]            
            elif cl == 'classe2':
                nameC = ["2","2Censorship"]            
            elif cl == 'classe3':
                nameC = ["3","3Censorship"]           
            elif cl == 'classe4':
                nameC = ["4","4Censorship"]           
            elif cl == 'classe5':
                nameC = ["5","5Censorship"]            
            elif cl == 'classe6':
                nameC = ["6","6Censorship"]
            for classes in nameC:
                listeClasses.append(y_grp.loc[y_grp['group'] == classes].index)
            rcc=[[] for i in range(len(rates))]
            r=0
            for cc in range(len(listeClasses)):
                pa = 0
                while pa <len(listeClasses[cc]) and len(listeClasses[cc])!=0:
                    rcc[r].append(listeClasses[cc][pa])
                    pa+=1
                    r+=1
                    if r >=len(rates):
                        r=0
    
            rcc=[random.sample(rcc[i],len(rcc[i])) for i in range(len(rcc))]
            for i in range(len(rcc)):
                np.save(doss+"r"+ str(i)+".csv",np.array(rcc[i]))
    else:  
        listeClasses=[]
        y_grp = classifyPatients(List_of_Patient,outnames= outnames,sep=sep)
        if cl == 'all':
            nameC = ["0","1","2","3","4","5","6","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
        elif cl == 'sans0':
            nameC = ["1","2","3","4","5","6","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
        elif cl == 'sans1':
            nameC = ["0","2","3","4","5","6","0Censorship","2Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
        elif cl == 'sans2':
            nameC = ["0","1","3","4","5","6","0Censorship","1Censorship","3Censorship","4Censorship","5Censorship","6Censorship"]
        elif cl == 'sans3':
            nameC = ["0","1","2","4","5","6","0Censorship","1Censorship","2Censorship","4Censorship","5Censorship","6Censorship"]
        elif cl == 'sans4':
            nameC = ["0","1","2","3","5","6","0Censorship","1Censorship","2Censorship","3Censorship","5Censorship","6Censorship"]
        elif cl == 'sans5':
            nameC = ["0","1","2","3","4","6","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","6Censorship"]
        elif cl == 'sans6':
            nameC = ["0","1","2","3","4","5","0Censorship","1Censorship","2Censorship","3Censorship","4Censorship","5Censorship"]
        elif cl == 'classe0':
            nameC = ["0","0Censorship"]
        elif cl == 'classe1':
            nameC = ["1","1Censorship"]            
        elif cl == 'classe2':
            nameC = ["2","2Censorship"]            
        elif cl == 'classe3':
            nameC = ["3","3Censorship"]           
        elif cl == 'classe4':
            nameC = ["4","4Censorship"]           
        elif cl == 'classe5':
            nameC = ["5","5Censorship"]            
        elif cl == 'classe6':
            nameC = ["6","6Censorship"]
        for classes in nameC:
            listeClasses.append(y_grp.loc[y_grp['group'] == classes].index)
        rcc=[[] for i in range(len(rates))]
        r=0
        for cc in range(len(listeClasses)):
            pa = 0
            while pa <len(listeClasses[cc]) and len(listeClasses[cc])!=0:
                rcc[r].append(listeClasses[cc][pa])
                pa+=1
                r+=1
                if r >=len(rates):
                    r=0

        rcc=[random.sample(rcc[i],len(rcc[i])) for i in range(len(rcc))]
        for i in range(len(rcc)):
            np.save(doss+"r"+ str(i),np.array(rcc[i]))
    return rcc



def classifyPatients(List_of_Patient,outnames = "./output.csv",sep=';') :
    filename = outnames
    data = pd.read_csv(filename,sep=sep,names=['patient','event','time'])
    patient = list(data.loc[:,'patient'])
    for i in range(len(data.loc[:,'patient'])):
        data=data.rename({i : patient[i]})
    data=data.drop(['patient'],axis = 1)
    y = data.copy()
    for pat in data.index:
        if pat not in List_of_Patient :
            y=y.drop(pat)
    
    y_grp = y.copy()
    y_grp = y_grp.drop('time',axis=1)
    y_grp=y_grp.rename(columns={'event': 'group'})
    
    
    for pat in y_grp.index:
        if y.loc[pat,'event'] =='1' or y.loc[pat,'event'] == 1:
            y_grp.loc[pat,'group'] = str(int(y.loc[pat,'time'])//365)
        else:
            y_grp.loc[pat,'group'] = str(int(y.loc[pat,'time'])//365)+'Censorship'
    return y_grp




def BalancedSeparation(List_of_Patient,file,rates=[0.5,0.25,0.25],classe = 'YEAR', change = True,outnames = "./output.csv"): 
    random.shuffle(List_of_Patient)
    
    #open the file with survival quantities
    filename = outnames
    data = pd.read_csv(filename,sep=';',names=['patient','event','time'])
    patient = list(data.loc[:,'patient'])
    for i in range(len(data.loc[:,'patient'])):
        data=data.rename({i : patient[i]})
    data=data.drop(['patient'],axis = 1)

    if classe=='surv':
        noC= 7
        y = data.copy()
        for pat in data.index:
            if pat not in List_of_Patient :
                y=y.drop(pat)
        y_grp = y.copy()
        y_grp = y_grp.drop('time',axis=1)
        y_grp=y_grp.rename(columns={'event': 'group'})
        
        for pat in y_grp.index:
            y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
            
    else:
        # correspondance between classe and number of group
        if classe == 'YEAR':   
            noC = 7
        elif classe == 'YEARWC':
            noC = 6
        elif classe == 'TWO':
            noC = 3
        else:
            noC = 2
        y_grp = YProcessing(data,List_of_Patient,classe)
    
    C= []
    for i in range(noC):
        C.append([])
    for i,n in enumerate(List_of_Patient):
        C[y_grp.loc[n][0]].append(n)
    name1 = file+"Balance.npy"
    name2 = file+"Ate.npy"
    name3 = file+"Aval.npy"

    if os.path.exists(name1) and os.path.exists(name2) and os.path.exists(name3) and change!= True:        
        Atr= np.load(name1)
        Ate = np.load(name2)
        Aval = np.load(name3)
    else:
        Atr =[]
        Ate = []
        Aval = []
        for j in range(len(C)):
            if len(C[j]) >=2:
                A = int(0.5*len(C[j]))
            else:
                A= len(C[j])
            B = int((len(C[j]) - A )/2)
            for k in range(A):
                Atr.append(C[j][k])
            for k in range(B):
                b = k + A
                Ate.append(C[j][b])
            for k in range(B):
                b = k + A + B
                Aval.append(C[j][b])                
        Atr = np.random.permutation(Atr)
        Ate = np.random.permutation(Ate)
        Aval = np.random.permutation(Aval)
        np.save(name3,Aval)
        np.save(name2,Ate)
        np.save(name1,Atr)
    return [Atr,Ate,Aval]



def saving(fitting,title,eval_train, eval_val,resu,file,metrics):
    if fitting != None:
        # Plot training And validation accuracy values
        plt.figure(figsize=(15,5), dpi=70)
        if metrics == 'accuracy':
            plt.subplot(1, 2 ,2)
            plt.plot(fitting.history['acc'], label="train")
            plt.plot(fitting.history['val_acc'], label="valid")
            titleA = title + '_accuracy'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            
        if metrics=='rankAndMSE':
            plt.subplot(2,2 ,3)
            plt.plot(fitting.history['mse'], label="train mse")
            plt.plot(fitting.history['val_mse'], label="valid mse")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_mse loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('mse loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2,2 ,4)
            plt.plot(fitting.history['rank'], label="train rank")
            plt.plot(fitting.history['val_rank'], label="valid rank")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_rank loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('rank loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2 ,2)
            plt.plot(fitting.history['tf_cindexT'], label="train")
            plt.plot(fitting.history['val_tf_cindexT'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
            
        elif metrics =='coxAndrank':
            plt.subplot(2,2 ,3)
            plt.plot(fitting.history['rank'], label="train rank")
            plt.plot(fitting.history['val_rank'], label="valid rank")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_rank loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('rank loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2,2 ,4)
            plt.plot(fitting.history['cox'], label="train cox")
            plt.plot(fitting.history['val_rank'], label="valid cox")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cox loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('cox loss')
            plt.xlabel('Epoch')
            plt.subplot(2, 2 ,2)
            plt.plot(fitting.history['tf_cindexR'], label="train")
            plt.plot(fitting.history['val_tf_cindexR'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
        elif metrics =='coxAndclassif':
            plt.subplot(3,2 ,6)
            plt.plot(fitting.history['Partial_likelihood_tf_cindexR'], label="train")
            plt.plot(fitting.history['val_Partial_likelihood_tf_cindexR'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_tf_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('cindex')
            plt.xlabel('Epoch')
            
            plt.subplot(3,2 ,5)
            plt.plot(fitting.history['Cross_entropy_loss_acc'], label="train")
            plt.plot(fitting.history['val_Cross_entropy_loss_acc'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_accuracy'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('accuracy')
            plt.xlabel('Epoch')
            
            plt.subplot(3,2 ,4)
            plt.plot(fitting.history['Cross_entropy_loss_loss'], label="classif loss")
            plt.plot(fitting.history['val_Cross_entropy_loss_loss'], label="classif loss")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_calssif_loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('classif loss')
            plt.xlabel('Epoch')
            
            plt.subplot(3, 2 ,3)
            plt.plot(fitting.history['Partial_likelihood_loss'], label="train")
            plt.plot(fitting.history['val_Partial_likelihood_loss'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cox_loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('cox loss')
            plt.xlabel('Epoch')
     
        elif metrics =='coxAndmse':
            plt.subplot(2,2 ,3)
            plt.plot(fitting.history['mse'], label="train mse")
            plt.plot(fitting.history['val_mse'], label="valid mse")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_mse loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('mse loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2,2 ,4)
            plt.plot(fitting.history['cox'], label="train cox")
            plt.plot(fitting.history['val_cox'], label="valid cox")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cox loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('cox loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2 ,2)
            plt.plot(fitting.history['tf_cindexR'], label="train")
            plt.plot(fitting.history['val_tf_cindexR'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
            
        elif metrics=='tf_cindex':
            plt.subplot(1, 2 ,2)
            plt.plot(fitting.history['tf_cindexR'], label="train")
            plt.plot(fitting.history['val_tf_cindexR'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
        elif metrics=='tf_cindexTD':
            plt.subplot(1, 2 ,2)
            plt.plot(fitting.history['tf_cindexTD'], label="train")
            plt.plot(fitting.history['val_tf_cindexTD'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
        elif metrics=='triplet':
            plt.subplot(2, 2 ,2)
            plt.plot(fitting.history['tf_cindexT'], label="train")
            plt.plot(fitting.history['val_tf_cindexT'], label="valid")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
            
            plt.subplot(2,2 ,4)            
            plt.plot(fitting.history['acc'], label="train")
            plt.plot(fitting.history['val_acc'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_accuracy'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
        elif metrics == 'RankAnddiscret':
            plt.subplot(2,2 ,3)
            plt.plot(fitting.history['rankL'], label="train rank")
            plt.plot(fitting.history['val_rankL'], label="valid rank")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_rank loss'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('rank loss')
            plt.xlabel('Epoch')
            
            plt.subplot(2,2 ,4)
            plt.plot(fitting.history['tf_cindexTD'], label="train")
            plt.plot(fitting.history['val_tf_cindexTD'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_cindex'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
            
            plt.subplot(2, 2 ,2)
            plt.plot(fitting.history['discretcox'], label="train")
            plt.plot(fitting.history['val_discretcox'], label="valid")
            #plt.plot(fitting.history['test_acc'], label="test")
            titleA = title + '_surv_likelihood'
            plt.title(titleA)
            pylab.legend(loc='upper left')
            plt.ylabel('C index')
            plt.xlabel('Epoch')
            
            
        if metrics == 'cox'or metrics == 'accuracy' or metrics == 'tf_cindexTD':
            plt.subplot(1, 2 ,1)
        elif metrics == 'coxAndclassif':
            plt.subplot(3,2 ,1)
        else:
            plt.subplot(2, 2 ,1)
        plt.plot(fitting.history['loss'], label="train")
        plt.plot(fitting.history['val_loss'], label="valid")
        #  plt.plot(fitting.history['test_loss'], label="test")
        titleL = title + '_loss'
        plt.title(titleL)
        pylab.legend(loc='upper left')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        #plt.show()
        pathSaveFIG = file+ '/' +title + '_curves.png'
        plt.savefig(pathSaveFIG)
    plt.close('all')
    results = pd.DataFrame(np.zeros((2, 2)),columns=['Loss',metrics],index = ['train','val'])
    results.loc["train",metrics] =eval_train[1]
    results.loc["train",'Loss'] = eval_train[0]
    results.loc["val",metrics] = eval_val[1]
    results.loc["val",'Loss'] = eval_val[0]
    if resu != None:
        results.loc["test",metrics] = resu
#    PATHEV= file + '/'  + title + 'evaluation.csv'
#    results.to_csv(PATHEV)
    return results


def makeTifFromPile(pathToPile,correct):
    '''Takes an absolute path containing a pile of masks, compute the resulting .tif mask and
     output the path to the created .tif mask'''
    list_pileFiles = []
    for dirpath, dirnames, fileNames in os.walk(pathToPile):
        for fileName in fileNames:
            if not fileName.startswith('.'):
                list_pileFiles.append(os.path.join(dirpath, fileName))
    change_name(list_pileFiles)

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
        shapeLocalMask = getWords(tifFile.readline())  
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


def predictionShow(name,y,prediction,writer,sess, file, titleP,log_dir, number=1000):
    if len(y) < number:
        number=len(y)
    if len(np.shape(prediction)) == 1:
        prediction = np.reshape(np.array(prediction), (-1 ,1))
    # y = np.array(y)
    print(np.shape(y))
    print(np.shape(prediction))
    rank = prediction[0:number,0].argsort() #il faut aller chercher le .. puis le .. puis le ...
    
    array=np.array([y[rank,0],y[rank,1],prediction[rank,0]])                  

    evalu = [tf.convert_to_tensor([str(array[0,i]) for i in range(number)]),
             tf.convert_to_tensor([str(array[1,i]) for i in range(number)]),
            tf.convert_to_tensor([str(array[2,i]) for i in range(number)])         ]
    s = sess.run(tf.summary.text('Ranked{}Prediction'.format(name), tf.stack(evalu)))
    writer.add_summary(s)
    
    axi=plt.plot(array[2,np.where(array[0,:]==0)],array[1,np.where(array[0,:]==0)],'ro',label='with censorship')                    
    axi=plt.plot(array[2,np.where(array[0,:]==1)],array[1,np.where(array[0,:]==1)],'bo',label='without censorship')
    axi=plt.xlabel('{} prediction'.format(name))
    axi=plt.ylabel('real time')
    axi=plt.title('time according to {} prediction (red: censorship)'.format(name))
    axi.get_figure().savefig(file+'/' + titleP +'{}Pred.png'.format(name))
    plt.close()
    curves = plt.imread(file+'/' + titleP +'{}Pred.png'.format(name))
    
    writer = tf.summary.FileWriter(log_dir)
    fi = tf.convert_to_tensor(np.reshape( curves,(1,np.shape(curves)[0],np.shape(curves)[1],np.shape(curves)[2])))
    s = sess.run(tf.summary.image('{}predictions'.format(name), tf.stack(fi)))
    writer.add_summary(s)
    plt.close()
    return axi

def findBestSep(ytrain,predictiontrain):
    if len(np.shape(predictiontrain))==2:
        predd = predictiontrain[:,0]
    else:
        predd = predictiontrain
    df = pd.DataFrame(np.array([ytrain[:,0],ytrain[:,1],predd,np.zeros((len(ytrain[:,0])))]).T)
    df.columns=['event','time','prediction','groupe']
    #pour chaque separation calculer log rank
    bestPval = 1000
    testStat = 0
    bestPval2 = 1000
    testStat2 = 0
    bestsep2 = 1000
    bestsep =1000
    numG02=0
    numG12=0
    sep = (np.max(predd) - np.min(predd))/20
    for ki in range(1,20):
        ssp = np.min(predd) + sep*ki
        groupe1 = df[df['prediction']<=ssp]
        groupe2 = df[df['prediction']>ssp]
        if len(groupe1) >= 0.1*len(df) and len(groupe2) >= 0.1*len(df):
            # group1=df[df['groupe']==0]
            # group2=df[df['groupe']==1]
            T1=groupe1['time']
            E1=groupe1['event']
            T2=groupe2['time']
            E2=groupe2['event']
        
            results=logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
            if results.p_value < bestPval:
                bestPval = results.p_value
                testStat = results.test_statistic
                bestsep = ssp
                numG0= len(groupe1)
                numG1 = len(groupe2)
        elif len(groupe1) >= 2 and len(groupe2) >= 2:
            T1=groupe1['time']
            E1=groupe1['event']
            T2=groupe2['time']
            E2=groupe2['event']
        
            results=logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
            if results.p_value < bestPval2:
                bestPval2 = results.p_value
                testStat2 = results.test_statistic
                bestsep2 = ssp
                numG02= len(groupe1)
                numG12 = len(groupe2)
            #○results.print_summary()
    if bestPval == 1000:
        bestPval  =bestPval2
        testStat =testStat2
        bestsep=     bestsep2
        numG0=numG02
        numG1=numG12
            
    return bestPval,testStat,bestsep,numG0,numG1 

def groupeByBestSep(ytrain,predictiontrain,bestsep,file,titleP,name,sess,typee = 'risk'):    
    if len(np.shape(predictiontrain))==2:
        predd = predictiontrain[:,0]
    else:
        predd = predictiontrain
    df = pd.DataFrame(np.array([ytrain[:,0],ytrain[:,1],predd,np.zeros((len(ytrain[:,0])))]).T)
    df.columns=['event','time','prediction','groupe']
    groupe1 = df[df['prediction']<=bestsep]
    groupe2 = df[df['prediction']>bestsep]
    results=logrank_test(groupe1['time'],groupe2['time'],event_observed_A=groupe1['event'], event_observed_B=groupe2['event'])
    pval = results.p_value
    numG0= len(groupe1)
    numG1 = len(groupe2)
    if typee == 'time':
        kmf = KaplanMeierFitter()
        kmf.fit(groupe1['time'], groupe1['event'], label='bad prognosis')
        axa = kmf.plot()
        kmf = KaplanMeierFitter()
        kmf.fit(groupe2['time'], groupe2['event'], label='good prognosis')
        axa = kmf.plot(ax=axa)
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(groupe1['time'], groupe1['event'], label='good prognosis')
        axa = kmf.plot()
        kmf = KaplanMeierFitter()
        kmf.fit(groupe2['time'], groupe2['event'], label='bad prognosis')
        axa = kmf.plot(ax=axa)
    
    axa.get_figure().savefig(file+'/' + titleP +'{}Kaplan.png'.format(name))   
    plt.close()
    curves2 = plt.imread(file+'/' + titleP +'{}Kaplan.png'.format(name))
    
    log_dir = file+'/' + "logs/fit/" +titleP +'/' #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    writer = tf.summary.FileWriter(log_dir)
    fi2 = tf.convert_to_tensor(np.reshape( curves2,(1,np.shape(curves2)[0],np.shape(curves2)[1],np.shape(curves2)[2])))
    s = sess.run(tf.summary.image('{}kaplanMeier'.format(name), tf.stack(fi2)))
    writer.add_summary(s)
    plt.close()
    return  pval,bestsep,numG0,numG1 