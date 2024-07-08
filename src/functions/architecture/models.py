# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
"""

import tensorflow.keras.backend as K
import tensorflow as tf
import re
from pathlib import Path
import numpy as np
import keras
from keras_contrib.layers.normalization.instancenormalization  import InstanceNormalization
from keras import regularizers
from keras.utils import to_categorical
from tensorflow  import  keras
from keras.layers import Dense,LeakyReLU, Conv2D,Conv3D,Input, Dropout,MaxPool2D ,MaxPool3D, BatchNormalization,GlobalAveragePooling3D,AveragePooling2D,AveragePooling3D,GlobalAveragePooling2D
from keras.layers import Lambda
from keras.models import Model,Sequential

from BprocessFunction import patchPHN, patchP, ground_truth_processing
from attention_module import attach_attention_module
from attention_module3D import attach_attention_module3D

import data_processing as dpr
from tensorflow.python.ops import gen_nn_ops
from typing import Any

def outer_product(x:list[Any]):
    """calculate outer-products of 2 tensors
    
    Args    
        x (list[Any]):list of 2 tensors assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    
    """
    outer = keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 
    
    return outer


def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)


def L2_norm(x:Any, axis: int=-1):
    ''' L2 normalisation 
    Args
        x (Any): input to normalise
        axis (int): Axis for normalisation. Default: -1
    '''
    return keras.backend.l2_normalize(x, axis=axis)


def expend_dimension(x):
    ''' Expend the dimension
    '''
    rgb_x= [[[[x[i,h,w],0,0] for w in range(np.shape(x)[2])]  for h in range(np.shape(x)[1])] for i in range(np.shape(x)[0])]
    
    expended = np.array(rgb_x, dtype=np.float32)
    
    return expended


def getWords(text):
    '''From a text input as a string, get the words separated by a space 
    and return it as a list of strings
    '''
    
    return re.compile('\w+').findall(text)


def ConvSurvBefore(interpolation_size:int, D3:str = 'False', spp:bool = True): #box shape (B,4) box : (xmin,ymin, h,w)
    """Create the base model

    Args:
        interpolation_size (int): size of the input
        D3 (str, optional): If True, the input is 3D, if False the input is 2D, if 25 the input is a stack of 3 2D images Defaults to False.
        spp (bool, optional): If True, use SPP block. Defaults to True.

    """
    if D3=='False': 
        input = Input(shape=(interpolation_size, interpolation_size, 1), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,1))
        modelCN = conv_netFirst(interpolation_size,False,spp=spp)
        
    elif D3=='25': 
        input = Input(shape=(interpolation_size, interpolation_size, 3,1), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,3,1))
        modelCN = conv_netFirst(interpolation_size,True,spp=spp)
        
    else:
        input = Input(shape=(interpolation_size, interpolation_size, interpolation_size,1), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,interpolation_size,1))
        modelCN = conv_netFirst(interpolation_size,True,spp=spp)
        
    output = modelCN(input) 
    
    if spp:
        spp = Lambda(SPP, name='spp')([output,mask]) 
    else:
        spp = output 
    model = Model(inputs = [input,mask], outputs = spp)
    
    return model
  
  
def PrepDataCrossVal(list_pat_ref:list,
                     path_patches: Path, 
                     interpolation_size:int = 36,
                     reload_patches: bool = True, 
                     classes:str = 'surv',
                     D3:str = 'False',
                     method:str='all',
                     dataAug:int=0,
                     npatch:int=9,
                     dataset:str='mm'):
    """Data preparation

    Args:
        list_pat_ref (list): list of patients
        path_patches (Path): path to the patches
        interpolation_size (int, optional):  size of the input. Defaults to 36.
        reload_patches (bool, optional): reload patches. Defaults to True.
        classes (str, optional): type of ground truth. Defaults to 'surv'.
        D3 (str, optional): If True, the input is 3D, if False the input is 2D, if 25 the input is a stack of 3 2D images Defaults to False.
        method (str, optional): Method for patches. Defaults to 'all'.
        dataAug (int, optional): Number of images for data augmentation. Defaults to 0.
        npatch (int, optional): number of patches to extract. Defaults to 9.
        dataset (str, optional): dataset. Defaults to 'mm'.

    """
    if dataset == 'hn':
        data, liste_patches, liste_ref, liste_patients,errors,liste_mask = patchPHN(path_patches,reload_patches,interpolation_size,method,D3,npatch,dataset='hn')
    else:
        data, liste_patches, liste_ref, liste_patients,errors,liste_mask = patchP(path_patches,reload_patches,interpolation_size,method,D3,npatch)
    
    y_grp = ground_truth_processing(data, liste_patients, classes)
    
    X,Y,M,Ref = ([[] for i in range(len(list_pat_ref))], ) * 4 
    
    for i in range(len(liste_ref)):
        for k in range(len(list_pat_ref)):
            if liste_ref[i] in list_pat_ref[k]:
                X[k].append(liste_patches[i])
                M[k].append(liste_mask[i])
                Ref[k].append(liste_ref[i])

    for k in range(len(list_pat_ref)) :   
        if D3=='False':
            X[k] = dpr.normalize(np.expand_dims(np.array(X[k]),axis = 3)) 
            M[k] = dpr.normalize(np.expand_dims(np.array(M[k]),axis = 3)) 
        
        else:
            X[k] = dpr.normalize(np.expand_dims(np.array(X[k]),axis = 4)) 
            M[k] = dpr.normalize(np.expand_dims(np.array(M[k]),axis = 4)) 
        
        if classes != 'surv':
            Y[k] = to_categorical(np.asarray(y_grp.loc[Ref[k]]))
        
        else:
            Y[k] = np.asarray(y_grp.loc[Ref[k]])
            
    return X, Y, M, Ref

  
def conv_netFirst(size: int, D3 = 'False', spp:bool = True):
    """Convolutions block

    Args:
        size (int): input size
        D3 (str, optional): If 3D, 2D or 2.5. Defaults to 'False'.
        spp (bool, optional): If True, SPP block will be used. Defaults to True.
    """
    if D3 == 'False': 
        if D3 =='25':
            third_axis = 3
        else:
            third_axis = 1
        model = Sequential()
#        model.add(Dropout(rate=0.17))
        model.add(Conv2D(filters=16, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,third_axis), 
                                   name="conv1",kernel_regularizer=regularizers.l2(0.001)))
#        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())

        if not spp:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2
        model.add(Conv2D(filters=32, kernel_size=5,
                                   strides=1, padding="same", input_shape=(size,size,third_axis), 
                                   name="conv2",kernel_regularizer=regularizers.l2(0.001)))
#        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.1))
        #model.add(InstanceNormalization())

        if not spp:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2    
            
        model.add(Conv2D(filters=64, kernel_size=3,
                                   strides=1, padding="same",input_shape=(size,size,third_axis), 
                                   name="conv3",kernel_regularizer=regularizers.l2(0.001)))  
#        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.1))
        #model.add(InstanceNormalization())

        if not spp:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2
        
    else: 
        third_axis = size
        model = Sequential() #filtre 1 depm 16
        model.add(Conv3D(filters=16, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,third_axis,1), 
                                   name="conv1",kernel_regularizer=regularizers.l2(0.001)))
                                   
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        
        if not spp:
            model.add(AveragePooling3D(pool_size=(2,2,2))) # dephm 3 ou 2
        model.add(Conv3D(filters=32, kernel_size=5,
                                   strides=1, padding="same", input_shape=(size,size,third_axis,1), 
                                   name="conv2",kernel_regularizer=regularizers.l2(0.001)))
        
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        
        if not spp:
            model.add(AveragePooling3D(pool_size=(2,2,2)))    
        model.add(Conv3D(filters=64, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,third_axis,1), 
                                   name="conv3",kernel_regularizer=regularizers.l2(0.001)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        
        if not spp:
            model.add(AveragePooling3D(pool_size=(2,2,2)))    
            
    return model  


def deepConvSurv(interpolation_size:int,
                 num:int = 3, 
                 mode:str = 'cox',
                 D3 = 'False',
                 spp:bool = True,
                 attention:str = 'False',
                 rate:float = 0.17,
                 conv1:bool = False,
                 l11:float = 0.001):
    """_summary_

    Args:
        interpolation_size (int): _description_
        num (int, optional): _description_. Defaults to 3.
        mode (str, optional): _description_. Defaults to 'cox'.
        D3 (str, optional): _description_. Defaults to 'False'.
        spp (bool, optional): _description_. Defaults to True.
        attention (str, optional): _description_. Defaults to 'False'.
        rate (float, optional): _description_. Defaults to 0.17.
        conv1 (bool, optional): _description_. Defaults to False.
        l11 (float, optional): _description_. Defaults to 0.001.

    Returns:
        _type_: _description_
    """
    if D3=='False': 
        input = Input(shape=(interpolation_size, interpolation_size, 1), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,1))
        modelCN = conv_netFirst(interpolation_size,False,spp=spp)
    
    elif D3=='25':
        input = Input(shape=(interpolation_size, interpolation_size, 3), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,3))
        modelCN = conv_netFirst(interpolation_size,'25',spp=spp)
        
    else: 
        input = Input(shape=(interpolation_size, interpolation_size, interpolation_size,1), name='input')
        mask = Input(shape=(interpolation_size,interpolation_size,interpolation_size,1))
        modelCN = conv_netFirst(interpolation_size,True,spp=spp)
    
    output = modelCN(input) 
    
    if spp == 'B4':
        spp21= Lambda(SPP4, name='spp')([output,mask]) 
        if  D3=='False':
            numm = 1024            
            output = keras.layers.Reshape([4,4,64],name=('resh11'))(spp21)

        else:
            numm = 4096
            output = keras.layers.Reshape([4,4,4,64],name=('resh11'))(spp21)

    elif spp == 'B42':
        spp21= Lambda(SPP, name='spp')([output,mask]) 
        
        if  D3=='False':
            numm = 1280
        elif D3=='25':
            numm = 1280
        else:
            numm = 4608
            
        if attention != 'False':
            if  D3=='True':
                output = keras.layers.Reshape([8,8,8,9],name=('resh8889'))(spp21)
            else:                
                output = keras.layers.Reshape([8,8,9],name=('resh8889'))(spp21)

        else:
            output = keras.layers.Reshape([numm],name=('resh22'))(spp21)
            
    if attention != 'False':
        if  D3=='False' or D3=='25':
            y = attach_attention_module(output, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
        else:
            y = attach_attention_module3D(output, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
        
        x = keras.layers.add([output, y])
        output = keras.layers.Activation('relu')(x)
    
    if spp == 'A4':
        spp21= Lambda(SPP4, name='spp')([output,mask]) 
        if  D3=='False' or D3=='25':
            numm = 1024
        else:
            numm = 4096
        output = keras.layers.Reshape([numm],name=('resh33'))(spp21)

    elif spp == 'A42':
        spp21= Lambda(SPP, name='spp')([output,mask]) 
        if  D3=='False':
            numm = 1280
        elif D3=='25':
            numm = 1280
        else:
            numm = 4608
        output = keras.layers.Reshape([numm],name=('resh44'))(spp21)
    
    elif spp == 'False' or not spp:
        if D3=='False' or D3=='25':
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh48'))(output)
        else: #(?,46656)
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]*K.int_shape(output)[4]],name=('resh48'))(output)  
    
    elif spp == 'B4' or spp=='B42':
        if D3=='False' or D3=='25':
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh55'))(output)
        else: #(?,46656)
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]*K.int_shape(output)[4]],name=('resh55'))(output)  
    
    if mode == 'tripletAnddiscret' or mode == 'tripletAndcox' or mode == 'tripletContinueAndcox' :
        model2 = Dense(100, activation='softmax', name="triplet_output",kernel_regularizer=regularizers.l2(0.001))(output)
        
        
    if mode != 'triplet' :
        if not conv1:
            model = Dense(100, name="fc3",kernel_regularizer=regularizers.l2(0.01),
                          bias_initializer='zeros')(output)
        else:
            if D3=='True':
                model = Conv3D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,4,1), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            elif D3=='False':
                model = Conv2D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,1), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            else:
                model = Conv2D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,3), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(rate = rate)(model)
        
        if conv1:
            if D3=='True':
                model = GlobalAveragePooling3D()(model)
            else:
                model = GlobalAveragePooling2D()(model)
                
        if mode == 'classif':
            model = Dense(num,activation = 'softmax', name="fc4",kernel_regularizer=regularizers.l2(l11))(model)
        
        elif mode == 'coxAndclassif':
            model1 = Dense(1, activation='linear', name="Partial_likelihood",kernel_regularizer=regularizers.l2(l11))(model)
            model2 = Dense(num,activation = 'softmax', name="Cross_entropy_loss",kernel_regularizer=regularizers.l2(l11))(model)
        
        elif mode== 'discret':
            stride = 365
            breaks=np.arange(0.,365*7,stride)
            n_intervals=len(breaks)
            model= Dense(n_intervals,  name="output",activation = 'softmax',kernel_regularizer=regularizers.l2(l11))(model)

        else:
            model = Dense(1, activation='linear', name="cox_output",kernel_regularizer=regularizers.l2(l11))(model)
        
        if mode == 'tripletAnddiscret' or mode == 'tripletAndcox'or mode == 'tripletContinueAndcox' :
            extractor2 = Model(inputs=[input,mask], outputs=[model,model2])
        else:
            extractor2 = Model(inputs=[input,mask], outputs=model)
    # elif mode=='tripletContinue':
    #     model1 = Dense(4096, activation='softmax', name="triplet_output",kernel_regularizer=regularizers.l2(0.001))(output)
    #     extractor2 = Model(inputs=[input,mask], outputs=model1)

    else:
        extractor2 = Model(inputs=[input,mask], outputs=output)
    return extractor2



def Prep25(x, m, y, r):
    x2, m2 =([],)*2
    
    for i in range(len(x)):
        x2.append([x[i,18,:,:,:],x[i,:,18,:,:],x[i,:,:,18,:]])
        m2.append([m[i,18,:,:,:],m[i,:,18,:,:],m[i,:,:,18,:]])
        
    x2= np.array(x2).transpose(0,2,3,1,4)
    m2= np.array(m2).transpose(0,2,3,1,4)
    
    return x2, m2, y, r

def PrepData(rtrain, rtest, rval,PATH,interpolation_size=36,reload_patches=True,classes='surv',D3 = 'False',method='all',dataAug=0,npatch=9):
        
    data, liste_patches, liste_ref, liste_patients,errors,liste_mask = patchP(PATH,reload_patches,interpolation_size,method,D3,npatch)
    y_grp = ground_truth_processing(data,liste_patients,classes)
    xtrain, reftr, xtest, refte , xval, refv, mtrain, mtest, mval =([],)*9
    
    for i in range(len(liste_ref)):
        if liste_ref[i] in rtrain:
            xtrain.append(liste_patches[i])
            mtrain.append(liste_mask[i])
            reftr.append(liste_ref[i])
        if liste_ref[i] in rtest:
            xtest.append(liste_patches[i])
            mtest.append(liste_mask[i])
            refte.append(liste_ref[i])
        if liste_ref[i] in rval:
            xval.append(liste_patches[i])
            mval.append(liste_mask[i])
            refv.append(liste_ref[i])
    
    dict_d3 = {'False':3,
               'True':4,
               '25':4}
    axis_dim = dict_d3[D3]
    
    for x_temp in [xtest,xtrain,xval, mtest, mtrain, mval]:
        x_temp = dpr.normalize(np.expand_dims(np.array(x_temp),axis = axis_dim)) 
             
    yval = np.asarray(y_grp.loc[refv])
    ytest = np.asarray(y_grp.loc[refte])
    ytrain = np.asarray(y_grp.loc[reftr])
    
    if classes != 'surv':
        yval = to_categorical(y_grp.loc[refv])
        ytest = to_categorical(y_grp.loc[refte])
        ytrain = to_categorical(y_grp.loc[reftr])

    return ytrain, ytest, yval, xtrain, xtest, xval, reftr, refte, refv, mtrain, mtest, mval



def maxPool3DFixeSize(input: Any, n:int, ksize:int, strides:int):
    l=0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a=tf.reduce_max(input[:,i*strides[0]:i*strides[0]+ksize[0],
                   j*strides[1]:j*strides[1]+ksize[1],k*strides[2]:k*strides[2]+ksize[2],:],axis=(1,2,3))                 

                if l!=0:
                    out = tf.concat((out,a),axis=0)
                else:
                    out=a
                l=l+1
    out = tf.reshape(out,(1,n,n,n,K.int_shape(input)[4]))
            
    return out

def while_conditionF(i,output,new,box,bins):
    return tf.less(i, tf.shape(output)[0]) # surement None, a changer

def SPP_Net(input, pyramid_Levels,D3 = 'False'):
    input_shape = tf.shape(input) # (14,14,3)
    pyramid_list=[]
    Pyramid = [2,4]
    
    for n in Pyramid:
        stride_0 = tf.cast(tf.floor(tf.cast(input_shape[1] / n, tf.float32)),tf.int32)
        stride_1 = tf.cast(tf.floor(tf.cast(input_shape[2] / n, tf.float32)),tf.int32)
        
        Kernel_size_0 = stride_0 + (input_shape[1] % n)
        Kernel_size_1 = stride_1 + (input_shape[2] % n)
        
        if D3=='False':
            pool = gen_nn_ops.max_pool_v2(input,
                                ksize=[ 1,Kernel_size_0, Kernel_size_1, 1],
                                strides=[1,stride_0, stride_1, 1],
                                padding='SAME')
        else: 
            stride_2 = tf.cast(tf.floor(tf.cast(input_shape[3] / n, tf.float32)),tf.int32)
            Kernel_size_2 = stride_2 + (input_shape[3] % n)
            
            pool = maxPool3DFixeSize(input,n,(Kernel_size_0,Kernel_size_1, Kernel_size_2), (stride_0,stride_1, stride_2))
       
        pyramid_list.append(tf.reshape(pool, [1, -1]))
    
    spp_out_fixed_size = tf.concat(pyramid_list, axis=1)
    
    return spp_out_fixed_size


def SPP_Net4(input, Pyramid_Levels,D3 = 'False'):
    
    input_shape = tf.shape(input) # (14,14,3)
    
    pyramid_list=[]
    Pyramid = [4]
    
    for n in Pyramid:
        stride_0 = tf.cast(tf.floor(tf.cast(input_shape[1] / n, tf.float32)),tf.int32)
        stride_1 = tf.cast(tf.floor(tf.cast(input_shape[2] / n, tf.float32)),tf.int32)
        Kernel_size_0 = stride_0 + (input_shape[1] % n)
        Kernel_size_1 = stride_1 + (input_shape[2] % n)
        if D3=='False':
            pool = gen_nn_ops.max_pool_v2(input,
                                ksize=[ 1,Kernel_size_0, Kernel_size_1, 1],
                                strides=[1,stride_0, stride_1, 1],
                                padding='SAME')
        else: 
            stride_2 = tf.cast(tf.floor(tf.cast(input_shape[3] / n, tf.float32)),tf.int32)
            Kernel_size_2 = stride_2 + (input_shape[3] % n)
            pool=maxPool3DFixeSize(input,n,(Kernel_size_0,Kernel_size_1, Kernel_size_2), (stride_0,stride_1, stride_2))
        pyramid_list.append(tf.reshape(pool, [1, -1]))
    spp_out_fixed_size = tf.concat(pyramid_list, axis=1)
    return spp_out_fixed_size


def true(h2):
    return 36-h2

def true28(h2):
    return 28-h2
def false(h2):
    return h2+tf.cast(4-(h2%4),dtype=tf.int32)

def bodyF(i,output,new,mask,bins):
    """ Pour chaque individu on va prendre l'image et la box, changer taille image, max pooling et remettre dans new"""
    if K.int_shape(mask)[3] == 36 or K.int_shape(mask)[3] == 28:
        maski = mask[i,:,:,:,0]#36*36
    else:
        maski = mask[i,:,:,0]#36*36
    g=tf.not_equal(maski,tf.zeros(tf.shape(maski)))
    a=tf.where(g)#36*36
    a0 =a[:,0]
    a1 = a[:,1]
    if K.int_shape(mask)[3] ==36 or K.int_shape(mask)[3] ==28 :
        a2 = a[:,2]
    xmin = tf.cast(tf.reduce_min(a0), dtype = tf.int32)
    ymin = tf.cast(tf.reduce_min(a1), dtype = tf.int32)
    h2 = tf.cast(tf.reduce_max(a0)-tf.reduce_min(a0), dtype = tf.int32)
    w2 = tf.cast( tf.reduce_max(a1)-tf.reduce_min(a1)  , dtype = tf.int32)
    h2=tf.cond(tf.equal(h2%4,tf.constant(0)),lambda: h2,lambda:false(h2))
    w2=tf.cond(tf.equal(w2%4,tf.constant(0)),lambda: w2,lambda:false(w2))
    h2=tf.cond(tf.less(h2,tf.constant(4)),lambda: tf.constant(4),lambda:h2)
    w2=tf.cond(tf.less(w2,tf.constant(4)),lambda: tf.constant(4),lambda:w2)
    if K.int_shape(mask)[2] ==28 : 
        xmin=tf.cond(tf.greater(h2+xmin,tf.constant(27)),lambda: true28(h2),lambda:xmin)
        ymin=tf.cond(tf.greater(w2+ymin,tf.constant(27)),lambda: true28(w2),lambda:ymin)
    elif K.int_shape(mask)[2] ==36:
        xmin=tf.cond(tf.greater(h2+xmin,tf.constant(35)),lambda: true(h2),lambda:xmin)
        ymin=tf.cond(tf.greater(w2+ymin,tf.constant(35)),lambda: true(w2),lambda:ymin)
    if K.int_shape(mask)[3] ==36 or K.int_shape(mask)[3] ==28:
        zmin = tf.cast(tf.reduce_min(a2), dtype = tf.int32)
        p2 = tf.cast( tf.reduce_max(a2)-tf.reduce_min(a2)  , dtype = tf.int32)
        p2=tf.cond(tf.equal(p2%4,tf.constant(0)),lambda: p2,lambda:false(p2))
        p2=tf.cond(tf.less(p2,tf.constant(4)),lambda: tf.constant(4),lambda:p2)
        if K.int_shape(mask)[2] ==28 : 
            zmin=tf.cond(tf.greater(p2+zmin,tf.constant(27)),lambda: true28(p2),lambda:zmin)
        elif K.int_shape(mask)[2] ==36:
            zmin=tf.cond(tf.greater(p2+zmin,tf.constant(35)),lambda: true(p2),lambda:zmin)

    output2 = tf.gather_nd(output,[[i]])
    
    
    if K.int_shape(mask)[3]==36 or K.int_shape(mask)[3]==28: 
        new2=tf.slice(output2,[0,xmin,ymin,zmin,0],[1,h2,w2,p2,K.int_shape(output2)[4]],name='new23D')
        D3 = 'True'
    else:
        new2=tf.slice(output2,[0,xmin,ymin,0],[1,h2,w2,K.int_shape(output2)[3]],name='new22D')
        D3 = 'False'
    new3=new2
    spp_out_fixed_size = SPP_Net(new3,bins,D3)
    g2=tf.cond(tf.equal(i,tf.constant(0)), lambda:spp_out_fixed_size, lambda:tf.concat([new,spp_out_fixed_size],axis=0 ) )
    return [tf.add(i, 1),output,g2,mask,bins]

def bodyF4(i,output,new,mask,bins):
    """ Pour chaque individu on va prendre l'image et la box, changer taille image, max pooling et remettre dans new"""
    if K.int_shape(mask)[3] == 36 or K.int_shape(mask)[3] == 28:
        maski = mask[i,:,:,:,0]#36*36
    else:
        maski = mask[i,:,:,0]#36*36
    g=tf.not_equal(maski,tf.zeros(tf.shape(maski)))
    a=tf.where(g)#36*36
    a0 =a[:,0]
    a1 = a[:,1]
    if K.int_shape(mask)[3] ==36 or K.int_shape(mask)[3] ==28 :
        a2 = a[:,2]
    xmin = tf.cast(tf.reduce_min(a0), dtype = tf.int32)
    ymin = tf.cast(tf.reduce_min(a1), dtype = tf.int32)
    h2 = tf.cast(tf.reduce_max(a0)-tf.reduce_min(a0), dtype = tf.int32)
    w2 = tf.cast( tf.reduce_max(a1)-tf.reduce_min(a1)  , dtype = tf.int32)
    h2=tf.cond(tf.equal(h2%4,tf.constant(0)),lambda: h2,lambda:false(h2))
    w2=tf.cond(tf.equal(w2%4,tf.constant(0)),lambda: w2,lambda:false(w2))
    h2=tf.cond(tf.less(h2,tf.constant(4)),lambda: tf.constant(4),lambda:h2)
    w2=tf.cond(tf.less(w2,tf.constant(4)),lambda: tf.constant(4),lambda:w2)
    if K.int_shape(mask)[2] ==28 : 
        xmin=tf.cond(tf.greater(h2+xmin,tf.constant(27)),lambda: true28(h2),lambda:xmin)
        ymin=tf.cond(tf.greater(w2+ymin,tf.constant(27)),lambda: true28(w2),lambda:ymin)
    elif K.int_shape(mask)[2] ==36:
        xmin=tf.cond(tf.greater(h2+xmin,tf.constant(35)),lambda: true(h2),lambda:xmin)
        ymin=tf.cond(tf.greater(w2+ymin,tf.constant(35)),lambda: true(w2),lambda:ymin)
    if K.int_shape(mask)[3] ==36 or K.int_shape(mask)[3] ==28:
        zmin = tf.cast(tf.reduce_min(a2), dtype = tf.int32)
        p2 = tf.cast( tf.reduce_max(a2)-tf.reduce_min(a2)  , dtype = tf.int32)
        p2=tf.cond(tf.equal(p2%4,tf.constant(0)),lambda: p2,lambda:false(p2))
        p2=tf.cond(tf.less(p2,tf.constant(4)),lambda: tf.constant(4),lambda:p2)
        if K.int_shape(mask)[2] ==28 : 
            zmin=tf.cond(tf.greater(p2+zmin,tf.constant(27)),lambda: true28(p2),lambda:zmin)
        elif K.int_shape(mask)[2] ==36:
            zmin=tf.cond(tf.greater(p2+zmin,tf.constant(35)),lambda: true(p2),lambda:zmin)

    output2 = tf.gather_nd(output,[[i]])
    
    
    if K.int_shape(mask)[3]==36 or K.int_shape(mask)[3]==28: 
        new2=tf.slice(output2,[0,xmin,ymin,zmin,0],[1,h2,w2,p2,K.int_shape(output2)[4]],name='new23D')
        D3 = 'True'
    else:
        new2=tf.slice(output2,[0,xmin,ymin,0],[1,h2,w2,K.int_shape(output2)[3]],name='new22D')
        D3 = 'False'
    new3=new2
    spp_out_fixed_size = SPP_Net4(new3,bins,D3)
    g2=tf.cond(tf.equal(i,tf.constant(0)), lambda:spp_out_fixed_size, lambda:tf.concat([new,spp_out_fixed_size],axis=0 ) )
    return [tf.add(i, 1),output,g2,mask,bins]


def SPP(output1):
    """Spacial pyramide pooling layer
    
    Input: output = output of the bounding box layer
    output.shape = (Batch, h,w, c)
    
    Bin: number of bin
    """    
    bins = tf.constant([2,4])
    output = output1[0]
    mask = output1[1]
    i = tf.constant(0)
    if  K.int_shape(mask)[3] != 1:
        
        numm = 4608
    else:
        numm = 1280
    new = tf.stack(tf.zeros([1,numm]),name='new')    #new: (B,sh)
    r = tf.while_loop(while_conditionF, bodyF, loop_vars=[i,output,new,mask,bins], shape_invariants=[i.get_shape(),output.get_shape(), tf.TensorShape([None,None]), mask.get_shape(), bins.get_shape()])           
    return r[2]



def SPP4(output1):
    """Spacial pyramide pooling layer
    
    Input: output = output of the bounding box layer
    output.shape = (Batch, h,w, c)
    
    Bin: number of bin
    """    
    bins = tf.constant([4])
    output = output1[0]
    mask = output1[1]
    i = tf.constant(0)
    if  K.int_shape(mask)[3] != 1:
        numm = 4096
    else:
        numm = 1024
    new = tf.stack(tf.zeros([1,numm]),name='new')    #new: (B,sh)
    r = tf.while_loop(while_conditionF, bodyF4, loop_vars=[i,output,new,mask,bins], shape_invariants=[i.get_shape(),output.get_shape(), tf.TensorShape([None,None]), mask.get_shape(), bins.get_shape()])           
    return r[2]
