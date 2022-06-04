# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
"""

import tensorflow.keras.backend as K
import tensorflow as tf
import re
import numpy as np
from skimage import io
from keras.layers import Input, Dense, Conv2D, Flatten,  Dropout,MaxPool3D #,# BatchNormalization#,GlobalAveragePooling2D
import keras
from triplet_loss import batch_hard_triplet_loss
import losses as lss
import BprocessFunction as pf
import pandas as pd
from vis.utils import utils
from attention_module import attach_attention_module
from attention_module3D import attach_attention_module3D
from keras import regularizers
from sklearn.model_selection import ParameterGrid
from keras_contrib.layers.normalization.instancenormalization  import InstanceNormalization
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import accuracy_score
import pylab
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow  import  keras
from keras.layers import Dense,LeakyReLU, Conv1D, Conv2D,Conv3D,Input, Flatten, Dropout,MaxPool2D ,MaxPool3D, BatchNormalization,GlobalAveragePooling3D,AveragePooling2D,AveragePooling3D,GlobalAveragePooling2D
from keras.optimizers import adam
from keras.layers import Lambda
from keras.applications.vgg16 import VGG16
from keras.models import Model,Sequential,load_model#, load_model
import time
import tensorflow
import data_processing as dpr
from tensorflow.python.ops import gen_nn_ops



def outer_product(x):
    """     calculate outer-products of 2 tensors
        args    x =      list of 2 tensors assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    ff=keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 
    return ff

def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    ''' L2 normalisation '''
    return keras.backend.l2_normalize(x, axis=axis)

def expDim(xtest):
    ''' Expend the dimension'''
    rgbXtest= [[[[xtest[i,h,w],0,0] for w in range(np.shape(xtest)[2])]  for h in range(np.shape(xtest)[1])] for i in range(np.shape(xtest)[0])]
    a=np.array(rgbXtest, dtype=np.float32)
    return a

def getWords(text):
    '''From a text input as a string, get the words separated by a space and return it as a list of strings'''
    return re.compile('\w+').findall(text)
    


def ConvSurvBefore(interpolSize, D3=False,spp=True): #box shape (B,4) box : (xmin,ymin, h,w)
    if D3==False: 
        input = Input(shape=(interpolSize, interpolSize, 1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,False,spp=spp)
    elif D3=='25': 
        input = Input(shape=(interpolSize, interpolSize, 3,1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,3,1))
        modelCN = conv_netFirst(interpolSize,True,spp=spp)
    else:
        input = Input(shape=(interpolSize, interpolSize, interpolSize,1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,True,spp=spp)
    output = modelCN(input) 
    if spp == True:
        spp= Lambda(SPP, name='spp')([output,mask]) 
    else:
        spp = output # keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh44'))(output)
    model = Model(inputs = [input,mask], outputs = spp)
    return model
  
def PrepDataCrossVal(RR,PATH,change=False,interpolSize=36,reload=True,classes='surv',D3=False,method='all',dataAug=0,npatch=9,dataset='mm'):
    if dataset=='hn':
        data, liste_patches, liste_ref, liste_patients,errors,liste_mask = pf.patchPHN(PATH,reload,interpolSize,method,D3,npatch,dataset='hn')
    else:
        data, liste_patches, liste_ref, liste_patients,errors,liste_mask = pf.patchP(PATH,reload,interpolSize,method,D3,npatch)
    y_grp = pf.YProcessing(data,liste_patients,classes)
    X,Y,M,Ref = [[] for i in range(len(RR))], [[] for i in range(len(RR))],[[] for i in range(len(RR))],[[] for i in range(len(RR))]
    for i in range(len(liste_ref)):
        for k in range(len(RR)):
            if liste_ref[i] in RR[k]:
                X[k].append(liste_patches[i])
                M[k].append(liste_mask[i])
                Ref[k].append(liste_ref[i])
    print(np.shape(X[0]))
    for k in range(len(RR)) :   
        print(k)
        if D3==False:
            X[k] = dpr.normalize(np.expand_dims(np.array(X[k]),axis = 3)) 
            M[k] = dpr.normalize(np.expand_dims(np.array(M[k]),axis = 3)) 
        else:
            X[k] = dpr.normalize(np.expand_dims(np.array(X[k]),axis = 4)) 
            M[k] = dpr.normalize(np.expand_dims(np.array(M[k]),axis = 4)) 
        if classes != 'surv':
            Y[k] = to_categorical(np.asarray(y_grp.loc[Ref[k]]))
        else:
            Y[k] = np.asarray(y_grp.loc[Ref[k]])
    return X,Y,M,Ref

  
def conv_netFirst(size,D3=False,spp=True,maxi=False):
    if D3!=True: 
        if D3=='25':
            Third = 3
        else:
            Third = 1
        model = Sequential()
#        model.add(Dropout(rate=0.17))
        model.add(Conv2D(filters=16, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,Third), 
                                   name="conv1",kernel_regularizer=regularizers.l2(0.001)))
#        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())

        if spp==False:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2
        model.add(Conv2D(filters=32, kernel_size=5,
                                   strides=1, padding="same", input_shape=(size,size,Third), 
                                   name="conv2",kernel_regularizer=regularizers.l2(0.001)))
#        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.1))
        #model.add(InstanceNormalization())

        if spp==False:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2    
            
        model.add(Conv2D(filters=64, kernel_size=3,
                                   strides=1, padding="same",input_shape=(size,size,Third), 
                                   name="conv3",kernel_regularizer=regularizers.l2(0.001)))  
#        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.1))
        #model.add(InstanceNormalization())

        if spp==False:
            model.add(AveragePooling2D(pool_size=(2,2))) # dephm 3 ou 2
        
    else: 
        Third = size
        model = Sequential() #filtre 1 depm 16
        model.add(Conv3D(filters=16, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,Third,1), 
                                   name="conv1",kernel_regularizer=regularizers.l2(0.001)))
                                   # kernel_initializer='zeros',
                                   #bias_initializer='zeros'))
                                   
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        if spp==False:
            model.add(AveragePooling3D(pool_size=(2,2,2))) # dephm 3 ou 2
        model.add(Conv3D(filters=32, kernel_size=5,
                                   strides=1, padding="same", input_shape=(size,size,Third,1), 
                                   name="conv2",kernel_regularizer=regularizers.l2(0.001)))
                                    #kernel_initializer='zeros',
                                    #bias_initializer='zeros'
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        if spp==False:
            model.add(AveragePooling3D(pool_size=(2,2,2)))    
        model.add(Conv3D(filters=64, kernel_size=3,
                                   strides=1, padding="same", input_shape=(size,size,Third,1), 
                                   name="conv3",kernel_regularizer=regularizers.l2(0.001)))
                                    # kernel_initializer='zeros',
                                     #bias_initializer='zeros'   
        model.add(LeakyReLU(alpha=0.1))
        model.add(InstanceNormalization())
        
        if spp==False:
            model.add(AveragePooling3D(pool_size=(2,2,2)))    
    return model  

def deepConvSurv(interpolSize,num=3, mode = 'cox',D3=False,spp=True,attention='False',rate=0.17,conv1 = False,maxi=False,l11 = 0.001): #box shape (B,4) box : (xmin,ymin, h,w)
    if D3==False: 
        input = Input(shape=(interpolSize, interpolSize, 1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,False,spp=spp,maxi=maxi)
    elif D3=='25':
        input = Input(shape=(interpolSize, interpolSize, 3), name='input')
        mask = Input(shape=(interpolSize,interpolSize,3))
        modelCN = conv_netFirst(interpolSize,'25',spp=spp,maxi=maxi)
        
    else: 
        input = Input(shape=(interpolSize, interpolSize, interpolSize,1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,True,spp=spp,maxi=maxi)
    
    output = modelCN(input) 
    
    if spp == 'B4':
        spp21= Lambda(SPP4, name='spp')([output,mask]) 
        if  D3==False:
            numm = 1024            
            output = keras.layers.Reshape([4,4,64],name=('resh11'))(spp21)

        else:
            numm = 4096
            output = keras.layers.Reshape([4,4,4,64],name=('resh11'))(spp21)

    elif spp == 'B42':
        spp21= Lambda(SPP, name='spp')([output,mask]) 
        if  D3==False:
            numm = 1280
        elif D3=='25':
            numm = 1280
        else:
            numm = 4608
            
        if attention != 'False':
            if  D3==True:
                output = keras.layers.Reshape([8,8,8,9],name=('resh8889'))(spp21)
            else:                
                output = keras.layers.Reshape([8,8,9],name=('resh8889'))(spp21)

        else:
            output = keras.layers.Reshape([numm],name=('resh22'))(spp21)
            
    if attention != 'False':
        if  D3==False or D3=='25':
            y = attach_attention_module(output, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
        else:
            y = attach_attention_module3D(output, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
        x = keras.layers.add([output, y])
        output = keras.layers.Activation('relu')(x)
    
    if spp == 'A4':
        spp21= Lambda(SPP4, name='spp')([output,mask]) 
        if  D3==False or D3=='25':
            numm = 1024
        else:
            numm = 4096
        output = keras.layers.Reshape([numm],name=('resh33'))(spp21)

    elif spp == 'A42':
        spp21= Lambda(SPP, name='spp')([output,mask]) 
        if  D3==False:
            numm = 1280
        elif D3=='25':
            numm = 1280
        else:
            numm = 4608
        output = keras.layers.Reshape([numm],name=('resh44'))(spp21)
    elif spp == 'False' or spp == False:
        if D3==False or D3=='25':
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh48'))(output)
        else: #(?,46656)
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]*K.int_shape(output)[4]],name=('resh48'))(output)  
    elif spp == 'B4' or spp=='B42':
        if D3==False or D3=='25':
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh55'))(output)
        else: #(?,46656)
            output = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]*K.int_shape(output)[4]],name=('resh55'))(output)  
    if mode == 'tripletAnddiscret' or mode == 'tripletAndcox' or mode == 'tripletContinueAndcox' :
        model2 = Dense(100, activation='softmax', name="triplet_output",kernel_regularizer=regularizers.l2(0.001))(output)
        
        
    if mode != 'triplet' :
        if conv1 == False:
            model = Dense(100, name="fc3",kernel_regularizer=regularizers.l2(0.01),
                          bias_initializer='zeros')(output)
        else:
            if D3==True:
                model = Conv3D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,4,1), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            elif D3==False:
                model = Conv2D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,1), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            else:
                model = Conv2D(filters=64, kernel_size=1,
                          strides=1, padding="same", input_shape=(4,4,3), 
                          name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(output)
            
          
        #kernel_initializer='zeros',
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(rate = rate)(model)
        if conv1==True:
            if D3==True:
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




def Prep25(x,m,y,r):
    x2=[]
    m2=[]
    for i in range(len(x)):
        x2.append([x[i,18,:,:,:],x[i,:,18,:,:],x[i,:,:,18,:]])
        m2.append([m[i,18,:,:,:],m[i,:,18,:,:],m[i,:,:,18,:]])
    x2= np.array(x2).transpose(0,2,3,1,4)
    m2= np.array(m2).transpose(0,2,3,1,4)
    return x2, m2,y,r

def PrepData(rtrain, rtest, rval,PATH,change=False,interpolSize=36,reload=True,classes='surv',D3=False,method='all',dataAug=0,npatch=9):
    
    
    data, liste_patches, liste_ref, liste_patients,errors,liste_mask = pf.patchP(PATH,reload,interpolSize,method,D3,npatch)
    y_grp = pf.YProcessing(data,liste_patients,classes)
    xtrain, reftr, xtest, refte , xval, refv, mtrain, mtest, mval =[],[],[],[],[],[],[],[],[]
    for i in range(len(liste_ref)):
        # #print(i)
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
    if D3==False:

        xtest = dpr.normalize(np.expand_dims(np.array(xtest),axis = 3)) 
        xtrain = dpr.normalize(np.expand_dims(np.array(xtrain),axis = 3))
        xval = dpr.normalize(np.expand_dims(np.array(xval),axis = 3))
        
        mtest = dpr.normalize(np.expand_dims(np.array(mtest),axis = 3)) 
        mtrain = dpr.normalize(np.expand_dims(np.array(mtrain),axis = 3))
        mval = dpr.normalize(np.expand_dims(np.array(mval),axis = 3))
    else:
        xtest = dpr.normalize(np.expand_dims(np.array(xtest),axis = 4)) 
        xtrain = dpr.normalize(np.expand_dims(np.array(xtrain),axis = 4))
        xval = dpr.normalize(np.expand_dims(np.array(xval),axis = 4))
        
        mtest = dpr.normalize(np.expand_dims(np.array(mtest),axis = 4)) 
        mtrain = dpr.normalize(np.expand_dims(np.array(mtrain),axis = 4))
        mval = dpr.normalize(np.expand_dims(np.array(mval),axis = 4))
    if classes != 'surv':
        yval = to_categorical(np.asarray(y_grp.loc[refv]))
        ytest = to_categorical(np.asarray(y_grp.loc[refte]))
        ytrain = to_categorical(np.asarray(y_grp.loc[reftr]))
    else:
        yval = np.asarray(y_grp.loc[refv])
        ytest = np.asarray(y_grp.loc[refte])
        ytrain = np.asarray(y_grp.loc[reftr])
                
    return ytrain, ytest, yval, xtrain, xtest, xval, reftr, refte, refv, mtrain, mtest, mval



def maxPool3DFixeSize(input,n,ksize, strides):
    l=0
    # print(ksize[0].eval(),ksize[1].eval(),ksize[2].eval(),strides[0].eval(),strides[1].eval(),strides[2].eval())
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a=tf.reduce_max(input[:,i*strides[0]:i*strides[0]+ksize[0],
                   j*strides[1]:j*strides[1]+ksize[1],k*strides[2]:k*strides[2]+ksize[2],:],axis=(1,2,3))                 

                if l!=0:
                    out = tf.concat((out,a),axis=0)
                else:
                    out=a
                # print(i,j,k,tf.reduce_min(out).eval(),tf.reduce_min(a).eval())

                l=l+1
#    for i in range(n):
#        pool = gen_nn_ops.max_pool_v2(input,
#                                ksize=[ 1,Kernel_size_1, Kernel_size_2, 1],
#                                strides=[1,stride_1, stride_2, 1],
#                                padding='SAME')
    out = tf.reshape(out,(1,n,n,n,K.int_shape(input)[4]))
    
            
    return out

# def VGG():
#     input = Input(shape=(240, 240, 155), name='input')
#     vgg16 = VGG16(weights='imagenet', include_top=False)
#     output = vgg16(input) # sortie cnn1
#     x = Flatten(name='flatten')(output)
#     #### dense 512
#     x1 = Dense(256, activation='relu',name='dense1')(x)
#     extractor = Model(inputs=input, outputs=x1)
#     return extractor


def while_conditionF(i,output,new,box,bins):
    return tf.less(i, tf.shape(output)[0]) # surement None, a changer

def SPP_Net(input, Pyramid_Levels,D3=False):
    input_shape = tf.shape(input) # (14,14,3)
    pyramid_list=[]
    Pyramid = [2,4]
    for n in Pyramid:
        stride_0 = tf.cast(tf.floor(tf.cast(input_shape[1] / n, tf.float32)),tf.int32)
        stride_1 = tf.cast(tf.floor(tf.cast(input_shape[2] / n, tf.float32)),tf.int32)
        Kernel_size_0 = stride_0 + (input_shape[1] % n)
        Kernel_size_1 = stride_1 + (input_shape[2] % n)
        if D3==False:
            pool = gen_nn_ops.max_pool_v2(input,
                                ksize=[ 1,Kernel_size_0, Kernel_size_1, 1],
                                strides=[1,stride_0, stride_1, 1],
                                padding='SAME')
        else: 
            stride_2 = tf.cast(tf.floor(tf.cast(input_shape[3] / n, tf.float32)),tf.int32)
            Kernel_size_2 = stride_2 + (input_shape[3] % n)
#            a1=tf.stack(tf.zeros([Kernel_size_0,Kernel_size_1,Kernel_size_2]))
#            pool=MaxPool3D(pool_size=(Kernel_size_0, Kernel_size_1, Kernel_size_2), strides=(stride_0,stride_1,stride_2))(input)
            pool=maxPool3DFixeSize(input,n,(Kernel_size_0,Kernel_size_1, Kernel_size_2), (stride_0,stride_1, stride_2))
            
            # def tf_print(op, tensors, message=None):
            #     def print_message(x):
            #         sys.stdout.write(message + " %s\n" % x)
            #         return x
            
            #     prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
            #     with tf.control_dependencies(prints):
            #         op = tf.identity(op)
            #     return op
            # Kernel_size_0 = tf_print(Kernel_size_0, [Kernel_size_0], 'Kernel_size_0 = ')
            # Kernel_size_1 = tf_print(Kernel_size_1, [Kernel_size_1], 'Kernel_size_1 = ')
            # Kernel_size_2 = tf_print(Kernel_size_2, [Kernel_size_2], 'Kernel_size_2 = ')

            # stride_0 = tf_print(stride_0, [stride_0], 'stride_0 = ')
            # stride_1 = tf_print(stride_1, [stride_1], 'stride_1 = ')
            # stride_2 = tf_print(stride_2, [stride_2], 'stride_2 = ')

            # maxi=tf.nn.max_pool(input,
            #                     ksize=(Kernel_size_0,Kernel_size_1, Kernel_size_2,1),
            #                     strides=(stride_0,stride_1, stride_2,1), padding='SAME')
#            pool = gen_nn_ops.max_pool_v2(input,
#                                ksize=[ 1,Kernel_size_0,Kernel_size_1, Kernel_size_2, 1],
#                                strides=[1,stride_0,stride_1, stride_2, 1],
#                                padding='SAME')

        pyramid_list.append(tf.reshape(pool, [1, -1]))
    spp_out_fixed_size = tf.concat(pyramid_list, axis=1)
    return spp_out_fixed_size


def SPP_Net4(input, Pyramid_Levels,D3=False):
    input_shape = tf.shape(input) # (14,14,3)
    pyramid_list=[]
    Pyramid = [4]
    for n in Pyramid:
        stride_0 = tf.cast(tf.floor(tf.cast(input_shape[1] / n, tf.float32)),tf.int32)
        stride_1 = tf.cast(tf.floor(tf.cast(input_shape[2] / n, tf.float32)),tf.int32)
        Kernel_size_0 = stride_0 + (input_shape[1] % n)
        Kernel_size_1 = stride_1 + (input_shape[2] % n)
        if D3==False:
            pool = gen_nn_ops.max_pool_v2(input,
                                ksize=[ 1,Kernel_size_0, Kernel_size_1, 1],
                                strides=[1,stride_0, stride_1, 1],
                                padding='SAME')
        else: 
            stride_2 = tf.cast(tf.floor(tf.cast(input_shape[3] / n, tf.float32)),tf.int32)
            Kernel_size_2 = stride_2 + (input_shape[3] % n)
#            a1=tf.stack(tf.zeros([Kernel_size_0,Kernel_size_1,Kernel_size_2]))
#            pool=MaxPool3D(pool_size=(Kernel_size_0, Kernel_size_1, Kernel_size_2), strides=(stride_0,stride_1,stride_2))(input)
            pool=maxPool3DFixeSize(input,n,(Kernel_size_0,Kernel_size_1, Kernel_size_2), (stride_0,stride_1, stride_2))
            
            # def tf_print(op, tensors, message=None):
            #     def print_message(x):
            #         sys.stdout.write(message + " %s\n" % x)
            #         return x
            
            #     prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
            #     with tf.control_dependencies(prints):
            #         op = tf.identity(op)
            #     return op
            # Kernel_size_0 = tf_print(Kernel_size_0, [Kernel_size_0], 'Kernel_size_0 = ')
            # Kernel_size_1 = tf_print(Kernel_size_1, [Kernel_size_1], 'Kernel_size_1 = ')
            # Kernel_size_2 = tf_print(Kernel_size_2, [Kernel_size_2], 'Kernel_size_2 = ')

            # stride_0 = tf_print(stride_0, [stride_0], 'stride_0 = ')
            # stride_1 = tf_print(stride_1, [stride_1], 'stride_1 = ')
            # stride_2 = tf_print(stride_2, [stride_2], 'stride_2 = ')

            # maxi=tf.nn.max_pool(input,
            #                     ksize=(Kernel_size_0,Kernel_size_1, Kernel_size_2,1),
            #                     strides=(stride_0,stride_1, stride_2,1), padding='SAME')
#            pool = gen_nn_ops.max_pool_v2(input,
#                                ksize=[ 1,Kernel_size_0,Kernel_size_1, Kernel_size_2, 1],
#                                strides=[1,stride_0,stride_1, stride_2, 1],
#                                padding='SAME')

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
    #marche si on garde 4 et 2 ou juste 4 ou juste 2, sinon on doit trouver une soluton plus pereine
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
        D3=True
    else:
        new2=tf.slice(output2,[0,xmin,ymin,0],[1,h2,w2,K.int_shape(output2)[3]],name='new22D')
        D3=False
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
    #marche si on garde 4 et 2 ou juste 4 ou juste 2, sinon on doit trouver une soluton plus pereine
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
        D3=True
    else:
        new2=tf.slice(output2,[0,xmin,ymin,0],[1,h2,w2,K.int_shape(output2)[3]],name='new22D')
        D3=False
    new3=new2
    spp_out_fixed_size = SPP_Net4(new3,bins,D3)
    g2=tf.cond(tf.equal(i,tf.constant(0)), lambda:spp_out_fixed_size, lambda:tf.concat([new,spp_out_fixed_size],axis=0 ) )
    return [tf.add(i, 1),output,g2,mask,bins]

 
# print(xmin.eval(),ymin.eval(),zmin.eval())
# print(h2.eval(),w2.eval(),p2.eval())
# da = np.reshape(xtrain[1,:,:,:,0],(36,36))[xmin:xmin+h2,ymin:ymin+w2,zmin:zmin+p2]
# print(np.min(da),np.max(da))



def SPP(output1):
    """Spacial pyramide pooling layer
    
    Input: output = output of the bounding box layer
    output.shape = (Batch, h,w, c)
    
    Bin: number of bin
    """    
    bins = tf.constant([2,4])
    output = output1[0]
    mask = output1[1]
    # size = tf.map_fn(lambda x: K.int_shape(output)[3]*x*x , bins) #[64*4,64*16] if bins=[2,4] 
    # sh= tf.reduce_sum(size)
    i = tf.constant(0)
    if  K.int_shape(mask)[3] != 1:
        
        numm = 4608
    else:
        numm = 1280
    new = tf.stack(tf.zeros([1,numm]),name='new')    #new: (B,sh)
    r = tf.while_loop(while_conditionF, bodyF, loop_vars=[i,output,new,mask,bins], shape_invariants=[i.get_shape(),output.get_shape(), tf.TensorShape([None,None]), mask.get_shape(), bins.get_shape()])           
    # if  K.int_shape(mask)[3] == 36:
    #     rr = tf.reshape(r[2],[tf.shape(r[2])[0],8,8,8,9])
    # else:
    #     rr = tf.reshape(r[2],[tf.shape(r[2])[0],8,10,16])
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
    # size = tf.map_fn(lambda x: K.int_shape(output)[3]*x*x , bins) #[64*4,64*16] if bins=[2,4] 
    # sh= tf.reduce_sum(size)
    i = tf.constant(0)
    if  K.int_shape(mask)[3] != 1:
        numm = 4096
    else:
        numm = 1024
    new = tf.stack(tf.zeros([1,numm]),name='new')    #new: (B,sh)
    r = tf.while_loop(while_conditionF, bodyF4, loop_vars=[i,output,new,mask,bins], shape_invariants=[i.get_shape(),output.get_shape(), tf.TensorShape([None,None]), mask.get_shape(), bins.get_shape()])           
    # if  K.int_shape(mask)[3] == 36:
    #     rr = tf.reshape(r[2],[tf.shape(r[2])[0],8,8,8,9])
    # else:
    #     rr = tf.reshape(r[2],[tf.shape(r[2])[0],8,10,16])
    return r[2]

######### Bilinear model ########
def BilinearModel(interpolSize,modelCN = None,modelCN2 = None,name2 = "small_2",num=3, mode = 'cox',D3=False,spp=True,attention='False'):
    if D3==False: 
        input = Input(shape=(interpolSize, interpolSize, 1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,1))
    else: 
        input = Input(shape=(interpolSize, interpolSize, interpolSize,1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,interpolSize,1))
    modelCN2.name=name2
    output1 = modelCN([input,mask]) # sortie cnn1 ##shape = (?,9, 9, 64) en non spp
    output2 = modelCN2([input,mask])
    output1_shape = K.int_shape(output1)    
    output2_shape = K.int_shape(output2)
    if D3==False:
        output1 = keras.layers.Reshape([output1_shape[1]*output1_shape[2], output1_shape[-1]])(output1) # (None, 81, 64)  
        output2 = keras.layers.Reshape([output2_shape[1]*output2_shape[2], output2_shape[-1]])(output2)
    else:
        output1 = keras.layers.Reshape([output1_shape[1]*output1_shape[2]*output1_shape[3], output1_shape[-1]])(output1) # (None, 81, 64)  
        output2 = keras.layers.Reshape([output2_shape[1]*output2_shape[2]*output1_shape[3], output2_shape[-1]])(output2)
    inputs = [output1, output2]
    bilinearProduct = Lambda(outer_product)(inputs)  #(64*64)
    
    res = keras.layers.Reshape([output1_shape[-1]*output2_shape[-1]])(bilinearProduct)
    x = keras.layers.Lambda(signed_sqrt)(res)
    x = keras.layers.Lambda(L2_norm)(x)
#    model = Dense(500,  name="fc1",kernel_regularizer=regularizers.l2(0.001))(x)
#    model = LeakyReLU(alpha=0.1)(model)
#    model = Dropout(rate = 0.17)(model)
#    model = Dense(1000,  name="fc2",kernel_regularizer=regularizers.l2(0.001))(model)
#    model = LeakyReLU(alpha=0.1)(model)
#    model = Dropout(rate = 0.17)(model)
    model = Dense(100,  name="fc3",kernel_regularizer=regularizers.l2(0.001))(x)
    model = LeakyReLU(alpha=0.1)(model)
    model = Dropout(rate = 0.17)(model)
    if mode == 'classif':
        model = Dense(num,activation = 'softmax', name="fc4",kernel_regularizer=regularizers.l2(0.001))(model)
    elif mode == 'cox&classif':
        model1 = Dense(1, activation='linear', name="Partial_likelihood",kernel_regularizer=regularizers.l2(0.001))(model)
        model2 = Dense(num,activation = 'softmax', name="Cross_entropy_loss",kernel_regularizer=regularizers.l2(0.001))(model)
    else:
        model = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(model)
    if mode == 'cox&classif':
        extractor2 = Model(inputs=[input,mask], outputs=[model1,model2])
    else:
        extractor2 = Model(inputs=[input,mask], outputs=model)
    return extractor2


############# Fusion with radiomics ###############

def deepConvSurv(interpolSize,num=3, mode = 'cox',D3=False,spp=True,attention='False',rate=0.17,conv1 = False,radiomics=False): #box shape (B,4) box : (xmin,ymin, h,w)
    #to use fusion with radiomics
    if D3==False: 
        input = Input(shape=(interpolSize, interpolSize, 1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,False,spp=spp)
    elif D3=='2.5':
        input = Input(shape=(interpolSize, interpolSize, 3), name='input')
        mask = Input(shape=(interpolSize,interpolSize,3))
        modelCN = conv_netFirst(interpolSize,'2.5',spp=spp)
        
    else: 
        input = Input(shape=(interpolSize, interpolSize, interpolSize,1), name='input')
        mask = Input(shape=(interpolSize,interpolSize,interpolSize,1))
        modelCN = conv_netFirst(interpolSize,True,spp=spp)
    output = modelCN(input) 
    if spp == True: 
        # spp21 = RoiPooling3D([output,mask])
        spp21= Lambda(SPP, name='spp')([output,mask]) 
        if  D3==False:
            numm = 1280
        elif D3=='2.5':
            numm = 1280
        else:
            numm = 4608
        spp2 = keras.layers.Reshape([numm],name=('resh44'))(spp21)
    else:
        if D3==False or D3=='2.5':
            spp2 = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]],name=('resh44'))(output)
        else: #(?,46656)
            spp2 = keras.layers.Reshape([K.int_shape(output)[1]*K.int_shape(output)[2]*K.int_shape(output)[3]*K.int_shape(output)[4]],name=('resh44'))(output)
    if D3 != True:
        if attention!='False':
            if spp== True:
                if D3=='2.5':
                    spp2=keras.layers.Reshape([8,10,16],name=('res'))(spp2)
                else:
                    spp2=keras.layers.Reshape([8,10,16],name=('res'))(spp2)
            
            else:
                spp2=keras.layers.Reshape([K.int_shape(output)[1],K.int_shape(output)[2],K.int_shape(output)[3]],name=('res'))(spp2)
            y = attach_attention_module(spp2, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
            x = keras.layers.add([spp2, y])

            
            ######## TO decomment if removing convolv 1*1 ##############
            if conv1 == False:
                x = keras.layers.Activation('relu')(x)

                spp2=keras.layers.Reshape([K.int_shape(spp2)[1]*K.int_shape(spp2)[2]*K.int_shape(spp2)[3]],name=('res2'))(x)
            else:
                spp2= keras.layers.Activation('relu')(x)

            #############################################################
    else:        
        if attention!='False':
            if spp== True:
                spp2=keras.layers.Reshape([8,8,8,9],name=('res'))(spp2)
            else:
                spp2=keras.layers.Reshape([K.int_shape(output)[1],K.int_shape(output)[2],K.int_shape(output)[3],K.int_shape(output)[4]],name=('res'))(spp2)

            y = attach_attention_module3D(spp2, 'cbam_block',order=attention) #https://github.com/kobiso/CBAM-keras
            x = keras.layers.add([spp2, y])

            x = keras.layers.Activation('relu')(x)
            ######## TO decomment if removing convolv 1*1 ##############
            if conv1 == False:            
                x = keras.layers.Activation('relu')(x)
                spp2=keras.layers.Reshape([K.int_shape(spp2)[1]*K.int_shape(spp2)[2]*K.int_shape(spp2)[3]*K.int_shape(spp2)[4]],name=('res2'))(x)
            else:
                spp2= keras.layers.Activation('relu')(x)

    spp2 = InstanceNormalization()(spp2)
            #############################################################
    radiomics_input = Input((11, ), name='radiomics_input')
    
    # Concatenate the features
    if radiomics == True:
        spp2 = Concatenate(name='concatenation')([spp2, radiomics_input]) 
            
    #    model = Dense(300,  name="fc1",kernel_regularizer=regularizers.l2(0.001))(spp2)
    #    model = LeakyReLU(alpha=0.1)(model)
    #    model = Dropout(rate = 0.17)(model)
    #    model = Dense(500,  name="fc2",kernel_regularizer=regularizers.l2(0.001))(model)
    #    model = LeakyReLU(alpha=0.1)(model)
    #    model = Dropout(rate = 0.17)(model)
    if conv1 == False:
        model = Dense(100, name="fc3",kernel_regularizer=regularizers.l2(0.01),
                      bias_initializer='zeros')(spp2)
    else:
        if D3==True:
            model = Conv3D(filters=64, kernel_size=1,
                      strides=1, padding="same", input_shape=(4,4,4,1), 
                      name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(spp2)
        elif D3==False:
            model = Conv2D(filters=64, kernel_size=1,
                      strides=1, padding="same", input_shape=(4,4,1), 
                      name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(spp2)
        else:
            model = Conv2D(filters=64, kernel_size=1,
                      strides=1, padding="same", input_shape=(4,4,3), 
                      name="conv1b1",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(spp2)
        
      
    #kernel_initializer='zeros',
    model = LeakyReLU(alpha=0.1)(model)
    model = Dropout(rate = rate)(model)
    if conv1==True:
        if D3==True:
            model = GlobalAveragePooling3D()(model)
        else:
            model = GlobalAveragePooling2D()(model)
    
    if mode == 'classif':
        model = Dense(num,activation = 'softmax', name="fc4",kernel_regularizer=regularizers.l2(0.001))(model)
    
    elif mode == 'cox&classif':
        model1 = Dense(1, activation='linear', name="Partial_likelihood",kernel_regularizer=regularizers.l2(0.001))(model)
        model2 = Dense(num,activation = 'softmax', name="Cross_entropy_loss",kernel_regularizer=regularizers.l2(0.001))(model)
    elif mode== 'discret':
        stride = 365
        breaks=np.arange(0.,365*7,stride)
        n_intervals=len(breaks)-1
        model= Dense(n_intervals,  name="output",activation = 'softmax',kernel_regularizer=regularizers.l2(0.001))(model)
    else:
        model = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(model)
    if radiomics == False:
        if mode == 'cox&classif':
            extractor2 = Model(inputs=[input,mask], outputs=[model1,model2])
        else:
            extractor2 = Model(inputs=[input,mask], outputs=model)
    else:
        extractor2 = Model(inputs=[input,mask,radiomics_input], outputs=model)

    return extractor2
