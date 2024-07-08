# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

"""
import os
from sklearn.utils import shuffle
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from functions.losses.triplet_loss import batch_hard_triplet_loss
import pandas as pd
from vis.utils import utils
from sklearn.model_selection import ParameterGrid
from keras_contrib.layers.normalization.instancenormalization  import InstanceNormalization
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import accuracy_score
from keras import regularizers
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense,LeakyReLU, Conv2D, Conv3D,Input, Flatten, MaxPool2D ,MaxPool3D, BatchNormalization#,GlobalAveragePooling2D
from keras.optimizers import adam
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import functions.architecture.models as mds
from functions.data_processing.data_processing import save_training_plots, ground_truth_processing2, normalize_batch, shuffle_and_interpolate
import functions.losses.losses as lss
import functions.data_processing.pretraining as ptg
import functions.data_processing.data_prep as dataprep
import functions.losses.survivalTripletloss as striplet

def display_data_stats(p:dict, list_y:dict, path_patients:Path ):
    """Display statistics and Kaplan Meier curves for a list of ground_truth vectors (one per fold)

    Args:
        p (dict): arguments
        list_y (dict): list of ground_truth vectors (one per fold)
        path_patients (Path): path to save curves
    """
    
    def stat(y,name):
        mean= np.mean(y[:,1])
        med = np.median(y[:,1])
        tauxCensor = (1 - np.mean(y[:,0]))*100
        print(f"{name}: Censorship {tauxCensor}%, mean time {mean}, median time {med}")

    for key, y in list_y.items(): # if not p.cross_validation, key = 'train', 'test' or 'val'
        stat(y, str(key))
        kmf = KaplanMeierFitter()
        kmf.fit(y[:,1], y[:,0], label=key)
        kmf.plot_survival_function()

    plt.title('Kaplan-Meier Survival Curves for each fold')
    plt.xlabel('Timeline')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig(os.path.join(path_patients,'datasep.png'))
    plt.show()
    plt.close()
    
def get_optimised_model(p:dict,parameters:dict, breaks:np.ndarray, dict_input:dict, sess:Any):
    n_intervals=len(breaks) 
        
    list_final_results = []
    path_csv_final_results = os.path.join(p.save_dir, f'finalEval_{p.training_dir}.csv')
    
    if os.path.exists(path_csv_final_results):
        list_final_results = pd.read_csv(path_csv_final_results, encoding='utf-8')
        list_final_results = list_final_results.iloc[:,1:].values.tolist()
    else:
        list_final_results = []
        
    binary=[]
    grid = ParameterGrid(parameters)
    
    num_opti = 0
    for params in grid:  
    num_opti=num_opti+1
    print('***************************************')
    print(f'****** run {grid} on {len(grid)} *******************')
    print('***************************************')
    print('***************************************')

    p.interpolation_size = params['image size']
    
    dict_param = { 'attention':[params['attention']],'dataAug':[params['dataAug']],'npatch': [params['npatch']],'D3': [params['D3']],'loss' : [params['Loss']], 'nb_classes':[params['nb_classes']],'pretrain': [params['pretrain_version']],'spp': [params['SPP']], 'Maxi':[params['Maxi']], 'LR Pretrain':[params['LR Pretrain']],'LR Finetune': [params['LR Finetune']],  'epoch F' : [params['epoch Finetune']], 'epoch T' : [params['epoch Train']],'interpolsize' : [p.interpolation_size] , 'batch': [params['batch size']],'Lrate_decay_learningP': [params['lrate_decay_learningF']],'lrate_decay_learningF': [params['lrate_decay_learningF']],'rate': [params['rate']],'noTrainLayer':[params['noTrainLayer']]}
    param = pd.DataFrame.from_dict(dict_param)
    print(dict_param)
                    
    list_mode=[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
    
    do = True        
        
    if params['LR Finetune'] > params['lrate_decay_learningF'] and do  : 
        lrd=str(params['lrate_decay_learningF'])[-1]
        if params['Loss'] == 'classif':
            metrics = 'accuracy'
        else: 
            metrics = lss.tf_cindexR
            
        if not p.apply_pretrain :
            if not p.cross_validation :
                if params['Loss']=='discret' or params['Loss'] == 'RankAnddiscret':
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest, ytrainT,yvalT,ytestT, rtrain, rtest,rval = dataprep.extract_dataIm(p, dict_input['list_refs'], sess, list_patients_ref, params)
                else :
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest , rtrain, rtest,rval = dataprep.extract_dataIm(p, dict_input['list_refs'], list_patients_ref, params['SPP'])
            else: 
                Xt,Yt,Mt,Rt, list_patients_ref = dataprep.extract_dataIm(p.cross_validation,dict_input['list_refs'],p.path_patients,p.path_data,sess,'all',p.distribution,p.interpolation_size,list_patients_ref,p.reload_inputs,params['SPP'],params['npatch'],params['D3'],params['Loss'])
            
            loss_train, loss_valid, AcurracyTrain, AcurracyVal, cindex_train,cindex_val,train_best, val_best =[],[],[],[],[],[],[],[]
            
        if not p.cross_validation :
            cv_number = 1
        else:
            cv_number = len(list_patients_ref)
            
        if p.use_test_set :
            cv_number = 1
            
        if not p.cross_validation :
            intermediate_folder = os.path.join(p.save_dir, p.training_dir, 'TrainTest')
            os.makedirs( intermediate_folder , exist_ok=True)
            
        for CV in range(0,cv_number):
            if p.cross_validation :
                intermediate_folder = os.path.join(p.save_dir, p.training_dir, f'CV{CV}')
                os.makedirs(intermediate_folder, exist_ok=True)
                
                path_full_df = os.path.join(intermediate_folder, f'FullCV_{CV}.csv')
                
                if os.path.exists(path_full_df):
                    cv_full = pd.read_csv(path_full_df, encoding='utf-8')
                    cv_full = cv_full.iloc[:,1:].values.tolist()
                else:
                    cv_full = []
                
            if p.cross_validation :
                if not p.apply_pretrain:
                    if len(p.distribution) > 4 :
                        if p.use_test_set:
                            xtest,ytest,mtest,rtest = Xt[CV],Yt[CV],Mt[CV],Rt[CV]
                            X=[Xt[1],Xt[2],Xt[3],Xt[4]]
                            M=[Mt[1],Mt[2],Mt[3],Mt[4]]
                            R=[Rt[1],Rt[2],Rt[3],Rt[4]]
                            Y=[Yt[1],Yt[2],Yt[3],Yt[4]]
                        else:
                            X=[Xt[0],Xt[1],Xt[2],Xt[3],Xt[4]]
                            M=[Mt[0],Mt[1],Mt[2],Mt[3],Mt[4]]
                            R=[Rt[0],Rt[1],Rt[2],Rt[3],Rt[4]]
                            Y=[Yt[0],Yt[1],Yt[2],Yt[3],Yt[4]]
                    else:
                        if p.use_test_set:
                            xtest,ytest,mtest,rtest=Xt[CV],Yt[CV],Mt[CV],Rt[CV]
                            X=[Xt[1],Xt[2],Xt[3]]
                            M=[Mt[1],Mt[2],Mt[3]]
                            R=[Rt[1],Rt[2],Rt[3]]
                            Y=[Yt[1],Yt[2],Yt[3]]
                        else:
                            X=[Xt[0],Xt[1],Xt[2],Xt[3]]
                            M=[Mt[0],Mt[1],Mt[2],Mt[3]]
                            R=[Rt[0],Rt[1],Rt[2],Rt[3]]
                            Y=[Yt[0],Yt[1],Yt[2],Yt[3]]
                    
                else:
                    if p.use_test_set:
                        if params['D3']=='True' :
                            Xb=[x_binary_train_3d[1],x_binary_train_3d[2],x_binary_train_3d[3],x_binary_train_3d[4]]
                            Yb=[y_binary_train_3d[1],y_binary_train_3d[2],y_binary_train_3d[3],y_binary_train_3d[4]]
                            # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                        else:
                            Xb=[x_binary_train_2d[1],x_binary_train_2d[2],x_binary_train_2d[3],x_binary_train_2d[4]]
                            Yb=[y_binary_train_2d[1],y_binary_train_2d[2],y_binary_train_2d[3],y_binary_train_2d[4]]
                    else:
                        if params['D3']=='True' :
                            Xb=[x_binary_train_3d[0],x_binary_train_3d[1],x_binary_train_3d[2],x_binary_train_3d[3]]
                            Yb=[y_binary_train_3d[0],y_binary_train_3d[1],y_binary_train_3d[2],y_binary_train_3d[3]]
                            # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                        else:
                            Xb=[x_binary_train_2d[0],x_binary_train_2d[1],x_binary_train_2d[2],x_binary_train_2d[3]]
                            Yb=[y_binary_train_2d[0],y_binary_train_2d[1],y_binary_train_2d[2],y_binary_train_2d[3]]
            if not p.cross_validation :
                KNum = 1
            else:
                KNum = len(list_patients_ref)-1
            if not p.use_test_set:
                KNum = len(list_patients_ref)

            for k in range(KNum):
            
                list_mode=[f'CV{CV}_K{k}']+[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
                
                for j in range(len(cv_full)):
                    fcv = [cv_full[j][0]]   + cv_full[j][-19:]  
                    fcv = [str(fcv[i]) for i in range(len(fcv))]
                    if np.array_equal(list_mode,fcv):
                        do=False
                                                
                na = f'{str(num_opti)}_{str(CV)}_{str(k)}' 
                
                titleP = f'{na}lrP{params['LR Pretrain']}_lrF{params['LR Finetune']}b{params['batch size']}_eF{params['epoch Finetune']}_lrd{lrd}_D{params['dataAug']}_r{params['rate']}'
                
                if params['Loss'] == 'classif' or params['Loss'] =='triplet':
                    loss_attribute = f"{params['Loss']}{str(params['nb_classes'])}"
                elif params['Loss'] == 'discret':
                    loss_attribute = 'disc' + str(n_intervals)
                else:
                    loss_attribute = params['Loss']
                                    
                loss_attribute = f"{loss_attribute}_Pre_{params['pretrain_version']}_"
                file_name = os.path.join(intermediate_folder, f'{loss_attribute}at{params['attention']}_spp{params['SPP']}_maxi{params['Maxi']}_D3{params['D3']}_n{params['noTrainLayer']}_dim{params['npatch']}')
                
                
                do=False 
                
                best_model_file = os.path.join(file_name,  f'{titleP}best_model.h5')
                
                if os.path.exists(best_model_file) or os.path.exists(best_model_file) :
                    do = True 
                    print('weight exist')
                if do :        
                    print(f'************* CV {CV} K {k}*********************')
                    print('***************************************')

                    if "sess" in locals():     # this code prevent creating many intance of
                        tf.reset_default_graph()
                        sess.close()           # tensorflow in memory (that can cause crashes)
                    K.clear_session()
                    
                    sess = tf.Session()
                    print(f'****** run {num_opti} on {len(grid)} *******************')
                    print('***************************************') 
                    path_full_df = os.path.join(intermediate_folder,f'FullCV_{CV}.csv')
                    
                    if os.path.exists(path_full_df):
                        cv_full = pd.read_csv(path_full_df, encoding='utf-8')
                        cv_full = cv_full.iloc[:,1:].values.tolist()
                    else:
                        cv_full = []
    
                    na = str(num_opti)+'_'+str(CV) +'_'+str(k) 
                    
                    if params['Loss'] == 'classif' or params['Loss'] =='triplet':
                        loss_attribute = params['Loss'] + str(params['nb_classes'])
                    elif params['Loss'] == 'discret':
                        loss_attribute = 'disc' + str(n_intervals)
                    else:
                        loss_attribute = params['Loss']
                        
                    loss_attribute = loss_attribute+'_Pre_'+params['pretrain_version']+ '_'
                    file_name = os.path.join(intermediate_folder,f'{loss_attribute}at{params['attention']}_spp{params['SPP']}_maxi{params['Maxi']}_D3{params['D3']}_n{params['noTrainLayer']}_dim{params['npatch']}')
                    os.makedirs(file_name, exist_ok=True)
                                            
                    log_dir = os.path.join(file_name, "logs","fit", titleP)
                    
                    if CV== 0 or CV == 2:
                        tensorboard = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                    else:
                        tensorboard = TensorBoard(log_dir=log_dir) 

                    
                    """  DATA AUGMENTATION ********************************************************************"""
                    if not p.apply_pretrain :
                        if  p.cross_validation :
                            xval,yval,mval,rval = X[k],Y[k], M[k], R[k]
                            if p.use_test_set :
                                if k==0:
                                    xtrain,mtrain,rtrain,ytrain=[X[1],X[2],X[3]],[M[1],M[2],M[3]],[R[1],R[2],R[3]],[Y[1],Y[2],Y[3]]
                                elif k==1:
                                    xtrain,mtrain,rtrain,ytrain=[X[0],X[2],X[3]],[M[0],M[2],M[3]],[R[0],R[2],R[3]],[Y[0],Y[2],Y[3]]
                                elif k==2:
                                    xtrain=[X[1],X[0],X[3]]
                                    mtrain=[M[1],M[0],M[3]]
                                    rtrain=[R[1],R[0],R[3]]
                                    ytrain=[Y[1],Y[0],Y[3]]
                                else:
                                    xtrain=[X[1],X[2],X[0]]
                                    mtrain=[M[1],M[2],M[0]]
                                    rtrain=[R[1],R[2],R[0]]
                                    ytrain=[Y[1],Y[2],Y[0]]
                            else:
                                if len(p.distribution)>4:
                                    if k==0:
                                        xtrain=[X[1],X[2],X[3],X[4]]
                                        mtrain=[M[1],M[2],M[3],M[4]]
                                        rtrain=[R[1],R[2],R[3],R[4]]
                                        ytrain=[Y[1],Y[2],Y[3],Y[4]]
                                    elif k==1:
                                        xtrain=[X[0],X[2],X[3],X[4]]
                                        mtrain=[M[0],M[2],M[3],M[4]]
                                        rtrain=[R[0],R[2],R[3],R[4]]
                                        ytrain=[Y[0],Y[2],Y[3],Y[4]]
                                    elif k==2:
                                        xtrain=[X[0],X[1],X[3],X[4]]
                                        mtrain=[M[0],M[1],M[3],M[4]]
                                        rtrain=[R[0],R[1],R[3],R[4]]
                                        ytrain=[Y[0],Y[1],Y[3],Y[4]]
                                    elif k==3:
                                        xtrain=[X[0],X[1],X[2],X[4]]
                                        mtrain=[M[0],M[1],M[2],M[4]]
                                        rtrain=[R[0],R[1],R[2],R[4]]
                                        ytrain=[Y[0],Y[1],Y[2],Y[4]]
                                    else:
                                        xtrain=[X[0],X[1],X[2],X[3]]
                                        mtrain=[M[0],M[1],M[2],M[3]]
                                        rtrain=[R[0],R[1],R[2],R[3]]
                                        ytrain=[Y[0],Y[1],Y[2],Y[3]]      
                                else:
                                    if k==0:
                                        xtrain=[X[1],X[2],X[3]]
                                        mtrain=[M[1],M[2],M[3]]
                                        rtrain=[R[1],R[2],R[3]]
                                        ytrain=[Y[1],Y[2],Y[3]]
                                    elif k==1:
                                        xtrain=[X[0],X[2],X[3]]
                                        mtrain=[M[0],M[2],M[3]]
                                        rtrain=[R[0],R[2],R[3]]
                                        ytrain=[Y[0],Y[2],Y[3]]
                                    elif k==2:
                                        xtrain=[X[0],X[1],X[3]]
                                        mtrain=[M[0],M[1],M[3]]
                                        rtrain=[R[0],R[1],R[3]]
                                        ytrain=[Y[0],Y[1],Y[3]]
                                    else:
                                        xtrain=[X[0],X[1],X[2]]
                                        mtrain=[M[0],M[1],M[2]]
                                        rtrain=[R[0],R[1],R[2]]
                                        ytrain=[Y[0],Y[1],Y[2]]      
                                
                            # xtrain,ytrain,mtrain,rtrain = np.delete(X,k),np.delete(Y,k),np.delete(M,k),np.delete(R,k)
                            if p.use_test_set  or len(p.distribution)==4:
                                xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2]))
                                mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2]))
                                ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2]))
                                rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2]))
                            else:
                                xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2],xtrain[3]))
                                mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2],mtrain[3]))
                                ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2],ytrain[3]))
                                rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2],rtrain[3]))
                            if params['dataAug']!=0:
                                if params['D3'] == 'False':#or params['D3']=='25'
                                    dict_transf_train = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                            images = xtrain,
                                                                                            y = ytrain,
                                                                                            masks = mtrain,
                                                                                            ref = rtrain,
                                                                                            D3 = params['D3'],
                                                                                            num_augm = params['dataAug'])
                                    
                                    dict_transf_val = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                        images = xval,
                                                                                        y = yval,
                                                                                        masks = mval,
                                                                                        ref = rval,
                                                                                        D3 = params['D3'],
                                                                                        num_augm = params['dataAug'])
                                    if p.use_test_set:
                                        dict_transf_test = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                        images = xtest,
                                                                                        y = ytest,
                                                                                        masks = mtest,
                                                                                        ref = rtest,
                                                                                        D3 = params['D3'],
                                                                                        num_augm = params['dataAug'])
                                                                                
                                    (xtrain, ytrain, mtrain,rtrain) = (dict_transf_train['images_transformed'], 
                                                        dict_transf_train['y_transformed'],
                                                        dict_transf_train['masks_transformed'],
                                                        dict_transf_train['ref_transformed']) 
                                    
                                    (xval, yval, mval) = (dict_transf_val['images_transformed'], 
                                                        dict_transf_val['y_transformed'],
                                                        dict_transf_val['masks_transformed'],
                                                        dict_transf_val['ref_transformed']) 

                                if p.use_test_set:
                                    (xtest, ytest, mtest) = (dict_transf_test['images_transformed'], 
                                        dict_transf_test['y_transformed'],
                                        dict_transf_test['masks_transformed']) 
                                    
                                xval, yval, mval,rval = shuffle_and_interpolate(xval, yval,rval,p.interpolation_size,apply_interpolation = False,list_masks =mval)
                                xtrain, ytrain, mtrain,rtrain = shuffle_and_interpolate(xtrain, ytrain,rtrain, p.interpolation_size,apply_interpolation = False,list_masks =mtrain)

                        else:
                            if params['dataAug']!=0:                                        
                                dict_transf_train = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                        images = xtrain,
                                                                                        y = ytrain,
                                                                                        masks = mtrain,
                                                                                        ref = rtrain,
                                                                                        D3 = params['D3'],
                                                                                        num_augm = params['dataAug'])
                                dict_transf_val = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                    images = xval,
                                                                                    y = yval,
                                                                                    masks = mval,
                                                                                    ref = rval,
                                                                                    D3 = params['D3'],
                                                                                    num_augm = params['dataAug'])
                                if p.use_test_set:
                                    dict_transf_test = dataprep.apply_data_augmentation(augm_config = augmentation_config,
                                                                                    images = xtest,
                                                                                    y = ytest,
                                                                                    masks = mtest,
                                                                                    ref = rtest,
                                                                                    D3 = params['D3'],
                                                                                    num_augm = params['dataAug'])
                                                                            
                                (xtrain, ytrain, mtrain) = (dict_transf_train['images_transformed'], 
                                    dict_transf_train['y_transformed'],
                                    dict_transf_train['masks_transformed']) 
                                
                                (xval, yval, mval) = (dict_transf_val['images_transformed'], 
                                    dict_transf_val['y_transformed'],
                                    dict_transf_val['masks_transformed']) 
                                
                                if p.use_test_set:
                                    (xtest, ytest, mtest) = (dict_transf_test['images_transformed'], 
                                        dict_transf_test['y_transformed'],
                                        dict_transf_test['masks_transformed']) 
                                        
                                xval, yval, mval = shuffle_and_interpolate(xval, yval, None, p.interpolation_size, apply_interpolation = False, list_masks =mval)
                                xtrain, ytrain, mtrain = shuffle_and_interpolate(xtrain, ytrain, None, p.interpolation_size, apply_interpolation = False, list_masks =mtrain)
                                
                                if p.use_test_set:
                        if p.use_test_set:
                            xtrain,xval,xtest = normalize_batch(xtrain),normalize_batch(xval),normalize_batch(xtest)
                        else:
                            xtrain,xval = normalize_batch(xtrain),normalize_batch(xval)

                        if params['D3']=='25':
                            xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                            mtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                            if p.use_test_set:
                                xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                                mtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                            xval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))
                            mval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))

                        if params['Loss'] == 'classif':
                            ytrain_grp = ground_truth_processing2(ytrain,params['nb_classes'])
                            yval_grp = ground_truth_processing2(yval,params['nb_classes'])
                            if p.use_test_set:
                                ytest_grp = ground_truth_processing2(ytest,params['nb_classes'])
                                
                        elif params['Loss'] == 'triplet':
                            ytrain_grp = np.transpose(np.concatenate(([ytrain[:,0]],[np.argmax(ground_truth_processing2(ytrain,params['nb_classes']),axis=1)]),axis=0))
                            yval_grp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(ground_truth_processing2(yval,params['nb_classes']),axis=1)]),axis=0))
                            if p.use_test_set:
                                ytest_grp =np.transpose(np.concatenate(([ytest[:,0]],[np.argmax(ground_truth_processing2(ytest,params['nb_classes']),axis=1)]),axis=0))
                        else:
                            ytrain_grp = uf.ground_truth_processing4(ytrain,'YEAR')
                            yval_grp = uf.ground_truth_processing4(yval,'YEAR')
                            if p.use_test_set:
                                ytest_grp = uf.ground_truth_processing4(ytest,'YEAR')
                                
                            ytrain_grpwc = uf.ground_truth_processing4(ytrain,'YEARWC')
                            yval_grpwc = uf.ground_truth_processing4(yval,'YEARWC')
                            if p.use_test_set:
                                ytest_grpwc = uf.ground_truth_processing4(ytest,'YEARWC')
                                
                            ytrain_grp3 = uf.ground_truth_processing4(ytrain,'THREE')
                            yval_grp3 = uf.ground_truth_processing4(yval,'THREE')
                            if p.use_test_set:
                                ytest_grp3 = uf.ground_truth_processing4(ytest,'THREE')
                                
                            ytrain_grpwc3 = uf.ground_truth_processing4(ytrain,'THREEWC')
                            yval_grpwc3 = uf.ground_truth_processing4(yval,'THREEWC')
                            if p.use_test_set:
                                ytest_grpwc3 = uf.ground_truth_processing4(ytest,'THREEWC')
                            
                        if params['Loss']=='discret' or params['Loss'] == 'RankAnddiscret':
                            if p.use_test_set:
                                ytrainT,yvalT,ytestT =  ytrain,yval,ytest
                                ytrain,yval,ytest= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks),uf.FromTimeToLong(ytest,breaks)
                            else:
                                ytrainT,yvalT =  ytrain,yval
                                ytrain,yval= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks)
                        
                        
                    """  model creation ********************************************************************"""    
                    
                    if params['pretrain_version'] == 'triplet':
                        param_tri_lr_decay_pretraining = [1e-8,1e-8,1e-12,1e-8]

                        ww = ['2_0_0lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model',
                            '1_0_1lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.17best_model',
                            '7_0_2lrP0.0001_lrF0.0001b10_eF20_lrd2_D15_r0.17best_model',
                            '2_0_3lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model']
                        param_tri_rate = [0.83,0.17,0.17,0.83]

                    if not params['SPP']:
                        conv_1=False
                    else:
                        conv_1=False
                    if params['bilinear'] :
                        model_cnn = ConvSurvBefore(p.interpolation_size,params['D3'],spp=params['SPP']) 
                        model_cnn_2 = ConvSurvBefore(p.interpolation_size,params['D3'],spp=params['SPP']) 
                        pred =  BilinearModel(p.interpolation_size,modelCN = model_cnn,model_cnn_2 = model_cnn_2, mode = params['Loss'],D3=params['D3'],spp=params['SPP'],attention=params['attention'],num=params['nb_classes'])
                    else: 
                        if params['pretrain_version'] == 'triplet':
                            # pred = deepConvSurv(p.interpolation_size,mode = 'triplet',D3=params['D3'],spp=params['SPP'],attention=params['attention'],num=params['nb_classes'],rate=param_tri_rate,conv1=conv_1)  
                            pred = deepConvSurv(p.interpolation_size,mode = 'triplet',D3=params['D3'],spp=params['SPP'],attention=params['attention'],num=params['nb_classes'],rate=params['rate'],conv1=conv_1,l11=params['l2reg'])  

                        elif params['pretrain_version'] == 'binary':
                            pred = deepConvSurv(p.interpolation_size,mode = 'classif',D3=params['D3'],spp=params['SPP'],attention=params['attention'],num=2,rate=rate,conv1=conv_1,l11=params['l2reg'])  
                        else:
                            pred = deepConvSurv(p.interpolation_size,mode = params['Loss'] ,D3=params['D3'],spp=params['SPP'],attention=params['attention'],num=params['nb_classes'],rate=params['rate'],conv1=conv_1,l11=params['l2reg'])  

                    ''' *****************************************'''
                    ''' **************PRETRAINING****************'''
                    ''' *****************************************'''                
                    if params['pretrain_version']=='binary':
                        if p.apply_pretrain :
                            xbival,ybival = Xb[k], Yb[k]
                            
                            if k==0:
                                xbitrain=[Xb[1],Xb[2],Xb[3]]
                                ybitrain=[Yb[1],Yb[2],Yb[3]]
                            elif k==1:
                                xbitrain=[Xb[0],Xb[2],Xb[3]]
                                ybitrain=[Yb[0],Yb[2],Yb[3]]
                            elif k==2:
                                xbitrain=[Xb[1],Xb[0],Xb[3]]
                                ybitrain=[Yb[1],Yb[0],Yb[3]]
                            else:
                                xbitrain=[Xb[1],Xb[2],Xb[0]]
                                ybitrain=[Yb[1],Yb[2],Yb[0]]
                                
                                
                            # xtrain,ytrain,mtrain,rtrain = np.delete(X,k),np.delete(Y,k),np.delete(M,k),np.delete(R,k)
                            xbitrain=np.concatenate((xbitrain[0],xbitrain[1],xbitrain[2]))
                            ybitrain=np.concatenate((ybitrain[0],ybitrain[1],ybitrain[2]))

                            mbitrain,mbival =np.ones(np.shape(xbitrain)),np.ones(np.shape(xbival))
            
                            
                            # change the last layer
                            # compile
                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                        pred.compile(optimizer=adam(lr=params['LR Pretrain'],decay=params['lrate_decay_learningF']), loss='categorical_crossentropy',metrics=['accuracy']) #cindex_score
                        folder_name = f"Binary3D{params['D3']}_spp{params['SPP']}_att{params['attention']}_maxi{params['Maxi']}"
                        
                        folder_save = os.path.join(p.save_dir , 'pretraining' , folder_name)
                        
                        os.makedirs(folder_save, exist_ok=True) 
                        
                        weights = os.path.join(folder_save, f"Weights{params['LR Pretrain']}_{params['lrate_decay_learningF']}_ratio{p.ratio_binary_pretraining}_drop{0.83}_CV{CV}K{k}.h5")
                        
                        if os.path.exists(weights) :
                            print('load weights')
                            pred.load_weights(weights)
                            fit = 0
                        else:
                            print('run pretraining .....')

                            log_dirP = os.path.join(folder_save, "logs","fit"+ f"Weights{params['LR Pretrain']}_{params['lrate_decay_learningF']}_ratio{p.ratio_binary_pretraining}_CV{CV}K{k}")
                            
                            tensorboardP = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                            
                            fitting = pred.fit([xbitrain,mbitrain], ybitrain, validation_data=([xbival,mbival], ybival), batch_size=params['batch size'],epochs=params['epoch Train'], callbacks=[tensorboardP])#,mcAc
                            pred.save_weights(weights)
                            print('pretraining finished')

                            fit = 1
                            print('--- evaluation')
                            PREeval_train = pred.evaluate([xbitrain,mbitrain], ybitrain)
                            print('eval train',PREeval_train)
                            PREeval_val = pred.evaluate([xbival,mbival], ybival)
                            print('eval val',PREeval_val)
                            
                            name = f'pretrainBinary_{params['attention']}_{params['D3']}_{params['Loss']}_{params['SPP']}_{params['LR Pretrain']}_{params['lrate_decay_learningF']}_{params['Maxi']}'
                            results = save_training_plots(fitting,name,PREeval_train, PREeval_val,None,p.save_dir,'accuracy')            
                            binary.append([PREeval_train[1],PREeval_val[1],params['LR Pretrain'],params['lrate_decay_learningF']])
                            np.savetxt(os.path.join(p.save_dir, 'pretraining', "pretrainBinaryResults.csv"),binary)
                    
                    if params['pretrain_version'] == 'triplet':
                        folder_name = os.path.join(p.save_dir, 'Best','CV0','triplet7_Pre_False_atc_sppFalse_maxiFalse_D3True_n3')

                        weights = folder_name + f'{ww[k]}.h5'
                        
                        pred.load_weights(weights)
    
                    if params['pretrain_version'] != 'triplet':
                        lay = pred.get_layer(index = -2).output
                    else:
                        lay = pred.get_layer(index = -1).output

                        lay = Dense(100, name="fc3",kernel_regularizer=regularizers.l2(0.01),
                                    bias_initializer='zeros')(lay)

                        lay = LeakyReLU(alpha=0.1)(lay)
                        lay = Dropout(rate = params['rate'])(lay)
                        
                    
                    if params['Loss'] == 'cox':
                        x1 = Dense(1, activation='linear', name="cox_output",kernel_regularizer=regularizers.l2(0.001))(lay)
    
                    elif params['Loss'] == 'discret'or params['Loss']=='RankAnddiscret':
                        x1= Dense(n_intervals,  name="output",activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.001))(lay)
                    
                    elif params['Loss'] == 'RankAndmse':
                        x1 = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
                    elif params['Loss'] == 'RankAndcox':
                        x1 = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
                    elif params['Loss'] == 'triplet':
                        x1 = Dense(params['nb_classes'], activation='softmax', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)

                    pred = Model(inputs=pred.input, outputs=x1) 
                    if params['pretrain_version'] == 'triplet' and params['noTrainLayer'] !=0:
                        for layer in pred.layers[1].layers[:params['noTrainLayer']]:
                            print(layer)
                            layer.trainable = False
                    
                    # print(pred.summary())
                    # for layer in pred.layers[:params['noTrainLayer']]:
                    #     print(layer)
                    #     layer.trainable = False
                    
                    # if os.path.exists(best_model_file) :
                    #     print('load weights')
                    #     pred.load_weights(best_model_file)
                    #     fit = 0

                    if not p.apply_pretrain:
                        """ COMPILE  ********************************************************************"""
                        match params['Loss']:     
                            case 'cox':  #,clipnorm=0.1
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']), loss=lss.__cox_loss(),metrics=[lss.tf_cindexR]) #cindex_score
                                metrics = 'tf_cindex'
                            case 'RankAndmse':
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']), loss=lss.__RankAndMse_loss(),metrics=[uf.tf_cindexT,lss.rank,lss.mse]) #cindex_score
                                metrics= 'rankAndMSE'
                            case 'RankAndcox':
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']),loss=lss.__RankAndCox_loss(),metrics=[lss.tf_cindexR,lss.rank,lss.cox]) #cindex_score
                                metrics= 'coxAndrank'
                            case 'coxAndclassif': 
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']),loss=[lss.__cox_loss(),'categorical_crossentropy'],metrics=[lss.tf_cindexR,'accuracy'],loss_weights = [1.0,1.5]) #cindex_score
                                metrics = 'coxAndclassif'
                            case 'discret':
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']), loss=ns.surv_likelihood(n_intervals),metrics=[uf.tf_cindexTD]) #cindex_score
                                metrics = 'tf_cindexTD'
                            case 'RankAnddiscret':
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']),loss=ns.RankingAndDiscret(n_intervals), metrics=[uf.tf_cindexTD,ns.discretcox,ns.rankL]) #cindex_score
                                metrics = 'RankAnddiscret'
                            case 'triplet':
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']), loss=batch_hard_triplet_loss(),metrics=[ lss.mse,uf.tf_cindexT]) #uf.tf_cindexT,  cindex_score
                                metrics = 'triplet'                              
                            case _:
                                pred.compile(optimizer=adam(lr=params['LR Finetune'],decay=params['lrate_decay_learningF']),
                                            loss='categorical_crossentropy',metrics=['accuracy',uf.tf_cindexT]) #cindex_score
                                metrics = 'triplet'
                                
                        """  FITTING ********************************************************************"""
                        writer = tf.summary.FileWriter(log_dir)
                        
                        hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in p.items()]
                        hyperpara = tf.summary.text('hyperparameters', tf.stack(hyperparameters))
                        
                        s = sess.run(hyperpara)
                        writer.add_summary(s)
                    
                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
                        # fit model    
                        match params['Loss']:           
                            case 'coxAndclassif':
                                fitting = pred.fit([xtrain,mtrain], [ytrain,ytrain_grp], validation_data=([xval,mval], [yval,yval_grp]), batch_size=params['batch size'],epochs=params['epoch Finetune'],   callbacks=[tensorboard]) #PlotLossesKeras(),
                                eval_val = pred.evaluate([xval,mval], [yval,yval_grp])
                                eval_train = pred.evaluate([xtrain,mtrain], [ytrain,ytrain_grp])
                                
                                if p.use_test_set:
                                    eval_test= pred.evaluate([xtest,mtest], [ytest,ytest_grp])
                                
                                Newmodel = pred
                                
                                if not p.cross_validation:
                                    if p.use_test_set:
                                        predictionBefore=Newmodel.predict([xtest,mtest]) 
                                    predictionBeforeclass = np.argmax(predictionBefore[1],axis=1)
                                    predictionBefore=predictionBefore[0]
                                    
                                predictions_train=Newmodel.predict([xtrain,mtrain])      
                                predictiontrainclass = np.argmax(predictions_train[1],axis=1)
                                predictions_train=predictions_train[0]                        
                                predictionval=Newmodel.predict([xval,mval])
                                predictionvalclass = np.argmax(predictionval[1],axis=1)
                                predictionval=predictionval[0]
                                
                            case 'classif' | 'triplet':
                
                                mc = ModelCheckpoint(best_model_file, monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)#'val_tf_cindexT'               
                                
                                fitting = pred.fit([xtrain,mtrain], ytrain_grp, validation_data=([xval,mval], yval_grp), batch_size=params['batch size'],epochs=params['epoch Finetune'], callbacks=[mc,tensorboard])#,mcAc
                                
                                if params['Loss'] == 'triplet':
                                    loss_name = batch_hard_triplet_loss()
                                else:
                                    loss_name = 'categorical_crossentropy'
                                
                                saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':loss_name, 'InstanceNormalization':InstanceNormalization})

                                eval_val = pred.evaluate([xval,mval], yval_grp)
                                eval_train = pred.evaluate([xtrain,mtrain], ytrain_grp)
                                
                                if p.use_test_set:
                                    eval_test= pred.evaluate([xtest,mtest], ytest_grp)
                                
                                # Newmodel = pred
                                # if not p.cross_validation :
                                #     predictionBefore=Newmodel.predict([xtest,mtest]) 
                                #     predictionBeforeclass = np.argmax(predictionBefore,axis=1)
                                # predictions_train=Newmodel.predict([xtrain,mtrain])      
                                # predictiontrainclass = np.argmax(predictions_train,axis=1)
                                # predictionval=Newmodel.predict([xval,mval])
                                # predictionvalclass = np.argmax(predictionval,axis=1)
                                
                                # if params['SPP'] :
                                #     eval_val_best = saved_model.evaluate([xval,mval], yval_grp)
                                #     eval_train_best = saved_model.evaluate([xtrain,mtrain], ytrain_grp)
                                #     eval_test_best= saved_model.evaluate([xtest,mtest], ytest_grp)
                                # else: 
                                #     eval_val_best = saved_model.evaluate(xval, yval_grp)
                                #     eval_train_best = saved_model.evaluate(xtrain, ytrain_grp)
                                #     eval_test_best= saved_model.evaluate(xtest, ytest_grp)
                                
                                if params['SPP']:
                                    predictionvalclass = np.argmax(saved_model.predict([xval,mval]),axis=1)      
                
                                    predictiontrainclass = np.argmax(saved_model.predict([xtrain,mtrain]),axis=1)
                                    if p.use_test_set:
                                        predictiontestclass = np.argmax(saved_model.predict([xtest,mtest]),axis=1)  
                
                                else: 
                                    predictionvalclass = np.argmax(saved_model.predict(xval),axis=1)
                                    predictiontrainclass = np.argmax(saved_model.predict(xtrain) ,axis=1)
                                    if p.use_test_set:
                                        predictiontest = saved_model.predict(xtest)   
                                        predictiontestclass = np.argmax(predictiontest,axis=1)
                            
                                if p.use_test_set:
                                    evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                                tf.convert_to_tensor(['loss', str(eval_train[0]),str(eval_val[0]),str(eval_test[0])]),
                                                tf.convert_to_tensor(['tf_cindexT', str(eval_train[1]),str(eval_val[1]),str(eval_test[1])])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                                tf.convert_to_tensor(['loss', str(eval_train[0]),str(eval_val[0])]),
                                                tf.convert_to_tensor(['tf_cindexT', str(eval_train[1]),str(eval_val[1])])]
    
                                s = sess.run(tf.summary.text('Eval', tf.stack(evalu)))
                                writer.add_summary(s)
                                
                        
                                if p.use_test_set:   
                                    evalu = [tf.convert_to_tensor(['', 'best train','best val','best test']) ,
                                            tf.convert_to_tensor(['loss', str(0),str(0),str(0)]),
                                            tf.convert_to_tensor(['tf_cindexT', str(uf.cindexT(ytrain[:,1],predictiontrainclass,ytrain[:,0])),str(uf.cindexT(yval[:,1],predictionvalclass,yval[:,0])),str(uf.cindexT(ytest[:,1],predictiontestclass ,ytest[:,0]))])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'best train','best val']) ,
                                            tf.convert_to_tensor(['loss', str(0),str(0)]),
                                            tf.convert_to_tensor(['tf_cindexT', str(uf.cindexT(ytrain[:,1],predictiontrainclass,ytrain[:,0])),str(uf.cindexT(yval[:,1],predictionvalclass,yval[:,0]))])]
    
                                s = sess.run(tf.summary.text('Eval_best', tf.stack(evalu)))
                                writer.add_summary(s)
                                
                                
                                model_name = os.path.join(file_name, f"{titleP}best.h5")        
                                saved_model.save_weights(model_name)
                                new = pred.load_weights(model_name)
                                
                            
                            case 'discret':
                                if os.path.exists(best_model_file) :                                    
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                    
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=params['batch size'],epochs=params['epoch Finetune'], callbacks=[mc,tensorboard])
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                            
                            case 'RankAnddiscret':
                                if os.path.exists(best_model_file) :
                                        saved_model = load_model(best_model_file,
                                        custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                        'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                        'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                        'InstanceNormalization':InstanceNormalization})    
                                else:
                                    
                                    mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                    
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=params['batch size'],epochs=params['epoch Finetune'],callbacks=[mc,tensorboard] )#
                                    saved_model = load_model(best_model_file,
                                            custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                            'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                            'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                            'InstanceNormalization':InstanceNormalization})
            
                            case 'RankAndmse':      
                                best_model_file=best_model_file
                                if os.path.exists(best_model_file) :
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
    
                                else:
                                    mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=params['batch size'],epochs=params['epoch Finetune'], callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
        
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
                                            
                            case 'RankAndcox': 
                                best_model_file=best_model_file
                                if os.path.exists(best_model_file) :
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=params['batch size'],epochs=params['epoch Finetune'], callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                        
                            case _:
                                best_model_file=best_model_file
                                if os.path.exists(best_model_file) :
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'tf_cindexR':lss.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=params['batch size'],epochs=params['epoch Finetune'], callbacks=[mc,tensorboard])#,hp.KerasCallback(log_dir,hparams)    
                                    saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})

                        if params['Loss']!= 'classif' and params['Loss'] != 'triplet':      
                            if params['SPP']:
                                eval_val_best = saved_model.evaluate([xval,mval], yval)
                                eval_train_best = saved_model.evaluate([xtrain,mtrain], ytrain)
                                
                                if p.use_test_set:  
                                    eval_test_best= saved_model.evaluate([xtest,mtest], ytest)
                            else: 
                                eval_val_best = saved_model.evaluate(xval, yval)
                                eval_train_best = saved_model.evaluate(xtrain, ytrain)
                            if params['SPP']:
                                predictionval = saved_model.predict([xval,mval])
                                predictions_train = saved_model.predict([xtrain,mtrain])
                                
                                if p.use_test_set:
                                    predictiontest = saved_model.predict([xtest,mtest])
                                
                            else: 
                                predictionval = saved_model.predict(xval)
                                predictions_train = saved_model.predict(xtrain)
                                
                                if p.use_test_set:
                                    predictiontest = saved_model.predict(xtest) 
                                        
                            if len(np.shape(predictionval))==2:
                                if params['Loss'] !='discret'and params['Loss']!='RankAnddiscret':
                                    predictionval = predictionval[:,0]
                                    predictiontrainExclude = predictions_train[:,0]
                                    
                            if params['Loss'] =='discret'or params['Loss']=='RankAnddiscret':    
                                if p.use_test_set:                      
                                    predictiontest = uf.FromLongToTime(predictiontest,n_intervals,None)
                                predictionval = uf.FromLongToTime(np.array(predictionval),n_intervals,None)
                                predictions_train = uf.FromLongToTime(predictions_train,n_intervals,None)

                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret': 
                                ppr = np.array([ppred / 365 for ppred in predictionval], dtype=np.int)
                                pYM = np.argmax(yval_grp,axis = 1)
                                acc_val = accuracy_score(pYM,ppr)
                                confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                df_fft=pd.DataFrame(confB)
                                df_fft.to_csv(os.path.join(file_name, f"{titleP}_confusion_val.csv"))

                            
                            """  Features ********************************************************************"""
                
                            flat =  saved_model.get_layer(index = utils.find_layer_idx(saved_model,'resh48'))#Newmodel.get_layer(index = -4)
                            featModel = Model(inputs=saved_model.input, outputs=flat.output)
                        
                            featurestrain=featModel.predict(xtrain)
                            pathSaveFFT = os.path.join(file_name, f"{titleP}_features_flatten_train")
                            np.save(pathSaveFFT,featurestrain)
                            pathCSV_FFT = os.path.join(file_name, f"{titleP}_features_flatten_train.csv")
                            df_fft=pd.DataFrame(featurestrain)
                            df_fft.to_csv(pathCSV_FFT)           
                            
                            featuresval=featModel.predict(xval)
                            pathSaveFFT = os.path.join(file_name, f"{titleP}_features_flatten_val")
                            np.save(pathSaveFFT,featuresval)
                            pathCSV_FFT = os.path.join(file_name, f"{titleP}_features_flatten_val.csv")
                            df_fft=pd.DataFrame(featuresval)
                            df_fft.to_csv(pathCSV_FFT)
                                                            
                            if params['Loss'] =='discret'or params['Loss']=='RankAnddiscret':  
                                ytrain = ytrainT
                                yval = yvalT
                                
                            '''Evaluate with kaplan'''
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                dict_best = get_optimal_separation(ytrain,predictions_train)
                                optimal_separation = dict_best['separation']
                                try:
                                    PvalueTrain,optimal_separation,numG0_train,numG1_train =group_by_optimal_separation(ytrain,predictions_train,optimal_separation,file_name,titleP,'train','time')
                                    print('kaplan worked')

                                except :
                                    PvalueTrain =optimal_separation=numG0_train=numG1_train = -100
                                    print('kaplan didn t work')
                                    pass
                                try:
                                    PvalueVal,optimal_separation,numG0_val,numG1_val =group_by_optimal_separation(yval,predictionval,optimal_separation,file_name,titleP,'val','time')
                                    print('kaplan worked')

                                except :
                                    PvalueVal = optimal_separation = numG0_val = numG1_val = -100
                                    print('kaplan didn t work')
                                    pass
                            else:       
                                dict_best = get_optimal_separation(ytrain,predictions_train)
                                optimal_separation = dict_best['separation']
                                try:
                                    PvalueTrain,optimal_separation,numG0_train,numG1_train =group_by_optimal_separation(ytrain,predictions_train,optimal_separation,file_name,titleP,'train')
                                    print('kaplan worked')

                                except :
                                    PvalueTrain =optimal_separation=numG0_train=numG1_train = -100
                                    print('kaplan didn t work')
                                    pass
                                try:
                                    PvalueVal,optimal_separation,numG0_val,numG1_val =group_by_optimal_separation(yval,predictionval,optimal_separation,file_name,titleP,'val')
                                    print('kaplan worked')

                                except :
                                    PvalueVal = optimal_separation = numG0_val = numG1_val = -100
                                    print('kaplan didn t work')
                                    pass
                            
                            
                            if params['Loss'] =='discret'or params['Loss']=='RankAnddiscret':    
                                ax=display_predictions('val',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                if p.use_test_set:
                                    ax=display_predictions('test',ytestT,predictiontest,writer)
                            else:
                                ax=display_predictions('val',yval,predictionval,writer)
                                if p.use_test_set:
                                    ax=display_predictions('test',ytest,predictiontest,writer)
                    
                            predictionval=np.array(predictionval,dtype='float64')

                            yval=np.array(yval,dtype='float64')

                            df2 = pd.DataFrame(np.array([yval[:,0],yval[:,1],predictionval]).T,index=rval)
                            df2.columns=['event','time','pred']     
                            grouper = df2.groupby([df2.index]).mean()
                            predictionval = grouper['pred']
                            yval=[grouper['event'],grouper['time']]
                            yval=np.array(yval).T
                            yvalL=  uf.FromTimeToLong(yval,breaks)
                            yvalT =  yval
                            if params['Loss'] == 'classif':
                                yval_grp = ground_truth_processing2(yval,params['nb_classes'])
                            elif params['Loss'] == 'triplet':
                                yval_grp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(ground_truth_processing2(yval,params['nb_classes']),axis=1)]),axis=0))
                            else:
                                yval_grp = uf.ground_truth_processing4(yval,'YEAR')
                                yval_grpwc = uf.ground_truth_processing4(yval,'YEARWC')
                                yval_grp3 = uf.ground_truth_processing4(yval,'THREE')
                                yval_grpwc3 = uf.ground_truth_processing4(yval,'THREEWC')
                            
                            if params['Loss'] =='discret'or params['Loss']=='RankAnddiscret':    
                                ax=display_predictions('valgrp',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                if p.use_test_set:
                                    ax=display_predictions('test',ytestT,predictiontest,writer)
                                ax=display_predictions('train',ytrainT,predictions_train,writer)
                            else:
                                ax=display_predictions('valgrp',yval,predictionval,writer)
                                if p.use_test_set:
                                    ax=display_predictions('test',ytest,predictiontest,writer)
                                ax=display_predictions('train',ytrain,predictions_train,writer)
                    
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                #super Cindex avec censure
                                cind_trainSuper = uf.cindexT(np.argmax(ytrain_grp,axis=1),predictions_train,ytrain[:,0])
                                if p.use_test_set:
                                    cindTestSuper = uf.cindexT(np.argmax(ytest_grp,axis=1),predictiontest,ytest[:,0])                                                                     
                                cind_valSuper = uf.cindexT(np.argmax(yval_grp,axis=1),np.reshape(np.array(predictionval),(-1,1)),yval[:,0]) 
                                
                            else:
                            
                                #super Cindex avec censure
                                cind_trainSuper = uf.cindexR(np.argmax(ytrain_grp,axis=1),predictions_train,ytrain[:,0])
                                if p.use_test_set:
                                    cindTestSuper = uf.cindexR(np.argmax(ytest_grp,axis=1),predictiontest,ytest[:,0])                                                                     
                                cind_valSuper = uf.cindexR(np.argmax(yval_grp,axis=1),np.array(predictionval),yval[:,0]) 
                                
        
                            '''Evaluate with kaplan grp'''
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                try:
                                    PvalueValgrp,optimal_separation,numG0_valgrp,numG1_valgrp =group_by_optimal_separation(yval,predictionval,optimal_separation,file_name,titleP,'valgrp','time')
                                    print('kaplan worked')
                                except :
                                    PvalueValgrp = numG0_valgrp = numG1_valgrp = -100
                                    print('kaplan didn t work')
                                    pass
                            else:      
                                try:
                                    PvalueValgrp,optimal_separation,numG0_valgrp,numG1_valgrp =group_by_optimal_separation(yval,predictionval,optimal_separation,file_name,titleP,'valgrp')
                                    print('kaplan worked')
                                except :
                                    PvalueValgrp = numG0_valgrp = numG1_valgrp = -100
                                    print('kaplan didn t work')
                                    pass
                            '''Evaluate with kaplan Exclude grp'''
                            
                            # for prediction train remove patients where prediction to far
                                                    
                            predictiontrainExclude=np.array(predictions_train,dtype='float64')
                            ytrainExclude=np.array(ytrain,dtype='float64')
                            if len(np.shape(predictiontrainExclude))==2:
                                df2train = pd.DataFrame(np.array([ytrainExclude[:,0],ytrainExclude[:,1],predictiontrainExclude[:,0]]).T,index=rtrain)
                            else:
                                df2train = pd.DataFrame(np.array([ytrainExclude[:,0],ytrainExclude[:,1],predictiontrainExclude]).T,index=rtrain)
                            df2train.columns=['event','time','pred']     
                            groupertrain = df2train.groupby([df2train.index]).mean()
                            groupertrainstd = df2train.groupby([df2train.index]).std()
                            predictiontrainExcludestd = groupertrainstd['pred']
                            
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                limm = 400
                            else:
                                limm=2
                            groupertrainExclude = groupertrain[predictiontrainExcludestd<limm]
                            predictiontrainExclude = groupertrain['pred']
                            ytrainExclude=[groupertrain['event'],groupertrain['time']]
                            ytrainExclude=np.array(ytrainExclude).T
                            ytrainExcludeL=  uf.FromTimeToLong(ytrainExclude,breaks)
                            ytrainExcludeT =  ytrainExclude

                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                dict_best_exclude  = get_optimal_separation(ytrainExclude,predictiontrainExclude)
                                optimal_separation_exclude = dict_best_exclude['separation']

                                try:
                                    PvalueTrainExclude,optimal_separation_exclude,numG0_trainExclude,numG1_trainExclude =group_by_optimal_separation(ytrain,predictions_train,optimal_separation_exclude,file_name,titleP,'trainExclude','time')
                                    print('kaplan worked')
                                except :
                                    PvalueTrainExclude=optimal_separation_exclude=numG0_trainExclude=numG1_trainExclude = -100
                                    print('kaplan didn t work')
                                    pass
                                try:
                                    PvalueValgrpExclude,optimal_separation_exclude,numG0_valgrpExclude,numG1_valgrpExclude =group_by_optimal_separation(yval,predictionval,optimal_separation_exclude,file_name,titleP,'valgrpExclude','time')
                                    print('kaplan worked')
                                except :
                                    PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                    print('kaplan didn t work')
                                    pass
                            else:       
                                dict_best_exclude  = get_optimal_separation(ytrainExclude,predictiontrainExclude)
                                optimal_separation_exclude = dict_best_exclude['separation']

                                try:
                                    PvalueTrainExclude,optimal_separation_exclude,numG0_trainExclude,numG1_trainExclude =group_by_optimal_separation(ytrain,predictions_train,optimal_separation_exclude,file_name,titleP,'trainExclude')
                                    print('kaplan worked')
                                except :
                                    PvalueTrainExclude=optimal_separation_exclude=numG0_trainExclude=numG1_trainExclude = -100
                                    print('kaplan didn t work')
                                    pass
                                try:
                                    PvalueValgrpExclude,optimal_separation_exclude,numG0_valgrpExclude,numG1_valgrpExclude =group_by_optimal_separation(yval,predictionval,optimal_separation_exclude,file_name,titleP,'valgrpExclude')
                                    print('kaplan worked')
                                except :
                                    PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                    print('kaplan didn t work')
                                    pass
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                cindex_val1 = uf.cindexT(yval[:,1],np.array(predictionval),yval[:,0])
                            else:
                                cindex_val1 = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0])
                    
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                ppr = np.array([ppred / 365 for ppred in predictions_train], dtype=np.int)
                                pYM = np.argmax(ytrain_grp,axis = 1)
                                acc_train = accuracy_score(pYM,ppr)
                                confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                df_fft=pd.DataFrame(confB)
                                df_fft.to_csv(os.path.join(file_name, f"{titleP}_confusion_train.csv"))
                                
                                ppr = np.array([ppred / 365 for ppred in predictionval], dtype=np.int)
                                pYM = np.argmax(yval_grp,axis = 1)
                                acc_valgrp = accuracy_score(pYM,ppr)
                                confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                df_fft=pd.DataFrame(confB)
                                df_fft.to_csv(os.path.join(file_name, f"{titleP}_confusion_valgrp.csv"))
                            if p.use_test_set:
                                evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                        tf.convert_to_tensor(['loss', str(eval_train_best[0]),str(eval_val_best[0]),str(eval_test_best[0])]),
                                        tf.convert_to_tensor(['superCindex', str(cind_trainSuper),str(cind_valSuper),str(cindTestSuper)]),
                                        tf.convert_to_tensor(['tf_cindexR', str(eval_train_best[1]),str(eval_val_best[1]),str(eval_test_best[1])])]
                            else:
                                evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                        tf.convert_to_tensor(['loss', str(eval_train_best[0]),str(eval_val_best[0])]),
                                        tf.convert_to_tensor(['superCindex', str(cind_trainSuper),str(cind_valSuper)]),
                                        tf.convert_to_tensor(['Pval', str(PvalueTrain),str(PvalueVal)]),
                                        tf.convert_to_tensor(['numG0', str(numG0_train),str(numG0_val)]),
                                        tf.convert_to_tensor(['numG1', str(numG1_train),str(numG1_val)]),
                                        tf.convert_to_tensor(['optimal_separation', str(optimal_separation),str(optimal_separation)]),
                                        tf.convert_to_tensor(['tf_cindexR', str(eval_train_best[1]),str(cindex_val1)])]
                            s = sess.run(tf.summary.text('EvalBest', tf.stack(evalu)))
                            writer.add_summary(s)
                            evalu = [tf.convert_to_tensor(['', 'train','val']) ,                        
                                    tf.convert_to_tensor(['Pvalgrp', str(PvalueTrain),str(PvalueValgrp)]),
                                        tf.convert_to_tensor(['optimal_separation', str(optimal_separation),str(optimal_separation)]),
                                        tf.convert_to_tensor(['numG0grp', str(numG0_train),str(numG0_valgrp)]),
                                        tf.convert_to_tensor(['numG1grp', str(numG1_train),str(numG1_valgrp)]),
                                    tf.convert_to_tensor(['PvalgrpExclude', str(PvalueTrainExclude),str(PvalueValgrpExclude)]),
                                        tf.convert_to_tensor(['optimal_separation', str(optimal_separation_exclude),str(optimal_separation_exclude)]),
                                        tf.convert_to_tensor(['numG0grpExclude', str(numG0_trainExclude),str(numG0_valgrpExclude)]),
                                        tf.convert_to_tensor(['numG1grpExclude', str(numG1_trainExclude),str(numG1_valgrpExclude)])]
                            s = sess.run(tf.summary.text('EvalBestkaplan', tf.stack(evalu)))
                            writer.add_summary(s)
                            
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
                                evalu = [tf.convert_to_tensor(['', 'train','val','valgrp']) ,                        
                                        tf.convert_to_tensor(['accuracy', str(acc_train),str(acc_val),str(acc_valgrp)])]
                                s = sess.run(tf.summary.text('accuracy', tf.stack(evalu)))
                                writer.add_summary(s)                                
                                
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':

                                [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='classe',order='T')
                                [cindex_val0,cindex_val1,cindex_val2,cindex_val3,cindex_val4,cindex_val5,cindex_val6] = lss.time_cindex(yval[:,1],np.array(predictionval),yval[:,0],cla='classe',order='T')
                            else:
                                [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='classe',order='R')
                                [cindex_val0,cindex_val1,cindex_val2,cindex_val3,cindex_val4,cindex_val5,cindex_val6] = lss.time_cindex(yval[:,1],np.array(predictionval),yval[:,0],cla='classe',order='R')
                        
                            if p.use_test_set:
                                [cindex_test0,cindex_test1,cindex_test2,cindex_test3,cindex_test4,cindex_test5,cindex_test6] = lss.time_cindex(ytest[:,1],predictiontest,ytest[:,0],cla='classe')
                                                            
                            
                            if p.use_test_set:
                                evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                        tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindex_val0),str(cindex_test0)]),
                                        tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindex_val1),str(cindex_test1)]),
                                        tf.convert_to_tensor(['cindex_2', str(cindex2) ,str(cindex_val2),str(cindex_test2)]),
                                        tf.convert_to_tensor(['cindex_3', str(cindex3) ,str(cindex_val3),str(cindex_test3)]),
                                        tf.convert_to_tensor(['cindex_4', str(cindex4) ,str(cindex_val4),str(cindex_test4)]),
                                        tf.convert_to_tensor(['cindex_5', str(cindex5) ,str(cindex_val5),str(cindex_test5)]),
                                        tf.convert_to_tensor(['cindex_6', str(cindex6) ,str(cindex_val6),str(cindex_test6)])]
                            else:
                                evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                        tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindex_val0)]),
                                        tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindex_val1)]),
                                        tf.convert_to_tensor(['cindex_2', str(cindex2) ,str(cindex_val2)]),
                                        tf.convert_to_tensor(['cindex_3', str(cindex3) ,str(cindex_val3)]),
                                        tf.convert_to_tensor(['cindex_4', str(cindex4) ,str(cindex_val4)]),
                                        tf.convert_to_tensor(['cindex_5', str(cindex5) ,str(cindex_val5)]),
                                        tf.convert_to_tensor(['cindex_6', str(cindex6) ,str(cindex_val6)])]
                            s = sess.run(tf.summary.text('EvalBestParTemps_7classes', tf.stack(evalu)))
                            writer.add_summary(s)
                            
                            if params['Loss'] == 'RankAndmse' or params['Loss'] == 'RankAnddiscret' or params['Loss'] == 'discret':
    
                                [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='0ouAutre',order='T')
                                [cindex_val0,cindex_val1] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre',order='T')
                            else:
                                [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='0ouAutre',order='R')
                                [cindex_val0,cindex_val1] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre',order='R')
                                                            
                            if p.use_test_set:
                                [cindex_test0,cindex_test1] = lss.time_cindex(ytest[:,1],predictiontest,ytest[:,0],cla='0ouAutre')

                            if p.use_test_set:
                                evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                        tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindex_val0),str(cindex_test0)]),
                                        tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindex_val1),str(cindex_test1)])]
                            else:
                                evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                        tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindex_val0)]),
                                        tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindex_val1)])]
                            s = sess.run(tf.summary.text('EvalBestParTemps_0ouAutre', tf.stack(evalu)))
                            writer.add_summary(s) 
                                
                            
                        curves = plt.imread(p.path_patients+'datasep.png')
                        if CV==0:
                            writer = tf.summary.FileWriter(log_dir)
                            cawt = tf.convert_to_tensor(np.reshape( curves,(1,np.shape(curves)[0],np.shape(curves)[1],np.shape(curves)[2])))
                            s = sess.run(tf.summary.image('survival_curves_folds', tf.stack(cawt)))
                            writer.add_summary(s)
                        
                        
                        # layerCAW = utils.find_layer_idx(pred,'spp')
                        # caw = Model(inputs=pred.inputs,outputs=pred.layers[layerCAW].output)
                        # fee = caw.predict([xtrain[0:2],mtrain[0:2]]) #shape (B,1,1,64)

                        # layerCAW = utils.find_layer_idx(pred,'res')
                        # caw = Model(inputs=pred.inputs,outputs=pred.layers[layerCAW].output)
                        # fee2 = caw.predict([xtrain[0:2],mtrain[0:2]]) #shape (B,1,1,64)

                        if params['attention']!='False' and CV==0:
                            """ attention model, fee = weight of each filter for each patient"""
                            layerCAW = utils.find_layer_idx(saved_model,'channel_attention_weight')
                            caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerCAW].output)
                            if params['SPP']:
                                fee = caw.predict([xtrain,mtrain]) #shape (B,1,1,64)
                                feetest = caw.predict([xval,mval]) #shape (B,1,1,64)
                            else:
                                fee = caw.predict(xtrain) #shape (B,1,1,64)
                                feetest = caw.predict(xval) #shape (B,1,1,64)
                            if params['D3'] == 'True':
                                fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[4]))
                            else:
                                fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[3]))
                                
                            if params['D3']  == 'True':
                                feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[4]))
                            else:
                                feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[3]))
                                
                            
                            if params['D3'] == 'False':
                                if params['attention'] != 'c':
                                    layerSAW = utils.find_layer_idx(saved_model,'spatial_attention_weight')
                                    saw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerSAW].output)
                                    if params['SPP']:
                                        fee2 = saw.predict([xtrain,mtrain])
                                    else:
                                        
                                        fee2 = saw.predict(xtrain)#shape (B,9,9,1)
                                    if params['D3']  == 'True':                         
                                        fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2],np.shape(fee2)[3]))
                                    else: 
                                        fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2]))

                                    
                                    if params['SPP']:
                                        fee2val = saw.predict([xval,mval])
                                    else:
                                        fee2val = saw.predict(xval)
                                    if params['D3']  == 'True':                         
                                        fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2],np.shape(fee2val)[3]))
                                    else: 
                                        fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2]))                                           
                                    
                                    if p.use_test_set:
                                        if params['SPP']:
                                            fee2test = saw.predict([xtest,mtest])
                                        else:
                                            fee2test = saw.predict(xtest)
                                        if params['D3']  == 'True':                         
                                            fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2],np.shape(fee2test)[3]))
                                        else: 
                                            fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2]))

                            caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerCAW].output)
                            if params['SPP']:
                                fee = caw.predict([xtrain,mtrain]) #shape (B,1,1,64)
                                feeval = caw.predict([xval,mval]) #shape (B,1,1,64)
                            else:
                                fee = caw.predict(xtrain) #shape (B,1,1,64)
                                feeval = caw.predict(xval) #shape (B,1,1,64)
                            # with tf.name_scope("layer1"):
                            
                            #     writer = tf.summary.FileWriter(log_dir)
                            #     s = sess.run(tf.summary.histogram('channel_attention_weight_train', fee))
                            #     writer.add_summary(s)
                            #     writer = tf.summary.FileWriter(log_dir)
                            #     i = tf.placeholder(tf.int32)
                            #     mean_moving_normal = tf.convert_to_tensor(fee)[i]
                            #     histfee = tf.histogram_fixed_width(
                            #         mean_moving_normal,
                            #         [0,63],
                            #         nbins=64,
                            #         dtype=tf.dtypes.int32,
                            #     )                    
                            #     summaries = tf.summary.histogram("hist_channel_attention_weight_train", histfee)
                            #     N = int(np.shape(fee)[0]/2)
                            #     for step in range(N):
                            #       summ = sess.run(summaries, feed_dict={i: step})
                            #       writer.add_summary(summ, global_step=step)
                                
                            #     plt.imshow(np.reshape(sumcawt,(1,64)))
                            #     plt.xticks(list(range(0,64,5)))
                        
                            if params['D3'] == 'False':
                                writer = tf.summary.FileWriter(log_dir)
                                cawt = [tf.convert_to_tensor(np.reshape( fee,(np.shape(fee)[0],np.shape(fee)[3],1)))]
                                s = sess.run(tf.summary.image('channel_attention_weight_train', tf.stack(cawt)))
                                writer.add_summary(s)
                            else:
                                writer = tf.summary.FileWriter(log_dir)
                                cawt = [tf.convert_to_tensor(np.reshape( fee,(np.shape(fee)[0],np.shape(fee)[4],1)))]
                                s = sess.run(tf.summary.image('channel_attention_weight_train', tf.stack(cawt)))
                                writer.add_summary(s)
                                
                            sumcawt = np.sum(fee,axis=0)
                            rankCAW = sumcawt.argsort()[::-1]
                            
                            evalu = tf.convert_to_tensor([str(rankCAW[i]) for i in range(len(rankCAW))])
                            # s = sess.run(tf.summary.text('ranked_filters', tf.stack(evalu)))
                            # writer.add_summary(s)
    
                            writer = tf.summary.FileWriter(log_dir)
                            sortedfilters = np.sort(sumcawt)[::-1]
                            if params['D3'] == 'False':
                            
                                cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[2],1)))]
                            else:
                                cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[3],1)))]

                            s = sess.run(tf.summary.image('Sum_channel_attention_weight_train_over_patients', tf.stack(cawt)))
                            writer.add_summary(s)
                                                
                            if params['D3'] == 'False':
                                cawt = [tf.convert_to_tensor(np.reshape( fee,(np.shape(fee)[0],np.shape(fee)[3],1)))]
                                s = sess.run(tf.summary.image('channel_attention_weight_train', tf.stack(cawt)))
                                writer.add_summary(s)
                            else:
                                cawt = [tf.convert_to_tensor(np.reshape( fee,(np.shape(fee)[0],np.shape(fee)[4],1)))]
                                s = sess.run(tf.summary.image('channel_attention_weight_train', tf.stack(cawt)))
                                writer.add_summary(s)
                            
                            # cawt = [tf.convert_to_tensor(fee2)]
                            # s = sess.run(tf.summary.histogram('spatial_attention_weight_train', tf.stack(cawt)))
                            # writer.add_summary(s)
                            
                            
                            if params['D3'] == 'False':
                                ##### Features map
                                if params['attention'] == 'cs':
                                    layerFeat = utils.find_layer_idx(saved_model,'res')
                                else:
                                    layerFeat = utils.find_layer_idx(saved_model,'multiply_1')
                                caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerFeat].output)
                                if params['SPP'] :
                                    featuresMap = caw.predict([xtrain,mtrain]) #shape (456, 9, 9, 64)
                                    featuresMapTest = caw.predict([xval,mval]) #shape (456, 9, 9, 64)
                                else:
                                    featuresMap = caw.predict(xtrain) #shape (456, 9, 9, 64)
                                    featuresMapTest = caw.predict(xval) #shape (456, 9, 9, 64)
                                
                                for patientn in range(20):
                                    writer = tf.summary.FileWriter(log_dir)
                                    with tf.name_scope(f'channel_feat_map_Pat_{patientn}'):
                                        for fi in range(len(sumcawt)): 
                                            image  = tf.convert_to_tensor(np.reshape(xtrain[patientn,:,:,fi],(1,np.shape(xtrain)[1],np.shape(xtrain)[2],1)))
                                            s = sess.run(tf.summary.image('xtrain',image))
                                            writer.add_summary(s)  
                                            
                                            image  = tf.convert_to_tensor(np.reshape(featuresMap[patientn,:,:,fi],(1,np.shape(featuresMap)[1],np.shape(featuresMap)[2],1)))
                                            s = sess.run(tf.summary.image('feature_map',image))
                                            writer.add_summary(s)  
                                            
                                            
                                #create base 
                                plt.clf() 
                                im = Image.new(mode='RGB',size=(1000,1000),color=(255,255,255))
                                begx = 15
                                begy = 15
                                for patientn in range(10):
                                    for fi in rankCAW[0,0,:20]:
                                        #number
                                        numb = Image.new('RGB', (15, 15),color=(255,255,255))
                                        ImageDraw.Draw(numb).text((0, 0), str(fi), fill=(0, 0, 0))
                                        im.paste(numb,(0,begy+15))
                                        
                                        #pat
                                        pat = Image.new('RGB', (45, 15),color=(255,255,255))
                                        ImageDraw.Draw(pat).text((0, 0),'P_' + str(patientn), fill=(0, 0, 0))
                                        im.paste(pat,(begx+10,0))
                                        
                                        #feat
                                        list_mode =np.array(featuresMap[patientn,:,:,fi]) 
                                        list_mode = cv2.resize(list_mode, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
                                        a_scaled = 255*(list_mode-np.min(list_mode))/(np.max(list_mode)-np.min(list_mode))
                                        im.paste(Image.fromarray(a_scaled),(begx,begy))
                                        
                                        #image
                                        list_mode =np.array(xtrain[patientn,:,:,0]) 
                                        a_scaled = 255*(list_mode-np.min(list_mode))/(np.max(list_mode)-np.min(list_mode))
                                        im.paste(Image.fromarray(a_scaled),(begx+38,begy))
                                        
                                        begy=begy + 50
                                    begx=begx + 84
                                    begy=15
                                
                                plt.axis('off')
                                rt = plt.imshow(im)
                                rt.figure.savefig('Importantfilters.svg', format='svg', dpi=5000)
                                
                                plt.clf() 
                                im = Image.new(mode='RGB',size=(1000,1000),color=(255,255,255))
                                begx = 15
                                begy = 15
                                for patientn in range(10,20):
                                    for fi in rankCAW[0,0,:20]:
                                        #number
                                        numb = Image.new('RGB', (15, 15),color=(255,255,255))
                                        ImageDraw.Draw(numb).text((0, 0), str(fi), fill=(0, 0, 0))
                                        im.paste(numb,(0,begy+15))
                                        
                                        #pat
                                        pat = Image.new('RGB', (45, 15),color=(255,255,255))
                                        ImageDraw.Draw(pat).text((0, 0),'P_' + str(patientn), fill=(0, 0, 0))
                                        im.paste(pat,(begx+10,0))
                                        
                                        #feat
                                        list_mode =np.array(featuresMap[patientn,:,:,fi]) 
                                        list_mode = cv2.resize(list_mode, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
                                        a_scaled = 255*(list_mode-np.min(list_mode))/(np.max(list_mode)-np.min(list_mode))
                                        im.paste(Image.fromarray(a_scaled),(begx,begy))
                                        
                                        #image
                                        list_mode =np.array(xtrain[patientn,:,:,0]) 
                                        a_scaled = 255*(list_mode-np.min(list_mode))/(np.max(list_mode)-np.min(list_mode))
                                        im.paste(Image.fromarray(a_scaled),(begx+38,begy))
                                        
                                        begy=begy + 50
                                    begx=begx + 84
                                    begy=15
                                
                                plt.axis('off')
                                rt = plt.imshow(im)
                                rt.figure.savefig('Importantfilters.svg', format='svg', dpi=5000)
        
        
                        if metrics != 'accuracy':
                            if params['Loss'] != 'discret' and params['Loss']!='RankAnddiscret':
                                if metrics == 'triplet':
                                    
                                    cind_train = uf.cindexR(ytrain[:,1],predictiontrainclass,ytrain[:,0])
                                    
                                    
                                    if p.use_test_set :
                                        cindTest = uf.cindexR(ytest[:,1],predictiontestclass,ytest[:,0])
                                    cind_val = uf.cindexR(yval[:,1],np.array(predictionvalclass),yval[:,0]) 
                                    
                                    [cindex_train1,cindex_train2,cindex_train3] = lss.time_cindex(ytrain[:,1],predictiontrainclass,ytrain[:,0])
                                    
                                    if p.use_test_set :
                                        [cindex_test1,cindex_test2,cindex_test3] = lss.time_cindex(ytest[:,1],predictiontestclass,ytest[:,0])
                                    [cindex_val1,cindex_val2,cindex_val3] = lss.time_cindex(yval[:,1],predictionvalclass,yval[:,0])
            
                                
                                else:
                                        
                                    cind_train = uf.cindexR(ytrain[:,1],predictions_train,ytrain[:,0])
                                    
                                    if p.use_test_set:
                                        cindTest = uf.cindexR(ytest[:,1],predictiontest,ytest[:,0])
                                    print(np.shape(predictionval),np.shape(yval))
                                    cind_val = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0]) 
                                    
                                    [cindex_train1,cindex_train2,cindex_train3] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0])
                                    
                                    if p.use_test_set:
                                        [cindex_test1,cindex_test2,cindex_test3] = lss.time_cindex(ytest[:,1],predictiontest,ytest[:,0])
                                    [cindex_val1,cindex_val2,cindex_val3] = lss.time_cindex(yval[:,1],predictionval,yval[:,0])
                                
                                    [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='classe')
                                    [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictions_train,ytrain[:,0],cla='0ouAutre')
                                    
                                    [cindex_val0,cindex_val1,cindex_val2,cindex_val3,cindex_val4,cindex_val5,cindex_val6] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='classe')
                                    [cindex_val0,cindex_val1] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre')


                            else:
                                cind_train = uf.cindexT(ytrainT[:,1],predictions_train,ytrainT[:,0])
                                
                                if p.use_test_set:   
                                    cindTest = uf.cindexT(ytestT[:,1],predictiontest,ytestT[:,0])
                                cind_val = uf.cindexT(yvalT[:,1],np.array(predictionval),yvalT[:,0]) 
                                
                                # [cindex_train1,cindex_train2,cindex_train3] = lss.time_cindex(uf.FromLongToTime(ytrain,n_intervals,None)[0],predictions_train,uf.FromLongToTime(ytrain,n_intervals,None)[1],'T')
                                # [cindex_test1,cindex_test2,cindex_test3] = lss.time_cindex(uf.FromLongToTime(ytest,n_intervals,None)[0],predictiontest,uf.FromLongToTime(ytest,n_intervals,None)[1],'T')
                                # [cindex_val1,cindex_val2,cindex_val3] = lss.time_cindex(uf.FromLongToTime(yval,n_intervals,None)[0],predictionval,uf.FromLongToTime(yval,n_intervals,None)[1],'T')
                            
                        # if params['Loss'] != 'classif' and params['Loss'] != 'triplet'  and params['Loss'] != 'discret':
                        #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_train")
                        #     np.save(pathSavepredTr,predictions_train)
                        #     pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_train")
                        #     np.save(pathSavetrueTr,ytrain[:,1])
                        #     pathSaveeventTr = os.path.join(file_name, f"{titleP}_event_train")
                        #     np.save(pathSaveeventTr,ytrain[:,0])
                                        
                        #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_val")
                        #     np.save(pathSavepredTr,predictionval)
                        #     pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_val")
                        #     np.save(pathSavetrueTr,yval[:,1])
                        #     pathSaveeventTr = os.path.join(file_name, f"{titleP}_event_val")
                        #     np.save(pathSaveeventTr,yval[:,0])
                        #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_test")
                        #     np.save(pathSavepredTr,predictiontest)
                        #     pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_test")
                        #     np.save(pathSavetrueTr,ytest[:,1])
                        #     pathSaveeventTr = os.path.join(file_name, f"{titleP}_event_test")
                        #     np.save(pathSaveeventTr,ytest[:,0])
                        
                        """  FILTERS  ********************************************************************"""
                        # if params['attention'] !='False':
                        #     flatw =  saved_model.get_layer(index = utils.find_layer_idx(saved_model,'sequential_1')).get_layer(index = -4 )#Newmodel.get_layer(index = -4)
                        #     filters = flatw.get_weights()
                        #     pathSaveFFT = os.path.join(file_name, f"{titleP}_filters")
                        #     np.save(pathSaveFFT,filters)
                        #featModel = Model(inputs=Newmodel.input, outputs=flat.input)
                        #featuresSMALL=featModel.predict([xtrain,mtrain])
            
                        #pathSaveFFT = os.path.join(file_name, f"{titleP}_features_visu_train.csv")
                        #np.savez(pathSaveFFT,featuresSMALL)
                        """  Features ********************************************************************"""
            
                        # flat =  Newmodel.get_layer(index = utils.find_layer_idx(Newmodel,'res2'))#Newmodel.get_layer(index = -4)
                        # featModel = Model(inputs=pred.input, outputs=flat.output)
                        # if not p.cross_validation :
                        #     featuresSMALL=featModel.predict([xtest,mtest] )
                        #     pathSaveFFT = os.path.join(file_name, f"{titleP}_features_flatten_test")
                        #     np.save(pathSaveFFT,featuresSMALL)
                        #     pathCSV_FFT = os.path.join(file_name, f"{titleP}_features_flatten_test.csv")
                        #     df_fft=pd.DataFrame(featuresSMALL)
                        #     df_fft.to_csv(pathCSV_FFT)                        
                        # featurestrain=featModel.predict([xtrain,mtrain])
                
                        # pathSaveFFT = os.path.join(file_name, f"{titleP}_features_flatten_train")
                        # np.save(pathSaveFFT,featurestrain)
                        # pathCSV_FFT = os.path.join(file_name, f"{titleP}_features_flatten_train.csv")
                        # df_fft=pd.DataFrame(featurestrain)
                        # df_fft.to_csv(pathCSV_FFT)                       
                        # featuresval=featModel.predict([xval,mval])
                
                        # pathSaveFFT = os.path.join(file_name, f"{titleP}_features_flatten_val")
                        # np.save(pathSaveFFT,featuresval)
                        # pathCSV_FFT = os.path.join(file_name, f"{titleP}_features_flatten_val.csv")
                        # df_fft=pd.DataFrame(featuresval)
                        # df_fft.to_csv(pathCSV_FFT)
                        
                        # if params['Loss'] == 'coxAndclassif' or params['Loss'] == 'classif' :
                        #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_class_train")
                        #     np.save(pathSavepredTr,predictiontrainclass)
                        #     pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_class_train")
                        #     np.save(pathSavetrueTr,ytrain_grp)
                                        
                        #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_class_val")
                        #     np.save(pathSavepredTr,predictionvalclass)
                        #     pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_class_val")
                        #     np.save(pathSavetrueTr,yval_grp)
                            
                        #     if p.use_test_set:
                        #         pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_class_test")
                        #         np.save(pathSavepredTr,predictionBeforeclass)
                        #         pathSavetrueTr = os.path.join(file_name, f"{titleP}_true_class_test")
                        #         np.save(pathSavetrueTr,ytest_grp)
                                        
                        # pred.save_weights(best_model_file)
                        #results = save_training_plots(fitting,titleP,eval_train, eval_val,None,file_name,metrics)
                        
                        if params['Loss']=='classif' or params['Loss'] =='coxAndclassif':
                            print('--- predict after fintune')
                            
                            if p.use_test_set:
                                ppr = predictionBeforeclass
                                pYM = np.argmax(ytest_grp,axis = 1)
                                resu1 = accuracy_score(pYM,ppr)
                                print('acc test',resu1)
                                confA = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                np.save(os.path.join(file_name, f"{titleP}confusion_test.npy"),confA)
                            
                            ppr = predictiontrainclass
                            pYM = np.argmax(ytrain_grp,axis = 1)
                            resu2 = accuracy_score(pYM,ppr)
                            print('acc train',resu2)
                            confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                            np.save(os.path.join(file_name, f"{titleP}confusion_after_train.npy"),confB)
                            
                            ppr = predictionvalclass
                            pYM = np.argmax(yval_grp,axis = 1)
                            resu3 = accuracy_score(pYM,ppr)
                            print('acc validation',resu3)
                            confC = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                            np.save(os.path.join(file_name, f"{titleP}confusion_after_val.npy"),confC)
                                                        
                        """ ******************** SAVING  *********************************"""
    
                        if p.cross_validation :
                            if p.use_test_set:
                                full_row=[f'CV{CV}_K{k}',eval_train_best[1],cindex_val1,eval_test_best[1]]
                            else:
                                full_row=[f'CV{CV}_K{k}',eval_train_best[1],cindex_val1]

                            for key, value in param.iteritems():
                                print(key)
                                temp = value[0]
                                full_row.append(temp)
                            cv_full.append(full_row)
                            if p.use_test_set:
                                npa =['CV_K','best_train_cindex','best_val_cindex','best_test_cindex']
                            else:                                
                                npa =['CV_K','best_train_cindex','best_val_cindex']

                            for key, value in param.iteritems():
                                temp = key
                                npa.append(temp)
                            path_full_df =  os.path.join(intermediate_folder, f'FullCV_{CV}.csv')
                            df_full=pd.DataFrame(cv_full,columns = npa)
                            df_full.to_csv(path_full_df)
    
                            # loss_train.append(eval_train[0])
                            # if params['Loss'] != 'classif' :
                            cindex_train.append(cind_train)
                            # else:
                                # cindex_train.append(0)
                            # loss_valid.append(eval_val[0])
                            # if params['Loss'] != 'classif':
                            cindex_val.append(cind_val)
                            # else:
                            #     cindex_val.append(0)
                            if params['Loss'] != 'classif' and params['Loss'] != 'coxAndclassif' and params['Loss']!='triplet':
                                AcurracyTrain.append(0)
                                AcurracyVal.append(0)
                            # elif metrics=='coxAndclassif':
                                # AcurracyTrain.append(eval_train[6])
                                # AcurracyVal.append(eval_val[6])
                                
                            else:
                                AcurracyTrain.append(resu2)
                                AcurracyVal.append(resu3)
                            if params['Loss'] == 'cox' or params['Loss'] == 'discret' :   
                                train_best.append(eval_train_best[1])
                                val_best.append(cindex_val1)
                            else:
                                train_best.append(eval_train_best[2])
                                val_best.append(cindex_val1)
                                
                        
                """ FIN K loop """
            if p.cross_validation:
                if not p.apply_pretrain :
                    df_full = pd.read_csv(path_full_df, encoding='utf-8')
                    path_csv_mean_over_fold =   os.path.join(intermediate_folder, f'MeanOverFoldCV_{str(CV)}.csv')
                    if os.path.exists(path_csv_mean_over_fold):
                        mofcv = pd.read_csv(path_csv_mean_over_fold, encoding='utf-8')
                        mofcv = mofcv.iloc[:,1:].values.tolist()
                    else:
                        mofcv = []
                    if p.use_test_set:
                        npa =['CV_K','best_train_cindex','best_val_cindex','best_test_cindex']
                    else:
                        npa =['CV_K','best_train_cindex','best_val_cindex']

                    numi=0
                    for k in range(len(X)):
                        kDF = df_full[df_full['CV_K'] == f'CV{str(CV)}_K{str(k)}']
                        dict_mode={'loss':list(set(kDF['loss'])),'D3':list(set(kDF['D3'])),'spp':list(set(kDF['spp'])),'attention':list(set(kDF['attention'])),'pretrain':list(set(kDF['pretrain'])),'npatch':list(set(kDF['npatch']))}
                        gridMode = ParameterGrid(dict_mode)
                        for pM in gridMode:
                            try:
                                l= df_full['loss']==pM['loss']
                                d= df_full['D3']==pM['D3']
                                s= df_full['spp']==pM['spp']
                                a= df_full['attention']==pM['attention']
                                pr= df_full['pretrain']==pM['pretrain']
                                npp= df_full['npatch']==pM['npatch']
                                list_mode=kDF[l & d & s & a & pr & npp]
                                indM=np.where(list_mode['best_val_cindex']==np.max(list_mode['best_val_cindex']))
                                
                                if numi==0:
                                    mofcv=list_mode.iloc[indM[0],1:]
                                else:
                                    mofcv=pd.concat([mofcv,list_mode.iloc[indM[0],1:]])
                                numi+=1
                            except:
                                pass
                    mofcv.to_csv(path_csv_mean_over_fold) 
                    
if __name__ == "__main__":

    """ ******************** OPTIONS ******************** """

    parser = argparse.ArgumentParser()
    #path arguments
    parser.add_argument("--training-dir", type=Path, help="name of the training folder", default = 'avecspatial')
    parser.add_argument("--save-dir", type=Path, help="name of the experiement folder", default = os.path.join('Results','ModeFinalRunVal' ))

    parser.add_argument("--cross-validation",  action="store_true", help="If True do cross validation, otherwise train/validation/test.")
    parser.add_argument("--ratio-binary-pretraining", type=int, help="Ratio for binary pretraining.", default=50)
    parser.add_argument("--stride", type=int, help="Discretisation (in days).", default=365)
    parser.add_argument("--interpolation-size", type=int, help="Size of the input image.", default=36)
    parser.add_argument("--use-test-set", action="store_true", help="If True have a test set.")
    parser.add_argument("--distribution", type=list, help="Distribution in the folders.", default=[0.25,0.25,0.25,0.25])
    parser.add_argument("--reload-inputs", action="store_true", help="Reload inputs.")
    parser.add_argument("--apply-pretrain",action="store_true", help="Apply pretraining.")

    p = parser.parse_args()
    
    ''' ******************** PATH ******************** '''
    
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    p.path_data = os.path.join(current_directory, "data","ALL_DATA","Patients")
   
    test_text = 'without_test' if not p.use_test_set else ''
    p.path_patients = Path(os.path.join('Results',f'PATIENTS_SEP_TO_KEEP_154CVT{test_text}'))
    
    for path in [p.path_patients, p.save_dir]:
        os.makedirs(path, exist_ok=True) 
          
    if "sess" in locals():     # this code prevent creating many intance of
        tf.compat.v1.reset_default_graph()
        sess.close()           # tensorflow in memory (that can cause crashes)
    K.clear_session()
    sess = tf.Session() #create new tensorflow session


    """ ******************** Data augmentation definition ********************"""
    path_data_augmentation = os.path.join('files','data_augmentation_config.npy')
    
    augmentation_config = dpr.get_augmentation_config(path = path_data_augmentation, num_augm = 30)
    
    """ ********************SURVIVAL MM DATA *********************************"""
    
    dict_input = dpr.image_mask_ref(p.path_data)

    # """ ******************** DATA STATISTICS *********************************"""
    # display_data_stats(p, list_y, p.path_patients)
            
    
    if p.apply_pretrain :
        """ ******************** BINARY DATA *********************************"""
        dict_binary = ptg.binary_extraction(dict_input['list_refs'], p.cross_validation, p.path_data)
        
        if not p.cross_validation :
            x_binary_train_2d, y_binary_train_2d = dict_binary[0]['train']['list_x_2d'], dict_binary[0]['train']['list_y_2d']
            x_binary_val_2d, y_binary_val_2d = dict_binary[0]['val']['list_x_2d'], dict_binary[0]['val']['list_y_2d']
        else:
            x_binary_train_2d = [[dict_binary[i]['train']['list_x_2d']] for i in len(dict_binary.keys())] 
            y_binary_train_2d = [[dict_binary[i]['train']['list_y_2d']] for i in len(dict_binary.keys())]  
            x_binary_train_3d = [[dict_binary[i]['train']['list_x_3d']] for i in len(dict_binary.keys())] 
            y_binary_train_3d = [[dict_binary[i]['train']['list_y_3d']] for i in len(dict_binary.keys())] 
    
        parameters = {'Maxi':[False], 
                      'SPP':['False'], 
                      'D3':['True'],
                      'Loss':['cox'],
                      'pretrain':['binary'] ,  #,'textureP','textureF', 'binaryP','binaryF'
                      'LR Pretrain':[1e-04], 
                      'LR Finetune': [1e-04],  
                      'rate' : [0.83], 
                      'Lrate_decay_learningP' :[1e-07],
                      'lrate_decay_learningF' :[1e-8],
                      'epoch Train' : [20] ,
                      'epoch Finetune': [40] , 
                      'batch size': [10], 
                      'image size':[36],
                      'noTrainLayer': [0] ,
                      'npatch':[3],
                      'dataAug':[15],
                      'attention':['s'],
                      'nb_classes':[7],
                      'bilinear':[False],
                      'l2reg':[0.001]}
    else:
        parameters = {'Maxi':[False], 
                      'SPP':['False'], 
                      'D3':['True'],
                      'Loss':['cox'],
                      'pretrain':['binary'], 
                      'LR Pretrain':[1e-04], 
                      'LR Finetune': [1e-04],  
                      'rate': [0.83,0.17,0.5], 
                      'Lrate_decay_learningP' :[1e-07],
                      'lrate_decay_learningF' :[1e-8,1e-10,1e-12],
                      'epoch Train': [20], 
                      'epoch Finetune': [40], 
                      'batch size': [10], 
                      'image size':[36],
                      'noTrainLayer': [0],
                      'npatch':[3],
                      'dataAug':[15,30],
                      'attention':['s'],
                      'nb_classes':[7],
                      'bilinear':[False],
                      'l2reg':[0.001]}
    
    breaks = np.arange(0.,365*7,p.stride)
    
    get_optimised_model(p, parameters, breaks, dict_input, sess)
