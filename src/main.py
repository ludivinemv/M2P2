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
import functions.data_processing.data_processing as dpr
import functions.losses.losses as lss
import functions.data_processing.pretraining as ptg
import functions.data_processing.data_prep as dataprep
import functions.losses.survivalTripletloss as striplet


if __name__ == "__main__":

    """ ########################### OPTIONS ###########################################"""

    parser = argparse.ArgumentParser()
    #path arguments
    parser.add_argument("--training-dir", type=Path, help="name of the training folder", default = 'avecspatial')
    parser.add_argument("--save-dir", type=Path, help="name of the experiement folder", default = os.path.join('Results','ModeFinalRunVal' ))

    parser.add_argument("--cross-validation",  action=bool, help="If True do cross validation, otherwise train/validation/test.", choices=[True,False], default = True)
    parser.add_argument("--ratio-binary-pretraining", type=int, help="Ratio for binary pretraining.", default=50)
    parser.add_argument("--stride", type=int, help="Discretisation (in days).", default=365)
    parser.add_argument("--interpolation-size", type=int, help="Size of the input image.", default=36)
    parser.add_argument("--use-test-set", type=bool, help="If True have a test set.", choices = [True, False], default=False)
    parser.add_argument("--distribution", type=list, help="Distribution in the folders.", default=[0.25,0.25,0.25,0.25])
    parser.add_argument("--reload-inputs", type=bool, help="Reload inputs.", choices = [True, False], default=True)
    parser.add_argument("--pretrain", type=bool, help="Apply pretraining.", choices = [True, False], default=False)

    p = parser.parse_args()
    
    '''########################### PATH ############################################# '''
    
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    path_data = os.path.join(current_directory, "data","ALL_DATA","Patients")
    path_patients = Path(os.path.join('Results','PATIENTS_SEP_TO_KEEP_154CVTsansTest'))
    
    for path in [path_patients, p.save_dir]
        os.makedirs(path, exist_ok=True) 
          
    '''########################### CODE ############################################# '''

    if "sess" in locals():     # this code prevent creating many intance of
        tf.compat.v1.reset_default_graph()
        sess.close()           # tensorflow in memory (that can cause crashes)
    K.clear_session()
    sess = tf.Session() #create new tensorflow session


    """ ******************** Data augmentation definition ********************"""
    path_data_augmentation = 'dataAug.npy'
    if os.path.exists(path_data_augmentation):
        augmentation_config = np.load(path_data_augmentation, allow_pickle=True)
    else:
        augmentation_config = dpr.createDataAug()
        np.save(path_data_augmentation,augmentation_config)
        
    """ ********************SURVIVAL MM DATA *********************************"""
    
    liste_image, liste_mask,liste_ref = dpr.image_mask_ref(path_data)
    
    linList = ['all'] #take all the discretised time classes
    
    for lin in linList:    
        print('pretrain2 : ',p.pretrain) 
        print(f'************* lin {lin} *********************')
        print('***************************************')
        test_text = ''
        
        if not p.use_test_set :
            test_text = 'sansTest'
            
        if lin != 'all':
            path_patients = Path(os.path.join('Results',f'PATIENTS_SEP_TO_KEEP_154CVT{lin}{test_text}'))
        else:
            path_patients = Path(os.path.join('Results',f'PATIENTS_SEP_TO_KEEP_154CVT{test_text}'))

        os.makedirs(path_patients, exist_ok=True) 
        
        cl = lin #'all'
        
        # """ ******************** DATA EVALUATION *********************************"""
        # if not p.cross_validation :
        #     def stat(y,name):
        #         mean= np.mean(y[:,1])
        #         med = np.median(y[:,1])
        #         tauxCensor = (1 - np.mean(y[:,0]))*100
        #         print(f"{name}:    Censorship {tauxCensor}%, mean time {mean}, median time {med}")
        #     stat(ytrainsurv,'train')
        #     stat(yvalsurv,'val')
        #     stat(ytestsurv,'test')
        #     print(len(ytrainsurv),len(ytestsurv))
        # else:
        #     def stat(y,name):
        #         mean = np.mean(y[:,1])
        #         med = np.median(y[:,1])
        #         tauxCensor = 100- (np.mean(y[:,0])*100)
        #         print(f"{name}:    Censorship {tauxCensor}%, mean time {mean}, median time {med}")
        #     for k in range(len(list_pat_ref)):
        #         stat(Ysurv[k],str(k))
                
        
        #     kmf = KaplanMeierFitter()
        #     kmf.fit(Ysurv[0][:,1], Ysurv[0][:,0], label='0')
        #     ax = kmf.plot()
        #     kmf = KaplanMeierFitter()
        #     kmf.fit(Ysurv[1][:,1], Ysurv[1][:,0], label='1')
        #     ax = kmf.plot(ax=ax)
        #     kmf = KaplanMeierFitter()
        #     kmf.fit(Ysurv[2][:,1], Ysurv[2][:,0], label='2')
        #     ax = kmf.plot(ax=ax)
        #     kmf = KaplanMeierFitter()
        #     kmf.fit(Ysurv[3][:,1], Ysurv[3][:,0], label='3')
        #     ax = kmf.plot(ax=ax)
            
        #     if len(distribution) >4 :
        #         kmf = KaplanMeierFitter()
        #         kmf.fit(Ysurv[4][:,1], Ysurv[4][:,0], label='4')
        #         ax = kmf.plot(ax=ax)
        #     ax.get_figure().savefig(path_patients+'datasep.png')
        #     plt.savefig(path_patients+'datasep.png')
        #     plt.close()

        print('pretrain : ',p.pretrain)
        
        if p.pretrain :
            """ ******************** BINARY DATA *********************************"""
            if not p.cross_validation :
                xbitrain2D, ybitrain2D, xbival2D, ybival2D = ptg.binary_extraction(list_pat_ref, p.cross_validation, path_data, sess, TEST=False, p.interpolation_size)
            else:
                Xb2D, Yb2D, Xb3D, Yb3D = ptg.binary_extraction(list_pat_ref,p.cross_validation,path_data,sess,TEST=False, p.interpolation_size)
        
        breaks=np.arange(0.,365*7,p.stride)
        
        if p.pretrain:
            parameters = {'Maxi':[False], 'SPP':['False'], 'D3':['True'],
                            'Loss':['cox'],'pretrain':['binary'] ,  #,'textureP','textureF', 'binaryP','binaryF'
                            'LR Pretrain':[1e-04], 'LR Finetune': [1e-04],  'rate' : [0.83] , 
                            'Lrate_decay_learningP' :[1e-07],
                            'lrate_decay_learningF' :[1e-8],'epoch Train' : [20] , 
                            'epoch Finetune': [40] , 'batch size': [10], 'image size':[36],
                            'noTrainLayer': [0] ,'npatch':[3],'dataAug':[15],'attention':['s'],'nb_classes':[7]}
        else:
            parameters = {'Maxi':[False], 'SPP':['False'], 'D3':['True'],
                            'Loss':['cox'],'pretrain':['binary'] ,  #,'textureP','textureF', 'binaryP','binaryF'
                            'LR Pretrain':[1e-04], 'LR Finetune': [1e-04],  'rate' : [0.83,0.17,0.5] , 
                            'Lrate_decay_learningP' :[1e-07],
                            'lrate_decay_learningF' :[1e-8,1e-10,1e-12],'epoch Train' : [20] , 
                            'epoch Finetune': [40] , 'batch size': [10], 'image size':[36],
                            'noTrainLayer': [0] ,'npatch':[3],'dataAug':[15,30],'attention':['s'],'nb_classes':[7]}
        
        gh=0
        
        l2reg=0.001
        
        n_intervals=len(breaks) 
           
        finalR = []
        pathCSV_finDF = os.path.join(p.save_dir,f'finalEval_{p.training_dir}.csv')
        
        if os.path.exists(pathCSV_finDF):
            finalR = pd.read_csv(pathCSV_finDF, encoding='utf-8')
            finalR = finalR.iloc[:,1:].values.tolist()
        else:
            finalR = []
            
        binary=[]
        
        bilinear=False
        Maxi= False
        
        count =0
        
        grid = ParameterGrid(parameters)
        
        for params in grid:  
            gh=gh+1
            print('***************************************')
            print(f'****** run {grid} on {len(grid)} *******************')
            print('***************************************')
            print('***************************************')

            nb_classes,attention,data_augm_size,npatch,D3,loss,pretrain, spp, Maxi , lratePre , lr_fintuning, epochF,p.interpolation_size,batch,epochT ,lr_decay_fintuning,LRDecayP, rate,noTrainLayer = params['nb_classes'],params['attention'],params['dataAug'],params['npatch'],params['D3'],params['Loss'],params['pretrain'],params['SPP'], params['Maxi'],params['LR Pretrain'],params['LR Finetune'], params['epoch Finetune'],params['image size'],params['batch size'],params['epoch Train'],params['lrate_decay_learningF'],params['Lrate_decay_learningP'], params['rate'], params['noTrainLayer']
        
            dict_param = { 'attention':[attention],'dataAug':[data_augm_size],'npatch': [npatch],'D3': [D3],'loss' : [loss], 'nb_classes':[nb_classes],'pretrain': [pretrain],'spp': [spp], 'Maxi':[Maxi], 'LR Pretrain':[lratePre],'LR Finetune': [lr_fintuning],  'epoch F' : [epochF], 'epoch T' : [epochT],'interpolsize' : [p.interpolation_size] , 'batch': [batch],'Lrate_decay_learningP': [LRDecayP],'lrate_decay_learningF': [lr_decay_fintuning],'rate': [rate],'noTrainLayer':[noTrainLayer]}
            param = pd.DataFrame.from_dict(dict_param)
            print(dict_param)
                          
            list_mode=[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
            
            do = True        
                
            if lr_fintuning > lr_decay_fintuning and do  : 
                lrd=str(lr_decay_fintuning)[-1]
                if loss == 'classif':
                    metrics = 'accuracy'
                else: 
                    metrics = lss.tf_cindexR
                    
                if not p.pretrain :
                    if not p.cross_validation :
                        if loss=='discret' or loss == 'RankAnddiscret':
                            xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest, ytrainT,yvalT,ytestT, rtrain, rtest,rval = dataprep.extract_dataIm(p.cross_validation,liste_ref,path_patients,path_data,sess,cl,distribution,p.interpolation_size,list_pat_ref,p.reload_inputs,spp,npatch,D3,loss)
                        else :
                            xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest , rtrain, rtest,rval = dataprep.extract_dataIm(p.cross_validation,liste_ref,path_patients,path_data,sess,cl,distribution,p.interpolation_size,list_pat_ref,p.reload_inputs,spp,npatch,D3,loss)
                    else: 
                        Xt,Yt,Mt,Rt, list_pat_ref = dataprep.extract_dataIm(p.cross_validation,liste_ref,path_patients,path_data,sess,cl,distribution,p.interpolation_size,list_pat_ref,p.reload_inputs,spp,npatch,D3,loss)
                    
                    loss_train, loss_valid, AcurracyTrain, AcurracyVal, cindex_train,cindex_val,train_best, val_best =[],[],[],[],[],[],[],[]
                    
                if not p.cross_validation :
                    cv_number = 1
                else:
                    cv_number = len(list_pat_ref)
                    
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
                        if not p.pretrain:
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
                                if D3=='True' :
                                    Xb=[Xb3D[1],Xb3D[2],Xb3D[3],Xb3D[4]]
                                    Yb=[Yb3D[1],Yb3D[2],Yb3D[3],Yb3D[4]]
                                    # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                                else:
                                    Xb=[Xb2D[1],Xb2D[2],Xb2D[3],Xb2D[4]]
                                    Yb=[Yb2D[1],Yb2D[2],Yb2D[3],Yb2D[4]]
                            else:
                                if D3=='True' :
                                    Xb=[Xb3D[0],Xb3D[1],Xb3D[2],Xb3D[3]]
                                    Yb=[Yb3D[0],Yb3D[1],Yb3D[2],Yb3D[3]]
                                    # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                                else:
                                    Xb=[Xb2D[0],Xb2D[1],Xb2D[2],Xb2D[3]]
                                    Yb=[Yb2D[0],Yb2D[1],Yb2D[2],Yb2D[3]]
                    if not p.cross_validation :
                        KNum = 1
                    else:
                        KNum = len(list_pat_ref)-1
                    if not p.use_test_set:
                        KNum = len(list_pat_ref)

                    for k in range(KNum):
                    
                        list_mode=[f'CV{CV}_K{k}']+[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
                        
                        for j in range(len(cv_full)):
                            fcv = [cv_full[j][0]]   + cv_full[j][-19:]  
                            fcv = [str(fcv[i]) for i in range(len(fcv))]
                            if np.array_equal(list_mode,fcv):
                                do=False
                                                        
                        na = f'{str(gh)}_{str(CV)}_{str(k)}' 
                        
                        titleP = f'{na}lrP{lratePre}_lrF{lr_fintuning}b{batch}_eF{epochF}_lrd{lrd}_D{data_augm_size}_r{rate}'
                        
                        if loss == 'classif' or loss =='triplet':
                            loss_attribute = f"{loss}{str(nb_classes)}"
                        elif loss == 'discret':
                            loss_attribute = 'disc' + str(n_intervals)
                        else:
                            loss_attribute = loss
                                            
                        loss_attribute = f"{loss_attribute}_Pre_{pretrain}_"
                        file_name = os.path.join(intermediate_folder, f'{loss_attribute}at{attention}_spp{spp}_maxi{Maxi}_D3{D3}_n{noTrainLayer}_dim{npatch}')
                        
                        
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
                            print(f'****** run {gh} on {len(grid)} *******************')
                            print('***************************************') 
                            print(f'****** lin {lin} *******************')
                            path_full_df = os.path.join(intermediate_folder,f'FullCV_{CV}.csv')
                            
                            if os.path.exists(path_full_df):
                                cv_full = pd.read_csv(path_full_df, encoding='utf-8')
                                cv_full = cv_full.iloc[:,1:].values.tolist()
                            else:
                                cv_full = []
            
                            na = str(gh)+'_'+str(CV) +'_'+str(k) 
                            
                            if loss == 'classif' or loss =='triplet':
                                loss_attribute = loss + str(nb_classes)
                            elif loss == 'discret':
                                loss_attribute = 'disc' + str(n_intervals)
                            else:
                                loss_attribute = loss
                                
                            loss_attribute = loss_attribute+'_Pre_'+pretrain+ '_'
                            file_name = os.path.join(intermediate_folder,f'{loss_attribute}at{attention}_spp{spp}_maxi{Maxi}_D3{D3}_n{noTrainLayer}_dim{npatch}')
                            os.makedirs(file_name, exist_ok=True)
                                                    
                            log_dir = os.path.join(file_name, "logs","fit", titleP)
                            
                            if CV== 0 or CV == 2:
                                tensorboard = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                            else:
                                tensorboard = TensorBoard(log_dir=log_dir) 
        
                            
                            """  DATA AUGMENTATION ********************************************************************"""
                            if not p.pretrain :
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
                                        if len(distribution)>4:
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
                                    if p.use_test_set  or len(distribution)==4:
                                        xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2]))
                                        mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2]))
                                        ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2]))
                                        rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2]))
                                    else:
                                        xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2],xtrain[3]))
                                        mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2],mtrain[3]))
                                        ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2],ytrain[3]))
                                        rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2],rtrain[3]))
                                    if data_augm_size!=0:
                                        if D3 == 'False':#or D3=='25'
                                            xtrain,ytrain,mtrain,rtrain=dataprep.dataAugmentation(augmentation_config,xtrain,ytrain,mtrain,rtrain,D3=D3,num =data_augm_size)
                                            xval,yval,mval,rval=dataprep.dataAugmentation(augmentation_config,xval,yval,mval,rval, D3=D3,num =data_augm_size)
                                        else:
                                            xtrain,ytrain,mtrain,rtrain=dataprep.DataAugmentation3D(augmentation_config,xtrain,ytrain,mtrain, rtrain, num =data_augm_size)
                                            xval,yval,mval,rval=dataprep.DataAugmentation3D(augmentation_config,xval,yval,mval,rval,num =data_augm_size)
                                            if p.use_test_set:
                                                xtestA,ytestA,mtestA,rtestA=dataprep.DataAugmentation3D(augmentation_config,xtest,ytest,mtest,rtest,num =data_augm_size)
        
                                        xval, yval, mval,rval = dpr.shuffle_and_interpolate(xval, yval,rval,p.interpolation_size,interStatue = False,liste_mask =mval)
                                        xtrain, ytrain, mtrain,rtrain = dpr.shuffle_and_interpolate(xtrain, ytrain,rtrain, p.interpolation_size,interStatue = False,liste_mask =mtrain)

                                else:
                                    if data_augm_size!=0:
                                        if D3 == 'False' :#or D3=='25'
                                            xtrain,ytrain,mtrain=dataprep.dataAugmentation(augmentation_config,xtrain,ytrain,mtrain,D3=D3,num=data_augm_size)
                                            xval,yval,mval=dataprep.dataAugmentation(augmentation_config,xval,yval,mval,D3=D3,num =data_augm_size)
                                            #xtest,ytest,mtest=dataprep.dataAugmentation(augmentation_config,xtest,ytest,mtest,D3=D3,num =data_augm_size)
                                        else:
                                            xtrain,ytrain,mtrain=dataprep.DataAugmentation3D(augmentation_config,xtrain,ytrain,mtrain, num =data_augm_size)
                                            xval,yval,mval=dataprep.DataAugmentation3D(augmentation_config,xval,yval,mval,num =data_augm_size)
                                            #xtest,ytest,mtest=uf.DataAugmentation3D(augmentation_config,xtest,ytest,mtest,num=data_augm_size)
                            
                                        xval, yval, mval = dpr.shuffle_and_interpolate2(xval, yval,p.interpolation_size,interStatue = False,liste_mask =mval)
                                        xtrain, ytrain, mtrain = dpr.shuffle_and_interpolate2(xtrain, ytrain,p.interpolation_size,interStatue = False,liste_mask =mtrain)
                                        #xtest, ytest, mtest = dpr.shuffle_and_interpolate2(xtest, ytest,p.interpolation_size,interStatue = False,liste_mask =mtest)
                                if p.use_test_set:
                                    xtrain,xval,xtest = dpr.normalize(xtrain),dpr.normalize(xval),dpr.normalize(xtest)
                                else:
                                    xtrain,xval = dpr.normalize(xtrain),dpr.normalize(xval)
        
                                if D3=='25':
                                    xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                                    mtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                                    if p.use_test_set:
                                        xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                                        mtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                                    xval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))
                                    mval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))
        
                                if loss == 'classif':
                                    ytrain_grp = dpr.YProcessing2(ytrain,nb_classes)
                                    yval_grp = dpr.YProcessing2(yval,nb_classes)
                                    if p.use_test_set:
                                        ytest_grp = dpr.YProcessing2(ytest,nb_classes)
                                        
                                elif loss == 'triplet':
                                    ytrain_grp = np.transpose(np.concatenate(([ytrain[:,0]],[np.argmax(dpr.YProcessing2(ytrain,nb_classes),axis=1)]),axis=0))
                                    yval_grp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(dpr.YProcessing2(yval,nb_classes),axis=1)]),axis=0))
                                    if p.use_test_set:
                                        ytest_grp =np.transpose(np.concatenate(([ytest[:,0]],[np.argmax(dpr.YProcessing2(ytest,nb_classes),axis=1)]),axis=0))
                                else:
                                    ytrain_grp = uf.YProcessing4(ytrain,'YEAR')
                                    yval_grp = uf.YProcessing4(yval,'YEAR')
                                    if p.use_test_set:
                                        ytest_grp = uf.YProcessing4(ytest,'YEAR')
                                        
                                    ytrain_grpwc = uf.YProcessing4(ytrain,'YEARWC')
                                    yval_grpwc = uf.YProcessing4(yval,'YEARWC')
                                    if p.use_test_set:
                                        ytest_grpwc = uf.YProcessing4(ytest,'YEARWC')
                                        
                                    ytrain_grp3 = uf.YProcessing4(ytrain,'THREE')
                                    yval_grp3 = uf.YProcessing4(yval,'THREE')
                                    if p.use_test_set:
                                        ytest_grp3 = uf.YProcessing4(ytest,'THREE')
                                        
                                    ytrain_grpwc3 = uf.YProcessing4(ytrain,'THREEWC')
                                    yval_grpwc3 = uf.YProcessing4(yval,'THREEWC')
                                    if p.use_test_set:
                                        ytest_grpwc3 = uf.YProcessing4(ytest,'THREEWC')
                                    
                                if loss=='discret' or loss == 'RankAnddiscret':
                                    if p.use_test_set:
                                        ytrainT,yvalT,ytestT =  ytrain,yval,ytest
                                        ytrain,yval,ytest= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks),uf.FromTimeToLong(ytest,breaks)
                                    else:
                                        ytrainT,yvalT =  ytrain,yval
                                        ytrain,yval= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks)
                                
                                
                            """  model creation ********************************************************************"""    
                            
                            if pretrain == 'triplet':
                                param_tri_lr_decay_pretraining = [1e-8,1e-8,1e-12,1e-8]

                                ww = ['2_0_0lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model',
                                    '1_0_1lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.17best_model',
                                    '7_0_2lrP0.0001_lrF0.0001b10_eF20_lrd2_D15_r0.17best_model',
                                    '2_0_3lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model']
                                param_tri_rate = [0.83,0.17,0.17,0.83]

                            if not spp:
                                conv_1=False
                            else:
                                conv_1=False
                            if bilinear :
                                model_cnn = ConvSurvBefore(p.interpolation_size,D3,spp=spp) 
                                model_cnn_2 = ConvSurvBefore(p.interpolation_size,D3,spp=spp) 
                                pred =  BilinearModel(p.interpolation_size,modelCN = model_cnn,model_cnn_2 = model_cnn_2, mode = loss,D3=D3,spp=spp,attention=attention,num=nb_classes)
                            else: 
                                if pretrain == 'triplet':
                                    # pred = deepConvSurv(p.interpolation_size,mode = 'triplet',D3=D3,spp=spp,attention=attention,num=nb_classes,rate=param_tri_rate,conv1=conv_1,maxi = Maxi)  
                                    pred = deepConvSurv(p.interpolation_size,mode = 'triplet',D3=D3,spp=spp,attention=attention,num=nb_classes,rate=rate,conv1=conv_1,maxi = Maxi,l11=l2reg)  

                                elif pretrain == 'binary':
                                    pred = deepConvSurv(p.interpolation_size,mode = 'classif',D3=D3,spp=spp,attention=attention,num=2,rate=rate,conv1=conv_1,maxi = Maxi,l11=l2reg)  
                                else:
                                    pred = deepConvSurv(p.interpolation_size,mode = loss ,D3=D3,spp=spp,attention=attention,num=nb_classes,rate=rate,conv1=conv_1,maxi = Maxi,l11=l2reg)  

                            ''' *****************************************'''
                            ''' **************PRETRAINING****************'''
                            ''' *****************************************'''                
                            if pretrain=='binary':
                                if p.pretrain :
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
                                pred.compile(optimizer=adam(lr=lratePre,decay=LRDecayP), loss='categorical_crossentropy',metrics=['accuracy']) #cindex_score
                                folder_name = f"Binary3D{D3}_spp{spp}_att{attention}_maxi{Maxi}"
                                
                                folder_save = os.path.join(p.save_dir , 'pretraining' , folder_name)
                                
                                os.makedirs(folder_save, exist_ok=True) 
                                
                                weights = os.path.join(folder_save, f"Weights{lratePre}_{LRDecayP}_ratio{p.ratio_binary_pretraining}_drop{0.83}_CV{CV}K{k}.h5")
                                
                                if os.path.exists(weights) :
                                    print('load weights')
                                    pred.load_weights(weights)
                                    fit = 0
                                else:
                                    print('run pretraining .....')
        
                                    log_dirP = os.path.join(folder_save, "logs","fit"+ f"Weights{lratePre}_{LRDecayP}_ratio{p.ratio_binary_pretraining}_CV{CV}K{k}")
                                    
                                    tensorboardP = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                                    
                                    fitting = pred.fit([xbitrain,mbitrain], ybitrain, validation_data=([xbival,mbival], ybival), batch_size=batch,epochs=epochT, callbacks=[tensorboardP])#,mcAc
                                    pred.save_weights(weights)
                                    print('pretraining finished')
        
                                    fit = 1
                                    print('--- evaluation')
                                    PREeval_train = pred.evaluate([xbitrain,mbitrain], ybitrain)
                                    print('eval train',PREeval_train)
                                    PREeval_val = pred.evaluate([xbival,mbival], ybival)
                                    print('eval val',PREeval_val)
                                    
                                    name = f'pretrainBinary_{attention}_{D3}_{loss}_{spp}_{lratePre}_{LRDecayP}_{Maxi}'
                                    results = dpr.saving(fitting,name,PREeval_train, PREeval_val,None,p.save_dir,'accuracy')            
                                    binary.append([PREeval_train[1],PREeval_val[1],lratePre,LRDecayP])
                                    np.savetxt(os.path.join(p.save_dir, 'pretraining', "pretrainBinaryResults.csv"),binary)
                            
                            if pretrain == 'triplet':
                                folder_name = os.path.join(p.save_dir, 'Best','CV0','triplet7_Pre_False_atc_sppFalse_maxiFalse_D3True_n3')

                                weights = folder_name + f'{ww[k]}.h5'
                                
                                pred.load_weights(weights)
            
                            if pretrain != 'triplet':
                                lay = pred.get_layer(index = -2).output
                            else:
                                lay = pred.get_layer(index = -1).output

                                lay = Dense(100, name="fc3",kernel_regularizer=regularizers.l2(0.01),
                                            bias_initializer='zeros')(lay)

                                lay = LeakyReLU(alpha=0.1)(lay)
                                lay = Dropout(rate = rate)(lay)
                                
                            
                            if loss == 'cox':
                                x1 = Dense(1, activation='linear', name="cox_output",kernel_regularizer=regularizers.l2(0.001))(lay)
            
                            elif loss == 'discret'or loss=='RankAnddiscret':
                                x1= Dense(n_intervals,  name="output",activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.001))(lay)
                            
                            elif loss == 'RankAndmse':
                                x1 = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
                            elif loss == 'RankAndcox':
                                x1 = Dense(1, activation='linear', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
                            elif loss == 'triplet':
                                x1 = Dense(nb_classes, activation='softmax', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
        
                            pred = Model(inputs=pred.input, outputs=x1) 
                            if pretrain == 'triplet' and noTrainLayer !=0:
                                for layer in pred.layers[1].layers[:noTrainLayer]:
                                    print(layer)
                                    layer.trainable = False
                            
                            # print(pred.summary())
                            # for layer in pred.layers[:noTrainLayer]:
                            #     print(layer)
                            #     layer.trainable = False
                            
                            # if os.path.exists(best_model_file) :
                            #     print('load weights')
                            #     pred.load_weights(best_model_file)
                            #     fit = 0

                            if p.pretrain == False:
                                """ COMPILE  ********************************************************************"""                
                                if loss == 'cox':  #,clipnorm=0.1
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning), loss=lss.__cox_loss(),metrics=[lss.tf_cindexR]) #cindex_score
                                    metrics = 'tf_cindex'
                                elif loss == 'RankAndmse':
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning), loss=lss.__RankAndMse_loss(),metrics=[uf.tf_cindexT,lss.rank,lss.mse]) #cindex_score
                                    metrics= 'rankAndMSE'
                                elif loss== 'RankAndcox':
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning),loss=lss.__RankAndCox_loss(),metrics=[lss.tf_cindexR,lss.rank,lss.cox]) #cindex_score
                                    metrics= 'coxAndrank'
                                elif loss == 'coxAndclassif': 
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning),loss=[lss.__cox_loss(),'categorical_crossentropy'],metrics=[lss.tf_cindexR,'accuracy'],loss_weights = [1.0,1.5]) #cindex_score
                                    metrics = 'coxAndclassif'
                                elif loss == 'discret':
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning), loss=ns.surv_likelihood(n_intervals),metrics=[uf.tf_cindexTD]) #cindex_score
                                    metrics = 'tf_cindexTD'
                                
                                elif loss == 'RankAnddiscret':
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning),loss=ns.RankingAndDiscret(n_intervals), metrics=[uf.tf_cindexTD,ns.discretcox,ns.rankL]) #cindex_score
                                    metrics = 'RankAnddiscret'
                                elif loss== 'triplet':
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning), loss=batch_hard_triplet_loss(),metrics=[ lss.mse,uf.tf_cindexT]) #uf.tf_cindexT,  cindex_score
                                    metrics = 'triplet'                              
                                else:
                                    pred.compile(optimizer=adam(lr=lr_fintuning,decay=lr_decay_fintuning),
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
                                if loss == 'coxAndclassif':
                                    fitting = pred.fit([xtrain,mtrain], [ytrain,ytrain_grp], validation_data=([xval,mval], [yval,yval_grp]), batch_size=batch,epochs=epochF,   callbacks=[tensorboard]) #PlotLossesKeras(),
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
                                        
                                    predictiontrain=Newmodel.predict([xtrain,mtrain])      
                                    predictiontrainclass = np.argmax(predictiontrain[1],axis=1)
                                    predictiontrain=predictiontrain[0]                        
                                    predictionval=Newmodel.predict([xval,mval])
                                    predictionvalclass = np.argmax(predictionval[1],axis=1)
                                    predictionval=predictionval[0]
                                    
                                elif loss == 'classif' or loss == 'triplet':
                    
                                    mc = ModelCheckpoint(best_model_file, monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)#'val_tf_cindexT'               
                                    
                                    fitting = pred.fit([xtrain,mtrain], ytrain_grp, validation_data=([xval,mval], yval_grp), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#,mcAc
                                    
                                    if loss == 'triplet':
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
                                    # predictiontrain=Newmodel.predict([xtrain,mtrain])      
                                    # predictiontrainclass = np.argmax(predictiontrain,axis=1)
                                    # predictionval=Newmodel.predict([xval,mval])
                                    # predictionvalclass = np.argmax(predictionval,axis=1)
                                    
                                    # if spp :
                                    #     eval_val_best = saved_model.evaluate([xval,mval], yval_grp)
                                    #     eval_train_best = saved_model.evaluate([xtrain,mtrain], ytrain_grp)
                                    #     eval_test_best= saved_model.evaluate([xtest,mtest], ytest_grp)
                                    # else: 
                                    #     eval_val_best = saved_model.evaluate(xval, yval_grp)
                                    #     eval_train_best = saved_model.evaluate(xtrain, ytrain_grp)
                                    #     eval_test_best= saved_model.evaluate(xtest, ytest_grp)
                                    
                                    if spp:
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
                                    
                                
                                elif loss == 'discret':
                                    if os.path.exists(best_model_file) :                                    
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                                    else:
                                        mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                        
                                        fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                                
                                elif loss == 'RankAnddiscret':
                                    if os.path.exists(best_model_file) :
                                            saved_model = load_model(best_model_file,
                                            custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                            'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                            'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                            'InstanceNormalization':InstanceNormalization})    
                                    else:
                                        
                                        mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                        
                                        fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF,callbacks=[mc,tensorboard] )#
                                        saved_model = load_model(best_model_file,
                                                custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                                'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                                'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                                'InstanceNormalization':InstanceNormalization})
                
                                elif loss == 'RankAndmse':      
                                    best_model_file=best_model_file
                                    if os.path.exists(best_model_file) :
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
        
                                    else:
                                        mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)
                                        fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
            
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
                                                
                                elif loss == 'RankAndcox': 
                                    best_model_file=best_model_file
                                    if os.path.exists(best_model_file) :
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                                    else:
                                        mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                        fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                                
                                else:
                                    best_model_file=best_model_file
                                    if os.path.exists(best_model_file) :
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'tf_cindexR':lss.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})
                                    else:
                                        mc = ModelCheckpoint( best_model_file, monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                        fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#,hp.KerasCallback(log_dir,hparams)    
                                        saved_model = load_model(best_model_file,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':lss.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})

                                if loss!= 'classif' and loss != 'triplet':      
                                    if spp:
                                        eval_val_best = saved_model.evaluate([xval,mval], yval)
                                        eval_train_best = saved_model.evaluate([xtrain,mtrain], ytrain)
                                        
                                        if p.use_test_set:  
                                            eval_test_best= saved_model.evaluate([xtest,mtest], ytest)
                                    else: 
                                        eval_val_best = saved_model.evaluate(xval, yval)
                                        eval_train_best = saved_model.evaluate(xtrain, ytrain)
                                    if spp:
                                        predictionval = saved_model.predict([xval,mval])
                                        predictiontrain = saved_model.predict([xtrain,mtrain])
                                        
                                        if p.use_test_set:
                                            predictiontest = saved_model.predict([xtest,mtest])
                                        
                                    else: 
                                        predictionval = saved_model.predict(xval)
                                        predictiontrain = saved_model.predict(xtrain)
                                        
                                        if p.use_test_set:
                                            predictiontest = saved_model.predict(xtest) 
                                                
                                    if len(np.shape(predictionval))==2:
                                        if loss !='discret'and loss!='RankAnddiscret':
                                            predictionval = predictionval[:,0]
                                            predictiontrainExclude = predictiontrain[:,0]
                                            
                                    if loss =='discret'or loss=='RankAnddiscret':    
                                        if p.use_test_set:                      
                                            predictiontest = uf.FromLongToTime(predictiontest,n_intervals,None)
                                        predictionval = uf.FromLongToTime(np.array(predictionval),n_intervals,None)
                                        predictiontrain = uf.FromLongToTime(predictiontrain,n_intervals,None)

                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret': 
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
                                                                    
                                    if loss =='discret'or loss=='RankAnddiscret':  
                                        ytrain = ytrainT
                                        yval = yvalT
                                        
                                    '''Evaluate with kaplan'''
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        bestPval,testStat,bestsep,numG0,numG1  = findBestSep(ytrain,predictiontrain)
                                        try:
                                            PvalueTrain,bestsep,numG0_train,numG1_train =groupeByBestSep(ytrain,predictiontrain,bestsep,file_name,titleP,'train','time')
                                            print('kaplan worked')

                                        except :
                                            PvalueTrain =bestsep=numG0_train=numG1_train = -100
                                            print('kaplan didn t work')
                                            pass
                                        try:
                                            PvalueVal,bestsep,numG0_val,numG1_val =groupeByBestSep(yval,predictionval,bestsep,file_name,titleP,'val','time')
                                            print('kaplan worked')

                                        except :
                                            PvalueVal = bestsep = numG0_val = numG1_val = -100
                                            print('kaplan didn t work')
                                            pass
                                    else:       
                                        bestPval,testStat,bestsep,numG0,numG1  = findBestSep(ytrain,predictiontrain)
                                        try:
                                            PvalueTrain,bestsep,numG0_train,numG1_train =groupeByBestSep(ytrain,predictiontrain,bestsep,file_name,titleP,'train')
                                            print('kaplan worked')

                                        except :
                                            PvalueTrain =bestsep=numG0_train=numG1_train = -100
                                            print('kaplan didn t work')
                                            pass
                                        try:
                                            PvalueVal,bestsep,numG0_val,numG1_val =groupeByBestSep(yval,predictionval,bestsep,file_name,titleP,'val')
                                            print('kaplan worked')

                                        except :
                                            PvalueVal = bestsep = numG0_val = numG1_val = -100
                                            print('kaplan didn t work')
                                            pass
                                    
                                    
                                    if loss =='discret'or loss=='RankAnddiscret':    
                                        ax=predictionShow('val',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                        if p.use_test_set:
                                            ax=predictionShow('test',ytestT,predictiontest,writer)
                                    else:
                                        ax=predictionShow('val',yval,predictionval,writer)
                                        if p.use_test_set:
                                            ax=predictionShow('test',ytest,predictiontest,writer)
                            
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
                                    if loss == 'classif':
                                        yval_grp = dpr.YProcessing2(yval,nb_classes)
                                    elif loss == 'triplet':
                                        yval_grp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(dpr.YProcessing2(yval,nb_classes),axis=1)]),axis=0))
                                    else:
                                        yval_grp = uf.YProcessing4(yval,'YEAR')
                                        yval_grpwc = uf.YProcessing4(yval,'YEARWC')
                                        yval_grp3 = uf.YProcessing4(yval,'THREE')
                                        yval_grpwc3 = uf.YProcessing4(yval,'THREEWC')
                                    
                                    if loss =='discret'or loss=='RankAnddiscret':    
                                        ax=predictionShow('valgrp',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                        if p.use_test_set:
                                            ax=predictionShow('test',ytestT,predictiontest,writer)
                                        ax=predictionShow('train',ytrainT,predictiontrain,writer)
                                    else:
                                        ax=predictionShow('valgrp',yval,predictionval,writer)
                                        if p.use_test_set:
                                            ax=predictionShow('test',ytest,predictiontest,writer)
                                        ax=predictionShow('train',ytrain,predictiontrain,writer)
                            
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        #super Cindex avec censure
                                        cind_trainSuper = uf.cindexT(np.argmax(ytrain_grp,axis=1),predictiontrain,ytrain[:,0])
                                        if p.use_test_set:
                                            cindTestSuper = uf.cindexT(np.argmax(ytest_grp,axis=1),predictiontest,ytest[:,0])                                                                     
                                        cind_valSuper = uf.cindexT(np.argmax(yval_grp,axis=1),np.reshape(np.array(predictionval),(-1,1)),yval[:,0]) 
                                        
                                    else:
                                    
                                        #super Cindex avec censure
                                        cind_trainSuper = uf.cindexR(np.argmax(ytrain_grp,axis=1),predictiontrain,ytrain[:,0])
                                        if p.use_test_set:
                                            cindTestSuper = uf.cindexR(np.argmax(ytest_grp,axis=1),predictiontest,ytest[:,0])                                                                     
                                        cind_valSuper = uf.cindexR(np.argmax(yval_grp,axis=1),np.array(predictionval),yval[:,0]) 
                                        
                
                                    '''Evaluate with kaplan grp'''
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        try:
                                            PvalueValgrp,bestsep,numG0_valgrp,numG1_valgrp =groupeByBestSep(yval,predictionval,bestsep,file_name,titleP,'valgrp','time')
                                            print('kaplan worked')
                                        except :
                                            PvalueValgrp = numG0_valgrp = numG1_valgrp = -100
                                            print('kaplan didn t work')
                                            pass
                                    else:      
                                        try:
                                            PvalueValgrp,bestsep,numG0_valgrp,numG1_valgrp =groupeByBestSep(yval,predictionval,bestsep,file_name,titleP,'valgrp')
                                            print('kaplan worked')
                                        except :
                                            PvalueValgrp = numG0_valgrp = numG1_valgrp = -100
                                            print('kaplan didn t work')
                                            pass
                                    '''Evaluate with kaplan Exclude grp'''
                                    
                                    # for prediction train remove patients where prediction to far
                                                            
                                    predictiontrainExclude=np.array(predictiontrain,dtype='float64')
                                    ytrainExclude=np.array(ytrain,dtype='float64')
                                    if len(np.shape(predictiontrainExclude))==2:
                                        df2train = pd.DataFrame(np.array([ytrainExclude[:,0],ytrainExclude[:,1],predictiontrainExclude[:,0]]).T,index=rtrain)
                                    else:
                                        df2train = pd.DataFrame(np.array([ytrainExclude[:,0],ytrainExclude[:,1],predictiontrainExclude]).T,index=rtrain)
                                    df2train.columns=['event','time','pred']     
                                    groupertrain = df2train.groupby([df2train.index]).mean()
                                    groupertrainstd = df2train.groupby([df2train.index]).std()
                                    predictiontrainExcludestd = groupertrainstd['pred']
                                    
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        limm = 400
                                    else:
                                        limm=2
                                    groupertrainExclude = groupertrain[predictiontrainExcludestd<limm]
                                    predictiontrainExclude = groupertrain['pred']
                                    ytrainExclude=[groupertrain['event'],groupertrain['time']]
                                    ytrainExclude=np.array(ytrainExclude).T
                                    ytrainExcludeL=  uf.FromTimeToLong(ytrainExclude,breaks)
                                    ytrainExcludeT =  ytrainExclude

                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        bestPvalExclude,testStatExclude,bestsepExclude,numG0Exclude,numG1Exclude  = findBestSep(ytrainExclude,predictiontrainExclude)
                                        
                                        try:
                                            PvalueTrainExclude,bestsepExclude,numG0_trainExclude,numG1_trainExclude =groupeByBestSep(ytrain,predictiontrain,bestsepExclude,file_name,titleP,'trainExclude','time')
                                            print('kaplan worked')
                                        except :
                                            PvalueTrainExclude=bestsepExclude=numG0_trainExclude=numG1_trainExclude = -100
                                            print('kaplan didn t work')
                                            pass
                                        try:
                                            PvalueValgrpExclude,bestsepExclude,numG0_valgrpExclude,numG1_valgrpExclude =groupeByBestSep(yval,predictionval,bestsepExclude,file_name,titleP,'valgrpExclude','time')
                                            print('kaplan worked')
                                        except :
                                            PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                            print('kaplan didn t work')
                                            pass
                                    else:       
                                        bestPvalExclude,testStatExclude,bestsepExclude,numG0Exclude,numG1Exclude  = findBestSep(ytrainExclude,predictiontrainExclude)
                                        try:
                                            PvalueTrainExclude,bestsepExclude,numG0_trainExclude,numG1_trainExclude =groupeByBestSep(ytrain,predictiontrain,bestsepExclude,file_name,titleP,'trainExclude')
                                            print('kaplan worked')
                                        except :
                                            PvalueTrainExclude=bestsepExclude=numG0_trainExclude=numG1_trainExclude = -100
                                            print('kaplan didn t work')
                                            pass
                                        try:
                                            PvalueValgrpExclude,bestsepExclude,numG0_valgrpExclude,numG1_valgrpExclude =groupeByBestSep(yval,predictionval,bestsepExclude,file_name,titleP,'valgrpExclude')
                                            print('kaplan worked')
                                        except :
                                            PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                            print('kaplan didn t work')
                                            pass
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        cindex_val1 = uf.cindexT(yval[:,1],np.array(predictionval),yval[:,0])
                                    else:
                                        cindex_val1 = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0])
                            
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        ppr = np.array([ppred / 365 for ppred in predictiontrain], dtype=np.int)
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
                                                tf.convert_to_tensor(['bestsep', str(bestsep),str(bestsep)]),
                                                tf.convert_to_tensor(['tf_cindexR', str(eval_train_best[1]),str(cindex_val1)])]
                                    s = sess.run(tf.summary.text('EvalBest', tf.stack(evalu)))
                                    writer.add_summary(s)
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,                        
                                            tf.convert_to_tensor(['Pvalgrp', str(PvalueTrain),str(PvalueValgrp)]),
                                                tf.convert_to_tensor(['bestsep', str(bestsep),str(bestsep)]),
                                                tf.convert_to_tensor(['numG0grp', str(numG0_train),str(numG0_valgrp)]),
                                                tf.convert_to_tensor(['numG1grp', str(numG1_train),str(numG1_valgrp)]),
                                            tf.convert_to_tensor(['PvalgrpExclude', str(PvalueTrainExclude),str(PvalueValgrpExclude)]),
                                                tf.convert_to_tensor(['bestsep', str(bestsepExclude),str(bestsepExclude)]),
                                                tf.convert_to_tensor(['numG0grpExclude', str(numG0_trainExclude),str(numG0_valgrpExclude)]),
                                                tf.convert_to_tensor(['numG1grpExclude', str(numG1_trainExclude),str(numG1_valgrpExclude)])]
                                    s = sess.run(tf.summary.text('EvalBestkaplan', tf.stack(evalu)))
                                    writer.add_summary(s)
                                    
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                        evalu = [tf.convert_to_tensor(['', 'train','val','valgrp']) ,                        
                                                tf.convert_to_tensor(['accuracy', str(acc_train),str(acc_val),str(acc_valgrp)])]
                                        s = sess.run(tf.summary.text('accuracy', tf.stack(evalu)))
                                        writer.add_summary(s)                                
                                        
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
        
                                        [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe',order='T')
                                        [cindex_val0,cindex_val1,cindex_val2,cindex_val3,cindex_val4,cindex_val5,cindex_val6] = lss.time_cindex(yval[:,1],np.array(predictionval),yval[:,0],cla='classe',order='T')
                                    else:
                                        [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe',order='R')
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
                                    
                                    if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
            
                                        [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre',order='T')
                                        [cindex_val0,cindex_val1] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre',order='T')
                                    else:
                                        [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre',order='R')
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
                                        
                                    
                                curves = plt.imread(path_patients+'datasep.png')
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
        
                                if attention!='False' and CV==0:
                                    """ attention model, fee = weight of each filter for each patient"""
                                    layerCAW = utils.find_layer_idx(saved_model,'channel_attention_weight')
                                    caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerCAW].output)
                                    if spp:
                                        fee = caw.predict([xtrain,mtrain]) #shape (B,1,1,64)
                                        feetest = caw.predict([xval,mval]) #shape (B,1,1,64)
                                    else:
                                        fee = caw.predict(xtrain) #shape (B,1,1,64)
                                        feetest = caw.predict(xval) #shape (B,1,1,64)
                                    if D3 == 'True':
                                        fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[4]))
                                    else:
                                        fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[3]))
                                        
                                    if D3  == 'True':
                                        feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[4]))
                                    else:
                                        feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[3]))
                                        
                                    
                                    if D3 == 'False':
                                        if attention != 'c':
                                            layerSAW = utils.find_layer_idx(saved_model,'spatial_attention_weight')
                                            saw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerSAW].output)
                                            if spp:
                                                fee2 = saw.predict([xtrain,mtrain])
                                            else:
                                                
                                                fee2 = saw.predict(xtrain)#shape (B,9,9,1)
                                            if D3  == 'True':                         
                                                fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2],np.shape(fee2)[3]))
                                            else: 
                                                fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2]))

                                            
                                            if spp:
                                                fee2val = saw.predict([xval,mval])
                                            else:
                                                fee2val = saw.predict(xval)
                                            if D3  == 'True':                         
                                                fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2],np.shape(fee2val)[3]))
                                            else: 
                                                fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2]))                                           
                                            
                                            if p.use_test_set:
                                                if spp:
                                                    fee2test = saw.predict([xtest,mtest])
                                                else:
                                                    fee2test = saw.predict(xtest)
                                                if D3  == 'True':                         
                                                    fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2],np.shape(fee2test)[3]))
                                                else: 
                                                    fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2]))

                                    caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerCAW].output)
                                    if spp:
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
                                
                                    if D3 == 'False':
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
                                    if D3 == 'False':
                                    
                                        cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[2],1)))]
                                    else:
                                        cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[3],1)))]
        
                                    s = sess.run(tf.summary.image('Sum_channel_attention_weight_train_over_patients', tf.stack(cawt)))
                                    writer.add_summary(s)
                                                        
                                    if D3 == 'False':
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
                                    
                                    
                                    if D3 == 'False':
                                        ##### Features map
                                        if attention == 'cs':
                                            layerFeat = utils.find_layer_idx(saved_model,'res')
                                        else:
                                            layerFeat = utils.find_layer_idx(saved_model,'multiply_1')
                                        caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerFeat].output)
                                        if spp :
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
                                    if loss != 'discret' and loss!='RankAnddiscret':
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
                                                
                                            cind_train = uf.cindexR(ytrain[:,1],predictiontrain,ytrain[:,0])
                                            
                                            if p.use_test_set:
                                                cindTest = uf.cindexR(ytest[:,1],predictiontest,ytest[:,0])
                                            print(np.shape(predictionval),np.shape(yval))
                                            cind_val = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0]) 
                                            
                                            [cindex_train1,cindex_train2,cindex_train3] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0])
                                            
                                            if p.use_test_set:
                                                [cindex_test1,cindex_test2,cindex_test3] = lss.time_cindex(ytest[:,1],predictiontest,ytest[:,0])
                                            [cindex_val1,cindex_val2,cindex_val3] = lss.time_cindex(yval[:,1],predictionval,yval[:,0])
                                        
                                            [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe')
                                            [cindex0,cindex1] = lss.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre')
                                            
                                            [cindex_val0,cindex_val1,cindex_val2,cindex_val3,cindex_val4,cindex_val5,cindex_val6] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='classe')
                                            [cindex_val0,cindex_val1] = lss.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre')


                                    else:
                                        cind_train = uf.cindexT(ytrainT[:,1],predictiontrain,ytrainT[:,0])
                                        
                                        if p.use_test_set:   
                                            cindTest = uf.cindexT(ytestT[:,1],predictiontest,ytestT[:,0])
                                        cind_val = uf.cindexT(yvalT[:,1],np.array(predictionval),yvalT[:,0]) 
                                        
                                        # [cindex_train1,cindex_train2,cindex_train3] = lss.time_cindex(uf.FromLongToTime(ytrain,n_intervals,None)[0],predictiontrain,uf.FromLongToTime(ytrain,n_intervals,None)[1],'T')
                                        # [cindex_test1,cindex_test2,cindex_test3] = lss.time_cindex(uf.FromLongToTime(ytest,n_intervals,None)[0],predictiontest,uf.FromLongToTime(ytest,n_intervals,None)[1],'T')
                                        # [cindex_val1,cindex_val2,cindex_val3] = lss.time_cindex(uf.FromLongToTime(yval,n_intervals,None)[0],predictionval,uf.FromLongToTime(yval,n_intervals,None)[1],'T')
                                    
                                # if loss != 'classif' and loss != 'triplet'  and loss != 'discret':
                                #     pathSavepredTr = os.path.join(file_name, f"{titleP}_ped_train")
                                #     np.save(pathSavepredTr,predictiontrain)
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
                                # if attention !='False':
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
                                
                                # if loss == 'coxAndclassif' or loss == 'classif' :
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
                                #results = dpr.saving(fitting,titleP,eval_train, eval_val,None,file_name,metrics)
                                
                                if loss=='classif' or loss =='coxAndclassif':
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
                                    # if loss != 'classif' :
                                    cindex_train.append(cind_train)
                                    # else:
                                        # cindex_train.append(0)
                                    # loss_valid.append(eval_val[0])
                                    # if loss != 'classif':
                                    cindex_val.append(cind_val)
                                    # else:
                                    #     cindex_val.append(0)
                                    if loss != 'classif' and loss != 'coxAndclassif' and loss!='triplet':
                                        AcurracyTrain.append(0)
                                        AcurracyVal.append(0)
                                    # elif metrics=='coxAndclassif':
                                        # AcurracyTrain.append(eval_train[6])
                                        # AcurracyVal.append(eval_val[6])
                                        
                                    else:
                                        AcurracyTrain.append(resu2)
                                        AcurracyVal.append(resu3)
                                    if loss == 'cox' or loss == 'discret' :   
                                        train_best.append(eval_train_best[1])
                                        val_best.append(cindex_val1)
                                    else:
                                        train_best.append(eval_train_best[2])
                                        val_best.append(cindex_val1)
                                        
                                
                        """ FIN K loop """
                    if p.cross_validation:
                        if not p.pretrain :
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