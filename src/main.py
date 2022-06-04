# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

"""
""" ########################### OPTIONS ###########################################"""
import os

reload = True
CrossVal = True # if True do cross validation, if False just train/validation/test
ratess = [0.25,0.25,0.25,0.25] #[0.2,0.2,0.2,0.2,0.2] # distribution in the folders
TEST  = False #if True have a test set
Maxi= False
ratio = 50 #ratio for binary pretraining
stride = 365 #number of days for discretisation
interpolSize = 36 #size of the input images

#Folder names
tt= 'avecspatial/'#name of the training folder 
dossier = './Results/ModeFinalRunVal/' # name of the experience folder 
os.makedirs(dossier, exist_ok=True)  # create the folder if it does not exist yet
my_path = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
PATH = my_path + "/data/ALL_DATA/Patients/"
reimpW = False
doss = './Results/PATIENTS_SEP_TO_KEEP_154CVTsansTest/'
os.makedirs(doss, exist_ok=True) 
PRETRAIN = False #if True run pretraining

"""############################  PACKAGES ########################################"""
import functions.losses.survivalTripletloss as striplet
from sklearn.utils import shuffle
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from functions.losses.triplet_loss import batch_hard_triplet_loss
import pandas as pd
from vis.utils import utils
from keras import regularizers
from keras.utils import to_categorical
from sklearn.model_selection import ParameterGrid
from keras_contrib.layers.normalization.instancenormalization  import InstanceNormalization
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense,LeakyReLU, Conv2D, Conv3D,Input, Flatten, MaxPool2D ,MaxPool3D, BatchNormalization#,GlobalAveragePooling2D
from keras.optimizers import adam
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import time
import functions.architecture.models as mds
import functions.data_processing.data_processing as dpr
import functions.losses.losses as lss
import functions.data_processing.pretraining as ptg
import functions.data_processing.data_prep



'''########################### CODE ############################################# '''

if "sess" in locals():     # this code prevent creating many intance of
    tf.compat.v1.reset_default_graph()
    sess.close()           # tensorflow in memory (that can cause crashes)
K.clear_session()
sess = tf.Session() #create new tensorflow session



""" ******************** Data augmentation definition ********************"""
pathda = 'dataAug.npy'
if os.path.exists(pathda):
    da=np.load(pathda,allow_pickle=True)
else:
    da = dpr.createDataAug()
    np.save(pathda,da)
""" ********************SURVIVAL MM DATA *********************************"""

liste_image, liste_mask,liste_ref = dpr.image_mask_ref(PATH)
linList = ['all'] #take all the discretised time classes
for lin in linList:    
    print('pretrain2 : ',PRETRAIN) 
    print('************* lin {} *********************'.format(lin))
    print('***************************************')
    testornot = ''
    if TEST ==False:
        testornot = 'sansTest'
    if lin != 'all':
        doss = './Results/PATIENTS_SEP_TO_KEEP_154CVT{}{}/'.format(lin,testornot)
    else:
        doss = './Results/PATIENTS_SEP_TO_KEEP_154CVT{}/'.format(testornot)
    os.makedirs(doss, exist_ok=True) 
    cl = lin #'all'
    # """ ******************** DATA EVALUATION *********************************"""
    # if CrossVal != True:
    #     def stat(y,name):
    #         mean= np.mean(y[:,1])
    #         med = np.median(y[:,1])
    #         tauxCensor = (1 - np.mean(y[:,0]))*100
    #         print("{}:    Censorship {}%, mean time {}, median time {}".format(name,tauxCensor,mean,med))
    #     stat(ytrainsurv,'train')
    #     stat(yvalsurv,'val')
    #     stat(ytestsurv,'test')
    #     print(len(ytrainsurv),len(ytestsurv))
    # else:
    #     def stat(y,name):
    #         mean= np.mean(y[:,1])
    #         med = np.median(y[:,1])
    #         tauxCensor = 100- (np.mean(y[:,0])*100)
    #         print("{}:    Censorship {}%, mean time {}, median time {}".format(name,tauxCensor,mean,med))
    #     for k in range(len(RR)):
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
        
    #     if len(ratess) >4 :
    #         kmf = KaplanMeierFitter()
    #         kmf.fit(Ysurv[4][:,1], Ysurv[4][:,0], label='4')
    #         ax = kmf.plot(ax=ax)
    #     ax.get_figure().savefig(doss+'datasep.png')
    #     plt.savefig(doss+'datasep.png')
    #     plt.close()

    print('pretrain : ',PRETRAIN)
    if PRETRAIN == True:
        """ ******************** BINARY DATA *********************************"""
        if CrossVal== False:
            xbitrain2D, ybitrain2D, xbival2D,ybival2D = ptg.binary_extraction(RR,CrossVal,PATH,sess,TEST=False, interpolSize)
        else:
            Xb2D, Yb2D, Xb3D, Yb3D = ptg.binary_extraction(RR,CrossVal,PATH,sess,TEST=False, interpolSize)
    
    
    breaks=np.arange(0.,365*7,stride)
    if PRETRAIN==True:
        parameters = {'Maxi':[False], 'SPP':['False'], 'D3':[True],
                          'Loss':['cox'],'pretrain':['binary'] ,  #,'textureP','textureF', 'binaryP','binaryF'
                          'LR Pretrain':[1e-04], 'LR Finetune': [1e-04],  'rate' : [0.83] , 
                          'Lrate_decay_learningP' :[1e-07],
                          'lrate_decay_learningF' :[1e-8],'epoch Train' : [20] , 
                          'epoch Finetune': [40] , 'batch size': [10], 'image size':[36],
                          'noTrainLayer': [0] ,'npatch':[3],'dataAug':[15],'attention':['s'],'nbClasses':[7]}
    else:
        parameters = {'Maxi':[False], 'SPP':['False'], 'D3':[True],
                          'Loss':['cox'],'pretrain':['binary'] ,  #,'textureP','textureF', 'binaryP','binaryF'
                          'LR Pretrain':[1e-04], 'LR Finetune': [1e-04],  'rate' : [0.83,0.17,0.5] , 
                          'Lrate_decay_learningP' :[1e-07],
                          'lrate_decay_learningF' :[1e-8,1e-10,1e-12],'epoch Train' : [20] , 
                          'epoch Finetune': [40] , 'batch size': [10], 'image size':[36],
                          'noTrainLayer': [0] ,'npatch':[3],'dataAug':[15,30],'attention':['s'],'nbClasses':[7]}
    gh=0
    l2reg=0.001
    n_intervals=len(breaks)    
    finalR = []
    pathCSV_finDF = dossier +'finalEval_' + tt + '.csv'
    if os.path.exists(pathCSV_finDF):
        finalR = pd.read_csv(pathCSV_finDF, encoding='utf-8')
        finalR = finalR.iloc[:,1:].values.tolist()
    else:
        finalR = []
    binary=[]
    bilinear=False
    count =0
    grid = ParameterGrid(parameters)

    for params in grid:  
        gh=gh+1
        print('***************************************')
        print('****** run {} on {} *******************'.format(gh,len(grid)))
        print('***************************************')
        print('***************************************')
        nbClasses,attention,dataAug,npatch,D3,loss,pretrain, spp ,Maxi , lratePre , lrateFine, epochF,interpolSize,batch,epochT ,LRDecayF,LRDecayP, rate,noTrainLayer = params['nbClasses'],params['attention'],params['dataAug'],params['npatch'],params['D3'],params['Loss'],params['pretrain'],params['SPP'], params['Maxi'],params['LR Pretrain'],params['LR Finetune'], params['epoch Finetune'],params['image size'],params['batch size'],params['epoch Train'],params['lrate_decay_learningF'],params['Lrate_decay_learningP'], params['rate'], params['noTrainLayer']
    
        p = { 'attention':[attention],'dataAug':[dataAug],'npatch': [npatch],'D3': [D3],'loss' : [loss], 'nbClasses':[nbClasses],'pretrain': [pretrain],'spp': [spp], 'Maxi':[Maxi], 'LR Pretrain':[lratePre],'LR Finetune': [lrateFine],  'epoch F' : [epochF], 'epoch T' : [epochT],'interpolsize' : [interpolSize] , 'batch': [batch],'Lrate_decay_learningP': [LRDecayP],'lrate_decay_learningF': [LRDecayF],'rate': [rate],'noTrainLayer':[noTrainLayer]}
        param = pd.DataFrame.from_dict(p)
        print(p)              
        aaa=[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
        do = True        
            
        if lrateFine > LRDecayF and do ==True : 
            lrd=str(LRDecayF)[-1]
            if loss == 'classif':
                metrics = 'accuracy'
            else: 
                metrics = uf.tf_cindexR
            if PRETRAIN == False:
                if CrossVal != True:
                    if loss=='discret' or loss == 'RankAnddiscret':
                        xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest, ytrainT,yvalT,ytestT, rtrain, rtest,rval =extract_dataIm(CrossVal,liste_ref,doss,PATH,sess,cl,ratess,interpolSize,RR,reload,spp,npatch,D3,loss)
                    else :
                        xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest , rtrain, rtest,rval =extract_dataIm(CrossVal,liste_ref,doss,PATH,sess,cl,ratess,interpolSize,RR,reload,spp,npatch,D3,loss)
                else: 
                    Xt,Yt,Mt,Rt, RR =extract_dataIm(CrossVal,liste_ref,doss,PATH,sess,cl,ratess,interpolSize,RR,reload,spp,npatch,D3,loss)
                
                LossTrain=[]
                LossVal=[]
                AcurracyTrain=[]
                AcurracyVal =[]
                CindexTrain=[]
                CindexVal =[]
                train_best=[]
                val_best=[]
                
            if CrossVal != True :
                CVNum = 1
            else:
                CVNum = len(RR)
            if TEST == True:
                CVnum = 1
            timess = '/' +tt 
            if CrossVal != True:
                intermDoss = dossier + timess+'TrainTest/'
                os.makedirs( intermDoss , exist_ok=True)
                
            for CV in range(0,CVNum):
                print(do)
                if CrossVal == True:
                    intermDoss = dossier + timess+'CV{}/'.format(CV)
                    os.makedirs(intermDoss, exist_ok=True)
                    
                    pathCSV_fullDF = intermDoss +'FullCV_{}'.format(CV) + '.csv'
                    if os.path.exists(pathCSV_fullDF):
                        fullcv = pd.read_csv(pathCSV_fullDF, encoding='utf-8')
                        fullcv = fullcv.iloc[:,1:].values.tolist()
                    else:
                        fullcv = []
                    
                if CrossVal == True:
                    if PRETRAIN== False:
                        if len(ratess) >4 :
                            if TEST != False:
                                xtest,ytest,mtest,rtest=Xt[CV],Yt[CV],Mt[CV],Rt[CV]
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
                            if TEST != False:
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
                        if TEST != False:
                            if D3 == True:
                                Xb=[Xb3D[1],Xb3D[2],Xb3D[3],Xb3D[4]]
                                Yb=[Yb3D[1],Yb3D[2],Yb3D[3],Yb3D[4]]
                                # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                            else:
                                Xb=[Xb2D[1],Xb2D[2],Xb2D[3],Xb2D[4]]
                                Yb=[Yb2D[1],Yb2D[2],Yb2D[3],Yb2D[4]]
                        else:
                            if D3 == True:
                                Xb=[Xb3D[0],Xb3D[1],Xb3D[2],Xb3D[3]]
                                Yb=[Yb3D[0],Yb3D[1],Yb3D[2],Yb3D[3]]
                                # Xb, Yb= np.delete(Xb3D,CV), np.delete(Yb3D,CV)
                            else:
                                Xb=[Xb2D[0],Xb2D[1],Xb2D[2],Xb2D[3]]
                                Yb=[Yb2D[0],Yb2D[1],Yb2D[2],Yb2D[3]]
                if CrossVal != True:
                    KNum = 1
                else:
                    KNum = len(RR)-1
                if TEST == False:
                    KNum = len(RR)

                for k in range(KNum):
                 
                    aaa=['CV{}_K{}'.format(CV,k)]+[str(param.iloc[0].values[i]) for i in range(len(param.iloc[0].values))]
                    for j in range(len(fullcv)):
                        fcv = [fullcv[j][0]]   + fullcv[j][-19:]  
                        fcv = [str(fcv[i]) for i in range(len(fcv))]
                        if np.array_equal(aaa,fcv):
                            do=False
                    print("do 1",do)
                    na = str(gh)+'_'+str(CV) +'_'+str(k) 
                    titleP = '{}lrP{}_lrF{}b{}_eF{}_lrd{}_D{}_r{}'.format(na,lratePre , lrateFine, batch,epochF,lrd,dataAug,rate)
                    
                    if loss == 'classif' or loss =='triplet':
                        lossAtrib=loss + str(nbClasses)
                    elif loss == 'discret':
                        lossAtrib = 'disc' + str(n_intervals)
                    else:
                        lossAtrib = loss
                    print(lossAtrib,pretrain)
                    lossAtrib = lossAtrib+'_Pre_'+pretrain+ '_'
                    file = intermDoss+ '{}at{}_spp{}_maxi{}_D3{}_n{}_dim{}'.format(lossAtrib,attention,spp,Maxi, D3,noTrainLayer,npatch)
                    we=file+'/' + titleP +'best_model.h5'
                    do=False 
                    if os.path.exists(we) or os.path.exists(file+'/' + titleP +'best_model.h5') :
                        do = True 
                        print('weight exist')
                    if do == True:        
                        print('************* CV {} K {}*********************'.format(CV,k))
                        print('***************************************')
    
                        if "sess" in locals():     # this code prevent creating many intance of
                            tf.reset_default_graph()
                            sess.close()           # tensorflow in memory (that can cause crashes)
                        K.clear_session()
                        
                        sess = tf.Session()
                        print('****** run {} on {} *******************'.format(gh,len(grid)))
                        print('***************************************') 
                        print('****** lin {} *******************'.format(lin))
                        pathCSV_fullDF = intermDoss +'FullCV_{}'.format(CV)  + '.csv'
                        if os.path.exists(pathCSV_fullDF):
                            fullcv = pd.read_csv(pathCSV_fullDF, encoding='utf-8')
                            fullcv = fullcv.iloc[:,1:].values.tolist()
                        else:
                            fullcv = []
        
                        na = str(gh)+'_'+str(CV) +'_'+str(k) 
                        titleP = '{}lrP{}_lrF{}b{}_eF{}_lrd{}_D{}_r{}'.format(na,lratePre , lrateFine, batch,epochF,lrd,dataAug,rate)
                        
                        if loss == 'classif' or loss =='triplet':
                            lossAtrib=loss + str(nbClasses)
                        elif loss == 'discret':
                            lossAtrib = 'disc' + str(n_intervals)
                        else:
                            lossAtrib = loss
                        lossAtrib = lossAtrib+'_Pre_'+pretrain+ '_'
                        file = intermDoss+ '{}at{}_spp{}_maxi{}_D3{}_n{}_dim{}'.format(lossAtrib,attention,spp,Maxi, D3,noTrainLayer,npatch)
                        os.makedirs(file, exist_ok=True)
                        PATHEV= file + '/'  + titleP + 'parameters.csv'
                    
                        log_dir = file+'/' + "logs/fit/" +titleP +'/' #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        if CV== 0 or CV == 2:
                            tensorboard = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                        else:
                            tensorboard = TensorBoard(log_dir=log_dir) 
    
                        
                        """  DATA AUGMENTATION ********************************************************************"""
                        if PRETRAIN == False:
                            if  CrossVal == True:
                                xval,yval,mval,rval = X[k],Y[k], M[k], R[k]
                                if TEST != False:
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
                                    if len(ratess)>4:
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
                                if TEST !=False or len(ratess)==4:
                                    xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2]))
                                    mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2]))
                                    ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2]))
                                    rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2]))
                                else:
                                    xtrain=np.concatenate((xtrain[0],xtrain[1],xtrain[2],xtrain[3]))
                                    mtrain=np.concatenate((mtrain[0],mtrain[1],mtrain[2],mtrain[3]))
                                    ytrain=np.concatenate((ytrain[0],ytrain[1],ytrain[2],ytrain[3]))
                                    rtrain=np.concatenate((rtrain[0],rtrain[1],rtrain[2],rtrain[3]))
                                if dataAug!=0:
                                    if D3== False :#or D3=='25'
                                        xtrain,ytrain,mtrain,rtrain=uf.dataAugmentation(da,xtrain,ytrain,mtrain,rtrain,D3=D3,num =dataAug)
                                        xval,yval,mval,rval=uf.dataAugmentation(da,xval,yval,mval,rval, D3=D3,num =dataAug)
                                    else:
                                        xtrain,ytrain,mtrain,rtrain=uf.DataAugmentation3D(da,xtrain,ytrain,mtrain, rtrain, num =dataAug)
                                        xval,yval,mval,rval=uf.DataAugmentation3D(da,xval,yval,mval,rval,num =dataAug)
                                        if TEST != False:
                                            xtestA,ytestA,mtestA,rtestA=uf.DataAugmentation3D(da,xtest,ytest,mtest,rtest,num =dataAug)
    
                                    xval, yval, mval,rval = dpr.shuffle_and_interpolate(xval, yval,rval,interpolSize,interStatue = False,liste_mask =mval)
                                    xtrain, ytrain, mtrain,rtrain = dpr.shuffle_and_interpolate(xtrain, ytrain,rtrain, interpolSize,interStatue = False,liste_mask =mtrain)

                            else:
                                if dataAug!=0:
                                    if D3== False :#or D3=='25'
                                        xtrain,ytrain,mtrain=uf.dataAugmentation(da,xtrain,ytrain,mtrain,D3=D3,num=dataAug)
                                        xval,yval,mval=uf.dataAugmentation(da,xval,yval,mval,D3=D3,num =dataAug)
                                        #xtest,ytest,mtest=uf.dataAugmentation(da,xtest,ytest,mtest,D3=D3,num =dataAug)
                                    else:
                                        xtrain,ytrain,mtrain=uf.DataAugmentation3D(da,xtrain,ytrain,mtrain, num =dataAug)
                                        xval,yval,mval=uf.DataAugmentation3D(da,xval,yval,mval,num =dataAug)
                                        #xtest,ytest,mtest=uf.DataAugmentation3D(da,xtest,ytest,mtest,num=dataAug)
                        
                                    xval, yval, mval = dpr.shuffle_and_interpolate2(xval, yval,interpolSize,interStatue = False,liste_mask =mval)
                                    xtrain, ytrain, mtrain = dpr.shuffle_and_interpolate2(xtrain, ytrain,interpolSize,interStatue = False,liste_mask =mtrain)
                                    #xtest, ytest, mtest = dpr.shuffle_and_interpolate2(xtest, ytest,interpolSize,interStatue = False,liste_mask =mtest)
                            if TEST != False:
                                xtrain,xval,xtest = uf.normalize(xtrain),uf.normalize(xval),uf.normalize(xtest)
                            else:
                                xtrain,xval = uf.normalize(xtrain),uf.normalize(xval)
    
                            if D3=='25':
                                xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                                mtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
                                if TEST != False:
                                    xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                                    mtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2],xtest.shape[3]))
                                xval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))
                                mval = np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],xval.shape[3]))
    
                            if loss == 'classif':
                                ytraingrp = uf.YProcessing2(ytrain,nbClasses)
                                yvalgrp = uf.YProcessing2(yval,nbClasses)
                                if TEST != False:
                                    ytestgrp = uf.YProcessing2(ytest,nbClasses)
                            elif loss == 'triplet':
                                ytraingrp = np.transpose(np.concatenate(([ytrain[:,0]],[np.argmax(uf.YProcessing2(ytrain,nbClasses),axis=1)]),axis=0))
                                yvalgrp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(uf.YProcessing2(yval,nbClasses),axis=1)]),axis=0))
                                if TEST != False:
                                    ytestgrp =np.transpose(np.concatenate(([ytest[:,0]],[np.argmax(uf.YProcessing2(ytest,nbClasses),axis=1)]),axis=0))
                            else:
                                ytraingrp = uf.YProcessing4(ytrain,'YEAR')
                                yvalgrp = uf.YProcessing4(yval,'YEAR')
                                if TEST != False:
                                    ytestgrp = uf.YProcessing4(ytest,'YEAR')
                                    
                                ytraingrpwc = uf.YProcessing4(ytrain,'YEARWC')
                                yvalgrpwc = uf.YProcessing4(yval,'YEARWC')
                                if TEST != False:
                                    ytestgrpwc = uf.YProcessing4(ytest,'YEARWC')
                                    
                                ytraingrp3 = uf.YProcessing4(ytrain,'THREE')
                                yvalgrp3 = uf.YProcessing4(yval,'THREE')
                                if TEST != False:
                                    ytestgrp3 = uf.YProcessing4(ytest,'THREE')
                                    
                                ytraingrpwc3 = uf.YProcessing4(ytrain,'THREEWC')
                                yvalgrpwc3 = uf.YProcessing4(yval,'THREEWC')
                                if TEST != False:
                                    ytestgrpwc3 = uf.YProcessing4(ytest,'THREEWC')
                                
                            if loss=='discret' or loss == 'RankAnddiscret':
                                if TEST != False:
                                    ytrainT,yvalT,ytestT =  ytrain,yval,ytest
                                    ytrain,yval,ytest= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks),uf.FromTimeToLong(ytest,breaks)
                                else:
                                    ytrainT,yvalT =  ytrain,yval
                                    ytrain,yval= uf.FromTimeToLong(ytrain,breaks), uf.FromTimeToLong(yval,breaks)
                               
                            
                        """  model creation ********************************************************************"""    
                        
                        if pretrain == 'triplet':
                            paramTriLRDecayP = [1e-8,1e-8,1e-12,1e-8]

                            ww = ['2_0_0lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model',
                                  '1_0_1lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.17best_model',
                                  '7_0_2lrP0.0001_lrF0.0001b10_eF20_lrd2_D15_r0.17best_model',
                                  '2_0_3lrP0.0001_lrF0.0001b10_eF20_lrd8_D15_r0.83best_model']
                            paramTrirate = [0.83,0.17,0.17,0.83]

                        if spp == False:
                            conv1=False
                        else:
                            conv1=False
                        if bilinear == True:
                            modelCN = ConvSurvBefore(interpolSize,D3,spp=spp) 
                            modelCN2 = ConvSurvBefore(interpolSize,D3,spp=spp) 
                            pred =  BilinearModel(interpolSize,modelCN = modelCN,modelCN2 = modelCN2, mode = loss,D3=D3,spp=spp,attention=attention,num=nbClasses)
                        else: 
                            if pretrain == 'triplet':
                                # pred = deepConvSurv(interpolSize,mode = 'triplet',D3=D3,spp=spp,attention=attention,num=nbClasses,rate=paramTrirate,conv1=conv1,maxi = Maxi)  
                                pred = deepConvSurv(interpolSize,mode = 'triplet',D3=D3,spp=spp,attention=attention,num=nbClasses,rate=rate,conv1=conv1,maxi = Maxi,l11=l2reg)  

                            elif pretrain == 'binary':
                                pred = deepConvSurv(interpolSize,mode = 'classif',D3=D3,spp=spp,attention=attention,num=2,rate=rate,conv1=conv1,maxi = Maxi,l11=l2reg)  
                            else:
                                pred = deepConvSurv(interpolSize,mode = loss ,D3=D3,spp=spp,attention=attention,num=nbClasses,rate=rate,conv1=conv1,maxi = Maxi,l11=l2reg)  

                        ''' *****************************************'''
                        ''' **************PRETRAINING****************'''
                        ''' *****************************************'''                
                        if pretrain=='binary':
                            if PRETRAIN == True:
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
                            doss2 = "Binary3D{}_spp{}_att{}_maxi{}/".format(D3,spp,attention,Maxi)
                            os.makedirs(dossier + '/' + 'pretraining/' + doss2, exist_ok=True) 
                            we = dossier + '/' + 'pretraining/' + doss2 + "Weights{}_{}_ratio{}_drop{}_CV{}K{}.h5".format(lratePre,LRDecayP,ratio,0.83,CV,k)
                            if os.path.exists(we) :
                                print('load weights')
                                pred.load_weights(we)
                                fit = 0
                            else:
                                print('run pretraining .....')
    
                                log_dirP = dossier + '/' + 'pretraining/' + doss2 + "logs/fit/"+ "Weights{}_{}_ratio{}_CV{}K{}/".format(lratePre,LRDecayP,ratio,CV,k)
                                tensorboardP = TensorBoard(log_dir=log_dir,histogram_freq=1) 
                                
                                fitting = pred.fit([xbitrain,mbitrain], ybitrain, validation_data=([xbival,mbival], ybival), batch_size=batch,epochs=epochT, callbacks=[tensorboardP])#,mcAc
                                pred.save_weights(we)
                                print('pretraining finished')
    
                                fit = 1
                                print('--- evaluation')
                                PREeval_train = pred.evaluate([xbitrain,mbitrain], ybitrain)
                                print('eval train',PREeval_train)
                                PREeval_val = pred.evaluate([xbival,mbival], ybival)
                                print('eval val',PREeval_val)
                                results = saving(fitting,'pretrainBinary{}_{}_{}_{}_{}_{}_{}'.format(attention,D3, loss,spp,lratePre,LRDecayP,Maxi),PREeval_train, PREeval_val,None,dossier,'accuracy')
                
                                results = saving(fitting,'pretrainBinary{}_{}_{}_{}_{}_{}_{}'.format(attention,D3, loss,spp,lratePre,LRDecayP,Maxi),PREeval_train, PREeval_val,None,file,'accuracy')
         
                                binary.append([PREeval_train[1],PREeval_val[1],lratePre,LRDecayP])
                                np.savetxt(dossier + '/' + 'pretraining/' + "pretrainBinaryResults.csv",binary)
                        if pretrain == 'triplet':
                            doss2 = dossier + './Best/CV0/triplet7_Pre_False_atc_sppFalse_maxiFalse_D3True_n3/'

                            we = doss2 + '{}.h5'.format(ww[k])
                            pred.load_weights(we)
        
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
                            x1 = Dense(nbClasses, activation='softmax', name="fc4",kernel_regularizer=regularizers.l2(0.001))(lay)
    
                        pred = Model(inputs=pred.input, outputs=x1) 
                        if pretrain == 'triplet' and noTrainLayer !=0:
                            for layer in pred.layers[1].layers[:noTrainLayer]:
                                print(layer)
                                layer.trainable = False
                        
                        # print(pred.summary())
                        # for layer in pred.layers[:noTrainLayer]:
                        #     print(layer)
                        #     layer.trainable = False
                        
                        # we = file+'/' + titleP +'best_model.h5'
                        # if os.path.exists(we) :
                        #     print('load weights')
                        #     pred.load_weights(we)
                        #     fit = 0

                        if PRETRAIN == False:
                            """ COMPILE  ********************************************************************"""                
                            if loss == 'cox':  #,clipnorm=0.1
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF), loss=lss.__cox_loss(),metrics=[uf.tf_cindexR]) #cindex_score
                                metrics = 'tf_cindex'
                            elif loss == 'RankAndmse':
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF), loss=lss.__RankAndMse_loss(),metrics=[uf.tf_cindexT,lss.rank,lss.mse]) #cindex_score
                                metrics= 'rankAndMSE'
                            elif loss== 'RankAndcox':
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF),loss=lss.__RankAndCox_loss(),metrics=[uf.tf_cindexR,lss.rank,lss.cox]) #cindex_score
                                metrics= 'coxAndrank'
                            elif loss == 'coxAndclassif': 
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF),loss=[lss.__cox_loss(),'categorical_crossentropy'],metrics=[uf.tf_cindexR,'accuracy'],loss_weights = [1.0,1.5]) #cindex_score
                                metrics = 'coxAndclassif'
                            elif loss == 'discret':
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF), loss=ns.surv_likelihood(n_intervals),metrics=[uf.tf_cindexTD]) #cindex_score
                                metrics = 'tf_cindexTD'
                            
                            elif loss == 'RankAnddiscret':
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF),loss=ns.RankingAndDiscret(n_intervals), metrics=[uf.tf_cindexTD,ns.discretcox,ns.rankL]) #cindex_score
                                metrics = 'RankAnddiscret'
                            elif loss== 'triplet':
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF), loss=batch_hard_triplet_loss(),metrics=[ lss.mse,uf.tf_cindexT]) #uf.tf_cindexT,  cindex_score
                                metrics = 'triplet'                              
                            else:
                                pred.compile(optimizer=adam(lr=lrateFine,decay=LRDecayF),
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
                                fitting = pred.fit([xtrain,mtrain], [ytrain,ytraingrp], validation_data=([xval,mval], [yval,yvalgrp]), batch_size=batch,epochs=epochF,   callbacks=[tensorboard]) #PlotLossesKeras(),
                                eval_val = pred.evaluate([xval,mval], [yval,yvalgrp])
                                eval_train = pred.evaluate([xtrain,mtrain], [ytrain,ytraingrp])
                                if TEST != False:
                                    eval_test= pred.evaluate([xtest,mtest], [ytest,ytestgrp])
                                Newmodel = pred
                                if CrossVal != True:
                                    if TEST != False:
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
                
                                mc = ModelCheckpoint( file+'/' + titleP +'best_model.h5', monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)#'val_tf_cindexT'               
                                fitting = pred.fit([xtrain,mtrain], ytraingrp, validation_data=([xval,mval], yvalgrp), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#,mcAc
                                if loss == 'triplet':
                                    
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':batch_hard_triplet_loss(), 'InstanceNormalization':InstanceNormalization})
    
                                else:
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':'categorical_crossentropy', 'InstanceNormalization':InstanceNormalization})
                
                
                                eval_val = pred.evaluate([xval,mval], yvalgrp)
                                eval_train = pred.evaluate([xtrain,mtrain], ytraingrp)
                                if TEST != False:
                                    eval_test= pred.evaluate([xtest,mtest], ytestgrp)
                                # Newmodel = pred
                                # if CrossVal != True:
                                #     predictionBefore=Newmodel.predict([xtest,mtest]) 
                                #     predictionBeforeclass = np.argmax(predictionBefore,axis=1)
                                # predictiontrain=Newmodel.predict([xtrain,mtrain])      
                                # predictiontrainclass = np.argmax(predictiontrain,axis=1)
                                # predictionval=Newmodel.predict([xval,mval])
                                # predictionvalclass = np.argmax(predictionval,axis=1)
                                
                                # if spp == True:
                                #     eval_val_best = saved_model.evaluate([xval,mval], yvalgrp)
                                #     eval_train_best = saved_model.evaluate([xtrain,mtrain], ytraingrp)
                                #     eval_test_best= saved_model.evaluate([xtest,mtest], ytestgrp)
                                # else: 
                                #     eval_val_best = saved_model.evaluate(xval, yvalgrp)
                                #     eval_train_best = saved_model.evaluate(xtrain, ytraingrp)
                                #     eval_test_best= saved_model.evaluate(xtest, ytestgrp)
                                if spp != False:
                                    predictionvalclass = np.argmax(saved_model.predict([xval,mval]),axis=1)      
                
                                    predictiontrainclass = np.argmax(saved_model.predict([xtrain,mtrain]),axis=1)
                                    if TEST != False:
                                        predictiontestclass = np.argmax(saved_model.predict([xtest,mtest]),axis=1)  
                
                                else: 
                                    predictionvalclass = np.argmax(saved_model.predict(xval),axis=1)
                                    predictiontrainclass = np.argmax(saved_model.predict(xtrain) ,axis=1)
                                    if TEST != False:
                                        predictiontest = saved_model.predict(xtest)   
                                        predictiontestclass = np.argmax(predictiontest,axis=1)
                            
                                if TEST != False:
                                    evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                                 tf.convert_to_tensor(['loss', str(eval_train[0]),str(eval_val[0]),str(eval_test[0])]),
                                                 tf.convert_to_tensor(['tf_cindexT', str(eval_train[1]),str(eval_val[1]),str(eval_test[1])])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                                 tf.convert_to_tensor(['loss', str(eval_train[0]),str(eval_val[0])]),
                                                 tf.convert_to_tensor(['tf_cindexT', str(eval_train[1]),str(eval_val[1])])]
    
                                s = sess.run(tf.summary.text('Eval', tf.stack(evalu)))
                                writer.add_summary(s)
                                
                          
                                if TEST != False:   
                                    evalu = [tf.convert_to_tensor(['', 'best train','best val','best test']) ,
                                             tf.convert_to_tensor(['loss', str(0),str(0),str(0)]),
                                             tf.convert_to_tensor(['tf_cindexT', str(uf.cindexT(ytrain[:,1],predictiontrainclass,ytrain[:,0])),str(uf.cindexT(yval[:,1],predictionvalclass,yval[:,0])),str(uf.cindexT(ytest[:,1],predictiontestclass ,ytest[:,0]))])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'best train','best val']) ,
                                             tf.convert_to_tensor(['loss', str(0),str(0)]),
                                             tf.convert_to_tensor(['tf_cindexT', str(uf.cindexT(ytrain[:,1],predictiontrainclass,ytrain[:,0])),str(uf.cindexT(yval[:,1],predictionvalclass,yval[:,0]))])]
    
                                s = sess.run(tf.summary.text('Eval_best', tf.stack(evalu)))
                                writer.add_summary(s)
                                
                                
                                we =  file +'/'+ titleP +"best.h5"        
                                saved_model.save_weights(we)
                                new = pred.load_weights(file+'/' + titleP +'best.h5')
                                
                            
                            elif loss == 'discret':
                                we=file+'/' + titleP +'best_model.h5'
                                if os.path.exists(we) :                                    
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( file+'/' + titleP +'best_model.h5', monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                    
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexTD':uf.tf_cindexTD,'loss':ns.surv_likelihood(n_intervals), 'InstanceNormalization':InstanceNormalization})
                            elif loss == 'RankAnddiscret':
                                  
                                we=file+'/' + titleP +'best_model.h5'
                                if os.path.exists(we) :
                                        saved_model = load_model(file+'/' + titleP +'best_model.h5',
                                        custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                        'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                        'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                        'InstanceNormalization':InstanceNormalization})    
                                else:
                                    
                                    mc = ModelCheckpoint( file+'/' + titleP +'best_model.h5', monitor='val_tf_cindexTD', mode='max', verbose=1, save_best_only=True)
                    
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF,callbacks=[mc,tensorboard] )#
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',
                                            custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,
                                            'spp':uf.SPP,'tf_cindexTD':uf.tf_cindexTD,'discretcox':ns.discretcox,
                                            'rankL':ns.rankL,'mse':lss.mse,'loss':ns.RankingAndDiscret(n_intervals), 
                                            'InstanceNormalization':InstanceNormalization})
              
                            elif loss == 'RankAndmse':      
                                we=file+'/' + titleP +'best_model.h5'
                                if os.path.exists(we) :
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
    
                                else:
                                    mc = ModelCheckpoint( file+'/' + titleP +'best_model.h5', monitor='val_tf_cindexT', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
        
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexT':uf.tf_cindexT,'loss':lss.__RankAndMse_loss(),'rank':lss.rank,'mse':lss.mse, 'InstanceNormalization':InstanceNormalization})
                                              
                            elif loss == 'RankAndcox': 
                                we=file+'/' + titleP +'best_model.h5'
                                if os.path.exists(we) :
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':uf.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( file+'/' + titleP +'best_model.h5', monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#mc,,hp.KerasCallback(log_dir,hparams)
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':uf.tf_cindexR,'loss':lss.__RankAndCox_loss(),'rank':lss.rank,'cox':lss.cox, 'InstanceNormalization':InstanceNormalization})
                            
                            else:
                                we=file+'/' + titleP +'best_model.h5'
                                                                                        # we = './Results/Pretrain/Pretrain/New28_11_2020/ALLGPU12_3Dbin/RankAndmse_Pre_binary_atc_sppTrue_bFalse_D3True_n3/' + titleP +'best_model.h5'
                                if os.path.exists(we) :
                                    saved_model = load_model(we,custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'tf_cindexR':uf.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})
                                else:
                                    mc = ModelCheckpoint( we, monitor='val_tf_cindexR', mode='max', verbose=1, save_best_only=True)
                                    fitting = pred.fit([xtrain,mtrain], ytrain, validation_data=([xval,mval], yval), batch_size=batch,epochs=epochF, callbacks=[mc,tensorboard])#,hp.KerasCallback(log_dir,hparams)    
                                    saved_model = load_model(file+'/' + titleP +'best_model.h5',custom_objects={'bodyF':uf.bodyF,'bodyF4':uf.bodyF4,'while_conditionF':uf.while_conditionF,'tf':tensorflow,'spp':uf.SPP,'spp4':uf.SPP4,'tf_cindexR':uf.tf_cindexR,'loss':lss.__cox_loss(), 'InstanceNormalization':InstanceNormalization})

                            if loss!= 'classif' and loss != 'triplet':      
                                if spp != False:
                                    eval_val_best = saved_model.evaluate([xval,mval], yval)
                                    eval_train_best = saved_model.evaluate([xtrain,mtrain], ytrain)
                                    
                                    if TEST != False:  
                                        eval_test_best= saved_model.evaluate([xtest,mtest], ytest)
                                else: 
                                    eval_val_best = saved_model.evaluate(xval, yval)
                                    eval_train_best = saved_model.evaluate(xtrain, ytrain)
                                if spp != False:
                                    predictionval = saved_model.predict([xval,mval])
                                    predictiontrain = saved_model.predict([xtrain,mtrain])
                                    
                                    if TEST != False:
                                        predictiontest = saved_model.predict([xtest,mtest])
                                    
                                else: 
                                    predictionval = saved_model.predict(xval)
                                    predictiontrain = saved_model.predict(xtrain)
                                    
                                    if TEST != False:
                                        predictiontest = saved_model.predict(xtest) 
                                            
                                if len(np.shape(predictionval))==2:
                                    if loss !='discret'and loss!='RankAnddiscret':
                                        predictionval = predictionval[:,0]
                                        predictiontrainExclude = predictiontrain[:,0]
                                        
                                if loss =='discret'or loss=='RankAnddiscret':    
                                    if TEST != False:                      
                                        predictiontest = uf.FromLongToTime(predictiontest,n_intervals,None)
                                    predictionval = uf.FromLongToTime(np.array(predictionval),n_intervals,None)
                                    predictiontrain = uf.FromLongToTime(predictiontrain,n_intervals,None)

                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret': 
                                    ppr = np.array([ppred / 365 for ppred in predictionval], dtype=np.int)
                                    pYM = np.argmax(yvalgrp,axis = 1)
                                    acc_val = accuracy_score(pYM,ppr)
                                    confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                    df_fft=pd.DataFrame(confB)
                                    df_fft.to_csv(file +'/' + titleP + '_confusion_val.csv')
                                # Calculate mean of val predictions
                                # if loss != 'classif' and loss != 'triplet'  and loss != 'discret':
                                #     pathSavepredTr = file+'/' + titleP + '_ped_train'
                                #     np.save(pathSavepredTr,predictiontrain)
                                #     pathSavetrueTr = file+'/' + titleP + '_true_train'
                                #     np.save(pathSavetrueTr,ytrain[:,1])
                                #     pathSaveeventTr = file+'/' + titleP + '_event_train'
                                #     np.save(pathSaveeventTr,ytrain[:,0])
                                #     pathSaveeventTr = file+'/' + titleP + '_r_train'
                                #     np.save(pathSaveeventTr,rtrain)
                                
                                                
                                #     pathSavepredTr = file+'/' + titleP + '_ped_val'
                                #     np.save(pathSavepredTr,predictionval)
                                #     pathSavetrueTr = file+'/' + titleP + '_true_val'
                                #     np.save(pathSavetrueTr,yval[:,1])
                                #     pathSaveeventTr = file+'/' + titleP + '_event_val'
                                #     np.save(pathSaveeventTr,yval[:,0])
                                #     pathSaveeventTr = file+'/' + titleP + '_r_val'
                                #     np.save(pathSaveeventTr,rval)
                                
                                
                                """  Features ********************************************************************"""
                    
                                flat =  saved_model.get_layer(index = utils.find_layer_idx(saved_model,'resh48'))#Newmodel.get_layer(index = -4)
                                featModel = Model(inputs=saved_model.input, outputs=flat.output)
                               
                                featurestrain=featModel.predict(xtrain)
                                pathSaveFFT = file+'/' + titleP + '_features_flatten_train'
                                np.save(pathSaveFFT,featurestrain)
                                pathCSV_FFT = file +'/' + titleP + '_features_flatten_train.csv'
                                df_fft=pd.DataFrame(featurestrain)
                                df_fft.to_csv(pathCSV_FFT)           
                                
                                featuresval=featModel.predict(xval)
                                pathSaveFFT = file+'/' + titleP + '_features_flatten_val'
                                np.save(pathSaveFFT,featuresval)
                                pathCSV_FFT = file +'/' + titleP + '_features_flatten_val.csv'
                                df_fft=pd.DataFrame(featuresval)
                                df_fft.to_csv(pathCSV_FFT)
                                                                
                                if loss =='discret'or loss=='RankAnddiscret':  
                                    ytrain = ytrainT
                                    yval = yvalT
                                    
                                
                                '''Evaluate with kaplan'''
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                    bestPval,testStat,bestsep,numG0,numG1  = findBestSep(ytrain,predictiontrain)
                                    try:
                                        PvalueTrain,bestsep,numG0_train,numG1_train =groupeByBestSep(ytrain,predictiontrain,bestsep,file,titleP,'train','time')
                                        print('kaplan worked')

                                    except :
                                        PvalueTrain =bestsep=numG0_train=numG1_train = -100
                                        print('kaplan didn t work')
                                        pass
                                    try:
                                        PvalueVal,bestsep,numG0_val,numG1_val =groupeByBestSep(yval,predictionval,bestsep,file,titleP,'val','time')
                                        print('kaplan worked')

                                    except :
                                        PvalueVal = bestsep = numG0_val = numG1_val = -100
                                        print('kaplan didn t work')
                                        pass
                                else:       
                                    bestPval,testStat,bestsep,numG0,numG1  = findBestSep(ytrain,predictiontrain)
                                    try:
                                        PvalueTrain,bestsep,numG0_train,numG1_train =groupeByBestSep(ytrain,predictiontrain,bestsep,file,titleP,'train')
                                        print('kaplan worked')

                                    except :
                                        PvalueTrain =bestsep=numG0_train=numG1_train = -100
                                        print('kaplan didn t work')
                                        pass
                                    try:
                                        PvalueVal,bestsep,numG0_val,numG1_val =groupeByBestSep(yval,predictionval,bestsep,file,titleP,'val')
                                        print('kaplan worked')

                                    except :
                                        PvalueVal = bestsep = numG0_val = numG1_val = -100
                                        print('kaplan didn t work')
                                        pass
                                
                                
                                if loss =='discret'or loss=='RankAnddiscret':    
                                    ax=predictionShow('val',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                    if TEST != False:
                                        ax=predictionShow('test',ytestT,predictiontest,writer)
                                else:
                                    ax=predictionShow('val',yval,predictionval,writer)
                                    if TEST != False:
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
                                    yvalgrp = uf.YProcessing2(yval,nbClasses)
                                elif loss == 'triplet':
                                    yvalgrp = np.transpose(np.concatenate(([yval[:,0]],[np.argmax(uf.YProcessing2(yval,nbClasses),axis=1)]),axis=0))
                                else:
                                    yvalgrp = uf.YProcessing4(yval,'YEAR')
                                    yvalgrpwc = uf.YProcessing4(yval,'YEARWC')
                                    yvalgrp3 = uf.YProcessing4(yval,'THREE')
                                    yvalgrpwc3 = uf.YProcessing4(yval,'THREEWC')
                                
                                if loss =='discret'or loss=='RankAnddiscret':    
                                    ax=predictionShow('valgrp',yvalT,np.reshape(np.array(predictionval),(-1,1)),writer)
                                    if TEST != False:
                                        ax=predictionShow('test',ytestT,predictiontest,writer)
                                    ax=predictionShow('train',ytrainT,predictiontrain,writer)
                                else:
                                    ax=predictionShow('valgrp',yval,predictionval,writer)
                                    if TEST != False:
                                        ax=predictionShow('test',ytest,predictiontest,writer)
                                    ax=predictionShow('train',ytrain,predictiontrain,writer)
                           
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                    #super Cindex avec censure
                                    cindTrainSuper = uf.cindexT(np.argmax(ytraingrp,axis=1),predictiontrain,ytrain[:,0])
                                    if TEST != False:
                                        cindTestSuper = uf.cindexT(np.argmax(ytestgrp,axis=1),predictiontest,ytest[:,0])                                                                     
                                    cindValSuper = uf.cindexT(np.argmax(yvalgrp,axis=1),np.reshape(np.array(predictionval),(-1,1)),yval[:,0]) 
                                    
                                else:
                                
                                    #super Cindex avec censure
                                    cindTrainSuper = uf.cindexR(np.argmax(ytraingrp,axis=1),predictiontrain,ytrain[:,0])
                                    if TEST != False:
                                        cindTestSuper = uf.cindexR(np.argmax(ytestgrp,axis=1),predictiontest,ytest[:,0])                                                                     
                                    cindValSuper = uf.cindexR(np.argmax(yvalgrp,axis=1),np.array(predictionval),yval[:,0]) 
                                    
               
                                '''Evaluate with kaplan grp'''
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                    try:
                                        PvalueValgrp,bestsep,numG0_valgrp,numG1_valgrp =groupeByBestSep(yval,predictionval,bestsep,file,titleP,'valgrp','time')
                                        print('kaplan worked')
                                    except :
                                        PvalueValgrp = numG0_valgrp = numG1_valgrp = -100
                                        print('kaplan didn t work')
                                        pass
                                else:      
                                    try:
                                        PvalueValgrp,bestsep,numG0_valgrp,numG1_valgrp =groupeByBestSep(yval,predictionval,bestsep,file,titleP,'valgrp')
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
                                        PvalueTrainExclude,bestsepExclude,numG0_trainExclude,numG1_trainExclude =groupeByBestSep(ytrain,predictiontrain,bestsepExclude,file,titleP,'trainExclude','time')
                                        print('kaplan worked')
                                    except :
                                        PvalueTrainExclude=bestsepExclude=numG0_trainExclude=numG1_trainExclude = -100
                                        print('kaplan didn t work')
                                        pass
                                    try:
                                        PvalueValgrpExclude,bestsepExclude,numG0_valgrpExclude,numG1_valgrpExclude =groupeByBestSep(yval,predictionval,bestsepExclude,file,titleP,'valgrpExclude','time')
                                        print('kaplan worked')
                                    except :
                                        PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                        print('kaplan didn t work')
                                        pass
                                else:       
                                    bestPvalExclude,testStatExclude,bestsepExclude,numG0Exclude,numG1Exclude  = findBestSep(ytrainExclude,predictiontrainExclude)
                                    try:
                                        PvalueTrainExclude,bestsepExclude,numG0_trainExclude,numG1_trainExclude =groupeByBestSep(ytrain,predictiontrain,bestsepExclude,file,titleP,'trainExclude')
                                        print('kaplan worked')
                                    except :
                                        PvalueTrainExclude=bestsepExclude=numG0_trainExclude=numG1_trainExclude = -100
                                        print('kaplan didn t work')
                                        pass
                                    try:
                                        PvalueValgrpExclude,bestsepExclude,numG0_valgrpExclude,numG1_valgrpExclude =groupeByBestSep(yval,predictionval,bestsepExclude,file,titleP,'valgrpExclude')
                                        print('kaplan worked')
                                    except :
                                        PvalueValgrpExclude = numG0_valgrpExclude = numG1_valgrpExclude = -100
                                        print('kaplan didn t work')
                                        pass
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                    cindexval1 = uf.cindexT(yval[:,1],np.array(predictionval),yval[:,0])
                                else:
                                    cindexval1 = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0])
                        
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
                                    ppr = np.array([ppred / 365 for ppred in predictiontrain], dtype=np.int)
                                    pYM = np.argmax(ytraingrp,axis = 1)
                                    acc_train = accuracy_score(pYM,ppr)
                                    confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                    df_fft=pd.DataFrame(confB)
                                    df_fft.to_csv(file +'/' + titleP + '_confusion_train.csv')
                                    
                                    ppr = np.array([ppred / 365 for ppred in predictionval], dtype=np.int)
                                    pYM = np.argmax(yvalgrp,axis = 1)
                                    acc_valgrp = accuracy_score(pYM,ppr)
                                    confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                    df_fft=pd.DataFrame(confB)
                                    df_fft.to_csv(file +'/' + titleP + '_confusion_valgrp.csv')
                                if TEST != False:
                                    evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                             tf.convert_to_tensor(['loss', str(eval_train_best[0]),str(eval_val_best[0]),str(eval_test_best[0])]),
                                             tf.convert_to_tensor(['superCindex', str(cindTrainSuper),str(cindValSuper),str(cindTestSuper)]),
                                             tf.convert_to_tensor(['tf_cindexR', str(eval_train_best[1]),str(eval_val_best[1]),str(eval_test_best[1])])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                             tf.convert_to_tensor(['loss', str(eval_train_best[0]),str(eval_val_best[0])]),
                                             tf.convert_to_tensor(['superCindex', str(cindTrainSuper),str(cindValSuper)]),
                                             tf.convert_to_tensor(['Pval', str(PvalueTrain),str(PvalueVal)]),
                                             tf.convert_to_tensor(['numG0', str(numG0_train),str(numG0_val)]),
                                             tf.convert_to_tensor(['numG1', str(numG1_train),str(numG1_val)]),
                                             tf.convert_to_tensor(['bestsep', str(bestsep),str(bestsep)]),
                                             tf.convert_to_tensor(['tf_cindexR', str(eval_train_best[1]),str(cindexval1)])]
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
    
                                    [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe',order='T')
                                    [cindexVal0,cindexVal1,cindexVal2,cindexVal3,cindexVal4,cindexVal5,cindexVal6] = uf.time_cindex(yval[:,1],np.array(predictionval),yval[:,0],cla='classe',order='T')
                                else:
                                    [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe',order='R')
                                    [cindexVal0,cindexVal1,cindexVal2,cindexVal3,cindexVal4,cindexVal5,cindexVal6] = uf.time_cindex(yval[:,1],np.array(predictionval),yval[:,0],cla='classe',order='R')
                               
                                if TEST != False:
                                    [cindexTest0,cindexTest1,cindexTest2,cindexTest3,cindexTest4,cindexTest5,cindexTest6] = uf.time_cindex(ytest[:,1],predictiontest,ytest[:,0],cla='classe')
                                                                
                                
                                if TEST != False:
                                    evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                             tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindexVal0),str(cindexTest0)]),
                                             tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindexVal1),str(cindexTest1)]),
                                             tf.convert_to_tensor(['cindex_2', str(cindex2) ,str(cindexVal2),str(cindexTest2)]),
                                             tf.convert_to_tensor(['cindex_3', str(cindex3) ,str(cindexVal3),str(cindexTest3)]),
                                             tf.convert_to_tensor(['cindex_4', str(cindex4) ,str(cindexVal4),str(cindexTest4)]),
                                             tf.convert_to_tensor(['cindex_5', str(cindex5) ,str(cindexVal5),str(cindexTest5)]),
                                             tf.convert_to_tensor(['cindex_6', str(cindex6) ,str(cindexVal6),str(cindexTest6)])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                             tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindexVal0)]),
                                             tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindexVal1)]),
                                             tf.convert_to_tensor(['cindex_2', str(cindex2) ,str(cindexVal2)]),
                                             tf.convert_to_tensor(['cindex_3', str(cindex3) ,str(cindexVal3)]),
                                             tf.convert_to_tensor(['cindex_4', str(cindex4) ,str(cindexVal4)]),
                                             tf.convert_to_tensor(['cindex_5', str(cindex5) ,str(cindexVal5)]),
                                             tf.convert_to_tensor(['cindex_6', str(cindex6) ,str(cindexVal6)])]
                                s = sess.run(tf.summary.text('EvalBestParTemps_7classes', tf.stack(evalu)))
                                writer.add_summary(s)
                                
                                if loss == 'RankAndmse' or loss == 'RankAnddiscret' or loss == 'discret':
        
                                    [cindex0,cindex1] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre',order='T')
                                    [cindexVal0,cindexVal1] = uf.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre',order='T')
                                else:
                                    [cindex0,cindex1] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre',order='R')
                                    [cindexVal0,cindexVal1] = uf.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre',order='R')
                                                                
                                if TEST != False:
                                    [cindexTest0,cindexTest1] = uf.time_cindex(ytest[:,1],predictiontest,ytest[:,0],cla='0ouAutre')

                                if TEST != False:
                                    evalu = [tf.convert_to_tensor(['', 'train','val','test']) ,
                                             tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindexVal0),str(cindexTest0)]),
                                             tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindexVal1),str(cindexTest1)])]
                                else:
                                    evalu = [tf.convert_to_tensor(['', 'train','val']) ,
                                             tf.convert_to_tensor(['cindex_0', str(cindex0) ,str(cindexVal0)]),
                                             tf.convert_to_tensor(['cindex_1', str(cindex1) ,str(cindexVal1)])]
                                s = sess.run(tf.summary.text('EvalBestParTemps_0ouAutre', tf.stack(evalu)))
                                writer.add_summary(s) 
                                      
                                
                                ''' ######### test TTA'''
                                # pour chaque goupe d'images, mean de prediction
                                # predTTA = []
                                # for i in range(0,len(rtest)):
                                #     print(rtest[i])
                                #     predtestA = saved_model.predict([xtestA[i*30:i*30+30],mtestA[i*30:i*30+30]])
                                #     ppp = np.mean(predtestA)
                                #     predTTA.append(ppp)
                                # #toEval = np.array([predTTA,np.ones(np.shape(predTTA))]).T
                                # cindexTestA = uf.cindexR(ytest[:,1],np.array(predTTA),ytest[:,0])
                                # cindexTestSa = uf.cindexR(ytest[:,1],np.array(predictiontest),ytest[:,0])
    
                            curves = plt.imread(doss+'datasep.png')
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
                                if spp != False:
                                    fee = caw.predict([xtrain,mtrain]) #shape (B,1,1,64)
                                    feetest = caw.predict([xval,mval]) #shape (B,1,1,64)
                                else:
                                    fee = caw.predict(xtrain) #shape (B,1,1,64)
                                    feetest = caw.predict(xval) #shape (B,1,1,64)
                                if D3==True:
                                    fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[4]))
                                else:
                                    fee=np.reshape(fee,(np.shape(fee)[0],np.shape(fee)[3]))
                                    
                                if D3==True:
                                    feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[4]))
                                else:
                                    feetest=np.reshape(feetest,(np.shape(feetest)[0],np.shape(feetest)[3]))
                                    
                                # pathSaveFFT = file+'/' + titleP + '_channel_attention_weight_train'
                                # np.save(pathSaveFFT,fee) 
                                
                                # pathSaveFFT = file+'/' + titleP + '_channel_attention_weight_test'
                                # np.save(pathSaveFFT,feetest) 
                                
                                if D3==False:
                                    if attention != 'c':
                                        layerSAW = utils.find_layer_idx(saved_model,'spatial_attention_weight')
                                        saw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerSAW].output)
                                        if spp!= False:
                                            fee2 = saw.predict([xtrain,mtrain])
                                        else:
                                            
                                            fee2 = saw.predict(xtrain)#shape (B,9,9,1)
                                        if D3==True:                         
                                            fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2],np.shape(fee2)[3]))
                                        else: 
                                            fee2=np.reshape(fee2,(np.shape(fee2)[0],np.shape(fee2)[1],np.shape(fee2)[2]))
                                        # pathSaveFFT = file+'/' + titleP + '_spatial_attention_weight_train'
                                        # np.save(pathSaveFFT,fee2) 
                                        
                                        if spp != False:
                                            fee2val = saw.predict([xval,mval])
                                        else:
                                            fee2val = saw.predict(xval)
                                        if D3==True:                         
                                            fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2],np.shape(fee2val)[3]))
                                        else: 
                                            fee2val=np.reshape(fee2val,(np.shape(fee2val)[0],np.shape(fee2val)[1],np.shape(fee2val)[2]))
                                        # pathSaveFFT = file+'/' + titleP + '_spatial_attention_weight_val'
                                        # np.save(pathSaveFFT,fee2val) 
                                        
                                        
                                        if TEST != False:
                                            if spp != False:
                                                fee2test = saw.predict([xtest,mtest])
                                            else:
                                                fee2test = saw.predict(xtest)
                                            if D3==True:                         
                                                fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2],np.shape(fee2test)[3]))
                                            else: 
                                                fee2test=np.reshape(fee2test,(np.shape(fee2test)[0],np.shape(fee2test)[1],np.shape(fee2test)[2]))
                                        # pathSaveFFT = file+'/' + titleP + '_spatial_attention_weight_test'
                                        # np.save(pathSaveFFT,fee2val) 
            
                                caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerCAW].output)
                                if spp != False:
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
                            
                                if D3==False:
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
                                if D3==False:
                                
                                    cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[2],1)))]
                                else:
                                    cawt = [tf.convert_to_tensor(np.reshape(sumcawt,(1,np.shape(sumcawt)[3],1)))]
    
                                s = sess.run(tf.summary.image('Sum_channel_attention_weight_train_over_patients', tf.stack(cawt)))
                                writer.add_summary(s)
                                                    
                                if D3==False:
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
                                
                                
                                if D3==False:
                                    ##### Features map
                                    if attention == 'cs':
                                        layerFeat = utils.find_layer_idx(saved_model,'res')
                                    else:
                                        layerFeat = utils.find_layer_idx(saved_model,'multiply_1')
                                    caw = Model(inputs=saved_model.inputs,outputs=saved_model.layers[layerFeat].output)
                                    if spp != False:
                                        featuresMap = caw.predict([xtrain,mtrain]) #shape (456, 9, 9, 64)
                                        featuresMapTest = caw.predict([xval,mval]) #shape (456, 9, 9, 64)
                                    else:
                                        featuresMap = caw.predict(xtrain) #shape (456, 9, 9, 64)
                                        featuresMapTest = caw.predict(xval) #shape (456, 9, 9, 64)
                                    
                                    for patientn in range(20):
                                        writer = tf.summary.FileWriter(log_dir)
                                        with tf.name_scope('channel_feat_map_Pat_{}'.format(patientn)):
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
                                            aaa =np.array(featuresMap[patientn,:,:,fi]) 
                                            aaa = cv2.resize(aaa, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
                                            a_scaled = 255*(aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
                                            im.paste(Image.fromarray(a_scaled),(begx,begy))
                                            
                                            #image
                                            aaa =np.array(xtrain[patientn,:,:,0]) 
                                            a_scaled = 255*(aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
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
                                            aaa =np.array(featuresMap[patientn,:,:,fi]) 
                                            aaa = cv2.resize(aaa, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)
                                            a_scaled = 255*(aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
                                            im.paste(Image.fromarray(a_scaled),(begx,begy))
                                            
                                            #image
                                            aaa =np.array(xtrain[patientn,:,:,0]) 
                                            a_scaled = 255*(aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
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
                                        
                                        cindTrain = uf.cindexR(ytrain[:,1],predictiontrainclass,ytrain[:,0])
                                        
                                        
                                        if TEST != False:
                                            cindTest = uf.cindexR(ytest[:,1],predictiontestclass,ytest[:,0])
                                        cindVal = uf.cindexR(yval[:,1],np.array(predictionvalclass),yval[:,0]) 
                                        
                                        [cindexTrain1,cindexTrain2,cindexTrain3] = uf.time_cindex(ytrain[:,1],predictiontrainclass,ytrain[:,0])
                                        
                                        if TEST != False:
                                            [cindexTest1,cindexTest2,cindexTest3] = uf.time_cindex(ytest[:,1],predictiontestclass,ytest[:,0])
                                        [cindexVal1,cindexVal2,cindexVal3] = uf.time_cindex(yval[:,1],predictionvalclass,yval[:,0])
                
                                    
                                    else:
                                            
                                        cindTrain = uf.cindexR(ytrain[:,1],predictiontrain,ytrain[:,0])
                                        
                                        if TEST != False:
                                            cindTest = uf.cindexR(ytest[:,1],predictiontest,ytest[:,0])
                                        print(np.shape(predictionval),np.shape(yval))
                                        cindVal = uf.cindexR(yval[:,1],np.array(predictionval),yval[:,0]) 
                                        
                                        [cindexTrain1,cindexTrain2,cindexTrain3] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0])
                                        
                                        if TEST != False:
                                            [cindexTest1,cindexTest2,cindexTest3] = uf.time_cindex(ytest[:,1],predictiontest,ytest[:,0])
                                        [cindexVal1,cindexVal2,cindexVal3] = uf.time_cindex(yval[:,1],predictionval,yval[:,0])
                                       
                                        [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='classe')
                                        [cindex0,cindex1] = uf.time_cindex(ytrain[:,1],predictiontrain,ytrain[:,0],cla='0ouAutre')
                                        
                                        [cindexVal0,cindexVal1,cindexVal2,cindexVal3,cindexVal4,cindexVal5,cindexVal6] = uf.time_cindex(yval[:,1],predictionval,yval[:,0],cla='classe')
                                        [cindexVal0,cindexVal1] = uf.time_cindex(yval[:,1],predictionval,yval[:,0],cla='0ouAutre')


                                else:
                                    cindTrain = uf.cindexT(ytrainT[:,1],predictiontrain,ytrainT[:,0])
                                    
                                    if TEST != False:   
                                        cindTest = uf.cindexT(ytestT[:,1],predictiontest,ytestT[:,0])
                                    cindVal = uf.cindexT(yvalT[:,1],np.array(predictionval),yvalT[:,0]) 
                                    
                                    # [cindexTrain1,cindexTrain2,cindexTrain3] = uf.time_cindex(uf.FromLongToTime(ytrain,n_intervals,None)[0],predictiontrain,uf.FromLongToTime(ytrain,n_intervals,None)[1],'T')
                                    # [cindexTest1,cindexTest2,cindexTest3] = uf.time_cindex(uf.FromLongToTime(ytest,n_intervals,None)[0],predictiontest,uf.FromLongToTime(ytest,n_intervals,None)[1],'T')
                                    # [cindexVal1,cindexVal2,cindexVal3] = uf.time_cindex(uf.FromLongToTime(yval,n_intervals,None)[0],predictionval,uf.FromLongToTime(yval,n_intervals,None)[1],'T')
                                   
                            # if loss != 'classif' and loss != 'triplet'  and loss != 'discret':
                            #     pathSavepredTr = file+'/' + titleP + '_ped_train'
                            #     np.save(pathSavepredTr,predictiontrain)
                            #     pathSavetrueTr = file+'/' + titleP + '_true_train'
                            #     np.save(pathSavetrueTr,ytrain[:,1])
                            #     pathSaveeventTr = file+'/' + titleP + '_event_train'
                            #     np.save(pathSaveeventTr,ytrain[:,0])
                                            
                            #     pathSavepredTr = file+'/' + titleP + '_ped_val'
                            #     np.save(pathSavepredTr,predictionval)
                            #     pathSavetrueTr = file+'/' + titleP + '_true_val'
                            #     np.save(pathSavetrueTr,yval[:,1])
                            #     pathSaveeventTr = file+'/' + titleP + '_event_val'
                            #     np.save(pathSaveeventTr,yval[:,0])
                            #     pathSavepredTr = file+'/' + titleP + '_ped_test'
                            #     np.save(pathSavepredTr,predictiontest)
                            #     pathSavetrueTr = file+'/' + titleP + '_true_test'
                            #     np.save(pathSavetrueTr,ytest[:,1])
                            #     pathSaveeventTr = file+'/' + titleP + '_event_test'
                            #     np.save(pathSaveeventTr,ytest[:,0])
                            
                            """  FILTERS  ********************************************************************"""
                            # if attention !='False':
                            #     flatw =  saved_model.get_layer(index = utils.find_layer_idx(saved_model,'sequential_1')).get_layer(index = -4 )#Newmodel.get_layer(index = -4)
                            #     filters = flatw.get_weights()
                            #     pathSaveFFT = file+'/' + titleP + '_filters'
                            #     np.save(pathSaveFFT,filters)
                            #featModel = Model(inputs=Newmodel.input, outputs=flat.input)
                            #featuresSMALL=featModel.predict([xtrain,mtrain])
                
                            #pathSaveFFT = file+'/' + titleP + '_features_visu_train.csv'
                            #np.savez(pathSaveFFT,featuresSMALL)
                            """  Features ********************************************************************"""
                
                            # flat =  Newmodel.get_layer(index = utils.find_layer_idx(Newmodel,'res2'))#Newmodel.get_layer(index = -4)
                            # featModel = Model(inputs=pred.input, outputs=flat.output)
                            # if CrossVal != True:
                            #     featuresSMALL=featModel.predict([xtest,mtest] )
                            #     pathSaveFFT = file+'/' + titleP + '_features_flatten_test'
                            #     np.save(pathSaveFFT,featuresSMALL)
                            #     pathCSV_FFT = file +'/' + titleP + '_features_flatten_test.csv'
                            #     df_fft=pd.DataFrame(featuresSMALL)
                            #     df_fft.to_csv(pathCSV_FFT)                        
                            # featurestrain=featModel.predict([xtrain,mtrain])
                    
                            # pathSaveFFT = file+'/' + titleP + '_features_flatten_train'
                            # np.save(pathSaveFFT,featurestrain)
                            # pathCSV_FFT = file +'/' + titleP + '_features_flatten_train.csv'
                            # df_fft=pd.DataFrame(featurestrain)
                            # df_fft.to_csv(pathCSV_FFT)                       
                            # featuresval=featModel.predict([xval,mval])
                    
                            # pathSaveFFT = file+'/' + titleP + '_features_flatten_val'
                            # np.save(pathSaveFFT,featuresval)
                            # pathCSV_FFT = file +'/' + titleP + '_features_flatten_val.csv'
                            # df_fft=pd.DataFrame(featuresval)
                            # df_fft.to_csv(pathCSV_FFT)
                            
                            # if loss == 'coxAndclassif' or loss == 'classif' :
                            #     pathSavepredTr = file+'/' + titleP + '_ped_class_train'
                            #     np.save(pathSavepredTr,predictiontrainclass)
                            #     pathSavetrueTr = file+'/' + titleP + '_true_class_train'
                            #     np.save(pathSavetrueTr,ytraingrp)
                                            
                            #     pathSavepredTr = file+'/' + titleP + '_ped_class_val'
                            #     np.save(pathSavepredTr,predictionvalclass)
                            #     pathSavetrueTr = file+'/' + titleP + '_true_class_val'
                            #     np.save(pathSavetrueTr,yvalgrp)
                                
                            #     if TEST != False:
                            #         pathSavepredTr = file+'/' + titleP + '_ped_class_test'
                            #         np.save(pathSavepredTr,predictionBeforeclass)
                            #         pathSavetrueTr = file+'/' + titleP + '_true_class_test'
                            #         np.save(pathSavetrueTr,ytestgrp)
                                    
                            # we =  file + titleP +".h5"        
                            # pred.save_weights(we)
                            #results = saving(fitting,titleP,eval_train, eval_val,None,file,metrics)
                            
                            if loss=='classif' or loss =='coxAndclassif':
                                print('--- predict after fintune')
                                
                                if TEST != False:
                                    ppr = predictionBeforeclass
                                    pYM = np.argmax(ytestgrp,axis = 1)
                                    resu1 = accuracy_score(pYM,ppr)
                                    print('acc test',resu1)
                                    confA = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                    np.save(file+ '/'+titleP + "confusion_test.npy",confA)
                                
                                ppr = predictiontrainclass
                                pYM = np.argmax(ytraingrp,axis = 1)
                                resu2 = accuracy_score(pYM,ppr)
                                print('acc train',resu2)
                                confB = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                np.save(file+ '/'+titleP + "confusion_after_train.npy",confB)
                                
                                ppr = predictionvalclass
                                pYM = np.argmax(yvalgrp,axis = 1)
                                resu3 = accuracy_score(pYM,ppr)
                                print('acc validation',resu3)
                                confC = sklearn.metrics.confusion_matrix(pYM, ppr, labels=None, sample_weight=None)
                                np.save(file+ '/'+titleP + "confusion_after_val.npy",confC)
                                                            
                            """ ******************** SAVING  *********************************"""
        
                            if CrossVal == True:
                                if TEST != False:
                                    fullRow=['CV{}_K{}'.format(CV,k),eval_train_best[1],cindexval1,eval_test_best[1]]
                                else:
                                    fullRow=['CV{}_K{}'.format(CV,k),eval_train_best[1],cindexval1]
    
                                for key, value in param.iteritems():
                                    print(key)
                                    temp = value[0]
                                    fullRow.append(temp)
                                fullcv.append(fullRow)
                                if TEST != False:
                                    npa =['CV_K','best_train_cindex','best_val_cindex','best_test_cindex']
                                else:                                
                                    npa =['CV_K','best_train_cindex','best_val_cindex']
    
                                for key, value in param.iteritems():
                                    temp = key
                                    npa.append(temp)
                                pathCSV_fullDF = intermDoss +'FullCV_{}'.format(CV) + '.csv'
                                fullDF=pd.DataFrame(fullcv,columns = npa)
                                fullDF.to_csv(pathCSV_fullDF)
        
                                # LossTrain.append(eval_train[0])
                                # if loss != 'classif' :
                                CindexTrain.append(cindTrain)
                                # else:
                                    # CindexTrain.append(0)
                                # LossVal.append(eval_val[0])
                                # if loss != 'classif':
                                CindexVal.append(cindVal)
                                # else:
                                #     CindexVal.append(0)
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
                                    val_best.append(cindexval1)
                                else:
                                    train_best.append(eval_train_best[2])
                                    val_best.append(cindexval1)
                                    
                            
                    """ FIN K loop """
                if CrossVal==True:
                    if PRETRAIN != True:
                        fullDF = pd.read_csv(pathCSV_fullDF, encoding='utf-8')
                        pathCSV_mofDF =   intermDoss +'MeanOverFoldCV_{}'.format(str(CV))  + '.csv'
                        if os.path.exists(pathCSV_mofDF):
                            mofcv = pd.read_csv(pathCSV_mofDF, encoding='utf-8')
                            mofcv = mofcv.iloc[:,1:].values.tolist()
                        else:
                            mofcv = []
                        if TEST != False:
                            npa =['CV_K','best_train_cindex','best_val_cindex','best_test_cindex']
                        else:
                            npa =['CV_K','best_train_cindex','best_val_cindex']
    
                        numi=0
                        for k in range(len(X)):
                            kDF = fullDF[fullDF['CV_K']== 'CV{}_K{}'.format(str(CV),str(k))]
                            listeMode={'loss':list(set(kDF['loss'])),'D3':list(set(kDF['D3'])),'spp':list(set(kDF['spp'])),'attention':list(set(kDF['attention'])),'pretrain':list(set(kDF['pretrain'])),'npatch':list(set(kDF['npatch']))}
                            gridMode = ParameterGrid(listeMode)
                            for pM in gridMode:
                                try:
                                   l= fullDF['loss']==pM['loss']
                                   d= fullDF['D3']==pM['D3']
                                   s= fullDF['spp']==pM['spp']
                                   a= fullDF['attention']==pM['attention']
                                   pr= fullDF['pretrain']==pM['pretrain']
                                   npp= fullDF['npatch']==pM['npatch']
                                   aaa=kDF[l & d & s & a & pr & npp]
                                   indM=np.where(aaa['best_val_cindex']==np.max(aaa['best_val_cindex']))
                                   
                                   if numi==0:
                                       mofcv=aaa.iloc[indM[0],1:]
                                   else:
                                       mofcv=pd.concat([mofcv,aaa.iloc[indM[0],1:]])
                                   numi+=1
                                except:
                                    pass
                        mofcv.to_csv(pathCSV_mofDF) 