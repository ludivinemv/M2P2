# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv

Processing of the images and true values
"""

import os
from patch_preprocessing import patch_preprocessing as pp
import numpy as np
import pandas as pd
#from PIL import Image
#import matplotlib.pyplot as plt
from keras.utils import to_categorical

"""###############################################################################
######################### Patch processing ####################################
###############################################################################"""


def patchP(PATH,reload,interS,method='all',D3=False,npatch=9,sess=None):
    """ 
    Load 9 patches with done with the bouding box and interpolate to interpolsize if method = box, 

    else, 9 patches of the lesion with the interpolsize size and the associatec mask 
    """
    if D3==True:
        na = "liste_patchs"+str(interS) + method+ str(npatch)+"3D.npy"
        na2 = "liste_maskP"+str(interS) + method+ str(npatch)+"3D.npy"
        na3 = "liste_patients"+str(interS) + method+ str(npatch)+"3D.npy"
        na4 = "liste_ref"+str(interS) + method+str(npatch)+ "3D.npy"
        na5 = "errors"+str(interS) + method+str(npatch)+ "3D.npy"
        na6 = "data"+str(interS) + method+str(npatch)+ "3D.npy"
    else: 
        na = "liste_patchs"+str(interS) + method+str(npatch)+ ".npy"
        na2 = "liste_maskP"+str(interS) + method+ str(npatch)+".npy"
        na3 = "liste_patients"+str(interS) + method+str(npatch)+ ".npy"
        na4 = "liste_ref"+str(interS) + method+ str(npatch)+".npy"
        na5 = "errors"+str(interS) + method+ str(npatch)+".npy"
        na6 = "data"+str(interS) + method+ str(npatch)+".npy"
    if os.path.exists(na) and os.path.exists(na3) and os.path.exists(na2) and os.path.exists(na5)and os.path.exists(na4) :
        if reload == True :
            print("---------Patch processing already done ----------")
            liste_patches = np.load(na)
            liste_patients = np.load(na3)
            liste_mask = np.load(na2)
            errors = np.load(na5)
            liste_ref = np.load(na4)
            data = pd.read_csv(na6)
            data.index =data.iloc[:,0]
            data =data.iloc[:,1:3]
            if np.shape(liste_patches)[0]==0 or np.shape(liste_patients)[0]==0 or  np.shape(liste_ref)[0]==0 or np.shape(errors)[0]==0 or np.shape(liste_patches)[1] != interS:
                print("---------reload because wrong size ----------")
                patch_pp= pp(PATH,interS,method,reload,D3,npatch,sess=sess)
                data, liste_patches, liste_ref, liste_patients, errors,liste_mask = patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors, patch_pp.liste_mask
                print('************** patch size',np.shape(liste_patches))
                np.save(na,liste_patches)
                np.save(na3,liste_patients)
                np.save(na2,liste_mask)
                np.save(na5,errors)
                np.save(na4,liste_ref)
                data.to_csv(na6)
                
        else :        
            print("---------restart patch calculation ----------")
            patch_pp= pp(PATH,interS,method,reload,D3,npatch,sess=sess)
            data, liste_patches, liste_ref, liste_patients, errors, liste_mask= patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors,patch_pp.liste_mask
            np.save(na,liste_patches)
            np.save(na3,liste_patients)
            np.save(na2,liste_mask)
            np.save(na5,errors)
            np.save(na4,liste_ref)
            data.to_csv(na6)
        
    else :
        print("--------- patches doesn t exist ----------")
        patch_pp= pp(PATH,interS,method,reload,D3,npatch,sess=sess)
        data, liste_patches, liste_ref, liste_patients, errors, liste_mask= patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors,patch_pp.liste_mask
        np.save(na,liste_patches)
        np.save(na3,liste_patients)
        np.save(na2,liste_mask)
        np.save(na5,errors)
        np.save(na4,liste_ref)
        data.to_csv(na6)
    print('************** patient shape',np.shape(liste_patients))

    return data, liste_patches, liste_ref, liste_patients, errors,liste_mask
    

def patchPHN(PATH,reload,interS,method='all',D3=False,npatch=9,dataset='mm'):
    """ 
    To use with the Head and Neck dataset
    Load 9 patches with done with the bouding box and interpolate to interpolsize if method = box, 

    else, 9 patches of the lesion with the interpolsize size and the associatec mask 
    """
    dataset='hn'
    if D3==True:
        na = "liste_patchs"+str(interS) + method+ str(npatch)+"3DHN.npy"
        na2 = "liste_maskP"+str(interS) + method+ str(npatch)+"3DHN.npy"
        na3 = "liste_patients"+str(interS) + method+ str(npatch)+"3DHN.npy"
        na4 = "liste_ref"+str(interS) + method+str(npatch)+ "3DHN.npy"
        na5 = "errors"+str(interS) + method+str(npatch)+ "3DHN.npy"
        na6 = "data"+str(interS) + method+str(npatch)+ "3DHN.npy"
    else: 
        na = "liste_patchs"+str(interS) + method+str(npatch)+ "HN.npy"
        na2 = "liste_maskP"+str(interS) + method+ str(npatch)+"HN.npy"
        na3 = "liste_patients"+str(interS) + method+str(npatch)+ "HN.npy"
        na4 = "liste_ref"+str(interS) + method+ str(npatch)+"HN.npy"
        na5 = "errors"+str(interS) + method+ str(npatch)+"HN.npy"
        na6 = "data"+str(interS) + method+ str(npatch)+"HN.npy"
    if os.path.exists(na) and os.path.exists(na3) and os.path.exists(na2) and os.path.exists(na5)and os.path.exists(na4) :
        if reload == True :
            print("---------Patch processing already done ----------")
            liste_patches = np.load(na)
            liste_patients = np.load(na3)
            liste_mask = np.load(na2)
            errors = np.load(na5)
            liste_ref = np.load(na4)
            data = pd.read_csv(na6)
            data.index =data.iloc[:,0]
            data =data.iloc[:,1:3]
            if np.shape(liste_patches)[0]==0 or np.shape(liste_patients)[0]==0 or  np.shape(liste_ref)[0]==0 or np.shape(errors)[0]==0 or np.shape(liste_patches)[1] != interS:
                print("---------reload because wrong size ----------")
                patch_pp= pp(PATH,interS,method,reload,D3,npatch)
                data, liste_patches, liste_ref, liste_patients, errors,liste_mask = patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors, patch_pp.liste_mask
                print('************** patch size',np.shape(liste_patches))
                np.save(na,liste_patches)
                np.save(na3,liste_patients)
                np.save(na2,liste_mask)
                np.save(na5,errors)
                np.save(na4,liste_ref)
                data.to_csv(na6)
                
        else :        
            print("---------restart patch calculation ----------")
            patch_pp= pp(PATH,interS,method,reload,D3,npatch)
            data, liste_patches, liste_ref, liste_patients, errors, liste_mask= patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors,patch_pp.liste_mask
            np.save(na,liste_patches)
            np.save(na3,liste_patients)
            np.save(na2,liste_mask)
            np.save(na5,errors)
            np.save(na4,liste_ref)
            data.to_csv(na6)
        
    else :
        print("--------- patches doesn t exist ----------")
        if dataset=='hn':
            nameI = 'PET.tif'
        else:
            nameI='image.tif'
        patch_pp= pp(PATH,interS,method,reload,D3,npatch,nameI=nameI)
        data, liste_patches, liste_ref, liste_patients, errors, liste_mask= patch_pp.data, patch_pp.liste_patches, patch_pp.liste_ref, patch_pp.liste_patients , patch_pp.errors,patch_pp.liste_mask
        np.save(na,liste_patches)
        np.save(na3,liste_patients)
        np.save(na2,liste_mask)
        np.save(na5,errors)
        np.save(na4,liste_ref)
        data.to_csv(na6)
    print('************** patient shape',np.shape(liste_patients))

    return data, liste_patches, liste_ref, liste_patients, errors,liste_mask
    

 
    
"""###############################################################################
######################### y processing ####################################
###############################################################################"""
   
def YProcessing(data,liste_patients,classes):    
    y = data.copy()
    
    for pat in data.index:
        if pat not in liste_patients :
            y=y.drop(pat)
    if classes != 'surv':
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
            elif classes == 'YEAR' or classes == 'YEARWC' : # group 1 : less than 1 year, 5 : between 4 and 5 years, 10 : censorship
                if y.loc[pat,'event'] == 1  and y.loc[pat,'time']//365 <= 5:
                    y_grp.loc[pat,'group'] = y.loc[pat,'time']//365
                elif y.loc[pat,'time']//365 >= 5:
                    y_grp.loc[pat,'group'] = 5
                else:
                    y_grp.loc[pat,'group'] = 6
            else:
                y_grp = data
    else:
        y_grp = data
    y_grp.to_csv('y_grp.csv')
    
    return y_grp

"""###############################################################################
######################### data splitting ####################################
###############################################################################"""
def splitting(liste_patients,y_grp,classes,change,liste_interpolate):
    # remove censored classes if asked

    if classes[-2:] == 'WC':
        liste_patientsWC = []
        liste_index = []
        if classes == 'TWOWC':
            noC = 2
        else:
            noC = 6
        for i,n in enumerate(liste_patients):
            if y_grp.loc[n,'group'] != noC:
                liste_patientsWC.append(str(n))
                for j in range(9):
                    liste_index.append(9*i+j)
        yy = y_grp.loc[liste_patientsWC] #y_grp sans censure
        g = [0,0,0,0,0,0] #nbre de patients par groups
        for i in range(len(yy)):
            h = yy.iloc[i,0] 
            g[h] = g[h] + 1
        liste_patients = liste_patientsWC
        liste_interpolate = liste_interpolate[liste_index,:,:]
        
        y_grp= yy
        
    #### create vector of indexes
    if change == True:
        rnd_data = np.random.permutation(len(liste_patients))
    else: 
        if classes[-2:] == 'WC':
            if classes == 'TWOWC':
                name = "ordre_patientsWCT.npy"
            else :
                name = "ordre_patientsWCY.npy"   
        else:
            name = "ordre_patients.npy"
                
        if os.path.exists(name):
            rnd_data = np.load(name)
        else:
            rnd_data = np.random.permutation(len(liste_patients))
            np.save(name,rnd_data)
    if classes == 'YEAR':   
        noC = 7
    elif classes == 'YEARWC':
        noC = 6
    elif classes == 'TWO':
        noC = 3
    else:
        noC = 2
    
    C= []
    for i in range(noC):
        C.append([])
    
    for i,n in enumerate(rnd_data):
        C[y_grp.iloc[n,0]].append(n)

    if classes[-2:] == 'WC':
            if classes == 'TWOWC':
                name1, name2, name3 = "AtrWCT.npy","AteWCT.npy","AvalWCT.npy" 
            else :
                name1, name2, name3 = "AtrWCY.npy","AteWCY.npy","AvalWCY.npy"  
    else:
        name1, name2, name3 = "Atr.npy","Ate.npy","Aval.npy"
                
    if os.path.exists(name1) and os.path.exists(name2) and os.path.exists(name3) and change != True:        
            Atr= np.load(name1)
            Ate = np.load(name2)
            Aval = np.load(name3)
    else:
        Atr, Ate, Aval  =[],[],[]
        for j in range(len(C)):
            A = int(0.5*len(C[j]))
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
            
   
    Atr9=[]
    for j in range(len(Atr)):
        for i in range(9):
            Atr9.append(Atr[j]*9+i)
    Ate9=[]
    for j in range(len(Ate)):
        for i in range(9):
            Ate9.append(Ate[j]*9+i)
    Aval9=[]
    for j in range(len(Aval)):
        for i in range(9):
            Aval9.append(Aval[j]*9+i) 
    Atr9 = np.random.permutation(Atr9)
    Ate9 = np.random.permutation(Ate9)
    Aval9 = np.random.permutation(Aval9)
    names_train, names_val, names_test=[],[],[]
    #train
    for j in Atr9:
        k=int(j/9)
        names_train.append([j,liste_patients[k]])
    for j in Aval9:            
        k=int(j/9)
        names_val.append([j,liste_patients[k]])
    for j in Ate9:
        k=int(j/9)
        names_test.append([j,liste_patients[k]])
    if classes[-2:] != 'WC':
        names_trainDF=pd.DataFrame(names_train)
        names_trainDF.to_csv('names_train.csv')
        np.save("names_train",names_train)
    
        names_valDF=pd.DataFrame(names_val)
        names_valDF.to_csv('names_val.csv')
        np.save("names_val",names_val)
    
        names_testDF=pd.DataFrame(names_test)
        names_testDF.to_csv('names_test.csv')
        np.save("names_test",names_test)
    else:
        if classes == 'TWOWC':
            names_trainDF=pd.DataFrame(names_train)
            names_trainDF.to_csv('names_trainWCT.csv')
            np.save("names_trainWCT",names_train)
        
            names_valDF=pd.DataFrame(names_val)
            names_valDF.to_csv('names_valWCT.csv')
            np.save("names_valWCT",names_val)
        
            names_testDF=pd.DataFrame(names_test)
            names_testDF.to_csv('names_testWCT.csv')
            np.save("names_testWCT",names_test)
        else:
            names_trainDF=pd.DataFrame(names_train)
            names_trainDF.to_csv('names_trainWCY.csv')
            np.save("names_trainWCT",names_train)
        
            names_valDF=pd.DataFrame(names_val)
            names_valDF.to_csv('names_valWCY.csv')
            np.save("names_valWCY",names_val)
        
            names_testDF=pd.DataFrame(names_test)
            names_testDF.to_csv('names_testWCY.csv')
            np.save("names_testWCY",names_test)
            
    xtrain, xtest ,ytrain , ytest, xval, yval =[],[],[],[],[],[]
    #train           
    for j in range(len(names_train)):
        if np.shape(y_grp)[1]!=2:
            ytrain.append(y_grp.loc[names_train[j][1],'group'])
        else:
            ytrain.append([y_grp.loc[names_train[j][1],'event'],y_grp.loc[names_train[j][1],'time']])
        xtrain.append(liste_interpolate[names_train[j][0]])

    for j in range(len(names_val)):
        if np.shape(y_grp)[1]!=2:
            yval.append(y_grp.loc[names_val[j][1],'group'])
        else:
            yval.append([y_grp.loc[names_val[j][1],'event'],y_grp.loc[names_val[j][1],'time']])
        xval.append(liste_interpolate[names_val[j][0]])

    for j in range(len(names_test)):
        if np.shape(y_grp)[1]!=2:
            ytest.append(y_grp.loc[names_test[j][1],'group'])
        else:
            ytest.append([y_grp.loc[names_test[j][1],'event'],y_grp.loc[names_test[j][1],'time']])
        xtest.append(liste_interpolate[names_test[j][0]])

    
    xtrain = np.asarray(xtrain, dtype=np.float32)
    xtest = np.asarray(xtest, dtype=np.float32)
    xval = np.asarray(xval, dtype=np.float32)

    size = xtest[0][0].size
    xtrain = xtrain.reshape(xtrain.shape[0],size,size,1)
    xtest = xtest.reshape(xtest.shape[0],size,size,1)
    xval = xval.reshape(xval.shape[0],size,size,1)
 
    yval = np.asarray(yval)
    ytest = np.asarray(ytest)
    ytrain = np.asarray(ytrain)
   
    if np.shape(y_grp)[1]!=2:
        #one-hot encode target column
        ytrain = to_categorical(ytrain)
        ytest = to_categorical(ytest)
        yval = to_categorical(yval)
    
    return ytrain, ytest, yval, xtrain, xtest, xval, size
