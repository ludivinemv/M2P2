# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:29:25 2021

@author: ludiv
"""

def extract_dataIm(CrossVal,liste_ref,doss,PATH,sess,cl,ratess,interpolSize,RR,reload,spp,npatch,D3,loss):   
    if CrossVal != True:
        [rtrain, rtest,rval] = dpr.BalancedSeparation(liste_ref,doss,rates=[0.5,0.25,0.25],classe='surv',change=False) 
    else:
        RR= dpr.separation(doss,liste_ref,rates=ratess,cl = cl)
        Xsurv,Ysurv,Msurv, Rsurv =  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',sess=sess)  

        for k in range(len(RR)):
            Xsurv[k], Ysurv[k], Msurv[k],Rsurv[k] = dpr.shuffle_and_interpolate(Xsurv[k], Ysurv[k], Rsurv[k],interpolSize,interStatue=False,liste_mask = Msurv[k],sess=sess)
    
    """ ********************3D SURVIVAL MM DATA *********************************"""
    if CrossVal != True:
        ytrainsurv3D, ytestsurv3D, yvalsurv3D, xtrainsurv3D, xtestsurv3D, xvalsurv3D, reftr, refte, refv, mtrain3D, mtest3D, mval3D =  mds.PrepData(rtrain, rtest, rval,PATH,classes='surv',D3=True,method='all')  
        xtestsurv3D, ytestsurv3D, mtestsurv3D = dpr.shuffle_and_interpolate(xtestsurv3D, ytestsurv3D,interpolSize,interStatue=False,liste_mask = mtest3D,sess=sess)
        xtrainsurv3D, ytrainsurv3D, mtrainsurv3D = dpr.shuffle_and_interpolate(xtrainsurv3D, ytrainsurv3D,interpolSize,interStatue = False,liste_mask =mtrain3D,sess=sess)
        xvalsurv3D, yvalsurv3D, mvalsurv3D = dpr.shuffle_and_interpolate(xvalsurv3D, yvalsurv3D,interpolSize,interStatue = False,liste_mask =mval3D,sess=sess)
    else:
        Xsurv3D,Ysurv3D,Msurv3D,Rsurv3D =  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',D3=True,method='all',sess=sess)  
        for k in range(len(RR)):
            Xsurv3D[k], Ysurv3D[k], Msurv3D[k], Rsurv3D[k] = dpr.shuffle_and_interpolate(Xsurv3D[k], Ysurv3D[k], Rsurv3D[k],interpolSize,interStatue=False,liste_mask = Msurv3D[k],sess=sess)
   
    """ ********************  without spp SURVIVAL MM DATA with one patch*********************************"""
    if CrossVal != True:
        ytrainsurvONEBOX, ytestsurvONEBOX, yvalsurvONEBOX, xtrainsurvONEBOX, xtestsurvONEBOX, xvalsurvONEBOX, reftr, refte, refv, mtrainONEBOX, mtestONEBOX, mvalONEBOX =  mds.PrepData(rtrain, rtest, rval,PATH,reload=reload,classes='surv',npatch=1,method='box')  
        xtestsurvONEBOX, ytestsurvONEBOX, mtestsurvONEBOX = dpr.shuffle_and_interpolate(xtestsurvONEBOX, ytestsurvONEBOX,interpolSize,interStatue=False,liste_mask = mtestONEBOX)
        xtrainsurvONEBOX, ytrainsurvONEBOX, mtrainsurvONEBOX = dpr.shuffle_and_interpolate(xtrainsurvONEBOX, ytrainsurvONEBOX,interpolSize,interStatue = False,liste_mask =mtrainONEBOX)
        xvalsurvONEBOX, yvalsurvONEBOX, mvalsurvONEBOX = dpr.shuffle_and_interpolate(xvalsurvONEBOX, yvalsurvONEBOX,interpolSize,interStatue = False,liste_mask =mvalONEBOX)
    else:
        XsurvONEBOX,YsurvONEBOX,MsurvONEBOX,RsurvONEBOX =  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',npatch=1,method='box',sess=sess)  
        for k in range(len(RR)):
            XsurvONEBOX[k], YsurvONEBOX[k], MsurvONEBOX[k],RsurvONEBOX[k] = dpr.shuffle_and_interpolate(XsurvONEBOX[k],YsurvONEBOX[k], RsurvONEBOX[k],interpolSize,interStatue=False,liste_mask = MsurvONEBOX[k],sess=sess)
    
    
    """ ********************SURVIVAL MM DATA with three patch*********************************"""
    if CrossVal != True:
        ytrainsurvTHREEBOX, ytestsurvTHREEBOX, yvalsurvTHREEBOX, xtrainsurvTHREEBOX, xtestsurvTHREEBOX, xvalsurvTHREEBOX, reftr, refte, refv, mtrainTHREEBOX, mtestTHREEBOX, mvalTHREEBOX =  mds.PrepData(rtrain, rtest, rval,PATH,reload=reload,classes='surv',npatch=3,method='box')  
        xtestsurvTHREEBOX, ytestsurvTHREEBOX, mtestsurvTHREEBOX = dpr.shuffle_and_interpolate(xtestsurvTHREEBOX, ytestsurvTHREEBOX,interpolSize,interStatue=False,liste_mask = mtestTHREEBOX,sess=sess)
        xtrainsurvTHREEBOX, ytrainsurvTHREEBOX, mtrainsurvTHREEBOX = dpr.shuffle_and_interpolate(xtrainsurvTHREEBOX, ytrainsurvTHREEBOX,interpolSize,interStatue = False,liste_mask =mtrainTHREEBOX,sess=sess)
        xvalsurvTHREEBOX, yvalsurvTHREEBOX, mvalsurvTHREEBOX = dpr.shuffle_and_interpolate(xvalsurvTHREEBOX, yvalsurvTHREEBOX,interpolSize,interStatue = False,liste_mask =mvalTHREEBOX,sess=sess)
    else:
        XsurvTHREEBOX,YsurvTHREEBOX,MsurvTHREEBOX,RsurvTHREEBOX =  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',npatch=3,method='box',sess=sess)  
        for k in range(len(RR)):
            XsurvTHREEBOX[k], YsurvTHREEBOX[k], MsurvTHREEBOX[k], RsurvTHREEBOX[k] = dpr.shuffle_and_interpolate(XsurvTHREEBOX[k], YsurvTHREEBOX[k], RsurvTHREEBOX[k],interpolSize,interStatue=False,liste_mask = MsurvTHREEBOX[k],sess=sess)
    
    """ ******************** without spp SURVIVAL MM DATA *********************************"""
    
    if CrossVal != True:
        ytrainsurvBOX, ytestsurvBOX, yvalsurvBOX, xtrainsurvBOX, xtestsurvBOX, xvalsurvBOX, reftr, refte, refv, mtrainBOX, mtestBOX, mvalBOX =  mds.PrepData(rtrain, rtest, rval,PATH,reload=reload,classes='surv',method ='box')  
        xtestsurvBOX, ytestsurvBOX, mtestsurvBOX = dpr.shuffle_and_interpolate(xtestsurvBOX, ytestsurvBOX,interpolSize,interStatue=False,liste_mask = mtestBOX)
        xtrainsurvBOX, ytrainsurvBOX, mtrainsurvBOX = dpr.shuffle_and_interpolate(xtrainsurvBOX, ytrainsurvBOX,interpolSize,interStatue = False,liste_mask =mtrainBOX)
        xvalsurvBOX, yvalsurvBOX, mvalsurvBOX = dpr.shuffle_and_interpolate(xvalsurvBOX, yvalsurvBOX,interpolSize,interStatue = False,liste_mask =mvalBOX)
    else:
        XsurvBOX,YsurvBOX,MsurvBOX,RsurvBOX =  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',method ='box',sess=sess)  
        for k in range(len(RR)):
            XsurvBOX[k], YsurvBOX[k], MsurvBOX[k], RsurvBOX[k] = dpr.shuffle_and_interpolate(XsurvBOX[k], YsurvBOX[k], RsurvBOX[k],interpolSize,interStatue=False,liste_mask = MsurvBOX[k],sess=sess)
        # """ ******************** without spp SURVIVAL MM DATA 3D *********************************"""
    if CrossVal != True:
        ytrainsurvBOX3D, ytestsurvBOX3D, yvalsurvBOX3D, xtrainsurvBOX3D, xtestsurvBOX3D, xvalsurvBOX3D, reftr, refte, refv, mtrainBOX3D, mtestBOX3D, mvalBOX3D =  mds.PrepData(rtrain, rtest, rval,PATH,reload=reload,classes='surv',method ='box',D3=True)  
        xtestsurvBOX3D, ytestsurvBOX3D, mtestsurvBOX3D = dpr.shuffle_and_interpolate(xtestsurvBOX3D, ytestsurvBOX3D,interpolSize,interStatue=False,liste_mask = mtestBOX3D)
        xtrainsurvBOX3D, ytrainsurvBOX3D, mtrainsurvBOX3D = dpr.shuffle_and_interpolate(xtrainsurvBOX3D, ytrainsurvBOX3D,interpolSize,interStatue = False,liste_mask =mtrainBOX3D)
        xvalsurvBOX3D, yvalsurvBOX3D, mvalsurvBOX3D = dpr.shuffle_and_interpolate(xvalsurvBOX3D, yvalsurvBOX3D,interpolSize,interStatue = False,liste_mask =mvalBOX3D)
    else:
        XsurvBOX3D,YsurvBOX3D,MsurvBOX3D ,RsurvBOX3D=  mds.PrepDataCrossVal(RR,PATH,reload=reload,classes='surv',method ='box',D3=True,sess=sess)  
        for k in range(len(RR)):
            XsurvBOX3D[k], YsurvBOX3D[k], MsurvBOX3D[k], RsurvBOX3D[k] = dpr.shuffle_and_interpolate(XsurvBOX3D[k], YsurvBOX3D[k], RsurvBOX3D[k],interpolSize,interStatue=False,liste_mask = MsurvBOX3D[k],sess=sess)
    
    """ ******************** without spp 2.5 SURVIVAL MM DATA *********************************"""
    if CrossVal != True:
        xtestsurvBOX25,  mtestsurvBOX25, ytestsurvBOX25 = mds.Prep25(xtestsurvBOX3D, mtestsurvBOX3D,ytestsurvBOX3D)
        xtrainsurvBOX25,  mtrainsurvBOX25, ytrainsurvBOX25= mds.Prep25(xtrainsurvBOX3D, mtrainsurvBOX3D,ytrainsurvBOX3D)
        xvalsurvBOX25,  mvalsurvBOX25 , yvalsurvBOX25= mds.Prep25(xvalsurvBOX3D, mvalsurvBOX3D,yvalsurvBOX3D)
    else:    
        XsurvBOX25,YsurvBOX25,MsurvBOX25,RsurvBOX25 = [[] for i in range(len(RR))], [[] for i in range(len(RR))],[[] for i in range(len(RR))],[[] for i in range(len(RR))]
    
        for k in range(len(RR)):
            XsurvBOX25[k], MsurvBOX25[k] , YsurvBOX25[k], RsurvBOX25[k]= mds.Prep25(XsurvBOX3D[k], MsurvBOX3D[k],YsurvBOX3D[k],RsurvBOX3D[k])
    
    if CrossVal != True:
        if D3== False:
            if spp!=False :
                if npatch==9:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurv,mtrainsurv, ytrainsurv, xvalsurv, mvalsurv,yvalsurv,xtestsurv,mtestsurv,ytestsurv 
                elif npatch==3:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvTHREE,mtrainsurvTHREE, ytrainsurvTHREE, xvalsurvTHREE, mvalsurvTHREE,yvalsurvTHREE,xtestsurvTHREE,mtestsurvTHREE,ytestsurvTHREE 
                else:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvONE,mtrainsurvONE, ytrainsurvONE, xvalsurvONE, mvalsurvONE,yvalsurvONE,xtestsurvONE,mtestsurvONE,ytestsurvONE 
            else:
                if npatch==9:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvBOX, mtrainsurvBOX, ytrainsurvBOX, xvalsurvBOX,mvalsurvBOX, yvalsurvBOX,xtestsurvBOX,mtestsurvBOX, ytestsurvBOX
                elif npatch==3:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvTHREEBOX,mtrainsurvTHREEBOX, ytrainsurvTHREEBOX, xvalsurvTHREEBOX, mvalsurvTHREEBOX,yvalsurvTHREEBOX,xtestsurvTHREEBOX,mtestsurvTHREEBOX,ytestsurvTHREEBOX 
                else:
                    xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvONEBOX,mtrainsurvONEBOX, ytrainsurvONEBOX, xvalsurvONEBOX, mvalsurvONEBOX,yvalsurvONEBOX,xtestsurvONEBOX,mtestsurvONEBOX,ytestsurvONEBOX 
        elif D3=='25':
            if spp!= False :
                xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurv25,mtrainsurv25, ytrainsurv25, xvalsurv25, mvalsurv25,yvalsurv25,xtestsurv25,mtestsurv25,ytestsurv25 
                
            else:
                xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvBOX25, mtrainsurvBOX25, ytrainsurvBOX25, xvalsurvBOX25,mvalsurvBOX25, yvalsurvBOX25,xtestsurvBOX25,mtestsurvBOX25, ytestsurvBOX25

        else:
            if spp!=False :
                xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurv3D, mtrainsurv3D, ytrainsurv3D, xvalsurv3D,mvalsurv3D, yvalsurv3D,xtestsurv3D,mtestsurv3D, ytestsurv3D
            else:
                xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest = xtrainsurvBOX3D, mtrainsurvBOX3D, ytrainsurvBOX3D, xvalsurvBOX3D,mvalsurvBOX3D, yvalsurvBOX3D,xtestsurvBOX3D,mtestsurvBOX3D, ytestsurvBOX3D
        
        if loss=='discret' or loss == 'RankAnddiscret':
            ytrainT,yvalT,ytestT =  ytrain,yval,ytest
            ytrain,yval,ytest = lss.FromTimeToLong(ytrain,breaks), lss.FromTimeToLong(yval,breaks),lss.FromTimeToLong(ytest,breaks)
        
            return xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest, ytrainT,yvalT,ytestT, rtrain, rtest,rval
        else :
            return xtrain,mtrain,ytrain,xval,mval,yval,xtest,mtest,ytest , rtrain, rtest,rval
    else: 
        if D3== False:
            if spp!=False :
                if npatch==9:
                    Xt,Yt,Mt,Rt= Xsurv, Ysurv, Msurv,Rsurv
                elif npatch==3:
                    Xt,Yt,Mt,Rt = XsurvTHREE, YsurvTHREE, MsurvTHREE, RsurvTHREE
                else:
                    Xt,Yt,Mt,Rt = XsurvONE, YsurvONE, MsurvONE,RsurvONE
            else:
                if npatch==9:
                    Xt,Yt,Mt,Rt = XsurvBOX, YsurvBOX, MsurvBOX,RsurvBOX
                elif npatch==3:
                    Xt,Yt,Mt,Rt = XsurvTHREEBOX, YsurvTHREEBOX, MsurvTHREEBOX,RsurvTHREEBOX
                else:
                    Xt,Yt,Mt,Rt = XsurvONEBOX, YsurvONEBOX, MsurvONEBOX,RsurvONEBOX
        elif D3=='25':
            if spp!= False :
                Xt,Yt,Mt,Rt = Xsurv25, Ysurv25, Msurv25, Rsurv25
            else:
                Xt,Yt,Mt,Rt = XsurvBOX25, YsurvBOX25, MsurvBOX25, RsurvBOX25
            
        else:
            if spp!=False :
                Xt,Yt,Mt,Rt = Xsurv3D, Ysurv3D, Msurv3D, Rsurv3D
            else:
                Xt,Yt,Mt,Rt = XsurvBOX3D, YsurvBOX3D, MsurvBOX3D, RsurvBOX3D
        return Xt,Yt,Mt,Rt, RR