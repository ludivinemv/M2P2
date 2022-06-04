# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
"""


from __future__ import print_function
import numpy as np
import keras.backend as K
import tensorflow as tf

import keras



def discretcox(y_true, y_pred):
    """
    Returns
        metrics to evaluate the evolution of the discret cox loss when it is combined with another loss
    """
    n_intervals=7 # choosed number of intervals n_intervals = len(breaks)
    cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
    uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
    return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood

def tf_FromLongToTime(y,num_iter,breaks=None,size=1):
    """
    Input
         y : Long input (discret values of y)
         breaks : If breaks = list of breaks in the interval np.arange(0,stride*n_intervals,stride)
    Returns
        metrics to evaluate the evolution of the discret cox loss when it is combined with another loss
    """
    stride = 365 #choosed size of the intervals
    if breaks == None: 
        breaks =tf.linspace(tf.cast(0,tf.float32),tf.cast((num_iter-1)*stride,tf.float32),num_iter)
        breaks = tf.cast(breaks,dtype = tf.int32)
    if size == 1:
        binary = tf.cast(  tf.round(y),dtype=tf.int32)
        cumprod = tf.cast( tf.cumprod(binary,axis=1),dtype=tf.int32)
        cumprod = tf.cast(cumprod,dtype=tf.int32)
        ra = tf.range(tf.shape(y)[0])
        yTime = tf.map_fn(lambda x: tf.cast(breaks[tf.reduce_sum(cumprod[x,:])-1],dtype=tf.int32), ra,name='map2224') 
        return yTime
    else:
        yt = y[:,0:num_iter]
        ye = y[:,num_iter:num_iter*2]
        binary = tf.cast(  tf.round(yt),dtype=tf.int32)
        cumprod = tf.cast( tf.cumprod(binary,axis=1),dtype=tf.int32)
        cumprod = tf.cast(cumprod,dtype=tf.int32)        
        ra = tf.range(tf.shape(y)[0])
        yTime = tf.map_fn(lambda x: tf.cast(breaks[tf.reduce_sum(cumprod[x,:])-1],dtype=tf.int32), ra,name='map2225') 
        yE = tf.map_fn(lambda x: tf.cast(tf.reduce_max(ye[x,:]),dtype=tf.int32) , tf.cast(tf.range(tf.shape(ye)[0]),dtype=tf.int32),name='map123')
        return yTime,yE
    
def RankingAndDiscret(n_intervals):
  """Create custom Keras loss function for neural network survival model. 
  Arguments
      n_intervals: the number of survival time intervals
  Returns
      Custom loss function that can be used with Keras
  """
  def loss(y_true, y_pred):
    """
    Required to have only 2 arguments by Keras.
    Arguments
        y_true: Tensor.
          First half of the values is 1 if individual survived that interval, 0 if not.
          Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
          See make_surv_array function.
        y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
    Returns
        Vector of losses for this minibatch.
    """
    cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
    uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
    ldiscret = K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
    
    time, event = tf_FromLongToTime(y_true,n_intervals,size = 2)
    score = tf_FromLongToTime(y_pred,n_intervals,size = 1)
    score = tf.cast(score, tf.float32)
    event=tf.cast(event, tf.float32)
    time=tf.cast(time, tf.float32)
    n= tf.shape(score)[0]
    c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1 
    on =tf.ones(tf.shape(c1),dtype=tf.float32)
    ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
            
    c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
    c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
    c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
    c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
    comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
    Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
    P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
    I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
    at = tf.square(Y - P)*I 
    lrank =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
    a=tf.expand_dims(lrank, axis=0)
    b = tf.expand_dims(n, axis=0)
    
    lrankExt = tf.tile(a, b)
    loss = ldiscret + tf.cast(lrankExt*2e-5,tf.float32)
    return  loss
  return loss

def rankL(y_true,y_pred):
    """
    Returns
        metrics to evaluate the evolution of the ranking loss when it is combined with another loss
    Attributs
        y_true : true values (times, censorship)
        y_pred : predicted values
    """
    n_intervals=7
    time, event = tf_FromLongToTime(y_true,n_intervals,size = 2)
    score = tf_FromLongToTime(y_pred,n_intervals,size = 1)
    score = tf.cast(score, tf.float32)
    event=tf.cast(event, tf.float32)
    time=tf.cast(time, tf.float32)
    n= tf.shape(score)[0]
    c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1 
    on =tf.ones(tf.shape(c1),dtype=tf.float32)
    ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
            
    c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
    c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
    c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
    c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
    comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
    Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
    P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
    I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
    at = tf.square(Y - P)*I 
    lrank =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
    a=tf.expand_dims(lrank, axis=0)
    b = tf.expand_dims(n, axis=0)
    
    lrankExt = tf.tile(a, b)
    return lrankExt


def tf_cindexR(y_true, y_pred):# for risk cindex
    """
    Returns
        c-index when prediction is a risk 
        tensorflow version
    Attributs
        y_true : true values (times, censorship)
        y_pred : predicted values
    """
    yp=tf.cast(y_pred[:,0],dtype=tf.float32)
    yt,ye = tf.cast(y_true[:,1],dtype=tf.float32),tf.cast(y_true[:,0],dtype=tf.float32)
    ge = tf.subtract(tf.expand_dims(ye, -1), ye,name='ge')  #(B,B)
    in1=tf.expand_dims(ye, -1,name='in1') #(B,1)
    in2=tf.add(in1, ye,name='in2') #(B,B)
    in3=tf.equal(in2,2,name='in3') #(B,B)   
    ge = tf.where(in3,tf.ones(tf.shape(ge),dtype=tf.float32)*2,tf.cast(ge,dtype=tf.float32),name='ge2') #(B,B)
    in5=tf.subtract(tf.expand_dims(yt, -1), yt,name='in5')#(B,B)
    g = tf.where(in5<0,-1*tf.ones(tf.shape(in5),dtype=tf.float32),in5) #(B,B)
    g = tf.where(g>0,tf.ones(tf.shape(ge),dtype=tf.float32),g)#(B,B)
    in7=tf.cast(tf.subtract(tf.expand_dims(yp, -1), yp),dtype=tf.float32)
    in8=-1*tf.ones(tf.shape(in7),dtype=tf.float32)#(B,B)
    gp = tf.where(in7<0,in8,in7)#(B,B)
    zer=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    on=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    gp = tf.where(gp>0,on,gp) #(B,B)
    in15=tf.multiply(ge,g) #(B,B)
    zeri=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    oni=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    in15= tf.reshape(in15,(tf.shape(in15)[0],tf.shape(in15)[1]))
    pair = tf.where( tf.equal(in15,1), zeri,oni) #(B,B) select 4
    pair = tf.where(tf.equal(in15,0) , zeri, pair) #(B,B) select 5
    a= tf.cast(tf.equal(g,0),dtype=tf.float32) #determine ou cest egal
    c= tf.cast(tf.equal(tf.multiply(a,ge),0),dtype=tf.float32) 
    pair =  tf.multiply(pair, tf.cast(c,dtype=tf.float32)) #(B,B)
    in10=tf.eye(tf.shape(pair)[1])
    pair= (1-tf.cast(in10,dtype=tf.float32))*pair #(B,B)
    permissible= tf.reduce_sum(pair)
    dif = tf.where(  tf.equal(g,0), zeri,oni) # la ou ti!=tj #(B,B) select 6
    in11=tf.multiply(tf.cast(g,dtype=tf.float32),gp)
    in12=tf.cast(tf.equal(gp,0.0), tf.float32)
    sum = tf.cast(tf.multiply( in11,tf.cast(dif,dtype=tf.float32))< 0.0, tf.float32)*1 + 0.5*tf.cast(tf.multiply(in12,tf.cast(dif,dtype=tf.float32)), tf.float32)
    a=tf.cast(tf.equal(g,0), tf.float32) #Ti=Tj
    b=tf.cast(tf.equal(gp,0.0), tf.float32) #pj=pi
    d=tf.cast(tf.equal(ge,2), tf.float32) # both are death
    c=tf.where( tf.equal(gp,0.0), zer,on)  #different prediction
    sum=sum+ a*b*d*1 + a*c*d*0.5    
    d2=tf.cast(tf.where( tf.equal(ge,2), zeri,oni),dtype=tf.float32)
    ee = tf.where(tf.multiply(a,tf.cast(ge,dtype=tf.float32))<0, tf.ones(tf.shape(ge),dtype=tf.float32),tf.zeros(tf.shape(ge),dtype=tf.float32))
    ee = tf.where(tf.equal(tf.multiply(a,tf.cast(ge,dtype=tf.float32)),2.0), tf.ones(tf.shape(ee),dtype=tf.float32),ee)
    in16=tf.cast(tf.equal(tf.multiply(gp,tf.cast(ge,dtype=tf.float32)),1.0), tf.float32)
    sum = sum+1*tf.multiply(a,d2)*ee+0.5*tf.multiply(a,d2)*in16
    in13= tf.reduce_sum(sum*tf.cast(pair,dtype=tf.float32),name='in13')
    in14=in13/tf.cast(permissible,dtype=tf.float32)
    final= tf.where(tf.equal(tf.cast(permissible,dtype=tf.float32),0.0), 0.0, in14)
    return final

# y_pred=tf.constant([[10],[20],[30],[40],[50]])
# y_true=tf.constant([[1,10],[1,20],[1,30],[1,40],[1,50]])

def tf_cindexT(y_true, y_pred):# for time cindex
    """
    Returns
        c-index when prediction is a time 
        tensorflow version
    Attributs
        y_true : true values (times, censorship)
        y_pred : predicted values
    """
    yp=tf.cast(y_pred[:,0],dtype=tf.float32)
    yt,ye = tf.cast(y_true[:,1],dtype=tf.float32),tf.cast(y_true[:,0],dtype=tf.float32)
    ge = tf.subtract(tf.expand_dims(ye, -1), ye,name='ge')  #(B,B)
    in1=tf.expand_dims(ye, -1,name='in1') #(B,1)
    in2=tf.add(in1, ye,name='in2') #(B,B)
    in3=tf.equal(in2,2,name='in3') #(B,B)   
    ge = tf.where(in3,tf.ones(tf.shape(ge),dtype=tf.float32)*2,tf.cast(ge,dtype=tf.float32),name='ge2') #(B,B)
    in5=tf.subtract(tf.expand_dims(yt, -1), yt,name='in5')#(B,B)
    g = tf.where(in5<0,-1*tf.ones(tf.shape(in5),dtype=tf.float32),in5) #(B,B)
    g = tf.where(g>0,tf.ones(tf.shape(ge),dtype=tf.float32),g)#(B,B)
    in7=tf.cast(tf.subtract(tf.expand_dims(yp, -1), yp),dtype=tf.float32)
    in8=-1*tf.ones(tf.shape(in7),dtype=tf.float32)#(B,B)
    gp = tf.where(in7<0,in8,in7)#(B,B)
    zer=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    on=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    gp = tf.where(gp>0,on,gp) #(B,B)
    in15=tf.multiply(ge,g) #(B,B)
    zeri=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    oni=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    in15= tf.reshape(in15,(tf.shape(in15)[0],tf.shape(in15)[1]))
    pair = tf.where( tf.equal(in15,1), zeri,oni) #(B,B) select 4
    pair = tf.where(tf.equal(in15,0) , zeri, pair) #(B,B) select 5
    a= tf.cast(tf.equal(g,0),dtype=tf.float32) #determine ou cest egal
    c= tf.cast(tf.equal(tf.multiply(a,ge),0),dtype=tf.float32) 
    pair =  tf.multiply(pair, tf.cast(c,dtype=tf.float32)) #(B,B)
    in10=tf.eye(tf.shape(pair)[1])
    pair= (1-tf.cast(in10,dtype=tf.float32))*pair #(B,B)
    permissible= tf.reduce_sum(pair)
    dif = tf.where(  tf.equal(g,0), zeri,oni) # la ou ti!=tj #(B,B) select 6
    in11=tf.multiply(tf.cast(g,dtype=tf.float32),gp)
    in12=tf.cast(tf.equal(gp,0.0), tf.float32)
    sum = tf.cast(tf.multiply( in11,tf.cast(dif,dtype=tf.float32))> 0.0, tf.float32)*1 + 0.5*tf.cast(tf.multiply(in12,tf.cast(dif,dtype=tf.float32)), tf.float32)
    a=tf.cast(tf.equal(g,0), tf.float32) #Ti=Tj
    b=tf.cast(tf.equal(gp,0.0), tf.float32) #pj=pi
    d=tf.cast(tf.equal(ge,2), tf.float32) # both are death
    c=tf.where( tf.equal(gp,0.0), zer,on)  #different prediction
    sum=sum+ a*b*d*1 + a*c*d*0.5    
    d2=tf.cast(tf.where( tf.equal(ge,2), zeri,oni),dtype=tf.float32)
    ee = tf.where(tf.multiply(a,tf.cast(ge,dtype=tf.float32))<0, tf.ones(tf.shape(ge),dtype=tf.float32),tf.zeros(tf.shape(ge),dtype=tf.float32))
    ee = tf.where(tf.equal(tf.multiply(a,tf.cast(ge,dtype=tf.float32)),2.0), tf.ones(tf.shape(ee),dtype=tf.float32),ee)
    in16=tf.cast(tf.equal(tf.multiply(gp,tf.cast(ge,dtype=tf.float32)),1.0), tf.float32)
    sum = sum+1*tf.multiply(a,d2)*ee+0.5*tf.multiply(a,d2)*in16
    in13= tf.reduce_sum(sum*tf.cast(pair,dtype=tf.float32),name='in13')
    in14=in13/tf.cast(permissible,dtype=tf.float32)
    final= tf.where(tf.equal(tf.cast(permissible,dtype=tf.float32),0.0), 0.0, in14)
    return final


    
def tf_cindexTD(y_true, y_pred):# discret time cindex 
    """
    Returns
        c-index when prediction is a discret time 
        tensorflow version
    Attributs
        y_true : true values (times, censorship)
        y_pred : predicted values
    """   
    yt= tf_FromLongToTime(y_true,tf.shape(y_pred)[1],size=2)[0]
    ye= tf_FromLongToTime(y_true,tf.shape(y_pred)[1],size=2)[1]  
    yp= tf_FromLongToTime(y_pred,tf.shape(y_pred)[1],size=1)  
    yp = -tf.cast(yp,dtype=tf.float32)
    ye = tf.cast(ye,dtype=tf.float32)
    yt = tf.cast(yt,dtype=tf.float32)
    ge = tf.subtract(tf.expand_dims(ye, -1), ye,name='ge')  #(B,B)
    in1=tf.expand_dims(ye, -1,name='in1') #(B,1)
    in2=tf.add(in1, ye,name='in2') #(B,B)
    in3=tf.equal(in2,2,name='in3') #(B,B)   
    ge = tf.where(in3,tf.ones(tf.shape(ge),dtype=tf.float32)*2,tf.cast(ge,dtype=tf.float32),name='ge2') #(B,B)
    in5=tf.subtract(tf.expand_dims(yt, -1), yt,name='in5')#(B,B)
    g = tf.where(in5<0,-1*tf.ones(tf.shape(in5),dtype=tf.float32),in5) #(B,B)
    g = tf.where(g>0,tf.ones(tf.shape(ge),dtype=tf.float32),g)#(B,B)
    in7=tf.cast(tf.subtract(tf.expand_dims(yp, -1), yp),dtype=tf.float32)
    in8=-1*tf.ones(tf.shape(in7),dtype=tf.float32)#(B,B)
    gp = tf.where(in7<0,in8,in7)#(B,B)
    zer=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    on=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    gp = tf.where(gp>0,on,gp) #(B,B)
    in15=tf.multiply(ge,g) #(B,B)
    zeri=tf.zeros(tf.shape(gp),dtype=tf.float32) #(B,B)
    oni=tf.ones(tf.shape(gp),dtype=tf.float32) #(B,B)
    in15= tf.reshape(in15,(tf.shape(in15)[0],tf.shape(in15)[1]))
    pair = tf.where( tf.equal(in15,1), zeri,oni) #(B,B) select 4
    pair = tf.where(tf.equal(in15,0) , zeri, pair) #(B,B) select 5
    a= tf.cast(tf.equal(g,0),dtype=tf.float32) #determine ou cest egal
    c= tf.cast(tf.equal(tf.multiply(a,ge),0),dtype=tf.float32) 
    pair =  tf.multiply(pair, tf.cast(c,dtype=tf.float32)) #(B,B)
    in10=tf.eye(tf.shape(pair)[1])
    pair= (1-tf.cast(in10,dtype=tf.float32))*pair #(B,B)
    permissible= tf.reduce_sum(pair)
    dif = tf.where(  tf.equal(g,0), zeri,oni) # la ou ti!=tj #(B,B) select 6
    in11=tf.multiply(tf.cast(g,dtype=tf.float32),gp)
    in12=tf.cast(tf.equal(gp,0.0), tf.float32)
    sum = tf.cast(tf.multiply( in11,tf.cast(dif,dtype=tf.float32))> 0.0, tf.float32)*1 + 0.5*tf.cast(tf.multiply(in12,tf.cast(dif,dtype=tf.float32)), tf.float32)
    a=tf.cast(tf.equal(g,0), tf.float32) #Ti=Tj
    b=tf.cast(tf.equal(gp,0.0), tf.float32) #pj=pi
    d=tf.cast(tf.equal(ge,2), tf.float32) # both are death
    c=tf.where( tf.equal(gp,0.0), zer,on)  #different prediction
    sum=sum+ a*b*d*1 + a*c*d*0.5    
    d2=tf.cast(tf.where( tf.equal(ge,2), zeri,oni),dtype=tf.float32)
    ee = tf.where(tf.multiply(a,tf.cast(ge,dtype=tf.float32))<0, tf.ones(tf.shape(ge),dtype=tf.float32),tf.zeros(tf.shape(ge),dtype=tf.float32))
    ee = tf.where(tf.equal(tf.multiply(a,tf.cast(ge,dtype=tf.float32)),2.0), tf.ones(tf.shape(ee),dtype=tf.float32),ee)
    in16=tf.cast(tf.equal(tf.multiply(gp,tf.cast(ge,dtype=tf.float32)),1.0), tf.float32)
    sum = sum+1*tf.multiply(a,d2)*ee+0.5*tf.multiply(a,d2)*in16
    in13= tf.reduce_sum(sum*tf.cast(pair,dtype=tf.float32),name='in13')
    in14=in13/tf.cast(permissible,dtype=tf.float32)
    final= tf.where(tf.equal(tf.cast(permissible,dtype=tf.float32),0.0), 0.0, in14)
    return final


def cindexR(yt,yp,ye):# for risk
    """
    Returns
        c-index when prediction is a risk 
        numpy version
    Attributs
        yt : true values
        yp : predicted values
        ye : censorship
    """
    # yp=yp[:,0]
    if len(np.shape(yp))==2: 
        yp=np.reshape(yp,yt.shape)

    ge = np.subtract(np.expand_dims(ye, -1), ye) 
    ge=np.where(np.add(np.expand_dims(ye, -1), ye)==2,2,ge)
    g = np.where(np.subtract(np.expand_dims(yt, -1), yt)<0,-1,np.subtract(np.expand_dims(yt, -1), yt))
    g = np.where(g>0,1,g)
    gp = np.where(np.subtract(np.expand_dims(yp, -1), yp)<0,-1,np.subtract(np.expand_dims(yp, -1), yp))
    gp = np.where(gp>0,1,gp)
    """
    **ge: si -1 alors b mort si 1 c'est a et si 0 aucun 
        si les deux mort: 2
    **g: -1 si B superieur, 1 si A sup, 0 si egaux
    **gp: -1 si B superieur, 1 si A sup, 0 si egaux

    si A<B et pas de mort:   O            -1 * 0
    +si A<B et B mort :       + 1          -1* -1
    =si A<B et A mort :       - 1          -1 * 1
    =si A<B et B mort et A mort:  -2     -1* 2
    =si B<A et B mort:        -1            1* -1
    +si B<A et A mort:         1            1 * 1
    =si B<A et A mort B mort:   2      1 * 2
    si B<A et pas de mort:   0             1 * 0

    **Omit those pairs whose shorter survival time is censored."""
    pair = np.where(ge*g ==1 , 0, 1)
    pair = np.where(ge*g ==0 , 0, pair)

    """**Omit pairs i and j if Ti =Tj unless at least one is a death."""
    a= np.where(g==0, 1,0) #determine ou cest egal
    """	a*ge determine si egal et au moins une censure= 1 ou -1
            si  egal et 2 morts: 2
            si pas egal : 0
            si egal et pas de mort 0"""
    c=np.where(a*ge==0, 1,0)
    pair =  np.multiply(pair, c)
    pair= (1-np.eye(np.shape(pair)[0],np.shape(pair)[1]))*pair
    """**Let Permissible denote the total number of permissible pairs."""
    permissible= np.sum(pair)

    """**For each permissible pair where Ti 6= Tj ,"""
    dif=np.array(g != 0.0, np.float32) # la ou ti!=tj
    """**count 1 if the shorter survival time has worse predicted outcome;
    si A<B et C<D: - * -
    si A<B et D<C: - * +
    si B<A et C<D: + * -
    si B<A et D<C: + * +
    si negatif = ok
    **count 0.5 if predicted outcomes are tied"""
    sum = np.array(g*gp*dif < 0.0, np.float32)*1 + 0.5*np.array(np.array(gp==0, np.float32)*dif, np.float32)

    """**where Ti=Tj and both are deaths, count 1 if predicted outcomes are tied,otherwise, count 0.5."""
    a=np.array(g==0, np.float32) #Ti=Tj
    b=np.array(gp==0, np.float32) #pj=pi
    d=np.array(ge==2, np.float32) # both are death
    c=np.array(gp!=0, np.float32) #different prediction
    sum=sum+ a*b*d*1 + a*c*d*0.5
    """where Ti = Tj , but not both are deaths, count 1 if the death has worse predicted outcome; otherwise, count 0.5"""
    d2=np.array(ge!=2, np.float32)
    ee = np.where(a*ge<0, 1,0)
    ee = np.where(a*ge==2, 1,ee)
    sum = sum+1*a*d2*ee+0.5*a*d2*np.array(gp*ge==1, np.float32)
    final= np.where(permissible == 0, 0.0, np.sum(sum*pair)/permissible)
    return final

def cindexT(yt,yp,ye):# for risk
    """
    Returns
        c-index when prediction is a time 
        numpy version
    Attributs
        yt : true values
        yp : predicted values
        ye : censorship
    """
    # yp=yp[:,0]
    yp=np.array(yp)
    yt=np.array(yt)
    ye=np.array(ye)
    if len(yp.shape)==2: 
        yp=np.reshape(yp,yt.shape)

    ge = np.subtract(np.expand_dims(ye, -1), ye) 
    ge=np.where(np.add(np.expand_dims(ye, -1), ye)==2,2,ge)
    g = np.where(np.subtract(np.expand_dims(yt, -1), yt)<0,-1,np.subtract(np.expand_dims(yt, -1), yt))
    g = np.where(g>0,1,g)
    gp = np.where(np.subtract(np.expand_dims(yp, -1), yp)<0,-1,np.subtract(np.expand_dims(yp, -1), yp))
    gp = np.where(gp>0,1,gp)
    pair = np.where(ge*g ==1 , 0, 1)
    pair = np.where(ge*g ==0 , 0, pair)
    a= np.where(g==0, 1,0) #determine ou cest egal
    c=np.where(a*ge==0, 1,0)
    pair =  np.multiply(pair, c)
    pair= (1-np.eye(np.shape(pair)[0],np.shape(pair)[1]))*pair
    permissible= np.sum(pair)
    dif=np.array(g != 0.0, np.float32) # la ou ti!=tj
    sum = np.array(g*gp*dif > 0.0, np.float32)*1 + 0.5*np.array(np.array(gp==0, np.float32)*dif, np.float32)
    """**where Ti=Tj and both are deaths, count 1 if predicted outcomes are tied,otherwise, count 0.5."""
    a=np.array(g==0, np.float32) #Ti=Tj
    b=np.array(gp==0, np.float32) #pj=pi
    d=np.array(ge==2, np.float32) # both are death
    c=np.array(gp!=0, np.float32) #different prediction
    sum=sum+ a*b*d*1 + a*c*d*0.5
    """where Ti = Tj , but not both are deaths, count 1 if the death has worse predicted outcome; otherwise, count 0.5"""
    d2=np.array(ge!=2, np.float32)
    ee = np.where(a*ge<0, 1,0)
    ee = np.where(a*ge==2, 1,ee)
    sum = sum+1*a*d2*ee+0.5*a*d2*np.array(gp*ge==1, np.float32)
    final= np.where(permissible == 0, 0.0, np.sum(sum*pair)/permissible)
    return final

#from sklearn.metrics import accuracy_score

def time_cindex(yt,yp,ye,order='R',cla = '3'):
    """
    Returns
        c-index when prediction is a time. Evaluation of the ordering of discret time (years,classes) instead of continious time (days) 
        numpy version
        
    Attributs
        yt : true values
        yp : predicted values
        ye : censorship
        order : 'R' = risk and 'T' = time
        cla = number of classes. If '3' " different classes", if 'classe' 7 different classes (one per year), if '0ouAutre' 2 classes with one at 0 years and one with all other times.
    """
    yp=np.array(yp)
    yt=np.array(yt)
    ye=np.array(ye)
    yp=np.reshape(yp,yt.shape)
    if cla == '3':
        th=2.2*365#(np.max(yt) - np.min(yt))//3
        yt1,yp1,ye1=yt[np.where(yt<th)],yp[np.where(yt<th)],ye[np.where(yt<th)]
        yt3,yp3,ye3=yt[np.where(yt>=th*2)],yp[np.where(yt>=th*2)],ye[np.where(yt>=th*2)]
        yt2,yp2,ye2=yt[np.where((yt>=th) & (yt<th*2 ))],yp[np.where((yt>=th) & (yt<th*2 ))],ye[np.where((yt>=th) & (yt<th*2 ))]
        if order == 'R':
            cindex1=cindexR(yt1,yp1,ye1)
            cindex2=cindexR(yt2,yp2,ye2)
            cindex3=cindexR(yt3,yp3,ye3)
        else:
            cindex1=cindexT(yt1,yp1,ye1)
            cindex2=cindexT(yt2,yp2,ye2)
            cindex3=cindexT(yt3,yp3,ye3)
        return [cindex1,cindex2,cindex3]
    elif cla == 'classe':
        th = 365
        yt0,yp0,ye0=yt[np.where(yt<th)],yp[np.where(yt<th)],ye[np.where(yt<th)]
        yt1,yp1,ye1=yt[np.where((yt>=th) & (yt<th*2 ))],yp[np.where((yt>=th) & (yt<th*2 ))],ye[np.where((yt>=th) & (yt<th*2 ))]
        yt2,yp2,ye2=yt[np.where((yt>=th) & (yt<th*3 ))],yp[np.where((yt>=th) & (yt<th*3 ))],ye[np.where((yt>=th) & (yt<th*3 ))]
        yt3,yp3,ye3=yt[np.where((yt>=th) & (yt<th*4 ))],yp[np.where((yt>=th) & (yt<th*4 ))],ye[np.where((yt>=th) & (yt<th*4 ))]
        yt4,yp4,ye4=yt[np.where((yt>=th) & (yt<th*5 ))],yp[np.where((yt>=th) & (yt<th*5 ))],ye[np.where((yt>=th) & (yt<th*5 ))]
        yt5,yp5,ye5=yt[np.where((yt>=th) & (yt<th*6 ))],yp[np.where((yt>=th) & (yt<th*6 ))],ye[np.where((yt>=th) & (yt<th*6 ))]
        yt6,yp6,ye6=yt[np.where(yt>=th*6)],yp[np.where(yt>=th*6)],ye[np.where(yt>=th*6)]
        if order == 'R':
            cindex0=cindexR(yt0,yp0,ye0)
            cindex1=cindexR(yt1,yp1,ye1)
            cindex2=cindexR(yt2,yp2,ye2)
            cindex3=cindexR(yt3,yp3,ye3)
            cindex4=cindexR(yt4,yp4,ye4)
            cindex5=cindexR(yt5,yp5,ye5)
            cindex6=cindexR(yt6,yp6,ye6)
        else:
            cindex0=cindexT(yt0,yp0,ye0)
            cindex1=cindexT(yt1,yp1,ye1)
            cindex2=cindexT(yt2,yp2,ye2)
            cindex3=cindexT(yt3,yp3,ye3)
            cindex4=cindexT(yt4,yp4,ye4)
            cindex5=cindexT(yt5,yp5,ye5)
            cindex6=cindexT(yt6,yp6,ye6)      
        return [cindex0,cindex1,cindex2,cindex3,cindex4,cindex5,cindex6]
    elif cla == '0ouAutre':
        th=365#(np.max(yt) - np.min(yt))//3
        yt0,yp0,ye0=yt[np.where(yt<th)],yp[np.where(yt<th)],ye[np.where(yt<th)]
        yt1,yp1,ye1=yt[np.where(yt>=th)],yp[np.where(yt>=th)],ye[np.where(yt>=th)]
        if order == 'R':
            cindex1=cindexR(yt1,yp1,ye1)
            cindex0=cindexR(yt0,yp0,ye0)
        else:
            cindex1=cindexT(yt1,yp1,ye1)
            cindex0=cindexT(yt0,yp0,ye0)
        return [cindex0,cindex1]
        


def FromTimeToLong(y,breaks):
    if y.shape[1]==2:
        return make_surv_array(y[:,1],y[:,0],breaks)#to_categorical(np.array(y[:,1]))
    else:
        return make_surv_array(y,np.ones((y.shape)),breaks)

def FromLongToTime(y,num_iter,breaks=None):
    stride = 365
    if breaks == None: 
        breaks =np.arange(num_iter)*stride
    if num_iter == len(y[0,:]): 
        cumprod = np.cumprod(np.round(y),axis=1)
        yTime = [breaks[int(np.sum(cumprod[i,:])-1)] for i in range(len(y))] 
        return yTime
    else:
        yt = y[:,0:num_iter]
        ye = y[:,num_iter:num_iter*2]
        cumprod = np.cumprod(np.round(yt),axis=1)
        yTime = [breaks[int(np.sum(cumprod[i,:])-1)] for i in range(len(y))] 
        yE = [int(np.where(np.sum(ye[i,:]) !=0,1,0))for i in range(len(ye))] 
        return yTime,yE


def __cox_loss(): #https://github.com/maycuiyan/deep-survival-model/blob/master/models.py
    def loss(y_true,y_pred):
        ## cox regression computes the risk score, we want the opposite
        score = y_pred
        event=y_true[:,0]
        time=y_true[:,1]
        ## find index i satisfying event[i]==1
        ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]
        ## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
        sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)
        ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
        p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
        loss = -tf.reduce_mean(p_lik) / tf.reduce_sum(event)
        return loss
    return loss

def __meanSquare_loss(): #https://github.com/maycuiyan/deep-survival-model/blob/master/models.py
    def loss(y_true,y_pred):
        score = tf.cast(y_pred[:,0], tf.float32)
        event=tf.cast(y_true[:,0], tf.float32)
        time=tf.cast(y_true[:,1], tf.float32)
        n= tf.shape(score)[0]
        # write the L1 loss (mean square error)
        ag =  tf.equal(event,0)
        ze =tf.zeros(tf.shape(event))
        on=tf.ones(tf.shape(event))
        ag2  = tf.where( tf.less(score,time),on,ze)
        I = event + tf.where(ag, on,ze)* ag2
        ix = tf.where(tf.cast(I, tf.bool))
        loss = tf.cast(1/n, tf.float32) * tf.reduce_sum(tf.square(tf.gather(score,ix) - tf.gather(time,ix)))
        
        return loss
    return loss

def __RankTime_loss(): #https://github.com/maycuiyan/deep-survival-model/blob/master/models.py
    def loss(y_true,y_pred):
        score = tf.cast(y_pred[:,0], tf.float32)
        event=tf.cast(y_true[:,0], tf.float32)
        time=tf.cast(y_true[:,1], tf.float32)
        n= tf.shape(score)[0]
        c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1 
        on =tf.ones(tf.shape(c1),dtype=tf.float32)
        ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
        
        c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
        c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
        c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
        c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
        comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
        Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
        P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
        I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
        at = tf.square(Y - P)*I 
        loss =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
           
        return loss
    return loss

def __RankAndMse_loss(): #https://github.com/maycuiyan/deep-survival-model/blob/master/models.py

    def loss(y_true,y_pred):
        score = tf.cast(y_pred[:,0], tf.float32)
        event=tf.cast(y_true[:,0], tf.float32)
        time=tf.cast(y_true[:,1], tf.float32)
        n= tf.shape(score)[0]
        c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1 
        on =tf.ones(tf.shape(c1),dtype=tf.float32)
        ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
                
        c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
        c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
        c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
        c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
        comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
        Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
        P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
        I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
        at = tf.square(Y - P)*I 
        lrank =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
        
        
        ag =  tf.equal(event,0)
        ze =tf.zeros(tf.shape(event))
        on=tf.ones(tf.shape(event))
        ag2  = tf.where( tf.less(score,time),on,ze)
        I = event + tf.where(ag, on,ze)* ag2
        ix = tf.where(tf.cast(I, tf.bool))
        lmse = tf.cast(1/n, tf.float32) * tf.reduce_sum(tf.square(tf.gather(score,ix) - tf.gather(time,ix)))
        
        loss=2* lrank/100000 + lmse/10000
        return loss
    return loss

def __RankAndCox_loss(): #https://github.com/maycuiyan/deep-survival-model/blob/master/models.py
    def loss(y_true,y_pred):
        ## cox regression computes the risk score, we want the opposite
        score = tf.cast(y_pred[:,0], tf.float32)
        event=tf.cast(y_true[:,0], tf.float32)
        time=tf.cast(y_true[:,1], tf.float32)
        n= tf.shape(score)[0]
        c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1 
        on =tf.ones(tf.shape(c1),dtype=tf.float32)
        ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
                
        c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
        c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
        c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
        c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
        comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
        Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
        P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
        I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
        at = tf.square(Y - P)*I 
        lrank =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
        
        ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]
        ## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
        sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)
        ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
        p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
        lcox = -tf.reduce_mean(p_lik) / tf.reduce_sum(event)       
        
        loss = lcox*3 + lrank*1e-5#lcox*15 + lrank*1e-7
        return loss
    return loss


#score = tf.constant([9,19,32,42,29])
#event=tf.constant([0,1,0,1,0])
#time=tf.constant([10,20,30,40,50])


def rank(y_true, y_pred):
    ## cox regression computes the risk score, we want the opposite
    score = tf.cast(y_pred[:,0], tf.float32)
    event=tf.cast(y_true[:,0], tf.float32)
    time=tf.cast(y_true[:,1], tf.float32)
    n= tf.shape(score)[0]
    c1 = tf.expand_dims(event, -1)*event # 1 where ei and ej = 1
    on =tf.ones(tf.shape(c1),dtype=tf.float32)
    ze =tf.zeros(tf.shape(c1),dtype=tf.float32)
    c2a = tf.cast(tf.subtract(tf.expand_dims(event, -1), event,name='c2a'), tf.float32)  # 1 where ei =1 ej = 0
    c2b =  tf.cast(tf.subtract(tf.expand_dims(time, -1), time,name='c2b'), tf.float32) # - or 0: yi <= yj
    c2aw = tf.where(tf.equal(c2a,on) ,on,ze) 
    c2bw = tf.where(tf.greater(c2b,ze),ze,on) 
    comp = tf.where(tf.greater(c2bw*c2aw,ze),on,ze) 
    Y =  tf.cast(tf.subtract(time,tf.expand_dims(time, -1),name='Y'), tf.float32)
    P = tf.cast(tf.subtract(score,tf.expand_dims(score, -1), name='P'), tf.float32)
    I =  tf.where(tf.greater(Y,P) ,on,ze) *comp
    at = tf.square(Y - P)*I 
    loss =  tf.cast(1/n, tf.float32) * tf.reduce_sum(at)
    return loss
def coxL(y_true,y_pred):
    ## cox regression computes the risk score, we want the opposite
    score = y_pred
    event=y_true[:,0]
    time=y_true[:,1]
    ## find index i satisfying event[i]==1
    ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]
    ## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
    sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)
    ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
    p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
    loss = -tf.reduce_mean(p_lik) #/ tf.reduce_sum(event)
    return loss

def __coxPlusClassif_loss(alpha,beta):
    def loss(y_true,y_pred):
        cox= coxL(y_true,y_pred)
        cat = keras.losses.categorical_crossentropy(y_true,y_pred)
        loss=cox*alpha + beta* cat
        return loss
    return loss


def mse(y_true,y_pred):
    
    """
    Returns
        metrics Mean square error
    """
    score = tf.cast(y_pred[:,0], tf.float32)
    event=tf.cast(y_true[:,0], tf.float32)
    time=tf.cast(y_true[:,1], tf.float32)
    ag =  tf.equal(event,0)
    ag1 =tf.zeros(tf.shape(event))
    ag2  = tf.where( tf.less(score,time), tf.ones(tf.shape(time)),tf.zeros(tf.shape(time)))
    I = event + tf.where(ag, tf.ones(tf.shape(event)),ag1)* ag2
    n= tf.shape(score)[0]
    ix = tf.where(tf.cast(I, tf.bool))
    L1 = tf.cast(1/n, tf.float32)  * tf.reduce_sum(tf.square(tf.gather(score,ix) - tf.gather(time,ix)))
    return L1
def accuracy(y_true,y_pred):
    score = tf.cast(y_pred[:,0], tf.float32)
    #event=tf.cast(y_true[:,0], tf.float32)
    time=tf.cast(y_true[:,1], tf.float32)
    acc, acc_op = tf.metrics.accuracy(labels=time, 
                                  predictions=score)
    return acc
def cox(y_true,y_pred):
    """
    Returns
        metrics cox
    """
    ## cox regression computes the risk score, we want the opposite
    score = -y_pred
    event=y_true[:,0]
    time=y_true[:,1]
    ## find index i satisfying event[i]==1
    ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]
    ## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
    sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)
    ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
    p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
    loss = -tf.reduce_mean(p_lik) / tf.reduce_sum(event)
    return loss



""" 
surv_likelihood and  make_surv_array functions where produced by MGensheimer (https://github.com/MGensheimer/nnet-survival/blob/master/nnet_survival.py)

MIT License

Copyright (c) 2019 Michael Gensheimer
"""
def surv_likelihood(n_intervals):
  """Create custom Keras loss function for neural network survival model. 
  Arguments
      n_intervals: the number of survival time intervals
  Returns
      Custom loss function that can be used with Keras
  """
  def loss(y_true, y_pred):
    """
    Required to have only 2 arguments by Keras.
    Arguments
        y_true: Tensor.
          First half of the values is 1 if individual survived that interval, 0 if not.
          Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
          See make_surv_array function.
        y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
    Returns
        Vector of losses for this minibatch.
    """
    cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
    uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
    return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
  return loss

def make_surv_array(t,f,breaks):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  n_intervals=len(breaks)
  timegap = breaks[1:] - breaks[:-1]
  #breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks)[0][0]-1]=1 #mark failure at first bin where survival time < upper break-point
      else :
        y_train[i,-1]=1 #mark failure at first bin where survival time < upper break-point   
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks) #give credit for surviving each time interval where failure time >= upper limit
  return y_train

