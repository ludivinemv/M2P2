# -*- coding: utf-8 -*-
"""

MIT License

Copyright (c) 2021 ludivinemv
"""

import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:

        embeddings: tensor of shape (batch_size, embed_dim)

        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.

                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask =  tf.cast(tf.equal(distances, 0.0),dtype=tf.float32) 
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels,event,pairwise_dist):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label and 
    event = 1 or if event = 0 and labels =6
    

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]

    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)


    #check where event = 0
    event2 = tf.multiply(tf.expand_dims(event, 0),tf.expand_dims(event, 1))
    
    
    labels2 = tf.multiply(tf.expand_dims(labels, 0),tf.expand_dims(labels, 1))
    classe6 =  tf.equal(labels2,36)
    evAnd36 = event2 + tf.cast(classe6,tf.float32)    

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask0 = tf.multiply(tf.cast(labels_equal,tf.float32), evAnd36)
    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, tf.cast(tf.squeeze(mask0),tf.bool))

    return mask


def _get_anchor_negative_triplet_mask(labels,event):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        
        A-B
        si A <= B : labelMin - ou 0
        si A = 1 et b= 0   eventmin  = 1
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    labels_not_equal = tf.logical_not(labels_equal)
    event1AndLabelDiff =  tf.logical_and(labels_not_equal,tf.cast(event,tf.bool))
    
    event2 = tf.expand_dims(event, 0) + tf.expand_dims(event, 1)
    labAMinusB = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    
    eventEq = tf.equal(event2,1)
    labinf = tf.math.less(labAMinusB,0)
    labEq = tf.math.equal(labAMinusB,0)
    labNeg = tf.logical_or(labEq,labinf)
    fin = tf.logical_and(labNeg,eventEq)

    indices_not_equal = tf.logical_not(tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool))
    mask = tf.logical_and(indices_not_equal,fin)
    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask,dtype=tf.float32) 
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def select_triplet(labels, event):
    try:
        labels = labels[:,0]
    except:
        pass
    try: 
        event = event[:,0]
    except:
        pass
    
    #### search for triplet where A<P<N and ep=1 et ea=1 ###
    AminusP = tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)) # P-A
    AminusP =  tf.clip_by_value(AminusP, 0, 1) # put at 0 when negative or =0 (A>=P) and 1 otherwise
    Pbar = tf.expand_dims(labels, 0) *tf.expand_dims(tf.ones(tf.shape(labels)), 1)
    Pvalid = Pbar*AminusP
    PdivN = tf.divide(tf.expand_dims(Pvalid, 2),tf.expand_dims(labels, 0))
    #3D matrix of valid triplets according to A<P<N Diff0[A,P,N]
    Diff0 = tf.cast(tf.not_equal(PdivN,0),tf.int32)*tf.cast(tf.less(PdivN,1),tf.int32)
    #PdivN = tf.divide(tf.expand_dims(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), 2),tf.expand_dims(tf.constant([1,2,3,4]), 0))

    event_AP = tf.multiply(tf.expand_dims(event, 0), tf.expand_dims(event, 1))
    event_APN = tf.multiply(tf.expand_dims(event_AP,2),tf.expand_dims(tf.ones(tf.shape(event)), 0))
    
    ValidTriplet1 = tf.cast(event_APN,tf.int32) *Diff0
    
    #### search for triplet where P<A<N and ep=1 et ea=1 et PA<AN ###
    AminusP = tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)) # P-A
    AminusP = - tf.clip_by_value(AminusP, -1,0) # put at 0 when negative or =0 (A>=P) and 1 otherwise
    Pbar = tf.expand_dims(tf.ones(tf.shape(labels)), 0)*tf.expand_dims(labels, 1)
    Pvalid = Pbar*AminusP
    PdivN = tf.divide(tf.expand_dims(Pvalid, 2),tf.expand_dims(labels, 0))
    #3D matrix of valid triplets according to A<P<N Diff0[A,P,N]
    Diff0 = tf.cast(tf.greater(PdivN,1),tf.int32)
    #PdivN = tf.divide(tf.expand_dims(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), 2),tf.expand_dims(tf.constant([1,2,3,4]), 0))

    event_AP = tf.multiply(tf.expand_dims(event, 0), tf.expand_dims(event, 1))
    event_APN = tf.multiply(tf.expand_dims(event_AP,2),tf.expand_dims(tf.ones(tf.shape(event)), 0))
    
    AminusP = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    AminusN = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    
    AminusP3D = tf.multiply(tf.expand_dims(AminusP,2),tf.expand_dims(tf.ones(tf.shape(AminusP)), 0))   
    AminusN3D = tf.transpose(tf.multiply(tf.expand_dims(AminusN,2),tf.expand_dims(tf.ones(tf.shape(AminusN)), 0)), [0, 2, 1])
    PAminusAN = tf.clip_by_value(AminusN3D - AminusP3D,0,1)
    
    
    ValidTriplet2 = tf.cast(event_APN,tf.int32) *Diff0*tf.cast(PAminusAN,tf.int32)
    
          
    
    
    #### search for triplet where N<A<P et PA<AN et en=1 et ep=1 ###
    AminusP = tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)) # P-A
    AminusP = tf.clip_by_value(AminusP, 0,1) # put at 0 when negative or =0 (A>=P) and 1 otherwise
    Pbar =  tf.expand_dims(tf.ones(tf.shape(labels)), 0)*tf.expand_dims(labels, 1)
    Pvalid = Pbar*AminusP
    PdivN = tf.divide(tf.expand_dims(Pvalid, 2),tf.expand_dims(labels, 0)) #A/N
    #3D matrix of valid triplets according to A<P<N Diff0[A,P,N]
    Diff0 = tf.cast(tf.greater(PdivN,1),tf.int32)
    #PdivN = tf.divide(tf.expand_dims(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), 2),tf.expand_dims(tf.constant([1,2,3,4]), 0))

    event_AP = tf.multiply(tf.expand_dims(tf.ones(tf.shape(event)), 1), tf.expand_dims(event, 0))
    event_APN = tf.multiply(tf.expand_dims(event, 0),tf.expand_dims(event_AP,2))
      
    AminusP = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    AminusN = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    
    AminusP3D = tf.multiply(tf.expand_dims(AminusP,2),tf.expand_dims(tf.ones(tf.shape(AminusP)), 0))   
    AminusN3D = tf.transpose(tf.multiply(tf.expand_dims(AminusN,2),tf.expand_dims(tf.ones(tf.shape(AminusN)), 0)), [0, 2, 1])
    PAminusAN = tf.clip_by_value(AminusN3D - AminusP3D,0,1)
      
    ValidTriplet3 = tf.cast(event_APN,tf.int32) *Diff0*tf.cast(PAminusAN,tf.int32)
   
    
    #### search for triplet where N<P<A  et en=1 et ep=1 ###
    AminusP = tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)) # P-A
    AminusP = -tf.clip_by_value(AminusP, -1,0) # put at 0 when negative or =0 (A>=P) and 1 otherwise
    Pbar = tf.expand_dims(labels, 0) *tf.expand_dims(tf.ones(tf.shape(labels)), 1)
    Pvalid = Pbar*AminusP
    PdivN = tf.divide(tf.expand_dims(Pvalid, 2),tf.expand_dims(labels, 0))
    #3D matrix of valid triplets according to A<P<N Diff0[A,P,N]
    Diff0 = tf.cast(tf.greater(PdivN,1),tf.int32)
    #PdivN = tf.divide(tf.expand_dims(tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), 2),tf.expand_dims(tf.constant([1,2,3,4]), 0))

    event_AP = tf.multiply(tf.expand_dims(tf.ones(tf.shape(event)), 1), tf.expand_dims(event, 0))
    event_APN = tf.multiply(tf.expand_dims(event, 0),tf.expand_dims(event_AP,2))
      
    AminusP = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    AminusN = tf.abs(tf.subtract(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))) # P-A
    

  
    ValidTriplet4 = tf.cast(event_APN,tf.int32) *Diff0
    
    
    All = tf.clip_by_value(ValidTriplet1 + ValidTriplet2 + ValidTriplet3 +ValidTriplet4, 0,1) 
    return All

# labelsN = [50,150,600,1000,1200,1300,1900]
# eventN = [1,0,1,0,1,0,1]
# y_pred=np.random.rand(10,4096)
# event =tf.cast(tf.expand_dims(tf.constant(eventN), -1),dtype=tf.float32) 
# labels =tf.cast(tf.expand_dims(tf.constant(labelsN), -1),dtype=tf.float32) 
# embeddings =tf.cast(tf.constant(y_predN),dtype=tf.float32) 
# y_true=tf.transpose(tf.constant(np.array([eventN,labelsN]),dtype=tf.float32))
# y_pred=tf.constant(y_predN,dtype=tf.float32)

# lossT = loss(y_true,y_pred)
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
def batch_hard_triplet_loss(Ymin,Ymax,margin=0.001, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    def loss(y_true,y_pred):
        labels =   tf.cast(tf.expand_dims(y_true[:,1], -1),dtype=tf.float32) 
        event = tf.cast(tf.expand_dims(y_true[:,0], -1),dtype=tf.float32) 
        embeddings= tf.cast(tf.expand_dims(y_pred, -1),dtype=tf.float32)  
        embeddings= tf.cast(y_pred,dtype=tf.float32)  

        # embeddings= y_pred
        # Get the pairwise distance matrix
        labels =   tf.cast(labels,dtype=tf.float32) 
        event = tf.cast(event,dtype=tf.float32) 
        embeddings= tf.cast(embeddings,dtype=tf.float32)  
        
        pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    
        All_triplet_mask = select_triplet(labels, event)
        pairwise_dist_pos = tf.multiply(tf.expand_dims(pairwise_dist, 2),tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),0))
        pairwise_dist_neg = tf.multiply(tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),2),tf.expand_dims(pairwise_dist, 0))
        dap = tf.multiply(tf.cast(All_triplet_mask,tf.float32), pairwise_dist_pos)
        dan = tf.multiply(tf.cast(All_triplet_mask,tf.float32), pairwise_dist_neg)
        
        allyP = tf.multiply(tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),2),tf.expand_dims(labels, 0))
        allyA = tf.multiply(tf.expand_dims(labels, 2),tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),0))

        yA= tf.multiply(tf.cast(All_triplet_mask,tf.float32), allyA)
        yP= tf.multiply(tf.cast(All_triplet_mask,tf.float32), allyP)
    
        epsilon=0.1
        yAnorm = normalizeTriplet(yA,Ymin,Ymax)*tf.cast(All_triplet_mask,dtype=tf.float32)  
        yPnorm = normalizeTriplet(yP,Ymin,Ymax)*tf.cast(All_triplet_mask,dtype=tf.float32)  
        wanp = (((1+epsilon)/(tf.abs(yAnorm-yPnorm)+epsilon))-1)*tf.cast(All_triplet_mask,dtype=tf.float32)  
       
        aaadan =tf.where(tf.equal(dan,0), tf.ones(tf.shape(dan),dtype=tf.float32)*(tf.reduce_max(dan)+100),dan)
        miindan = tf.reduce_min(aaadan)
        maaxdan = tf.reduce_max(dan)
        
        
        aaadap =tf.where(tf.equal(dap,0), tf.ones(tf.shape(dap),dtype=tf.float32)*(tf.reduce_max(dap)+100),dap)
        miindap = tf.reduce_min(aaadap)
        maaxdap = tf.reduce_max(dap)
        
        dannorm = normalizeTriplet(dan,miindan,maaxdan)*tf.cast(All_triplet_mask,dtype=tf.float32) 
        dapnorm = normalizeTriplet(dap,miindap,maaxdap)*tf.cast(All_triplet_mask,dtype=tf.float32) 
        
        tn=1
        tp=0
        dn=tf.exp(dannorm)/(tf.exp(dapnorm)+tf.exp(dannorm)) 
        dp=tf.exp(dapnorm)/(tf.exp(dapnorm)+tf.exp(dannorm)) 
     
        LT= -tf.log(dn) *tf.cast(All_triplet_mask,dtype=tf.float32)          
        # LT= tf.exp(dapnorm)/tf.exp(dannorm) *tf.cast(All_triplet_mask,dtype=tf.float32)  

        #LT= -tn*tf.log(dn) - tp*tf.log(dp)

        # Get final mean triplet loss
        loss = tf.reduce_mean(wanp * LT)# 1/T sum w(fa,fp,fn)*LT(d+,d-)
        
        return loss

    return loss


def normalizeTriplet(y,Ymin,Ymax):
    return (y-Ymin)/(Ymax-Ymin)


def tripletContinue_metrics(y_true,y_pred):
    Ymin=49
    Ymax=2362
    squared=False        
    labels =   tf.cast(tf.expand_dims(y_true[:,1], -1),dtype=tf.float32) 
    event = tf.cast(tf.expand_dims(y_true[:,0], -1),dtype=tf.float32) 
    embeddings= tf.cast(tf.expand_dims(y_pred, -1),dtype=tf.float32)  
    embeddings= tf.cast(y_pred,dtype=tf.float32)  

    # embeddings= y_pred
    # Get the pairwise distance matrix
    labels =   tf.cast(labels,dtype=tf.float32) 
    event = tf.cast(event,dtype=tf.float32) 
    embeddings= tf.cast(embeddings,dtype=tf.float32)  
    
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    All_triplet_mask = select_triplet(labels, event)
    pairwise_dist_pos = tf.multiply(tf.expand_dims(pairwise_dist, 2),tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),0))
    pairwise_dist_neg = tf.multiply(tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),2),tf.expand_dims(pairwise_dist, 0))
    dap = tf.multiply(tf.cast(All_triplet_mask,tf.float32), pairwise_dist_pos)
    dan = tf.multiply(tf.cast(All_triplet_mask,tf.float32), pairwise_dist_neg)
    
    allyP = tf.multiply(tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),2),tf.expand_dims(labels, 0))
    allyA = tf.multiply(tf.expand_dims(labels, 2),tf.expand_dims(tf.ones(tf.shape(pairwise_dist)),0))

    yA= tf.multiply(tf.cast(All_triplet_mask,tf.float32), allyA)
    yP= tf.multiply(tf.cast(All_triplet_mask,tf.float32), allyP)

    epsilon=0.1
    yAnorm = normalizeTriplet(yA,Ymin,Ymax)*tf.cast(All_triplet_mask,dtype=tf.float32)  
    yPnorm = normalizeTriplet(yP,Ymin,Ymax)*tf.cast(All_triplet_mask,dtype=tf.float32)  
    wanp = (((1+epsilon)/(tf.abs(yAnorm-yPnorm)+epsilon))-1)*tf.cast(All_triplet_mask,dtype=tf.float32)  
   
    aaadan =tf.where(tf.equal(dan,0), tf.ones(tf.shape(dan),dtype=tf.float32)*(tf.reduce_max(dan)+100),dan)
    miindan = tf.reduce_min(aaadan)
    maaxdan = tf.reduce_max(dan)
    
    
    aaadap =tf.where(tf.equal(dap,0), tf.ones(tf.shape(dap),dtype=tf.float32)*(tf.reduce_max(dap)+100),dap)
    miindap = tf.reduce_min(aaadap)
    maaxdap = tf.reduce_max(dap)
    
    dannorm = normalizeTriplet(dan,miindan,maaxdan)*tf.cast(All_triplet_mask,dtype=tf.float32) 
    dapnorm = normalizeTriplet(dap,miindap,maaxdap)*tf.cast(All_triplet_mask,dtype=tf.float32) 
    
    tn=1
    tp=0
    dn=tf.exp(dannorm)/(tf.exp(dapnorm)+tf.exp(dannorm)) 
    dp=tf.exp(dapnorm)/(tf.exp(dapnorm)+tf.exp(dannorm)) 
 
    LT= -tf.log(dn) *tf.cast(All_triplet_mask,dtype=tf.float32)          
    # LT= tf.exp(dapnorm)/tf.exp(dannorm) *tf.cast(All_triplet_mask,dtype=tf.float32)  

    #LT= -tn*tf.log(dn) - tp*tf.log(dp)

    # Get final mean triplet loss
    loss = tf.reduce_mean(wanp * LT)# 1/T sum w(fa,fp,fn)*LT(d+,d-)
    return loss
    