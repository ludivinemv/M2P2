# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 ludivinemv
code found in https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
and adapted to survival analysis
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


def _get_anchor_positive_triplet_mask(labels,event):
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


def batch_hard_triplet_loss(margin=0.001, squared=False):
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
        # embeddings= tf.cast(tf.expand_dims(y_pred, -1),dtype=tf.float32)  
        embeddings= tf.cast(y_pred,dtype=tf.float32)  

        # embeddings= y_pred
        # Get the pairwise distance matrix
        labels =   tf.cast(labels,dtype=tf.float32) 
        event = tf.cast(event,dtype=tf.float32) 
        embeddings= tf.cast(embeddings,dtype=tf.float32)  
        
        pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    
        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels,event)
        
        mask_anchor_positive = tf.cast(mask_anchor_positive, dtype = tf.float32)
    
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))
    
        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels,event)
        mask_anchor_negative =  tf.cast(mask_anchor_negative, dtype = tf.float32)
    
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    
        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    
        # Get final mean triplet loss
        loss = tf.reduce_mean(triplet_loss)
        return loss

    return loss

def triplet_metrics(y_true,y_pred):
        margin=0.001
        squared=False
        labels =   tf.cast(tf.expand_dims(y_true[:,1], -1),dtype=tf.float32) 
        event = tf.cast(tf.expand_dims(y_true[:,0], -1),dtype=tf.float32) 
        embeddings= tf.cast(tf.expand_dims(y_pred, -1),dtype=tf.float32)  
        # embeddings= y_pred
        # Get the pairwise distance matrix
        labels =   tf.cast(labels,dtype=tf.float32) 
        event = tf.cast(event,dtype=tf.float32) 
        # embeddings= tf.cast(embeddings,dtype=tf.float32)  
        embeddings= tf.cast(y_pred,dtype=tf.float32)  

        pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    
        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels,event)
        
        mask_anchor_positive = tf.cast(mask_anchor_positive, dtype = tf.float32)
    
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))
    
        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels,event)
        mask_anchor_negative =  tf.cast(mask_anchor_negative, dtype = tf.float32)
    
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    
        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    
        # Get final mean triplet loss
        loss = tf.reduce_mean(triplet_loss)
        return loss