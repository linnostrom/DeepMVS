from os import listdir
from os.path import isfile, join
import sys
import os 
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import argparse
#import tensorflow as tf

def get_metrics(d,d_gt,mask):
    #d = tf.convert_to_tensor(d, dtype = tf.float32)
    #d_gt = tf.convert_to_tensor(d_gt, dtype = tf.float32)
    #mask = tf.convert_to_tensor(mask, dtype = tf.bool)
    d = d[mask]

    d_gt = d_gt[mask]

    scale = np.divide(d_gt ,d)

    d = d * np.median(scale)

    e = AbsRel(d,d_gt)
  
    e_vec = [AbsRel(d,d_gt), AbsDiff(d,d_gt), SqRel(d,d_gt), RMSE(d,d_gt), log_scale_inv_RMSE(d, d_gt), accuracy_under_thres(d, d_gt),  accuracy_under_thres2(d, d_gt), accuracy_under_thres3(d, d_gt)]

    return e_vec

def AbsRel(d,d_gt):
    e = 1.0 / np.prod(np.shape(d))
    e = e * np.sum(np.divide(np.abs(d-d_gt),d_gt))

    return e

def AbsDiff(d,d_gt):

    e = 1.0 / np.prod(np.shape(d))

    e = e * np.sum(np.abs(d-d_gt))
    return e

def SqRel(d,d_gt):
    e = 1.0 / np.prod(np.shape(d))
    e = e* np.sum(np.divide((d-d_gt)**2, d_gt))
    return e

def RMSE(d,d_gt):
    e = 1.0 / np.prod(np.shape(d))
    e = e * np.sum((d-d_gt)**2)
    e = np.sqrt(e)
    return e

def log_scale_inv_RMSE(d, d_gt):
    EPS = 1e-16

    alpha = 1.0/(2.0*np.prod(np.shape(d)))
    alpha = alpha * np.sum(np.log(d_gt)-np.log(d+EPS)) 
    e = 1.0/(2.0*np.prod(np.shape(d)))
    
    e = e * np.sum((np.log(d+EPS)-np.log(d_gt)+alpha)**2)
    return e

def accuracy_under_thres(d, d_gt):
    thres = 1.1

    e = np.sum(np.maximum(np.divide(d,d_gt), np.divide(d_gt,d)) < thres)/np.prod(np.shape(d)).astype(np.float32) * 100.0
    return e

def accuracy_under_thres2(d, d_gt):
    thres = 1.25

    e = np.sum(np.maximum(np.divide(d,d_gt), np.divide(d_gt,d)) < thres)/np.prod(np.shape(d)).astype(np.float32) * 100.0
    #e = tf.reduce_sum(tf.cast(tf.math.maximum(d/d_gt, d_gt/d) < thres, dtype = tf.float32))/tf.cast(tf.reduce_prod(tf.shape(d)), dtype = tf.float32) * 100
    return e

def accuracy_under_thres3(d, d_gt):
    thres = 1.25**2
    e = np.sum(np.maximum(np.divide(d,d_gt), np.divide(d_gt,d)) < thres)/np.prod(np.shape(d)).astype(np.float32) * 100.0
    #e = tf.reduce_sum(tf.cast(tf.math.maximum(d/d_gt, d_gt/d) < thres, dtype = tf.float32))/tf.cast(tf.reduce_prod(tf.shape(d)), dtype = tf.float32) * 100
    return e

