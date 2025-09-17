import os
import sys
import time
import argparse

import numpy as np 
import tensorflow as tf
import custom_callbacks
import custom_functions as func
import rebuild_layers as rl
import rebuild_filters as rf
import criteria_filter as cf
import criteria_layer as cl

from datetime import datetime

import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from keras.layers import *
from keras.activations import *
from tensorflow.data import Dataset

from sklearn.utils import gen_batches
from sklearn.metrics._classification import accuracy_score

X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=False)

def pruneByLayer(model, criteria, p_layer):
    allowed_layers = rl.blocks_to_prune(model)
    layer_method = cl.criteria(criteria)
    scores = layer_method.scores(model, X_train, y_train, allowed_layers)    
    
    return rl.rebuild_network(model, scores, p_layer)

def pruneByFilter(model, criteria, p_filter):
    allowed_layers_filters = rf.layer_to_prune_filters(model)
    filter_method = cf.criteria(criteria)
    scores = filter_method.scores(model, X_train, y_train, allowed_layers_filters)    
    
    return  rf.rebuild_network(model, scores, p_filter)
             

