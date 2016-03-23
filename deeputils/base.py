'''
Created on 23.03.2016

@author: mario
'''
import numpy as np

def softmax(x):
    '''
    Takes an array of values and returns
    np array of its softmax value
    '''
    earr = np.exp(np.array(x))
    scores = earr / np.sum(earr,axis=0)
    return scores