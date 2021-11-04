# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:48:11 2021

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:26:34 2021

@author: a
"""

import HMM
import numpy as np
import scipy.stats
import math

###
# Inference part of the HMM model, with highly missing data
###

# Data Generation
# MC is a HMM object 
# Length is the required length of each sequence
# size is the number of samples
def Hidden_Generator(MC,length,size):
    hidden_sample=[]
    for i in range(0,size):
        hidden_sample.append(MC.sample('A',length))
    return np.array(hidden_sample)

def Obs_Generator(MC,hidden_sample):
    obs_sample=[]
    for i in range(0,len(hidden_sample[:,0])):
        obs_sample.append(MC.sample_obs(hidden_sample[i,:]))
    return np.array(obs_sample)

hidden_data=Hidden_Generator(HMM.MC,15,3600)
data=Obs_Generator(HMM.MC,hidden_data)

# Generate missing data
# p: missing rate
def Missing(obs,p):
    for i in range(0,obs.shape[0]):
        for j in range(0,obs.shape[1]):
            if np.random.binomial(1,p):
                obs[i][j]=None
    return obs
    
#data=Missing(obs_data,p=0.1)

#Generate missing but continuous data
#p: missing rate
# The idea for this generation is that we only assume the observed data is a continuous string
# but other data apart from this string is missing
# So we generate randomly along each sequence with expected value length*p
def seq_missing(data,p):
    for i in range(0,data.shape[0]):
        # construct the average observed length for this sequence
        obs_length=data.shape[1]-np.random.binomial(data.shape[1],p)
        # decide where this subsequence starts (other data set to be 'None')
        if obs_length>0:
            start_point=np.random.randint(0,data.shape[1])
            data[i,0:start_point]='None'
            data[i,min(start_point+obs_length,data.shape[1]):data.shape[1]]='None'
    return data

data=seq_missing(data,p=0.4)