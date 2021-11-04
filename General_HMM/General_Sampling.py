# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:56:36 2021

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:00:51 2021

@author: s1155151972
"""


import General_HMM as HMM
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

hidden_data=Hidden_Generator(HMM.MC,20,4000)
obs_data=Obs_Generator(HMM.MC,hidden_data)

# Generate missing data
# p: missing rate
def Missing(obs,p):
    for i in range(0,obs.shape[0]):
        for j in range(0,obs.shape[1]):
            if np.random.binomial(1,p):
                obs[i][j]=None
    return obs
    
data=Missing(obs_data,p=0.1)