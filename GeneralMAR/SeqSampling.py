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


# HMM construction
transition=np.array(
        [[0.4,0.6,0,0,0],[0,0.7,0.3,0,0],[0,0,0.8,0.2,0],[0,0,0,0.35,0.65],[0,0,0,0,1]]
        )           

state=np.array(['A','B','C','D','E'])
hidden_state=state
obs_state=np.array(['Blue','Red','Green','Purple','Grey'])
# Dirichelet (2,2,2,2,2)
'''
 obs_prob=np.array([[0.12921188, 0.10229518, 0.1940095 , 0.39287558, 0.18160785],
       [0.20193517, 0.43696288, 0.11614156, 0.12895052, 0.11600988],
       [0.40871969, 0.19663098, 0.1581776 , 0.13207572, 0.10439601],
       [0.21058348, 0.24257849, 0.20008727, 0.26959809, 0.07715268],
       [0.25477796, 0.13732367, 0.14608296, 0.26345009, 0.19836532]])
    '''
    
obs_prob=np.array([[0.7,0.3,0,0,0],[0.05,0.7,0.2,0.05,0],
                   [0.05,0.1,0.5,0.35,0.00],[0.0,0.03,0.07,0.6,0.3],
                   [0.01,0.04,0.05,0.1,0.8]
        ])

pi=[1,0,0,0,0]

MC=HMM.HMM(hidden_state,obs_state,transition,obs_prob,pi)





# Data Generation
# MC is a HMM object 
# Length is the required length of each sequence
# size is the number of samples
def Hidden_Generator(MC,length,size):
    hidden_sample=[]
    for i in range(0,size):
        hidden_sample.append(MC.sample(length))
    return np.array(hidden_sample)

def Obs_Generator(MC,hidden_sample):
    obs_sample=[]
    for i in range(0,len(hidden_sample[:,0])):
        obs_sample.append(MC.sample_obs(hidden_sample[i,:]))
    return np.array(obs_sample)

hidden_data=Hidden_Generator(MC,20,4000)
data=Obs_Generator(MC,hidden_data)

# Generate missing data
# p: missing rate
def Missing(obs,p):
    for i in range(0,obs.shape[0]):
        for j in range(0,obs.shape[1]):
            if np.random.binomial(1,p):
                obs[i][j]=None
    return obs
    



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
        if obs_length==data.shape[1]:
            data[i,:]=data[i,:]
        elif obs_length==0:
            data[i,:]='None'
        else:
            start_point=np.random.randint(0,data.shape[1]-obs_length)
            #start_point=np.random.randint(0,data.shape[1])
            data[i,0:start_point]='None'
            data[i,start_point+obs_length:data.shape[1]]='None'
            #data[i,min(start_point+obs_length,data.shape[1]):data.shape[1]]='None'
    return data

#data=seq_missing(data,p=0.3)
data=Missing(data,p=0.7)