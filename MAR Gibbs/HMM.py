# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:57:16 2021

@author: a
"""

import numpy as np
import scipy.stats
import math

class Markov:
    # state: states of a Markov Chain
    # transition: transition matrix
    # pi: original distribution
    def __init__(self,state,transition):
        self.state=state
        self.transition=transition
        self.pi=None
    
    # start: the state that starts
    # length: number of the path length
    def sample(self,start, length):
        sta=start
        path=[sta]
        for i in range(0,length-1):
            index=np.where(self.state==sta)[0][0]
            sta=np.random.choice(self.state,1,p=self.transition[index,:])[0]
            path.append(sta)
        path=np.array(path)
        return path


class HMM(Markov):
    # obs_prob: matrix that transform hidden state to obs state
    def __init__(self,h_state,o_state,transition,obs_prob):
        self.h_state=h_state
        self.state=h_state
        self.o_state=o_state
        self.transition=transition
        self.obs_prob=obs_prob
    
    
    def sample_obs(self,hidden_path):
        obs=[]
        for i in range(0,len(hidden_path)):
            index=np.where(self.state==hidden_path[i])[0][0]
            new_obs=np.random.choice(self.o_state,1,p=self.obs_prob[index,:])[0]
            obs.append(new_obs)
        obs=np.array(obs)
        return obs
    

          
transition=np.array(
        [[0.3,0.7,0,0,0],[0,0.8,0.2,0,0],[0,0,0.4,0.6,0],[0,0,0,0.8,0.2],[0,0,0,0,1]]
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

MC=HMM(hidden_state,obs_state,transition,obs_prob)
        

    