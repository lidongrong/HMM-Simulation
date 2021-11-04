# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:04:42 2021

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:00:40 2021

@author: s1155151972
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
        [[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.7,0.3,0],[0,0,0,0.2,0.8],[0,0,0,0,1]]
        )       

spec_transition=np.array(
    [[0.3,0.3,0.2,0.1,0.1],
     [0.2,0.4,0.4,0,0],
     [0.1,0.1,0.5,0.2,0.1],
     [0.3,0.1,0.1,0.3,0.2],
     [0.05,0.1,0.1,0.25,0.5]
     ]
    )
    

state=np.array(['A','B','C','D','E'])
hidden_state=state
obs_state=np.array(['Blue','Red','Green','Purple','Grey'])


    
obs_prob=np.array([[0.7,0.2,0.05,0.04,0.01],[0.1,0.7,0.1,0.09,0.01],
                   [0.05,0.1,0.4,0.4,0.05],[0.0,0.03,0.07,0.6,0.3],
                   [0.01,0.04,0.05,0.1,0.8]
        ])

MC=HMM(hidden_state,obs_state,spec_transition,obs_prob)