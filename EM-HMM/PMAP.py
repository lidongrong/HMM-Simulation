# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:50:16 2022

@author: lidon
"""


# implement the pointwise MAP inference algorithm for HMM


import numpy as np
from ZMARGibbs import*
from HMM import*
from SeqSampling import*
from EMHMM import*


transition=np.array([[0.6,0.3,0.1],[0.1,0.6,0.3],[0.3,0.1,0.6]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

pi=np.array([0.7,0.2,0.1])

# pointwise MAP implementation
def PMAP(obs,A,B,pi,hidden_state,obs_state):
    alph,bet=us_alpha_and_beta(obs,A,B,pi,hidden_state,obs_state)
    
    # compute p(z_t|y_o)
    cond_prob=[alph[t,:]*bet[t,:]/np.sum(alph[t,:]*bet[t,:]) for t in range(0,len(obs))]
    cond_prob=np.array(cond_prob)
    # find the index of the PMAP estimate of z
    out=[]
    for i in range(0,len(obs)):
        out.append(hidden_state[np.argmax(cond_prob[i,:])])
    out=np.array(out)
    return out

# evaluate PMAP on the whole dataset
# y is the whole dataset
def PMAP_dataset(y,A,B,pi,hidden_state,obs_state):
    n=y.shape[0]
    z=[]
    for i in range(0,n):
        new_z=PMAP(y[i],A,B,pi,hidden_state,obs_state)
        z.append(new_z)
    z=np.array(z)
    return z