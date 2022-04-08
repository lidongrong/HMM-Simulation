# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:04:46 2022

@author: lidon
"""

import numpy as np
from ZMARGibbs import*
from HMM import*
from SeqSampling import*
from EMHMM import*
import math


transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.05,0.05,0.9]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

pi=np.array([0.1875, 0.3125, 0.5   ])



# calculate the entropy
def entropy(x):
    return -np.dot(x,np.log2(x))

# calculate hamming epsilon ball
# T: length
# eps: epsilon
# z: alphabet size
def eps_ball(T,eps,z):
    maximum=np.floor(T*eps)
    maximum=int(maximum)
    s=0
    for k in range(0,maximum):
        s=s+(math.factorial(T)/(math.factorial(k)*math.factorial(T-k)))*((z-1)**k)
    return s

# #entropy of a markov chain
# # T: length
# def MC_entropy(T,A,B,pi):
#     s=entropy(pi)
#     for k in range(1,T):
#         prob=np.dot(pi,np.linalg.matrix_power(A,k))
#         ent=np.array([entropy(A[i]) for i in range(0,A.shape[0])])
#         s=s+np.dot(prob,ent)
#     return s


#entropy of a markov chain
# T: length
def MC_entropy(T,A,B,pi):
    s=entropy(pi)
    for k in range(1,T):
        prob=pi
        ent=np.array([entropy(A[i]) for i in range(0,A.shape[0])])
        s=s+np.dot(prob,ent)
    return s


eps=0.35
T=20
z=3
# obs rate
q=0.9
# p=(MC_entropy(T,transition,obs_prob,pi)-1-T*q*(np.log2(3)-entropy(obs_prob[0]))-np.log2(eps_ball(T,eps,z)))
# p=p/np.log2((3**T)/eps_ball(T,eps,z)-1)


prec=[]

for q in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    p=(MC_entropy(T,transition,obs_prob,pi)-1-T*q*(np.log2(3)-entropy(obs_prob[0]))-np.log2(eps_ball(T,eps,z)))
    p=p/np.log2((3**T)/eps_ball(T,eps,z)-1)
    prec.append(p)
    
