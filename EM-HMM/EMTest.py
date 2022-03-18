# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:51:05 2022

@author: lidon
"""

import numpy as np
import HMM
import SeqSampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math
import os
import multiprocessing as mp
from EMHMM import*
from ZMARGibbs import*

transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

#pi=np.array([0.7,0.2,0.1])
pi=np.array([0.25,0.55,0.2])

if __name__=='__main__':
    # Use multicore CPU
    p=Pool(16)
    rate=0.3
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
    data1=[]
    for i in range(0,len(data)):
        data1.append(data[i][data[i]!='None'])
    #data=data1
    data1=np.array(data1)
    A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    pi0=np.array([0.4,0.3,0.3])
    # testing code
    #A=transition
    #B=obs_prob
    #pi0=pi
    
    prob=y_prob(data[0:1000],transition,obs_prob,pi,hidden_state,obs_state,p)
    print('True Obj Func: ',sum(np.log(prob)))
    
    #a,b,pp,func=SEMTrain(A,B,pi0,data,50,25,hidden_state,obs_state,p)
    at,bt,pit,func=EMTrain(A,B,pi0,data[0:1000],0.0001,hidden_state,obs_state,p)
    
    
    
    
    