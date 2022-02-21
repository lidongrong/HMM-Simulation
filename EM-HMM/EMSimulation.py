# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:38:01 2022

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

transition=np.array([[0.6,0.3,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

pi=np.array([0.7,0.2,0.1])

if __name__=='__main__':
    
    
    # Use multicore CPU
    p=Pool(mp.cpu_count())
    rate=0
    A,B,pi,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
    A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    B=A.copy()
    pi0=np.array([0.4,0.3,0.3])
    print('done!')
    EMTrain(A,B,pi0,data[0:600],500,hidden_state,obs_state,p)




