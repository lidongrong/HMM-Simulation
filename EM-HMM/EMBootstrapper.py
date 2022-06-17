# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:36:06 2022

@author: lidon
"""

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
from datetime import datetime

transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
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
    p=Pool(12)
    rate=0.3
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
    A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    pi0=np.array([0.4,0.3,0.3])
    # testing code
    #A=transition
    #B=obs_prob
    #pi0=pi
    
    file_name=time.time()
    out_obj=[]
    os.mkdir(f'BootstrapResult_{file_name}')
    #os.mkdir('NaiveResult')
    for rate in [0.3,0.5,0.7,0.9]:
        #os.mkdir(f'Missingrate{rate}')
        # bootstrap numbers
        num=16
        out_obj=[]
        A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
        post_A=[]
        post_B=[]
        post_pi=[]
        for i in range(0,num):
            # Same data, different initial
            #A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
            
            #A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #pi0=np.array([0.4,0.3,0.3])
            # A=np.random.dirichlet((8,8,8),3)
            # B=np.random.dirichlet((8,8,8),3)
            # pi0=np.random.dirichlet((8,8,8),1)[0]
            A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            pi0=np.array([0.4,0.3,0.3])
            
            # construct bootstrapped sample
            boot_index=np.random.choice(np.arange(data.shape[0]),data.shape[0],replace=True)
            data1=data[boot_index,:]
            
            prob=y_prob(data1[0:750],transition,obs_prob,pi,hidden_state,obs_state,p)
            # data1=[]
            # for j in range(0,len(data)):
            #     data1.append(data[j][data[j]!='None'])
            # data1=np.array(data1)
            
            print('True Obj Func: ',sum(np.log(prob)))
            at,bt,pit,func=EMTrain(A,B,pi0,data1[0:750],0.0005,hidden_state,obs_state,p)
            
            post_A.append(at[len(at)-1])
            post_B.append(bt[len(bt)-1])
            post_pi.append(pit[len(pit)-1])
            
            # prob=y_prob(data1[0:1500],transition,obs_prob,pi,hidden_state,obs_state,p)
            # print('True Obj Func: ',sum(np.log(prob)))
            # at1,bt1,pit1,func1=EMTrain(A,B,pi0,data1[0:1500],0.0005,hidden_state,obs_state,p)
            
            #Save our model
        post_A=np.array(post_A)
        post_B=np.array(post_B)
        post_pi=np.array(post_pi)
        os.chdir(f'BootstrapResult_{file_name}')
        os.mkdir(f'Missingrate{rate}')
        # Save the results
        np.save(f'Missingrate{rate}/post_A.npy',post_A)
        np.save(f'Missingrate{rate}/post_B.npy',post_B)
        np.save(f'Missingrate{rate}/post_pi.npy',post_pi)
        os.chdir('..')
      


