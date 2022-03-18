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
    p=Pool(16)
    rate=0.3
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
    A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    pi0=np.array([0.4,0.3,0.3])
    # testing code
    #A=transition
    #B=obs_prob
    #pi0=pi
    
    out_obj=[]
    os.mkdir('SimulationResult')
    os.mkdir('NaiveResult')
    for rate in [0.3,0.5,0.7,0.9]:
        #os.mkdir(f'Missingrate{rate}')
        num=2
        out_obj=[]
        A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
        for i in range(0,num):
            # Same data, different initial
            #A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
            
            #A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #pi0=np.array([0.4,0.3,0.3])
            A=np.random.dirichlet((8,8,8),3)
            B=np.random.dirichlet((8,8,8),3)
            pi0=np.random.dirichlet((8,8,8),1)[0]
            at,bt,pit,func=EMTrain(A,B,pi0,data[0:2000],0.0001,hidden_state,obs_state,p)
            
            data1=[]
            for i in range(0,len(data)):
                data1.append(data[i][data[i]!='None'])
            data1=np.array(data)
            at1,bt1,pit1,func1=EMTrain(A,B,pi0,data1[0:2000],0.0001,hidden_state,obs_state,p)
            
            #Save our model
            
            os.chdir('SimulationResult')
            os.mkdir(f'Missingrate{rate}')
            os.mkdir(f'Missingrate{rate}/Experiment{i}')
            # Save the results
            np.save(f'Missingrate{rate}/Experiment{i}/at.npy',at)
            np.save(f'Missingrate{rate}/Experiment{i}/bt.npy',bt)
            np.save(f'Missingrate{rate}/Experiment{i}/pit.npy',pit)
            np.save(f'Missingrate{rate}/Experiment{i}/data.npy',data)
            np.save(f'Missingrate{rate}/Experiment{i}/TrueHidden.npy',hidden_data)
            np.save(f'Missingrate{rate}/Experiment{i}/ObjFunc.npy',func)
            os.chdir('..')
            
            #Save the Naive Model
            os.chdir('NaiveResult')
            os.mkdir(f'Missingrate{rate}')
            os.mkdir(f'Missingrate{rate}/Experiment{i}')
            np.save(f'Missingrate{rate}/Experiment{i}/at.npy',at1)
            np.save(f'Missingrate{rate}/Experiment{i}/bt.npy',bt1)
            np.save(f'Missingrate{rate}/Experiment{i}/pit.npy',pit1)
            np.save(f'Missingrate{rate}/Experiment{i}/data.npy',data)
            np.save(f'Missingrate{rate}/Experiment{i}/TrueHidden.npy',hidden_data)
            np.save(f'Missingrate{rate}/Experiment{i}/ObjFunc.npy',func1)
            os.chdir('..')
            
        #print('Analyzing the result...')
        
        #os.chdir(f'Missingrate{rate}')
        '''
        output=read_data(num)
        result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p)
        os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        '''




