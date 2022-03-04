# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:58:41 2022

@author: s1155151972
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
from NaiveHMM import *
from ZResultAnalysis import*


# Main program for HMM simulation 
# Data Generation + Missing value + Gibbs Sampling

# HMM construction
'''
transition=np.array(
        [[0.6,0.2,0.1,0.05,0.05],[0.05,0.6,0.2,0.1,0.05],[0.05,0.05,0.6,0.2,0.1],[0.05,0.05,0.1,0.6,0.2],
         [0.05,0.05,0.1,0.2,0.6]]
        )           
state=np.array(['A','B','C','D','E'])
hidden_state=state
obs_state=np.array(['Blue','Red','Green','Purple','Grey'])
    
obs_prob=np.array([[0.7,0.2,0.05,0.04,0.01],[0.01,0.7,0.2,0.05,0.04],[0.04,0.01,0.7,0.2,0.05],
                   [0.05,0.04,0.01,0.7,0.2],[0.2,0.05,0.04,0.01,0.7]
        ])
pi=[0.5,0.2,0.2,0.1,0]
'''


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
    p=Pool(mp.cpu_count())
    
    # Define the output object class
    # Which is a list of Out objects 
    out_obj=[]
    
    for rate in [0,0.3,0.5,0.7,0.9]:
        os.mkdir(f'Missingrate{rate}')
        num=4
        out_obj=[]
        for i in range(0,num):
            A,B,pi,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
            #print(data[0])
            post_A,post_B,post_pi,latent_seq,log_prob=parallel_Gibbs(data,I,A,B,pi,10000,hidden_state,obs_state,p)
            out_obj.append(Out(data,post_A,post_B,post_pi,latent_seq,log_prob,hidden_data))
            os.mkdir(f'Missingrate{rate}/Experiment{i}')
            # Save the results
            np.save(f'Missingrate{rate}/Experiment{i}/Post_A.npy',out_obj[i].post_A)
            np.save(f'Missingrate{rate}/Experiment{i}/Post_B.npy',out_obj[i].post_B)
            np.save(f'Missingrate{rate}/Experiment{i}/Post_pi.npy',out_obj[i].post_pi)
            np.save(f'Missingrate{rate}/Experiment{i}/latent_seq.npy',out_obj[i].latent_seq)
            np.savetxt(f'Missingrate{rate}/Experiment{i}/log_prob.txt',out_obj[i].log_prob)
            np.save(f'Missingrate{rate}/Experiment{i}/data.npy',out_obj[i].data)
            np.save(f'Missingrate{rate}/Experiment{i}/TrueHidden.npy',out_obj[i].true_hidden)
    
            
        print('Analyzing the result...')
        
        os.chdir(f'Missingrate{rate}')
        
        output=read_data(num)
        result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p)
        os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))