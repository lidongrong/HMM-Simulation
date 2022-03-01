# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:12:43 2022

@author: lidon
"""

import numpy as np
from ZMARGibbs import*
from HMM import*
from SeqSampling import*
from EMHMM import*
from PMAP import*
from Test_Viterbi import*
import matplotlib.pyplot as plt
import os

# predict + plot the result
# rate is an array, listing the missing rates
def pred(rate,A,B,pi,hidden_state,obs_state):
    h=hidden_state
    o=obs_state
    
    os.mkdir('PredResult')
    
    # the overall accuracy rate of viterbi and PMAP
    # rg stands for random guess
    vtb_overall_acc=[]
    pmap_overall_acc=[]
    rg_overall_acc=[]
    
    # the accuracy rate on observed data
    vtb_obs_acc=[]
    pmap_obs_acc=[]
    rg_obs_acc=[]
    
    # the accuracy rate on missing data
    vtb_mis_acc=[]
    pmap_mis_acc=[]
    rg_mis_acc=[]
    
    
    for r in rate:
        # generate data
        A0,B0,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,r)
        # prediction
        z_vtb=vtb(data,A,B,pi,h,o)
        z_pmap=PMAP_dataset(data,A,B,pi,h,o)
        z_rg=random_guesser(data,h)
        
        # test overall accuracy
        vtb_overall_acc.append(np.sum(z_vtb==hidden_data)/(data.shape[0]*data.shape[1]))
        pmap_overall_acc.append(np.sum(z_pmap==hidden_data)/(data.shape[0]*data.shape[1]))
        rg_overall_acc.append(np.sum(z_rg==hidden_data)/(data.shape[0]*data.shape[1]))
        # accuracy on observed data
        vtb_obs_acc.append(np.sum((z_vtb==hidden_data)&(data!='None'))/np.sum(data!='None'))
        pmap_obs_acc.append(np.sum((z_pmap==hidden_data)&(data!='None'))/np.sum(data!='None'))
        rg_obs_acc.append(np.sum((z_rg==hidden_data)&(data!='None'))/np.sum(data!='None'))
        # accuracy on missing data
        if r==0:
            vtb_mis_acc.append(np.sum(z_vtb==hidden_data)/(data.shape[0]*data.shape[1]))
            pmap_mis_acc.append(np.sum(z_pmap==hidden_data)/(data.shape[0]*data.shape[1]))
            rg_mis_acc.append(np.sum(z_rg==hidden_data)/(data.shape[0]*data.shape[1]))
        else:
            vtb_mis_acc.append(np.sum((z_vtb==hidden_data)&(data=='None'))/np.sum(data=='None'))
            pmap_mis_acc.append(np.sum((z_pmap==hidden_data)&(data=='None'))/np.sum(data=='None'))
            rg_mis_acc.append(np.sum((z_rg==hidden_data)&(data=='None'))/np.sum(data=='None'))
    
    # plot overall accuracy
    plt.plot(rate,vtb_overall_acc,'blue',label=f'Viterbi')
    plt.plot(rate,pmap_overall_acc,'green',label=f'PMAP')
    plt.plot(rate,rg_overall_acc,'red',label=f'Random Guess')
    plt.xlabel('missing rate')
    plt.legend(loc='best')
    plt.savefig(f'PredResult/OverallAccuracy.png')
    plt.close('all')
    
    # plot accuracy wrt observation
    plt.plot(rate,vtb_obs_acc,'blue',label=f'Viterbi')
    plt.plot(rate,pmap_obs_acc,'green',label=f'PMAP')
    plt.plot(rate,rg_obs_acc,'red',label=f'Random Guess')
    plt.xlabel('missing rate')
    plt.legend(loc='best')
    plt.savefig(f'PredResult/ObsAccuracy.png')
    plt.close('all')
    
    # plot accuracy wrt missing data
    plt.plot(rate,vtb_mis_acc,'blue',label=f'Viterbi')
    plt.plot(rate,pmap_mis_acc,'green',label=f'PMAP')
    plt.plot(rate,rg_mis_acc,'red',label=f'Random Guess')
    plt.xlabel('missing rate')
    plt.legend(loc='best')
    plt.savefig(f'PredResult/MisAccuracy.png')
    plt.close('all')  
        
rate=[0,0.3,0.5,0.7,0.9]

pred(rate,transition,obs_prob,pi,hidden_state,obs_state)    
        
        
        
    