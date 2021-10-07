# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:00:27 2021

@author: s1155151972
"""

# This program compares the efficiency of the Gibbs sampler at different missing levels



import numpy as np
import scipy.stats as stats
import math
import HMM
import Sampling
from multiprocessing import Pool
import time
from Gibbs_Sampler import*

obs_data=Sampling.obs_data

data1=Sampling.Missing(obs_data,p=0.1)
data2=Sampling.Missing(obs_data,p=0.3)
data3=Sampling.Missing(obs_data,p=0.5)
data4=Sampling.Missing(obs_data,p=0.65)
data5=Sampling.Missing(obs_data,p=0.8)

full_data=[data1,data2,data3,data4,data5]

class posterior:
    def __init__(self):
        post_A=None
        post_B=None


post1=posterior()
post2=posterior()
post3=posterior()
post4=posterior()
post5=posterior()

post=[post1,post2,post3,post4,post5]


if __name__=='__main__':
    p=Pool(8)
    for m in range(0,5):
        
        data=full_data[m]
        print('Data preprocessing...')
        miss_place=np.where(data=='None')
        miss_x=miss_place[0]
        miss_y=miss_place[1]
    
        print('Initializing parameters...')

        # Initialize transition matrix A
        A=np.array([[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.5,0.5,0],[0,0,0,0.5,0.5],[0,0,0,0,1]])

        # Initialize observation matrix B
        B=np.random.dirichlet((1,1,1,1,1),5)
    
        # Initialize omis by random labelling
        w=np.random.dirichlet((1,1,1,1,1),1)[0]
        for i in range(0,len(miss_x)):
            data[miss_x[i],miss_y[i]]=np.random.choice(HMM.obs_state,1,p=w)[0]
    
    
        I=[]

        for i in range(0,data.shape[0]):
            I.append(f_b_sampling(A,B,HMM.obs_state,data[i]))

        I=np.array(I)   
        print('Start Gibbs sampling...')
        #p=Pool(8)
        post[m].post_A,post[m].post_B=Gibbs(A,B,data,I,p,25000)
    
    
    
