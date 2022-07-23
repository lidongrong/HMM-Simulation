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

#transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
transition=np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.9,0.05,0.05],
                   [0.1,0.7,0.2],
                   [0.15,0.05,0.8]
    ])

#pi=np.array([0.7,0.2,0.1])
pi=np.array([0.6,0.3,0.1])

if __name__=='__main__':
    
    
    
    
    # Use multicore CPU
    p=Pool(16)
    # missing rate, used to generate full data
    r=0
    # total sample
    size=500
    # seq length
    length=10
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,0,size,length)
    # A=np.random.dirichlet((8,8,8),3)
    # B=np.random.dirichlet((8,8,8),3)
    # pi0=np.random.dirichlet((8,8,8),3)[0]
    # testing code
    #A=transition
    #B=obs_prob
    #pi0=pi
    # total experiments
    num=30
    
    # Construct list for starting time
    # for i th exp for different missing rate, use same initial
    A_start=[np.random.dirichlet((8,8,8),3) for i in range(0,num)]
    B_start=[np.random.dirichlet((8,8,8),3) for i in range(0,num)]
    pi_start=[np.random.dirichlet((8,8,8),3)[0] for i in range(0,num)]
    
    # acquire time
    t=datetime.now()
    time_list=[t.year,t.month,t.day,t.hour,t.minute,t.second]
    time_list=[str(x) for x in time_list]
    time_list='_'.join(time_list)
    #folder_name=f'BatchGMM{time_list}'
    
    # Construct output path
    out_obj=[]
    file_name=f'Result{time_list}'
    os.mkdir(file_name)
    os.mkdir(f'{file_name}/SimulationResult')
    sim_path=f'{file_name}/SimulationResult'
    os.mkdir(f'{file_name}/NaiveResult')
    naive_path=f'{file_name}/NaiveResult'
    os.mkdir(f'{file_name}/CompleteResult')
    complete_path=f'{file_name}/CompleteResult'
    for rate in [0.1,0.2,0.3,0.5,0.7,0.9]:
        os.mkdir(f'{sim_path}/Missingrate{rate}')
        os.mkdir(f'{naive_path}/Missingrate{rate}')
        os.mkdir(f'{complete_path}/Missingrate{rate}')
        print(os.getcwd())
        # total number of experiments
        
        out_obj=[]
        
        #size=5
        #length=10
        #r=0
        #generate complete data
        #A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,r,size,length)
        #os.chdir(file_name)
        
        data1=data.copy()
        data1=Sampling.Missing(data1,rate)
        
        # construct dataset for Naive method
        data2=[]
        for j in range(0,len(data1)):
            data2.append(data1[j][data1[j]!='None'])
        data2=np.array(data2)
        
        for i in range(0,num):
            # Same data, different initial
            #A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate)
            
            #A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
            #pi0=np.array([0.4,0.3,0.3])
            # for i th experiment under missing rate p, use same initial
            A=A_start[i]
            B=B_start[i]
            pi0=pi_start[i]
            #prob=y_prob(data,transition,obs_prob,pi,hidden_state,obs_state,p)
            
            
            
            
            # Our Approach
            #print('True Obj Func: ',sum(np.log(prob)))
            print('Our Approach')
            at,bt,pit,func=EMTrain(A,B,pi0,data1,0.0001,hidden_state,obs_state,p)
            
            # Naive Approach
            #prob=y_prob(data1,transition,obs_prob,pi,hidden_state,obs_state,p)
            #print('True Obj Func: ',sum(np.log(prob)))
            print('Naive Approach')
            at1,bt1,pit1,func1=EMTrain(A,B,pi0,data2,0.0001,hidden_state,obs_state,p)
            
            # Based on Complete data
            print('Approach Based on Complete Data')
            at2,bt2,pit2,func2=EMTrain(A,B,pi0,data,0.0001,hidden_state,obs_state,p)
            
            #Save our model
            #os.mkdir(f'Missingrate{rate}')
            #os.chdir(f'SimulationResult{time_list}')
            os.mkdir(f'{sim_path}/Missingrate{rate}/Experiment{i}')
            # Save the results
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/at.npy',at)
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/bt.npy',bt)
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/pit.npy',pit)
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/data.npy',data1)
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/TrueHidden.npy',hidden_data)
            np.save(f'{sim_path}/Missingrate{rate}/Experiment{i}/ObjFunc.npy',func)

            
            #Save the Naive Model
            #os.chdir(f'{naive_path}/NaiveResult{time_list}')
            #os.mkdir(f'Missingrate{rate}')
            os.mkdir(f'{naive_path}/Missingrate{rate}/Experiment{i}')
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/at.npy',at1)
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/bt.npy',bt1)
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/pit.npy',pit1)
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/data.npy',data2)
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/TrueHidden.npy',hidden_data)
            np.save(f'{naive_path}/Missingrate{rate}/Experiment{i}/ObjFunc.npy',func1)
            #os.chdir('..')
            
            #Save the Model based on complete data
            #os.chdir(f'{naive_path}/NaiveResult{time_list}')
            #os.mkdir(f'Missingrate{rate}')
            os.mkdir(f'{complete_path}/Missingrate{rate}/Experiment{i}')
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/at.npy',at2)
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/bt.npy',bt2)
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/pit.npy',pit2)
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/data.npy',data)
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/TrueHidden.npy',hidden_data)
            np.save(f'{complete_path}/Missingrate{rate}/Experiment{i}/ObjFunc.npy',func2)
            #os.chdir('..')
            
        #print('Analyzing the result...')
        
        #os.chdir(f'Missingrate{rate}')
        '''
        output=read_data(num)
        result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p)
        os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        '''




