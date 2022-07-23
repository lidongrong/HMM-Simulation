# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:19:30 2022

@author: lidon
"""
###
# Analyze the results of EM algorithm
###

import os
import matplotlib.pyplot as plt
import math
import numpy as np
from ZMARGibbs import*
from multiprocessing import Pool
import multiprocessing as mp
from EMHMM import*


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

class Out:
    def __init__(self,data,post_A,post_B,post_pi,obj_func,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.post_pi=post_pi
        self.true_hidden=true_hidden
        self.obj_func=obj_func

def read_data1(file_path,num):
    output=[]

    for i in range(0,num):
        post_A=np.load(f'{file_path}/Experiment{i}/at.npy',allow_pickle=True)
        post_B=np.load(f'{file_path}/Experiment{i}/bt.npy',allow_pickle=True)
        pi=np.load(f'{file_path}/Experiment{i}/pit.npy',allow_pickle=True)
        obj_func=np.load(f'{file_path}/Experiment{i}/ObjFunc.npy',allow_pickle=True)
        data=np.load(f'{file_path}/Experiment{i}/data.npy',allow_pickle=True)
        hidden_seq=np.load(f'{file_path}/Experiment{i}/TrueHidden.npy',allow_pickle=True)

        output.append(Out(data,post_A,post_B,pi,obj_func,hidden_seq))
    return output

# Read all data with different missing rates
# rate: a vector of different missing rates
def read_all(file_path,rate,num):
    out_result=[]
    for r in rate:
        path=file_path+f'/MissingRate{r}'
        output=read_data1(path,num)
        out_result.append(output)
    return out_result

# permute the output to the right position
# out is the output array
def permute(out,pi,transition,obs_prob):
    for output in out:
        est_pi=output.post_pi[-1]
        # if the estimated pi is not at the right order
        if np.any(-np.sort(-est_pi)!=est_pi):
            right_pi=-np.sort(-est_pi)
            permutation=[np.where(right_pi[i]==est_pi)[0][0] for i in range(0,len(est_pi))]
            for i in range(0,len(output.post_pi)):
                output.post_A[i]=output.post_A[i][permutation,:]
                output.post_A[i]=output.post_A[i][:,permutation]
                output.post_B[i]=output.post_B[i][permutation,:]
                #output.post_B[i]=output.post_B[i][:,permutation]
                output.post_pi[i]=output.post_pi[i][permutation]
    
    return out

# only apply to a vector of pi, A and B
def small_permute(post_pi,post_A,post_B,pi,transition,obs_prob):
    est_pi=post_pi[-1]
    # if the estimated pi is not at the right order
    if np.any(-np.sort(-est_pi)!=est_pi):
        right_pi=-np.sort(-est_pi)
        permutation=[np.where(right_pi[i]==est_pi)[0][0] for i in range(0,len(est_pi))]
        for i in range(0,len(post_pi)):
            post_A[i]=post_A[i][permutation,:]
            post_A[i]=post_A[i][:,permutation]
            post_B[i]=post_B[i][permutation,:]
            #output.post_B[i]=output.post_B[i][:,permutation]
            post_pi[i]=post_pi[i][permutation]
    return post_pi,post_A,post_B

#permute all observations with different missing rates
def permute_all(out,pi,transition,obs_prob,rate):
    for i in range(0,len(rate)):
        out[i]=permute(out[i],pi,transition,obs_prob)
        
    return out

# calculate Mean Absolute Error with respect to different experiments
# under different missing rates
def MAE(out,pi,transition,obs_prob):
    mae_pi=[]
    mae_A=[]
    mae_B=[]
    
    for i in range(0,len(out)):
        mae_pi_p=[]
        mae_pi_A=[]
        mae_pi_B=[]
        for j in range(0,len(out[i])):
            # total number of iteration
            iter_len=len(out[i,j].post_pi)
            mae_pi_p.append(np.sum(abs(out[i,j].post_pi[iter_len-1]-pi)))
            mae_pi_A.append(np.sum(abs(out[i,j].post_A[iter_len-1]-transition)))
            mae_pi_B.append(np.sum(abs(out[i,j].post_B[iter_len-1]-obs_prob)))
        mae_pi.append(mae_pi_p)
        mae_A.append(mae_pi_A)
        mae_B.append(mae_pi_B)
    return np.array(mae_pi),np.array(mae_A),np.array(mae_B)

