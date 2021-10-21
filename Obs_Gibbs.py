# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:53:10 2021

@author: s1155151972
"""


import numpy as np
import HMM
import Sampling
import time
from multiprocessing import pool


print('Data Preprocessing and Initializing Parameters...')
data=Sampling.data

# Initialize transition matrix A
A=np.array([[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.5,0.5,0],[0,0,0,0.5,0.5],[0,0,0,0,1]])

# Initialize observation matrix B
B=np.random.dirichlet((1,1,1,1,1),5)


# Sample the latent state using forward backward sampling
# Based on research note in 2021.10.21
# A,B: transition matrix and observation matrix
# state: observed sequence with missing observation
def f_b_sampling(A,B,obs):
    
    # Check if the whole sequence is missing
    if np.all(obs=='None'):
        return obs
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]
    # length of the index
    T=len(indexer)
    state=HMM.obs_state
    hidden_state=HMM.hidden_state
    # start to compute alpha recursively
    alpha=np.zeros((T,len(state)))
    
    # In this case, t1=t0
    if indexer[0]==0:
        alpha[0][0]=1
    else:
        # in this case, we initialize z_{t1} conditioned on the case that t0=1
        y=np.where(state==obs[indexer[0]])[0][0]
        initial=np.zeros(len(state))
        initial[0]=1
        alpha[0,:]=np.dot(initial,np.linalg.matrix_power(A, indexer[0]))*B[:,y]
    
    for i in range(1,T):
        y=np.where(state==obs[indexer[i]])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],np.linalg.matrix_power(A,indexer[i]-indexer[i-1]))*B[:,y]
        
    # initialize the output
    output=[]
    
    # First sample the last latent state
    w=alpha[T-1,:]/sum(alpha[T-1,:])
    output.append(np.random.choice(hidden_state,1,p=w)[0])
    
    # Then sample each latent state in sequence
    for t in range(1,T):
        # compute the index of hidden state z_{t_{i+1}}
        hidden_index=np.where(hidden_state==output[t-1])[0][0]
        # compute the transition matrix between the two observed states
        trans=np.linalg.matrix_power(A,indexer[T-t]-indexer[T-t-1])
        # generate the probability distribution
        w=trans[:,hidden_index]*alpha[T-1-t,:]/np.dot(trans[:,hidden_index],alpha[T-1-t,:])
        output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    seq=np.repeat('None',len(obs))
    seq[indexer]=output
    return seq

# sample the whole latent sequence out
def sample_latent_seq(data,I,A,B):
    for i in range(0,data.shape[0]):
        I[i,:]=f_b_sampling(A,B,data[i])
    return I


# initialize latent sequence I
I=[]
for i in range(0,data.shape[0]):
    I[i]=np.repeat('None',data.shape[1])

I=sample_latent_Seq(data,I,A,B)

# Used later in computing likelihood
I_buffer=I.copy()

# Sample B out in Gibbs Sampling
# B: B sampled in last iteration
def sample_B(data,I,B):
    for j in range(0,B.shape[0]):
        # for j th row of B, calculate the number of each 
        # observed states respectively
        n1=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[0]))
        n2=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[1]))
        n3=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[2]))
        n4=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[3]))
        n5=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[4]))
        B[j,:]=np.random.dirichlet((1+n1,1+n2,1+n3,1+n4,1+n5),1)[0]
    return B

# Sample A out using rejection sampling
# The code is based on research note on 2021.10.21
# A: transition matrix from last iteration
def sample_A(data,I,A):
    # Because we assume the last state is an absorbing state
    for j in range(0,A.shape[0]-1):
        # How many times the state remains or leave j from the latent space w.r.t. observed data
        stay_freq=0
        change_freq=0
        pass
    pass

        
        



    

