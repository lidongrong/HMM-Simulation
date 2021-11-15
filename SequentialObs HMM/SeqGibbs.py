


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:58:05 2021

@author: a
"""

import numpy as np
import HMM
import SeqSampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math


# initialize data, transition matrix and obs_matrix
def data_initializer():
    print('Data Preprocessing and Initializing Parameters...')
    data=Sampling.data

    # Initialize transition matrix A
    A=np.array([[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.5,0.5,0],[0,0,0,0.5,0.5],[0,0,0,0,1]])
    

    # Initialize observation matrix B
    B=np.random.dirichlet((1,1,1,1,1),5)
    return A,B,data

# Sample the latent state using forward backward sampling
# Based on research note in 2021.10.21
# A,B: transition matrix and observation matrix
# state: observed sequence with missing observation
# obs: the observed sequence
def f_b_sampling(A,B,obs):
    
    # Check if the whole sequence is missing
    if np.all(obs=='None'):
        return obs
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]
    # length of the whole observed sequence
    T=len(indexer)
    hidden_state=HMM.hidden_state
    obs_state=HMM.obs_state
    # start to compute alpha recursively
    alpha=np.zeros((T,len(hidden_state)))
    
    # we assume that the chain starts from A, i.e. the initial distribution pi is a trivial distribution
    # the i th row of P^n is the n step probability that starts from 
    # to initialize the first line of alpha, notice that alpha_1(i)=pi_i * b_i(y1)
    # so we have to acquire the state of the first observation 
    first_index=np.where(obs_state==obs[indexer[0]])[0][0]
    pi=np.linalg.matrix_power(A,indexer[0])[0,:]
    alpha[0,:]=pi*B[:,first_index]
    
    for i in range(1,T):
        index=np.where(obs_state==obs[indexer[i]])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,index]
    
    #print(alpha)
    # intialize the output
    output=[]
    
    # first sample z_T
    w=alpha[T-1,:]/sum(alpha[T-1,:])
    output.append(np.random.choice(hidden_state,1,p=w)[0])
    
    # then sample z_{t-1}, z_{t-2} in a backward manner
    for t in range(1,T):
        # compute the index of hidden state z_{t+1}
        hidden_index=np.where(hidden_state==output[t-1])[0][0]
        w=A[:,hidden_index]*alpha[T-1-t,:]/np.dot(A[:,hidden_index],alpha[T-1-t,:])
        output.append(np.random.choice(hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    
    seq=np.repeat('None',len(obs))
    seq[indexer]=output
    return seq
    
    
# sample the whole latent sequence out
# A,B:transition matrix and obs matrix
# data: partially observed data
# I: latent sequence from the last iteration
def sample_latent_seq(data,I,A,B):
    for i in range(0,data.shape[0]):
        I[i,:]=f_b_sampling(A,B,data[i])
    return I



# initialize the latent sequence as initial guess
def latent_seq_initializer(data,A,B):
    # initialize latent sequence I
    I=[]
    for i in range(0,data.shape[0]):
        I.append(np.repeat('None',data.shape[1]))
    I=np.array(I)

    I=sample_latent_seq(data,I,A,B)
    return I

# Make an initial guess of the parameters and latent variables
def initialize():
    A,B,data=data_initializer()
    I=latent_seq_initializer(data,A,B)
    return A,B,data,I



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
    B=np.array(B)
    new_B=B
    #print(B)
    return new_B









'''
# Sample A out based on the full latent sequence
# Algorithm based on Resarch note in 2021.10.29
# A: transition matrix from the last iteration
def sample_A(data,I,A):
    A=np.zeros((A.shape[0],A.shape[1]))
    A[4,4]=1
    # Count the total number that the chain stays at a state
    for j in range(0,4):
        #state_num=np.sum(I[:,0:9]==HMM.hidden_state[j])
        # how many times the state stays at state j
        stay_freq=0
        # how many time the state move away
        change_freq=0
        
        # Count how many times it happens that the chain stays 
        for k in range(0,I.shape[1]-1):
            #print(freq)
            a=(I[:,k]==I[:,k+1])
            b=(I[:,k]==HMM.hidden_state[j])
            stay_freq=stay_freq+np.sum(np.logical_and(a,b))
            c=(I[:,k]!=I[:,k+1])
            change_freq=change_freq+np.sum(np.logical_and(c,b))
                
            
        
        A_posterior=np.random.dirichlet((1+stay_freq,1+change_freq))
        A[j,j]=A_posterior[0]
        A[j,j+1]=A_posterior[1]
    return A

'''

### Sample A using Metropolis-Within-Gibbs
# A: transition matrix from the last iteration
def sample_A(data,I,A):
    
    for j in range(0,A.shape[0]-1):
        new_A=A.copy()
        a=A[j,j]
        
        new_a=np.random.beta(2,1/a,1)[0]
        new_A[j,j]=new_a
        new_A[j,j+1]=1-new_a
        
        # initialize the old and new log likelihood
        log_p=0
        log_new_p=0
        
        for i in range(0,I.shape[0]):
            # renew the log likelihood function based on each observation
            # indexer: indices that does not miss
            indexer=np.where(I[i]!='None')[0]
            if indexer.size!=0:
                # the starting state of the whole sequence
                pos=np.where(HMM.hidden_state==I[i][indexer[0]])[0][0]
                # renew the log likelihood
                log_p=log_p+np.log(np.linalg.matrix_power(A,indexer[0])[0][pos])
                log_new_p=log_new_p+np.log(np.linalg.matrix_power(new_A,indexer[0])[0][pos])
        
        
        
        
        stay_freq=0
        change_freq=0
        # Count how many times it happens that the chain stays 
        for k in range(0,I.shape[1]-1):
            #print(freq)
            tmp_a=(I[:,k]==I[:,k+1])
            tmp_b=(I[:,k]==HMM.hidden_state[j])
            stay_freq=stay_freq+np.sum(np.logical_and(tmp_a,tmp_b))
            tmp_c=(I[:,k]!=I[:,k+1])
            change_freq=change_freq+np.sum(np.logical_and(tmp_b,tmp_c))
        
        
        log_p=log_p+stay_freq*np.log(a)+change_freq*np.log(1-a)
        log_new_p=log_new_p+stay_freq*np.log(new_a)+change_freq*np.log(1-new_a)
        
        
        # Finally, perform M-H step
        
        if log_p==np.inf:
            print('oops!')
            r=0
        else:
            r=log_new_p+np.log(stats.beta.pdf(a,2,1/new_a))-log_p-np.log(stats.beta.pdf(new_a,2,1/a))
            r=min(0,r)
            
        u=np.random.uniform(0,1,1)[0]
        
        if np.log(u)<r:
            A[j,j]=new_a
            A[j,j+1]=1-new_a
        else:
            A[j,j]=a
            A[j,j+1]=1-a
            
        
    
    return A
    
      
        
        



# Gibbs Sampling accelerated by parallel computing
# input I,A,B: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,n):
    post_A=[]
    post_B=[]
    
    # calculate the data size
    ds=data.shape[0]
    
    for i in range(0,n):
        print(i)
        A=sample_A(data,I,A)
        B=sample_B(data,I,B)
        new_A=A.copy()
        post_A.append(new_A)
        print(A)
        post_B.append(B)
        
        
        
        I=p.starmap(sample_latent_seq,[(data[0:ds//8,:],I[0:ds//8,:],A,B),(data[ds//8:2*ds//8,:],
                                                                     I[ds//8:2*ds//8,:],A,B),(data[2*ds//8:3*ds//8,:],
                                                                                         I[2*ds//8:3*ds//8,:],A,B),
                                                                                         (data[3*ds//8:4*ds//8,:],
                                                                                          I[3*ds//8:4*ds//8,:],A,B),
                                                                                         (data[4*ds//8:5*ds//8,:],
                                                                                          I[4*ds//8:5*ds//8,:],A,B),
                                                                                         (data[5*ds//8:6*ds//8,:],
                                                                                          I[5*ds//8:6*ds//8,:],A,B),
                                                                                         (data[6*ds//8:7*ds//8,:],
                                                                                          I[6*ds//8:7*ds//8,:],A,B),
                                                                                         (data[7*ds//8:,:],
                                                                                          I[7*ds//8:,:],A,B)])
                                                                                        
        I=np.vstack((I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]))
        #I=sample_latent_seq(data,I,A,B)
        
        
        
        
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    
    return post_A,post_B
        
                  
if __name__=='__main__':
    A,B,data,I=initialize()
    
    
    p=Pool(8)
    post_A,post_B=parallel_Gibbs(data,I,A,B,6000)
    
