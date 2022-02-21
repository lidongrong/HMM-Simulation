# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:00:30 2022

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


# initialize data, transition matrix and obs_matrix
def data_initializer(transition,obs_prob,pi,hidden_state,obs_state,rate):
    #print('Data Preprocessing and Initializing Parameters...')
    #data=Sampling.data
    
    # Generate the HMM object
    MC=HMM.HMM(hidden_state,obs_state,transition,obs_prob,pi)


    # Construct Data
    hidden_data=Sampling.Hidden_Generator(MC,20,6000)
    data=Sampling.Obs_Generator(MC,hidden_data)

    # Missing at random
    data=Sampling.Missing(data,p=rate)
    
    # Initialize transition matrix A
    
    A=np.array([[1/transition.shape[1] for i in range(0,transition.shape[1])] for j in range(0,transition.shape[0])])
    for i in range(0,A.shape[0]):
        A[i,:]=A[i,:]/sum(A[i,:])
    
    # Initialize observation matrix B and pi
    # Dirichlet parameter of B
    
    #A=np.random.dirichlet(np.ones(transition.shape[1]),transition.shape[0])
    
    #alpha_B=np.array([1 for i in range(0,obs_prob.shape[1])])
    #B=np.random.dirichlet(alpha_B,obs_prob.shape[0])
        
    B=np.array([[1/obs_prob.shape[1] for i in range(0,obs_prob.shape[1])] for j in range(0,obs_prob.shape[0])])
    for i in range(0,B.shape[0]):
        B[i,:]=B[i,:]/sum(B[i,:])
    
    
    pi=np.array([1/len(pi) for i in range(0,len(pi))])
    pi=pi/sum(pi)
    
    
    
    
    return A,B,pi,data,hidden_data


# calculate alpha_t(z_t) in HMM models
def alpha_and_beta(obs,A,B,pi,hidden_state,obs_state):
    T=len(obs)
    alpha=np.zeros((T,len(hidden_state)))
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]

    # Handle the boundary case: the first line of alpha
    if obs[0]!='None':
        y=np.where(obs_state==obs[0])[0][0]
        
        alpha[0,:]=pi*B[:,y]
    else:
        alpha[0,:]=pi
    
    # scaling
    alpha[0,:]=alpha[0,:]/sum(alpha[0,:])
    
    
    for i in range(1,T):
        # The case when i is an observable data
        if i in indexer:
            y=np.where(obs_state==obs[i])[0][0]
            alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,y]
            alpha[i,:]=alpha[i,:]/sum(alpha[i,:])
        else:
            alpha[i,:]=np.dot(alpha[i-1,:],A)
            alpha[i,:]=alpha[i,:]/sum(alpha[i,:])
    
    T=len(obs)
    beta=np.zeros((T,len(hidden_state)))
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]
    beta[T-1,:]=[1 for i in range(0,len(hidden_state))]
    beta[T-1,:]=beta[T-1]/sum(alpha[T-1,:])
    for i in range(T-2,-1,-1):
        if (i+1) in indexer:
            y=np.where(obs_state==obs[i+1])[0][0]
            beta[i,:]=[np.sum(beta[i+1,:]*A[j,:]*B[:,y]) for j in range(0,len(hidden_state))]
            beta[i,:]=beta[i,:]/sum(alpha[i,:])
        else:
            beta[i,:]=np.dot(beta[i+1,:],A)
            beta[i,:]=beta[i,:]/sum(alpha[i,:])
    return alpha,beta

# calculate the probability that p(y_o,z_t,z_{t+1})
# z1,z2 is the respective hidden state
# t1,t2 is the position of z1 and z2, note that t2=t1+1
# alph, bet is the respective alpha and beta form of obs
def ksi(obs,A,B,pi,z1,z2,t1,t2,hidden_state,obs_state):
    z1_pos=np.where(hidden_state==z1)[0][0]
    z2_pos=np.where(hidden_state==z2)[0][0]
    
    alph,bet=alpha_and_beta(obs,A,B,pi,hidden_state,obs_state)
    
    indexer=np.where(obs!='None')[0]
    
    if t2 in indexer:
        y=np.where(obs_state==obs[t2])[0][0]
        return alph[t1,z1_pos]*A[z1_pos,z2_pos]*B[z2_pos,y]*bet[t2,z2_pos]
    else:
        
        return alph[t1,z1_pos]*A[z1_pos,z2_pos]*bet[t2,z2_pos]

# calculate ksi of the whole sequence given z1 and z2
def seq_ksi(obs,A,B,pi,z1,z2,hidden_state,obs_state):
    h=hidden_state
    o=obs_state
    ks=sum([ksi(obs,A,B,pi,z1,z2,t,t+1,h,o) for t in range(0,len(obs)-1)])
    return ks

# compute the probability of p(z_t|y)
# t: position of z
# z: latent sequence
def gamma(t,alph,bet,z,hidden_state):
    z_pos=np.where(hidden_state==z)
    return alph[t,z_pos]*bet[t,z_pos]/np.dot(alph[t,:],bet[t,:])


# compute p(y_o,z_t)
# return a vector
def ph(obs,A,B,pi,t,hidden_state,obs_state):
    alph,bet=alpha_and_beta(obs,A,B,pi,hidden_state,obs_state)
    #bet=beta(obs,A,B,pi,hidden_state,obs_state)
    
    return alph[t,:]*bet[t,:]
    
# a specifically designed parallel function for updating B
# compute p(y_o,z_t)*I(y_o_t==y_t), return a vector indicating each y
# j: the index of z
def ph_seq(obs,A,B,pi,j,hidden_state,obs_state):
    h=hidden_state
    o=obs_state
    #out=sum([ph(obs,A,B,pi,t,h,o)*((o==obs[t]).astype(np.int32)) for t in range(0,len(obs))])
    out1=[ph(obs,A,B,pi,t,h,o)[j] for t in range(0,len(obs))]
    out2=np.array([obs==obs_state[i] for i in range(0,len(obs_state))])
    out2=out2.astype(np.int)
    out=np.dot(out1,out2.T)
    return out

# traning, start from arbitrary A,B and pi
# e is the stopping rule
# n is the number of iterations
# p: multiprocessing core
def EMTrain(A,B,pi,data,n,hidden_state,obs_state,p):
    pi=pi
    A=A
    B=B
    h=hidden_state
    o=obs_state
    
    A_trace=[]
    B_trace=[]
    pi_trace=[]
    
    for i in range(0,n):
        print(i)
        new_pi=pi.copy()
        new_A=A.copy()
        new_B=B.copy()
        
        
        new_pi=sum(p.starmap(ph,[(data[i],A,B,pi,0,h,o) for i in range(0,data.shape[0])]))
        
        new_pi=new_pi/sum(new_pi)
        
        
        for j in range(0,A.shape[0]):
            for k in range(0,A.shape[1]):
                new_A[j,k]=sum(p.starmap(seq_ksi,
                                     [(data[i],A,B,pi,h[j],h[k],h,o) 
                                      for i in range(0,data.shape[0])]))
                
            new_A[j,:]=new_A[j,:]/np.sum(new_A[j,:])
        
        for j in range(0,B.shape[0]):
            new_B[j,:]=sum(p.starmap(ph_seq,[(data[i],A,B,pi,j,h,o) for i in range(0,data.shape[0])]))
            new_B[j,:]=new_B[j,:]/np.sum(new_B[j,:])
                
        # test code
        
        print(new_pi)
        print(new_A)
        print(new_B)
        A_trace.append(new_A)
        B_trace.append(new_B)
        pi_trace.append(new_pi)
        pi=new_pi.copy()
        B=new_B.copy()
        A=new_A.copy()
    A_trace=np.array(A_trace)
    B_trace=np.array(B_trace)
    pi_trace=np.array(pi_trace)
    
    return A_trace,B_trace,pi_trace


    



