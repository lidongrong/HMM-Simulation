# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:54:53 2022

@author: s1155151972
"""


######
# Perform MH within Gibbs Sampler
# Compare it to our method
######

import numpy as np
import HMM
import SeqSampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math
import os
import multiprocessing as mp
import torch # Will propose Langevin jump  


# initialize data, transition matrix and obs_matrix
def data_initializer(transition,obs_prob,pi,hidden_state,obs_state,rate):
    #print('Data Preprocessing and Initializing Parameters...')
    #data=Sampling.data
    
    # Generate the HMM object
    MC=HMM.HMM(hidden_state,obs_state,transition,obs_prob,pi)


    # Construct Data
    hidden_data=Sampling.Hidden_Generator(MC,20,5000)
    data=Sampling.Obs_Generator(MC,hidden_data)

    # Missing at random
    data=Sampling.Missing(data,p=rate)
    
    # Initialize transition matrix A
    
    A=np.array([[1/transition.shape[1] for i in range(0,transition.shape[1])] for j in range(0,transition.shape[0])])
    for i in range(0,A.shape[0]):
        A[i,:]=A[i,:]/sum(A[i,:])
    
    # Initialize observation matrix B and pi
    # Dirichlet parameter of B
    
    alpha_B=np.array([1 for i in range(0,obs_prob.shape[1])])
    B=np.random.dirichlet(alpha_B,obs_prob.shape[0])
    '''
    pi=np.array([1/len(pi) for i in range(0,len(pi))])
    pi=pi/sum(pi)
    '''
    
    # just testing!
    pi=pi
    
    return A,B,pi,data,hidden_data



# Sample the latent state using forward backward sampling
# Based on research note in 2021.10.21
# A,B: transition matrix and observation matrix
# pi: initial probability
# state: observed sequence with missing observation
def f_b_sampling(A,B,pi,obs,hidden_state,obs_state):
    
    # Check if the whole sequence is missing
    if np.all(obs=='None'):
        return obs
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]
    # length of the index
    T=len(indexer)
    #state=HMM.obs_state
    #state=obs_state
    #hidden_state=HMM.hidden_state
    #hidden_state=hidden_state
    
    # start to compute alpha recursively
    alpha=np.zeros((T,len(obs_state)))

    y=np.where(obs_state==obs[indexer[0]])[0][0]
    initial=pi
    alpha[0,:]=np.dot(initial,np.linalg.matrix_power(A,1))*B[:,y]
    
    
    for i in range(1,T):
        y=np.where(obs_state==obs[indexer[i]])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],np.linalg.matrix_power(A,1))*B[:,y]
        
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
        trans=np.linalg.matrix_power(A,1)
        # generate the probability distribution
        w=trans[:,hidden_index]*alpha[T-1-t,:]/np.dot(trans[:,hidden_index],alpha[T-1-t,:])
        #output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
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
# pi: initial probability
def sample_latent_seq(data,I,A,B,pi,hidden_state,obs_state):
    for i in range(0,data.shape[0]):
        I[i,:]=f_b_sampling(A,B,pi,data[i],hidden_state,obs_state)
    return I



# initialize the latent sequence as initial guess
def latent_seq_initializer(data,A,B,pi,hidden_state,obs_state):
    # initialize latent sequence I
    I=[]
    for i in range(0,data.shape[0]):
        I.append(np.repeat('None',data.shape[1]))
    I=np.array(I)

    I=sample_latent_seq(data,I,A,B,pi,hidden_state,obs_state)
    return I

# Make an initial guess of the parameters and latent variables
def initialize(hidden_state,obs_state,transition,obs_prob,pi,rate):
    A,B,pi,data,hidden_data=data_initializer(transition,obs_prob,pi,hidden_state,obs_state,rate)
    I=latent_seq_initializer(data,A,B,pi,hidden_state,obs_state)
    return A,B,pi,data,I,hidden_data


# Sample B out in Gibbs Sampling
# B: B sampled in last iteration
def sample_B(data,I,B,hidden_state,obs_state):
    for j in range(0,B.shape[0]):
        # for j th row of B, calculate the number of each 
        # observed states respectively
        
        obs_num=[]
        
        for k in range(0,B.shape[1]):
            n=np.sum(np.logical_and(I==hidden_state[j],data==obs_state[k]))
            obs_num.append(n)
        
        obs_num=np.array(obs_num)
        B[j,:]=np.random.dirichlet(1+obs_num,1)[0]
    B=np.array(B)
    new_B=B
    #print(B)
    return new_B


def sample_A(data,I,A,pi,hidden_state,obs_state,p,gam):
    tmp=I[I!='None']
    new_A=A.copy()
    for i in range(0,A.shape[0]):
        transform=[np.sum((tmp[:-1]==hidden_state[i])&(tmp[1:]==hidden_state[j])) for j in range(A.shape[1])]
        transform=np.array(transform)
        new_A[i,:]=np.random.dirichlet(1+transform,1)[0]
    return new_A

# obtain the first observation of a I
def first_obs(seq,obs,hidden_state,obs_state):
    indexer=np.where(obs!='None')[0]
    return seq[indexer[0]]

def sample_pi(pi,data,I,A,B,hidden_state,obs_state,p):
    first_observations=p.starmap(first_obs,[(I[i],data[i],hidden_state,obs_state) for i in range(0,data.shape[0])])
    first_observations=np.array(first_observations)
    trans=[sum(first_observations==hidden_state[i]) for i in range(0,len(hidden_state))]
    new_pi=np.random.dirichlet(1+trans,1)[0]
    return new_pi

# evaluate the likelihood of a sequqnce given A
# seq: hidden sequence
# A,pi: transition matrix and initial distribution
def p_seq(seq,A,pi,hidden_state,obs_state):
    indexer=np.where(seq!='None')[0]
    log_p=0
    
    # in case the whole sequence is missing
    if indexer.size==0:
        return 0
            
    # marginal condition
    if indexer.size==1:
        pos=np.where(hidden_state==seq[indexer[0]])[0][0]
        log_p=np.log(np.dot(pi,np.linalg.matrix_power(A,1)[:,pos]))
        return log_p
        
    if indexer.size>1:
        pos=np.where(hidden_state==seq[indexer[0]])[0][0]
        log_p=np.log(np.dot(pi,np.linalg.matrix_power(A,1)[:,pos]))
                
        for i in range(0,len(indexer)-1):
            current_pos=np.where(hidden_state==seq[indexer[i]])[0][0]
            future_pos=np.where(hidden_state==seq[indexer[i+1]])[0][0]
            log_p=log_p+np.log(np.linalg.matrix_power(A,1)[current_pos][future_pos])
    return log_p

# Locate the indexer of a specific element in hidden_state
def hidden_loc(element,hidden_state):
    return np.where(hidden_state==element)[0][0]

# Locate the indexer of a specific element in obs_seq
def obs_loc(element,obs_state):
    return np.where(obs_state==element)[0][0]

vhidden_loc=np.vectorize(hidden_loc,excluded=['hidden_state'])
vobs_loc=np.vectorize(obs_loc,excluded=['obs_state'])


# evaluate the likelihood of an observed sequence
# seq_hidden, seq_obs: latent sequqnce and observed sequqnce respectively
def p_observe(hidden_seq,obs_seq,B,hidden_state,obs_state):
    #hidden_state=HMM.hidden_state
    #obs_state=HMM.obs_state
    
    # indices of the observed data
    indexer=np.array(np.where(obs_seq!='None')[0])
    
    # initialize the log likelihood
    log_y=0
    
    # in case the whole sequqnce is missing
    if indexer.size>=1:
        '''
        for k in range(0,len(indexer)):
            z_pos=np.where(hidden_state==hidden_seq[indexer[k]])[0][0]
            y_pos=np.where(obs_state==obs_seq[indexer[k]])[0][0]
            log_y=log_y+np.log(B[z_pos,y_pos])
            '''
        
        #z_pos=vhidden_loc(hidden_seq[indexer],hidden_state)
        z_pos=np.array([hidden_loc(hidden_seq[k],hidden_state) for k in indexer])
        #y_pos=vobs_loc(obs_seq[indexer],obs_state)
        y_pos=np.array([obs_loc(obs_seq[k],obs_state) for k in indexer])
        log_y=np.sum(np.log(B[z_pos,y_pos]))
    
    return log_y

# evaluate the log-likelihood of each observation that helps selecting the prediction
# return a vector of log-prob
# p is the multiprocessing core
def p_sample(A,B,pi,I,data,hidden_state,obs_state,p):    
    #obs_state=HMM.obs_state
    #hidden_state=HMM.hidden_state
    # initialize the log likelihood
    
    
    # first compute log likelihood of B
    
    '''
    # Then compute the loglikelihood of Y|Z,B and Z
    for i in range(0,I.shape[0]):
    '''   
    # Note here p is the multiprocessing pool defined in global main() function
    
    # log_z is the log likelihood of the latent sequqnce
    log_z=p.starmap(p_seq,[(I[i],data[i],A,pi,hidden_state,obs_state) for i in range(0,I.shape[0])])
    # log_y is the log likelihood of the observed sequence
    log_y=p.starmap(p_observe,[(I[i],data[i],B,hidden_state,obs_state) for i in range(0,I.shape[0])])
    
    log_z=np.array(log_z)
    log_y=np.array(log_y)
    
    # Return the vector of log prob
    log_p=log_z+log_y
        
    
        
    return log_p


# Gibbs sampling using Metropolis within Gibbs algorithm (acceleration by parallel computing)
# input I,A,B,pi: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,pi,n,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    log_prob=[]
    post_pi=[]
    
    
    # construct a buffer to store the latent sequence with largest likelihood
    I_buffer=I.copy()
    log_p=p_sample(A,B,pi,I_buffer,data,hidden_state,obs_state,p)
    
    # log prob of each sample, help to select hidden state
    selector=p_sample(A,B,pi,I_buffer,data,hidden_state,obs_state,p)
    log_prob.append(log_p)
    
    #inv_permute=np.arange(A.shape[0])
    #permute=inv_permute
    
    for i in range(0,n):
        start=time.time()
        print(i)
        
        # opi=pi.copy()
        pi=sample_pi(I,A,pi,hidden_state,obs_state,p)
        
        
        #print('pi',pi)
        #post_pi.append(pi)
        #oB=B.copy()
        B=sample_B(data,I,B,hidden_state,obs_state)
        
        
        A=sample_A(data,I,A,hidden_state,obs_state,p)
        
        I=p.starmap(f_b_sampling,[(A,B,pi,data[i],hidden_state,obs_state) for i in range(0,I.shape[0])])
        I=np.array(I)
        
        
       
        '''
        # invert the permutations
        A=A[inv_permute,:]
        A=A[:,inv_permute]
        B=B[inv_permute,:]
        pi=pi[inv_permute]
        
        I=p.starmap(switch_seq,[(I[i],hidden_state,permute) for i in range(0,I.shape[0])])
        I=np.array(I)
        '''
        
        post_pi.append(pi)
        post_A.append(A)
        #print('A',A)
        post_B.append(B)
        
        
        print('pi',pi)
        print('A',A)
        print('B',B)
        
        '''
        # define the new random permutation
        permute=np.arange(A.shape[0])
        permute=np.random.permutation(permute)
        # and the inv permutation of the new permutation
        inv_permute=np.empty(permute.size,dtype=np.int32)
        for i in np.arange(permute.size):
            inv_permute[permute[i]]=i
        '''
        

        
        #new_log_p=p_evaluator(A,B,pi,I,data,hidden_state,obs_state,p)
        new_selector=p_sample(A,B,pi,I,data,hidden_state,obs_state,p)
        new_log_p=sum(new_selector)
        #new_log_p=1
        log_prob.append(new_log_p)
        
        '''
        if new_log_p>log_p:
            I_buffer=I.copy()
            log_p=new_log_p
        '''
        
        indicator=new_selector>selector
        print(np.sum(indicator))
        
        I_buffer[indicator]=I[indicator]
        selector[indicator]=new_selector[indicator]
        
        '''
        # permute!
        A=A[permute,:]
        A=A[:,permute]
        B=B[permute,:]
        pi=pi[permute]
        
        I=p.starmap(switch_seq,[(I[i],hidden_state,permute) for i in range(0,I.shape[0])])
        I=np.array(I)
        '''
        
        
        end=time.time()
        print(end-start)
        
        
        
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    log_prob=np.array(log_prob)
    post_pi=np.array(post_pi)
    
    return post_A,post_B,post_pi,I_buffer,log_prob


# define the output class of the experiments
class Out:
    def __init__(self,data,post_A,post_B,post_pi,latent_seq, log_prob,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.post_pi=post_pi
        self.latent_seq=latent_seq
        self.log_prob=log_prob
        self.true_hidden=true_hidden
    
    