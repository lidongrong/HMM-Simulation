# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:43:33 2022

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Try estimate the full latent sequence
"""




######
# Perform MH within Gibbs Sampler
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
    
    #A=np.random.dirichlet(np.ones(transition.shape[1]),transition.shape[0])
    
    #alpha_B=np.array([1 for i in range(0,obs_prob.shape[1])])
    #B=np.random.dirichlet(alpha_B,obs_prob.shape[0])
        
    B=np.array([[1/obs_prob.shape[1] for i in range(0,obs_prob.shape[1])] for j in range(0,obs_prob.shape[0])])
    for i in range(0,B.shape[0]):
        B[i,:]=B[i,:]/sum(B[i,:])
    
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
    T=len(obs)
    #state=HMM.obs_state
    #state=obs_state
    #hidden_state=HMM.hidden_state
    #hidden_state=hidden_state
    
    # start to compute alpha recursively
    alpha=np.zeros((T,len(hidden_state)))

    #y=np.where(obs_state==obs[indexer[0]])[0][0]
    initial=pi
    
    # Handle the boundary case: the first line of alpha
    if indexer[0]==0:
        y=np.where(obs_state==obs[0])[0][0]
        
        alpha[0,:]=initial*B[:,y]
    else:
        alpha[0,:]=initial
        
        
    #alpha[0,:]=np.dot(initial,np.linalg.matrix_power(A, indexer[0]))*B[:,y]
    
    
    for i in range(1,T):
        # The case when i is an observable data
        if i in indexer:
            y=np.where(obs_state==obs[i])[0][0]
            alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,y]
        else:
            alpha[i,:]=np.dot(alpha[i-1,:],A)
    
     
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
        trans=A
        # generate the probability distribution
        w=trans[:,hidden_index]*alpha[T-1-t,:]/np.dot(trans[:,hidden_index],alpha[T-1-t,:])
        #output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
        output.append(np.random.choice(hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    return output


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
        
        obs_num=np.array([np.sum(np.logical_and(I==hidden_state[j],data==obs_state[k])) for k in range(0,B.shape[1])])
        
        B[j,:]=np.random.dirichlet(1+obs_num,1)[0]
    B=np.array(B)
    new_B=B
    #print(B)
    return new_B


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
        log_p=np.log(np.dot(pi,np.linalg.matrix_power(A,indexer[0])[:,pos]))
        return log_p
        
    if indexer.size>1:
        pos=np.where(hidden_state==seq[indexer[0]])[0][0]
        log_p=np.log(np.dot(pi,np.linalg.matrix_power(A,indexer[0])[:,pos]))
                
        for i in range(0,len(indexer)-1):
            current_pos=np.where(hidden_state==seq[indexer[i]])[0][0]
            future_pos=np.where(hidden_state==seq[indexer[i+1]])[0][0]
            log_p=log_p+np.log(np.linalg.matrix_power(A,indexer[i+1]-indexer[i])[current_pos][future_pos])
    return log_p


# Sample A out using Metropolis within Gibbs with a log_normal propose
# log-likelihood is evaluated on the full data set
# p is the multiprocessing core
def sample_A(data,I,A,pi,hidden_state,obs_state,p):
    
    
    for i in range(0,A.shape[0]):
        
        transform=np.array([np.sum((I[:,:-1]==hidden_state[i])&(I[:,1:]==hidden_state[j])) for j in range(A.shape[1])])
    
        
        A[i]=np.random.dirichlet(1+transform,1)[0]
   
    
    return A


# Compute the probability of the first observable state in a hidden sequence give A & pi
def p_first_state(seq,A,pi,hidden_state,obs_state):
    indexer=np.where(seq!='None')[0]
    log_p=0
    
    # In case the whole seq is missing
    if indexer.size==0:
        return 0
    
    pos=np.where(hidden_state==seq[indexer[0]])[0][0]
    log_p=np.log(np.dot(pi,np.linalg.matrix_power(A,indexer[0])[:,pos]))
    return log_p

'''
# sample the initial distribution pi out
# pi: pi from last iteration
# Use Metropolis within Gibbs algorithm
# p is the multiprocessing core
def sample_pi(I,A,pi,hidden_state,obs_state,p):
    for i in range(0,len(pi)):
        
        new_pi=pi.copy()
        
        propose=np.random.beta(2,1/pi[i],1)[0]
        
        new_pi=pi*(1-propose)/(1-pi[i])
        new_pi[i]=propose
    
    
        log_p=sum(p.starmap(p_first_state,[(seq,A,pi,hidden_state,obs_state) for seq in I]))
        log_new_p=sum(p.starmap(p_first_state,[(seq,A,new_pi,hidden_state,obs_state) for seq in I]))
    
        # Metropolis Step
        if log_p==np.inf:
            u=0
        else:
            r=log_new_p+np.log(stats.beta.pdf(pi[i],2,1/new_pi[i]))-log_p-np.log(stats.beta.pdf(new_pi[i],2,1/pi[i]))
            r=min(0,r)
    
        # Metropolis step
        u=np.random.uniform(0,1,1)[0]
        if np.log(u)<r:
            pi=new_pi
    
    
    
    return pi
     
'''

# just testing
def sample_pi(I,A,pi,hidden_state,obs_state,p):
    return pi


'''
# sample the initial distribution pi out
# pi: pi from last iteration
# Use Metropolis within Gibbs algorithm
# p is the multiprocessing core
def sample_pi(I,A,pi,hidden_state,obs_state,p):
    
    
    
    new_pi=np.random.dirichlet(pi+1,1)[0]
    log_p=sum(p.starmap(p_first_state,[(seq,A,pi,hidden_state,obs_state) for seq in I]))
    log_new_p=sum(p.starmap(p_first_state,[(seq,A,new_pi,hidden_state,obs_state) for seq in I]))
    
    # Metropolis Step
    r=log_new_p+np.log(stats.dirichlet.pdf(pi,new_pi+1))-log_p-np.log(stats.dirichlet.pdf(new_pi,pi+1))
    r=min(0,r)
    
    # Metropolis step
    u=np.random.uniform(0,1,1)[0]
    if np.log(u)<r:
        pi=new_pi
    
    
    return pi
'''


   
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
            
   


# evaluate the log-likelihood of the estimation that helps selecting the prediction
# p is the multiprocessing core
def p_evaluator(A,B,pi,I,data,hidden_state,obs_state,p):    
    #obs_state=HMM.obs_state
    #hidden_state=HMM.hidden_state
    # initialize the log likelihood
    log_p=0
    
    # first compute log likelihood of B
    alpha=np.array([1 for i in range(0,B.shape[1])])
    
    log_p=np.sum([stats.dirichlet.logpdf(B[i,:],alpha) for i in range(0,B.shape[0])])
    
    '''
    # Then compute the loglikelihood of Y|Z,B and Z
    for i in range(0,I.shape[0]):
    '''   
    # Note here p is the multiprocessing pool defined in global main() function
    
    # log_z is the log likelihood of the latent sequqnce
    log_z=sum(p.starmap(p_seq,[(seq,A,pi,hidden_state,obs_state) for seq in I]))
    # log_y is the log likelihood of the observed sequence
    log_y=sum(p.starmap(p_observe,[(I[i],data[i],B,hidden_state,obs_state) for i in range(0,I.shape[0])]))
    
    log_p=log_p+log_z+log_y
        
    
        
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
    log_p=p_evaluator(A,B,pi,I_buffer,data,hidden_state,obs_state,p)
    log_prob.append(log_p)
    
    for i in range(0,n):
        start=time.time()
        print(i)
        
        pi=sample_pi(I,A,pi,hidden_state,obs_state,p)
        print('pi: ',pi)
        post_pi.append(pi)
        A=sample_A(data,I,A,pi,hidden_state,obs_state,p)
        
        B=sample_B(data,I,B,hidden_state,obs_state)
        new_A=A.copy()
        post_A.append(new_A)
        print('A',A)
        post_B.append(B)
        print('B',B)
        
        
        I=p.starmap(f_b_sampling,[(A,B,pi,data[i],hidden_state,obs_state) for i in range(0,I.shape[0])])
        I=np.array(I)

        
        new_log_p=p_evaluator(new_A,B,pi,I,data,hidden_state,obs_state,p)
        #new_log_p=1
        
        log_prob.append(new_log_p)
        
        if new_log_p>log_p:
            I_buffer=I.copy()
            log_p=new_log_p
        
        
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