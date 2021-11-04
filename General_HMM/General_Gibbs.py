# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:24:46 2021

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:17:07 2021

@author: a
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:01:45 2021
@author: s1155151972
"""


######
# Perform MH within Gibbs Sampler but only integrate the observed missing value out
# The Gibbs sampler performed on a general Markov Chain (not limited)
######


import numpy as np
import General_HMM as HMM
import General_Sampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math



# initialize data, transition matrix and obs_matrix
def data_initializer():
    print('Data Preprocessing and Initializing Parameters...')
    data=Sampling.data

    # Initialize transition matrix A
    A=np.random.dirichlet((1,1,1,1,1),5)

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
    
    # length of the whole sequence 
    T=len(obs)
    state=HMM.obs_state
    hidden_state=HMM.hidden_state
    # start to compute alpha recursively
    alpha=np.zeros((T,len(HMM.state)))
    
    '''
    # In this case, t1=t0
    if indexer[0]==0:
        alpha[0][0]=1
    else:
        # in this case, we initialize z_{t1} conditioned on the case that t0=1
        y=np.where(state==obs[indexer[0]])[0][0]
        initial=np.zeros(len(state))
        initial[0]=1
        alpha[0,:]=np.dot(initial,np.linalg.matrix_power(A, indexer[0]))*B[:,y]
        '''
    
    alpha[0][0]=1
    
    for i in range(1,T):
        # corresponds to the case that y_i is observable
        if i in indexer:
            y=np.where(state==obs[i])[0][0]
            alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,y]
        else:
            alpha[i,:]=np.dot(alpha[i-1,:],A)
        
        '''
        y=np.where(state==obs[indexer[i]])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],np.linalg.matrix_power(A,indexer[i]-indexer[i-1]))*B[:,y]
        '''
        
    # initialize the output
    output=[]
    
    # First sample the last latent state
    w=alpha[T-1,:]/sum(alpha[T-1,:])
    output.append(np.random.choice(hidden_state,1,p=w)[0])
    
    # Then sample each latent state in sequence
    for t in range(1,T):
        # compute the index of hidden state z_{t+1}
        hidden_index=np.where(HMM.hidden_state==output[t-1])[0][0]
        w=A[:,hidden_index]*alpha[T-1-t,:]/np.dot(A[:,hidden_index],alpha[T-1-t,:])
        output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    return output


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


# Sample A out based on the full latent sequence
# Algorithm for a general Markov Chain
# A: transition matrix from the last iteration
def sample_A(data,I,A):
    A=np.zeros((A.shape[0],A.shape[1]))
    #A[4,4]=1
    # Count the total number that the chain transits from j to k
    for j in range(0,A.shape[0]):
        
        count_of_change=[0 for i in range(0,A.shape[1])]
        
        for k in range(0,A.shape[1]):
            search_pattern=[HMM.hidden_state[j],HMM.hidden_state[k]]
            
            table=(I[:,:-1]==search_pattern[0])&(I[:,1:]==search_pattern[1])
            count_of_change[k]=sum(sum(table))
            
        
        # Generate dirichlet distribution
        dist=np.random.dirichlet((1+np.array(count_of_change)))
        
        A[j,:]=dist
        
        '''
        for r in range(0,A.shape[1]):
            A[j,r]=dist[r]
        '''
    
    return A
            
        
        







# Gibbs Sampling accelerated by parallel computing
# input I,A,B: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,n):
    post_A=[]
    post_B=[]
    for i in range(0,n):
        
        print(i)
        A=sample_A(data,I,A)
        B=sample_B(data,I,B)
        
        I=p.starmap(sample_latent_seq,[(data[0:500,:],I[0:500,:],A,B),(data[500:1000,:],
                                                                     I[500:1000,:],A,B),(data[1000:1500,:],
                                                                                         I[1000:1500,:],A,B),
                                                                                         (data[1500:2000,:],
                                                                                          I[1500:2000,:],A,B),
                                                                                         (data[2000:2500,:],
                                                                                          I[2000:2500,:],A,B),
                                                                                         (data[2500:3000,:],
                                                                                          I[2500:3000,:],A,B),
                                                                                         (data[3000:3500,:],
                                                                                          I[3000:3500,:],A,B),
                                                                                         (data[3500:4000,:],
                                                                                          I[3500:4000,:],A,B)])
                                                                                        
        I=np.vstack((I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]))
        #I=sample_latent_seq(data,I,A,B)
        
        print(A)
        post_A.append(A)
        post_B.append(B)
        
        
        
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    return post_A,post_B
        
                    
if __name__=='__main__':
    A,B,data,I=initialize()
    
    
    p=Pool(8)
    post_A,post_B=parallel_Gibbs(data,I,A,B,6000)
    
    
    print('Program finished')