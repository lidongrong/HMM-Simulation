# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:01:45 2021

@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:53:10 2021

@author: s1155151972
"""

######
# Perform MH within Gibbs Sampler with minibatch of the data
######


import numpy as np
import HMM
import Sampling
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


# Sample A out using Metropolis within Gibbs on a minibatch of the data
# Algorithm based on Resarch note in 2021.10.29
# A: transition matrix from the last iteration
def minibatch_sample_A(batch_size,data,I,A):
    
    initial_batch_index=np.random.choice(I.shape[0],batch_size,replace=False)
    batch_index=initial_batch_index
    
    # Because we assume the last state is an absorbing state
    for j in range(0,A.shape[0]-1):
        print(batch_index)
        
        # initialize the batch, which is from the last iteration
        batch=I[batch_index,:]
        
        # construct proposed batch index
        proposed_batch_index=np.random.choice(I.shape[0],batch_size,replace=False)
        proposed_batch=I[proposed_batch_index,:]
        
        # Tempering constant suggested by the algorithm in the JASA paper
        # A number smaller but close to sqrt(m), where m is the batch size
        cn=math.sqrt(len(batch_index))-0.1
        
        
        # acquire the parameter from the last iteration
        a=A[j,j]
        
        # sample from proposal distribution
        # prevent b from <0, so if b too small, truncate by .0000001
        #new_a=np.random.beta(2,max(2/a-2,0.001),1)[0]
        #new_a=np.random.beta(2,max(2/a-2,0.001),1)[0]
        new_a=np.random.uniform(0,1,1)[0]
        
        new_A=A.copy()
        new_A[j,j]=new_a
        new_A[j,j+1]=1-new_a
        
        # compute the likelihood up to a normalizing constant
        p=1
        new_p=1
        # evaluate the likelihood on both of the mini batches
        for k in range(0,batch_size):
            indexer=np.where(batch[k]!='None')[0]
            new_indexer=np.where(proposed_batch[k]!='None')[0]
            
            # in case that the whole sequence is missing
            if indexer.size>1:
                # We always assume start from a specific state
                if indexer[0]!=0:
                    pos=np.where(HMM.hidden_state==batch[k][indexer[0]])[0][0]
                    p=p*np.linalg.matrix_power(A,indexer[0])[0][pos]
                    
                
                for i in range(0,len(indexer)-1):
                    current_pos=np.where(HMM.hidden_state==batch[k][indexer[i]])[0][0]
                    future_pos=np.where(HMM.hidden_state==batch[k][indexer[i+1]])[0][0]
                    p=p*np.linalg.matrix_power(A,indexer[i+1]-indexer[i])[current_pos][future_pos]
                    
            if new_indexer.size>1:
                
                # We always assume start from a specific state
                if new_indexer[0]!=0:
                    pos=np.where(HMM.hidden_state==proposed_batch[k][new_indexer[0]])[0][0]
                    new_p=new_p*np.linalg.matrix_power(new_A,new_indexer[0])[0][pos]
                    
                
                for i in range(0,len(new_indexer)-1):
                    current_pos=np.where(HMM.hidden_state==proposed_batch[k][new_indexer[i]])[0][0]
                    future_pos=np.where(HMM.hidden_state==proposed_batch[k][new_indexer[i+1]])[0][0]
                    new_p=new_p*np.linalg.matrix_power(new_A,new_indexer[i+1]-new_indexer[i])[current_pos][future_pos]
                    
                
        
        
        # determine the metropolis acceptance rate
        # avoid dominator=0
        if p==0:
            u=1
        else:
            
            #r is the acceptance rate
            #r=(new_p*stats.beta.pdf(a,2,max((2/new_a-2),0.001)))/(p*stats.beta.pdf(new_a,2,max((2/a-2),0.001)))
            r=(new_p/p)**(cn/batch_size)
            
            #r=((new_p**(cn/batch_size))*stats.beta.pdf(a,2,max(2/new_a-2,0.001)))/((p**(cn/batch_size))*stats.beta.pdf(new_a,2,max(2/a-2,0.001)))
            u=min(1,r)
        
        unif=np.random.uniform(0,1,1)[0]
        if unif<u:
            A[j,j]=new_a
            A[j,j+1]=1-new_a
            batch_index=proposed_batch_index
    
    A=np.array(A)
    
    final_A=A
    
    return final_A

# Gibbs sampling using Metropolis within Gibbs algorithm
# input I,A,B: initial guesses of the parameter
# n: number of samples to draw
def Gibbs(data,I,A,B,n,batch_size):
    post_A=[]
    post_B=[]
    for i in range(0,n):
        print(i)
        A=minibatch_sample_A(batch_size,data,I,A)
        B=sample_B(data,I,B)
        I=sample_latent_seq(data,I,A,B)
        
        post_A.append(A)
        post_B.append(B)
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    return post_A,post_B

# Gibbs sampling using Metropolis within Gibbs algorithm (acceleration by parallel computing)
# input I,A,B: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,n,batch_size):
    post_A=[]
    post_B=[]
    for i in range(0,n):
        print(i)
        A=minibatch_sample_A(batch_size,data,I,A)
        B=sample_B(data,I,B)
        
        I=p.starmap(sample_latent_seq,[(data[0:450,:],I[0:450,:],A,B),(data[450:900,:],
                                                                     I[450:900,:],A,B),(data[900:1350,:],
                                                                                         I[900:1350,:],A,B),
                                                                                         (data[1350:1800,:],
                                                                                          I[1350:1800,:],A,B),
                                                                                         (data[1800:2250,:],
                                                                                          I[1800:2250,:],A,B),
                                                                                         (data[2250:2700,:],
                                                                                          I[2250:2700,:],A,B),
                                                                                         (data[2700:3150,:],
                                                                                          I[2700:3150,:],A,B),
                                                                                         (data[3150:3600,:],
                                                                                          I[3150:3600,:],A,B)])
                                                                                        
        I=np.vstack((I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]))
        #I=sample_latent_seq(data,I,A,B)
        
        
        
        post_A.append(A)
        post_B.append(B)
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    return post_A,post_B
        
                    
if __name__=='__main__':
    A,B,data,I=initialize()
    batch_size=15
    
    
    p=Pool(8)
    post_A,post_B=parallel_Gibbs(data,I,A,B,60,batch_size)
    
    
    print('Program finished')
    

        
        



    

