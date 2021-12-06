# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:23:42 2021

@author: s1155151972
"""


# Construct Gibbs sampler with HMM that is completely missing at random

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
    
    a=np.random.dirichlet((2,2),4)
    
    
    A=np.array([[a[0][0],1-a[0][0],0,0,0],[0,a[1][0],1-a[1][0],0,0],
                [0,0,a[2][0],1-a[2][0],0],[0,0,0,a[3][0],1-a[3][0]],[0,0,0,0,1]])

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


# evaluate the likelihood of a sequqnce given A
def p_seq(seq,A):
    hidden_state=HMM.hidden_state
    indexer=np.array(np.where(seq!='None')[0])
    log_p=0
            
    # in case that the whole sequence is missing
    if indexer.size==1:
        pos=np.where(hidden_state==seq[indexer[0]])[0][0]
        log_p=np.log(np.linalg.matrix_power(A,indexer[0]))[0][pos]
    
    if indexer.size>1:
        # We always assume start from a specific state
        if indexer[0]!=0:
            pos=np.where(hidden_state==seq[indexer[0]])[0][0]
            log_p=log_p+np.log(np.linalg.matrix_power(A,indexer[0])[0][pos])
            
                    
                
        for i in range(0,len(indexer)-1):
            current_pos=np.where(hidden_state==seq[indexer[i]])[0][0]
            future_pos=np.where(hidden_state==seq[indexer[i+1]])[0][0]
            log_p=log_p+np.log(np.linalg.matrix_power(A,indexer[i+1]-indexer[i])[current_pos][future_pos])
    return log_p



# Sample A out using Metropolis within Gibbs 
# log-likelihood is evaluated on the full data set
def sample_A(data,I,A):
    
    
    # Because we assume the last state is an absorbing state
    for j in range(0,A.shape[0]-1):
        # acquire the parameter from the last iteration
        a=A[j,j]
        
        # new proposal 
        new_a=np.random.beta(2,1/a,1)[0]
        
        
        new_A=A.copy()
        new_A[j,j]=new_a
        new_A[j,j+1]=1-new_a
        
        # compute the likelihood up to a normalizing constant
        log_p=0
        log_new_p=0
        # evaluate the likelihood of current state and proposed state
        
        # p is the pool variable in global main function
        
        log_p=sum(p.starmap(p_seq,[(seq,A) for seq in I]))
        log_new_p=sum(p.starmap(p_seq,[(seq,new_A) for seq in I]))
        
        
            
            
            
        
        # determine the metropolis acceptance rate
        # avoid dominator=0
        if log_p==np.inf:
            u=1
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


# Locate the indexer of a specific element in hidden_state
def hidden_loc(element):
    return np.where(HMM.hidden_state==element)[0][0]

# Locate the indexer of a specific element in obs_seq
def obs_loc(element):
    return np.where(HMM.obs_state==element)[0][0]

vhidden_loc=np.vectorize(hidden_loc)
vobs_loc=np.vectorize(obs_loc)

# evaluate the likelihood of an observed sequence
# seq_hidden, seq_obs: latent sequqnce and observed sequqnce respectively
def p_observe(hidden_seq,obs_seq,B):
    #hidden_state=HMM.hidden_state
    #obs_state=HMM.obs_state
    
    # indices of the observed data
    indexer=np.array(np.where(hidden_seq!='None')[0])
    
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
        
        z_pos=vhidden_loc(hidden_seq[indexer])
        y_pos=vobs_loc(obs_seq[indexer])
        log_y=np.sum(np.log(B[z_pos,y_pos]))
    
    return log_y
            
    


# evaluate the log-likelihood of the estimation that helps selecting the prediction
def p_evaluator(A,B,I,data):    
    #obs_state=HMM.obs_state
    #hidden_state=HMM.hidden_state
    # initialize the log likelihood
    log_p=0
    
    # first compute log likelihood of B
    alpha=np.array([1,1,1,1,1])
    log_p=stats.dirichlet.logpdf(B[0,:],alpha)+stats.dirichlet.logpdf(B[1,:],alpha)
    +stats.dirichlet.logpdf(B[2,:],alpha)+stats.dirichlet.logpdf(B[3,:],alpha)
    +stats.dirichlet.logpdf(B[4,:],alpha)
    
    '''
    # Then compute the loglikelihood of Y|Z,B and Z
    for i in range(0,I.shape[0]):
    '''   
    # Note here p is the multiprocessing pool defined in global main() function
    
    # log_z is the log likelihood of the latent sequqnce
    log_z=sum(p.starmap(p_seq,[(seq,A) for seq in I]))
    # log_y is the log likelihood of the observed sequence
    log_y=sum(p.starmap(p_observe,[(I[i],data[i],B) for i in range(0,I.shape[0])]))
    
    
        
    log_p=log_p+log_z+log_y
        
        
    
        
    return log_p


# Gibbs sampling using Metropolis within Gibbs algorithm (acceleration by parallel computing)
# input I,A,B: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,n):
    post_A=[]
    post_B=[]
    log_prob=[]
    
    # calculate the data size
    ds=data.shape[0]
    
    # construct a buffer to store the latent sequence with largest likelihood
    I_buffer=I.copy()
    log_p=p_evaluator(A,B,I_buffer,data)
    log_prob.append(log_p)
    
    for i in range(0,n):
        start=time.time()
        print(i)
        
        A=sample_A(data,I,A)
        
        B=sample_B(data,I,B)
        new_A=A.copy()
        post_A.append(new_A)
        print(A)
        post_B.append(B)
        
        
        '''
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
        '''
        
        I=p.starmap(f_b_sampling,[(A,B,data[i]) for i in range(0,I.shape[0])])
        I=np.array(I)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #I=np.vstack((I[0],I[1],I[2],I[3],I[4],I[5],I[6],I[7]))
        #I=sample_latent_seq(data,I,A,B)
        
        
        new_log_p=p_evaluator(new_A,B,I,data)
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
    
    return post_A,post_B,I_buffer,log_prob


# define the output class of the experiments
class Out:
    def __init__(self,data,post_A,post_B,latent_seq, log_prob,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.latent_seq=latent_seq
        self.log_prob=log_prob
        self.true_hidden=true_hidden
 


                   
if __name__=='__main__':
    
    # Code deployed on an 8-core CPU
    p=Pool(8)
    
    
    # Define the output object class
    # Which is a list of Out objects 
    out_obj=[]
    
    for i in range(0,10):
        A,B,data,I=initialize()
        post_A,post_B,latent_seq,log_prob=parallel_Gibbs(data,I,A,B,4200)
        out_obj.append(Out(data,post_A,post_B,latent_seq,log_prob,Sampling.hidden_data))
        
    
    
    
    
    print('Program finished')
