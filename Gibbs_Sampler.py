# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:07:13 2021

@author: a
"""

import numpy as np
import scipy.stats as stats
import math
import HMM
import Sampling


print('Data preprocessing...')

data=Sampling.data

miss_place=np.where(data=='None')
miss_x=miss_place[0]
miss_y=miss_place[1]


# Estimate using Gibbs sampling + Viterbi

x=np.random.dirichlet((2,2),1)[0]

# parameter initialization

print('Initializing parameters...')

# Initialize transition matrix A
A=np.array([[0.5,0.5,0,0,0],[0,0.5,0.5,0,0],[0,0,0.5,0.5,0],[0,0,0,0.5,0.5],[0,0,0,0,1]])

# Initialize observation matrix B
B=np.random.dirichlet((1,1,1,1,1),5)


print('Initializing latent variables...')

# Initialize omis by random labelling
w=np.random.dirichlet((1,1,1,1,1),1)[0]
for i in range(0,len(miss_x)):
    data[miss_x[i],miss_y[i]]=np.random.choice(HMM.obs_state,1,p=w)[0]

#  Initialize hidden state using Viterbi
    
# A: transition matrix
# B: observation matrix
# obs: observable sequence
# state: array of observable states
def Viterbi(A,B,state,obs):
    K=A.shape[0]
    T=len(obs)
    
    # the probability of the most likely path
    T1=np.empty((K,T))
    # the x_j-1 of the most likely path
    T2=np.empty((K,T))
    
    pi=np.array([1,0,0,0,0])
    
    T2[:,0]=0
    T1[:,0]=pi*B[:,np.where(state==obs[0])[0][0]]
    
    for i in range(1,T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis,:,
          np.where(state==obs[i])[0][0]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)
        #print(T1)
        #print(T2)
    
    x=np.zeros(T)
    x=np.int16(x)
    T2=np.int16(T2)
    #print(T2)
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    
    return HMM.hidden_state[x]

# create initial guess of hidden state
 

# Sample from the conditional distribution of I
# A,B: parameters
# obs: observed sequence
# state: array of observable states
def Sample_I(A,B,state,obs):
    T=len(obs)
    output=[]
    # always starts with A
    output.append(HMM.hidden_state[0])
    for i in range(1,T):
        # (i-1)th hidden state
        hidden_address=np.where(HMM.hidden_state==output[i-1])[0][0]
        # i th observed state
        observe_address=np.where(state==obs[i])[0][0]
        
        w=A[hidden_address,:]*B[:,observe_address].T/np.dot(A[hidden_address,:],
           B[:,observe_address])
        
        
        
        output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
    output=np.array(output)
    return output

# Based on the research note of 2021.10.2
# An alternative function for sampling the hidden path
# A,B: transition matrix and observation matrix
# obs: observed sequence
# state: array of observable states
def sample_I(A,B,state,obs):
    # Use backward and forward algorithm to compute the distribution of hidden 
    # T is the length of observed sequence
    T=len(obs)
    alpha=np.zeros((T,len(state)))
    beta=np.zeros((T,len(state)))
    
    
    #print(T)
    # intialize alpha and beta
    # here beta_T(i)=1 and alpha_0(1)=1
    alpha[0][0]=1
    tmp=np.where(state==obs[T-1])[0][0]
    beta[T-1,tmp]=1
    
    for i in range(1,T):
        index=np.where(state==obs[i])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,index]
        
        beta[T-1-i,:]=np.dot((A*B[:,index]),beta[T-i,:])
    #print('beta',beta)
    
    output=[]
    output.append(HMM.hidden_state[0])
    # add the path one by one
    for t in range(1,T):
        # last hidden state
        hidden_index=np.where(HMM.hidden_state==output[t-1])[0][0]
        # current observed state
        obs_index=np.where(state==obs[t])[0][0]
        
        dominator=alpha[t,hidden_index]*beta[t,hidden_index]/np.dot(alpha[t,:],beta[t,:]) 
        nominator=alpha[t,hidden_index]*(A[hidden_index,:]*B[:,obs_index])*beta[t+1,:]
        w=(nominator/dominator)/(alpha[t,hidden_index]*
          np.dot(A[hidden_index,:]*B[:,obs_index],beta[t+1,:]))
        print(w)
        #w=alpha[t,:]*beta[t,:]/np.dot(alpha[t,:],beta[t,:])
        output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
        print('w:',w)
    output=np.array(output)
    return output

I=[]

for i in range(0,data.shape[0]):
    I.append(Sample_I(A,B,HMM.obs_state,data[i]))

I=np.array(I)   

# Gibbs sampling, pass initial guess of A, B, data & I as parameters
# n: total number of iterations
def Gibbs(A,B,data,I,n):  
    post_A=[]
    post_B=[]
    #post_I=[]
    for i in range(0,n):
        print(i)
        # first, sample B
        B=np.zeros((B.shape[0],B.shape[1]))
        for j in range(0,B.shape[0]):
            # for j th row of B, calculate the number of each 
            # observed states respectively
            n1=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[0]))
            n2=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[1]))
            n3=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[2]))
            n4=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[3]))
            n5=np.sum(np.logical_and(I==HMM.hidden_state[j],data==HMM.obs_state[4]))
            #n1=len(np.where(data[np.where(I==HMM.hidden_state[j])]
            #==HMM.obs_state[0])[0])
            #n2=len(np.where(data[np.where(I==HMM.hidden_state[j])]
            #==HMM.obs_state[1])[0])
            #n3=len(np.where(data[np.where(I==HMM.hidden_state[j])]
            #==HMM.obs_state[2])[0])
            #n4=len(np.where(data[np.where(I==HMM.hidden_state[j])]
            #==HMM.obs_state[3])[0])
            #n5=len(np.where(data[np.where(I==HMM.hidden_state[j])]
            #==HMM.obs_state[4])[0])
            # Draw from posterior distribution
            
            B[j,:]=np.random.dirichlet((1+n1,1+n2,1+n3,1+n4,1+n5),1)[0]
        #print(B)
        post_B.append(B)
        
        # Then sample Omis
        for j in range(0,len(miss_x)):
            
            h=I[miss_x[j],miss_y[j]]
            #print(h)
            w=B[np.where(HMM.hidden_state==h)[0][0],:]
            data[miss_x[j],miss_y[j]]=np.random.choice(HMM.obs_state,1,p=w)[0]
        
        # Next, we sample A using Dirichlet distribution
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
            for k in range(0,9):
                #print(freq)
                a=(I[:,k]==I[:,k+1])
                b=(I[:,k]==HMM.hidden_state[j])
                stay_freq=stay_freq+np.sum(np.logical_and(a,b))
                c=(I[:,k]!=I[:,k+1])
                change_freq=change_freq+np.sum(np.logical_and(c,b))
                
            #freq=freq+np.sum(I[:,9]==HMM.hidden_state[j])
            
            A_posterior=np.random.dirichlet((1+stay_freq,1+change_freq),1)[0]
            #print('freq:',freq)
            #print('state:',state_num)
            #print(A_posterior)
            A[j,j]=A_posterior[0]
            A[j,j+1]=A_posterior[1]
        post_A.append(A)
        
        #print(A)
        
        # Finally, we sample I (unobserved states)
        for k in range(0,I.shape[0]):
            I[k,:]=Sample_I(A,B,HMM.obs_state,data[k])
            #I[k,:]=Viterbi(A,B,HMM.obs_state,data[k])
        #print(I[3])
    return post_A, post_B

print('Start Gibbs sampling...')
post_A,post_B=Gibbs(A,B,data,I,30000)
