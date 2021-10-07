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
from multiprocessing import Pool
import time

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
 


# Based on the research note of 2021.10.5
# An alternative function for sampling the hidden path
# Use forward-backward sampling algorithm
# A,B: transition matrix and observation matrix
# obs: observed sequence
# state: array of observable states
def f_b_sampling(A,B,state,obs):
    T=len(obs)
    alpha=np.zeros((T,len(state)))
    #beta=np.zeros((T,len(state)))
    
    # intialize alpha 
    # here beta_T(i)=1 and alpha_0(1)=1
    alpha[0][0]=1
    #tmp=np.where(state==obs[T-1])[0][0]
    #beta[T-1,tmp]=1
    
    for i in range(1,T):
        index=np.where(state==obs[i])[0][0]
        alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,index]
    
    #print(alpha)
    # intialize the output
    output=[]
    
    # first sample z_T
    w=alpha[T-1,:]/sum(alpha[T-1,:])
    output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
    
    # then sample z_{t-1}, z_{t-2} in a backward manner
    for t in range(1,T):
        # compute the index of hidden state z_{t+1}
        hidden_index=np.where(HMM.hidden_state==output[t-1])[0][0]
        w=A[:,hidden_index]*alpha[T-1-t,:]/np.dot(A[:,hidden_index],alpha[T-1-t,:])
        output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    return output


# Sample the whole I out using f_b_sampling        
def sampling_hidden(data,I,A,B):
  for i in range(0,data.shape[0]):
    I[i,:]=f_b_sampling(A,B,HMM.obs_state,data[i])
  return I
        

I=[]

for i in range(0,data.shape[0]):
    I.append(f_b_sampling(A,B,HMM.obs_state,data[i]))

I=np.array(I)   

# Gibbs sampling, pass initial guess of A, B, data & I as parameters
# n: total number of iterations
def Gibbs(A,B,data,I,p,n):  
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
        
            
        post_B.append(B)
        
        
        
        
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
            for k in range(0,I.shape[1]-1):
                #print(freq)
                a=(I[:,k]==I[:,k+1])
                b=(I[:,k]==HMM.hidden_state[j])
                stay_freq=stay_freq+np.sum(np.logical_and(a,b))
                c=(I[:,k]!=I[:,k+1])
                change_freq=change_freq+np.sum(np.logical_and(c,b))
                
            
            
            A_posterior=np.random.dirichlet((1+stay_freq,1+change_freq))
            #print('freq:',freq)
            #print('state:',state_num)
            #print(A_posterior)
            A[j,j]=A_posterior[0]
            A[j,j+1]=A_posterior[1]
            
        post_A.append(A)
        
        
        # Then sample Ymis
        for j in range(0,len(miss_x)):
            
            h=I[miss_x[j],miss_y[j]]
            w=B[np.where(HMM.hidden_state==h)[0][0],:]
            data[miss_x[j],miss_y[j]]=np.random.choice(HMM.obs_state,1,p=w)[0]
        
        # Finally, we sample I (unobserved states)
            
        #p=Pool(4)
        #s=time.time()
        # CPU acceleration
        I=p.starmap(sampling_hidden,[(data[0:450,:],I[0:450,:],A,B),(data[450:900,:],
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
        #e=time.time()
        #print(e-s)
        
        
        '''
        for k in range(0,I.shape[0]):
            #I[k,:]=Sample_I(A,B,HMM.obs_state,data[k])
            #I[k,:]=Viterbi(A,B,HMM.obs_state,data[k])
            I[k,:]=f_b_sampling(A,B,HMM.obs_state,data[k])
        '''
        
    return post_A, post_B

'''
if __name__=='__main__':
    
    print('Start Gibbs sampling...')
    # Parallel computing by CPU acceleration
    p=Pool(8)
    post_A,post_B=Gibbs(A,B,data,I,p,10000)
    '''


