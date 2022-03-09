# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:02:35 2022

@author: lidon
"""

###
# Analyze the results of EM algorithm
###

import os
import matplotlib.pyplot as plt
import math
import numpy as np
from ZMARGibbs import*
from multiprocessing import Pool
import multiprocessing as mp
from EMHMM import*


transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])

pi=np.array([0.7,0.2,0.1])


class Out:
    def __init__(self,data,post_A,post_B,post_pi,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.post_pi=post_pi
        self.true_hidden=true_hidden

        
num=4

def read_data(num):
    output=[]

    for i in range(0,num):
        post_A=np.load(f'Experiment{i}/at.npy')
        post_B=np.load(f'Experiment{i}/bt.npy')
        pi=np.load(f'Experiment{i}/pit.npy')
        
        data=np.load(f'Experiment{i}/data.npy')
        hidden_seq=np.load(f'Experiment{i}/TrueHidden.npy')
    
        output.append(Out(data,post_A,post_B,pi,hidden_seq))
    return output


color_bar=['red','blue','green','pink','k','violet','gold','brown','c','m']


# Make a new directory to store simulation result

def result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p):
    h=hidden_state
    o=obs_state

    os.mkdir('ResultAnalysis')
    
    
    print('Painting Trace plot of A...')
    
    # Paint the Trace Plot of entries of A
    for j in range(0,output[0].post_A[0].shape[0]):
        for k in range(0,output[0].post_A[0].shape[1]):
            for i in range(0,num):
                row=int(math.ceil(num/2))
                plt.subplot(row,2,i+1)
                plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,j,k],color_bar[i],label=f'Experiment{i}')
                plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[j,k],len(output[i].post_B)),'black',label='True Value')
                plt.xlabel('iteration')
                plt.legend(loc='best')
            plt.savefig(f'ResultAnalysis/A{j+1}{k+1}.png')
            plt.close('all')
    
    
    print('Painting Trace plot of B...')
    # Paint the trace plot of entries of B
    for j in range(0,output[0].post_B[0].shape[0]):
        for k in range(0,output[0].post_B[0].shape[1]):
            for i in range(0,num):
                row=int(math.ceil(num/2))
                plt.subplot(row,2,i+1)
                plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,j,k],color_bar[i],label=f'Experiment{i}')
                plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[j,k],len(output[i].post_B)),'black',label='True Value')
                plt.xlabel('iteration')
                plt.legend(loc='best')
            plt.savefig(f'ResultAnalysis/B{j+1}{k+1}.png')
            plt.close('all')
    
    print('Painting Trace Plot of pi...')
    # Trace plot of pi
    for j in range(0,output[0].post_pi[0].shape[0]):
        for k in range(0,num):
            row=int(math.ceil(num/2))
            plt.subplot(row,2,k+1)
            plt.plot(np.arange(0,len(output[k].post_pi)),output[k].post_pi[:,j],color_bar[k],label=f'Experiment{k}')
            plt.plot(np.arange(0,len(output[k].post_pi)),np.repeat(pi[j],len(output[k].post_pi)),'black',label='True Value')
            plt.xlabel('iteration')
            plt.legend(loc='best')
        plt.savefig(f'ResultAnalysis/pi{j+1}.png')
        plt.close('all')
    
    print('Painting Trace Plot of log prob...')
    # compute the trace plot of objective function
    for j in range(0,num):
        row=int(math.ceil(num/2))
        log_obs=[]
        # acquire the trace plot
        for k in range(0,len(output[j].post_A)):
            log_prob=y_prob(output[j].data,output[j].post_A[k],output[j].post_B[k],output[j].post_pi[k],h,o,p)
            log_prob=np.log(log_prob)
            log_prob=sum(log_prob)
            log_obs.append(log_prob)
        true_log_prob=y_prob(output[j].data,transition,obs_prob,pi,h,o,p)
        true_log_prob=np.log(true_log_prob)
        true_log_prob=sum(true_log_prob)
        plt.subplot(row,2,j+1)
        log_obs=np.array(log_obs)
        plt.plot(np.arange(0,len(log_obs)),log_obs[:],color_bar[j],label=f'Experiment{k}')
        plt.plot(np.arange(0,len(log_obs)),np.repeat(true_log_prob,len(log_obs)),'black',label='True Log Prob')
        plt.xlabel('iteration')
        plt.legend(loc='best')
    plt.savefig(f'ResultAnalysis/LogProb.png')
    plt.close('all')
    
    print('Painting a Zoomed version...')
    for j in range(0,num):
        row=int(math.ceil(num/2))
        log_obs=[]
        # acquire the trace plot
        for k in range(0,len(output[j].post_A)):
            log_prob=y_prob(output[j].data,output[j].post_A[k],output[j].post_B[k],output[j].post_pi[k],h,o,p)
            log_prob=np.log(log_prob)
            log_prob=sum(log_prob)
            log_obs.append(log_prob)
        true_log_prob=y_prob(output[j].data,transition,obs_prob,pi,h,o,p)
        true_log_prob=np.log(true_log_prob)
        true_log_prob=sum(true_log_prob)
        plt.subplot(row,2,j+1)
        log_obs=np.array(log_obs)
        plt.plot(np.arange(80,len(log_obs)),log_obs[80:],color_bar[j],label=f'Experiment{k}')
        plt.plot(np.arange(80,len(log_obs)),np.repeat(true_log_prob,len(log_obs)-80),'black',label='True Log Prob')
        plt.xlabel('iteration')
        plt.legend(loc='best')
    plt.savefig(f'ResultAnalysis/Zoomed_LogProb.png')
    plt.close('all')
    
    print('Painting 2-Dim trace plot...')
    # compute 2 dimensional trace plot
    
    print('computing 2-D trace plot of A...')
    for j in range(0,output[0].post_A[0].shape[0]):
        for i in range(0,num):
            row=int(math.ceil(num/2))
            plt.subplot(row,2,i+1)
            #plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,j,0],color_bar[i],label=f'Experiment{i}')
            plt.plot(output[i].post_A[:,j,0],output[i].post_A[:,j,1],color_bar[i],label=f'Experiment{i}')
            plt.text(transition[j,0],transition[j,1],'True Value')
            plt.plot(transition[j,0],transition[j,1],'go')
            plt.xlabel('iteration')
            plt.legend(loc='best')
        plt.savefig(f'ResultAnalysis/A{j}.png')
        plt.close('all')
    
    print('computing 2-D trace plot of B...')
    for j in range(0,output[0].post_A[0].shape[0]):
        for i in range(0,num):
            row=int(math.ceil(num/2))
            plt.subplot(row,2,i+1)
            #plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,j,0],color_bar[i],label=f'Experiment{i}')
            plt.plot(output[i].post_B[:,j,0],output[i].post_B[:,j,1],color_bar[i],label=f'Experiment{i}')
            plt.text(obs_prob[j,0],obs_prob[j,1],'True Value')
            plt.plot(obs_prob[j,0],obs_prob[j,1],'go')
            plt.xlabel('iteration')
            plt.legend(loc='best')
        plt.savefig(f'ResultAnalysis/B{j}.png')
        plt.close('all')
    
    print('Computing 2-D trace plot of pi...')
    for i in range(0,num):
        row=int(math.ceil(num/2))
        plt.subplot(row,2,i+1)
        #plt.plot(np.arange(0,len(output[i].post_pi)),output[i].post_pi[:,0],color_bar[i],label=f'Experiment{i}')
        plt.plot(output[i].post_pi[:,0],output[i].post_pi[:,1],color_bar[i],label=f'Experiment{i}')
        plt.text(pi[0],pi[1],'True Value')
        plt.plot(pi[0],pi[1],'go')
        plt.xlabel('iteration')
        plt.legend(loc='best')
    plt.savefig(f'ResultAnalysis/pi.png')
    plt.close('all')
    
if __name__=='__main__':
    
    
    # Use multicore CPU
    p=Pool(16)
    num=4
    os.chdir('Missingrate0')
    out=read_data(4)
    result_analysis(color_bar,out,num,transition,obs_prob,pi,hidden_state,obs_state,p)
    os.chdir('..')
    os.chdir('Missingrate0.3')
    out=read_data(4)
    result_analysis(color_bar,out,num,transition,obs_prob,pi,hidden_state,obs_state,p)
    os.chdir('..')
    os.chdir('Missingrate0.5')
    out=read_data(4)
    result_analysis(color_bar,out,num,transition,obs_prob,pi,hidden_state,obs_state,p)
    os.chdir('..')
    os.chdir('Missingrate0.7')
    out=read_data(4)
    result_analysis(color_bar,out,num,transition,obs_prob,pi,hidden_state,obs_state,p)
    os.chdir('..')
    os.chdir('Missingrate0.9')
    out=read_data(4)
    result_analysis(color_bar,out,num,transition,obs_prob,pi,hidden_state,obs_state,p)
            
            