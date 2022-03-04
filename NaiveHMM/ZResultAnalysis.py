# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:27:17 2022

@author: lidon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:42:56 2022
@author: s1155151972
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:55:01 2021
@author: s1155151972
"""


# The result analysis script is desinged to analyze all results in one-shot



import os
import matplotlib.pyplot as plt
import math
import numpy as np
from ZMARGibbs import*
from multiprocessing import Pool
import multiprocessing as mp
# Read the data


def read_data(num):
    output=[]

    for i in range(0,num):
        post_A=np.load(f'Experiment{i}/Post_A.npy')
        post_B=np.load(f'Experiment{i}/Post_B.npy')
        pi=np.load(f'Experiment{i}/Post_pi.npy')
        latent_seq=np.load(f'Experiment{i}/latent_seq.npy')
        log_prob=np.loadtxt(f'Experiment{i}/log_prob.txt')
        data=np.load(f'Experiment{i}/data.npy')
        hidden_seq=np.load(f'Experiment{i}/TrueHidden.npy')
    
        output.append(Out(data,post_A,post_B,pi,latent_seq,log_prob,hidden_seq))
    return output




color_bar=['red','blue','green','pink','k','violet','gold','brown','c','m']


# Make a new directory to store simulation result

def result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p):


    os.mkdir('ResultAnalysis')

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
    
    # Paint the trace plot of entries of B
    for j in range(0,output[0].post_B[0].shape[0]):
        for k in range(0,output[0].post_B[0].shape[1]):
            for i in range(0,num):
                row=int(math.ceil(num/2))
                plt.subplot(row,2,i+1)
                plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_A[:,j,k],color_bar[i],label=f'Experiment{i}')
                plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[j,k],len(output[i].post_B)),'black',label='True Value')
                plt.xlabel('iteration')
                plt.legend(loc='best')
            plt.savefig(f'ResultAnalysis/B{j+1}{k+1}.png')
            plt.close('all')
   



    # Paint the Trace Plot log-prob

    # Acquire true hidden seq
    true_log_p=[]
    for i in range(0,num):
        h=output[i].latent_seq.copy()
        for j in range(0,h.shape[0]):
            for k in range(0,h.shape[1]):
                if h[j,k]!='None':
                    h[j,k]=output[i].true_hidden[j,k]
        # compute true log prob
        true_log_p.append(p_evaluator(transition,obs_prob,pi,h,output[i].data,hidden_state,obs_state,p))

    for i in range(0,num):
        row=int(math.ceil(num/2))
        plt.subplot(row,2,i+1)
        plt.plot(np.arange(0,len(output[i].log_prob)),output[i].log_prob,color_bar[i],label=f'Experiment{i}')
        plt.plot(np.arange(0,len(output[i].log_prob)),np.repeat(true_log_p[i],len(output[i].log_prob)),'black',label='True Value')
        plt.xlabel('iteration')
        plt.legend(loc='best')
    plt.savefig('ResultAnalysis/logProb.png')
    plt.close('all')


    # Accuracy
    accuracy=[]
    for i in range(0,len(output)):
        acc=np.sum((output[i].latent_seq==output[i].true_hidden)&(output[i].data!='None'))/np.sum(output[i].data!='None')
        accuracy.append(acc)
    accuracy=np.array(accuracy)
    np.savetxt(f'ResultAnalysis/accuracy.txt',accuracy)


'''
# HMM construction
transition=np.array(
        [[0.6,0.2,0.1,0.05,0.05],[0.05,0.6,0.2,0.1,0.05],[0.05,0.05,0.6,0.2,0.1],[0.05,0.05,0.1,0.6,0.2],
         [0.05,0.05,0.1,0.2,0.6]]
        )           
state=np.array(['A','B','C','D','E'])
hidden_state=state
obs_state=np.array(['Blue','Red','Green','Purple','Grey'])
    
obs_prob=np.array([[0.7,0.2,0.05,0.04,0.01],[0.01,0.7,0.2,0.05,0.04],[0.04,0.01,0.7,0.2,0.05],
                   [0.05,0.04,0.01,0.7,0.2],[0.2,0.05,0.04,0.01,0.7]
        ])
pi=[0.5,0.2,0.2,0.1,0]
'''

'''
if __name__=='__main__':
    # Use multicore CPU
    p=Pool(mp.cpu_count())
    num=10
    output=read_data(num)
    result_analysis(color_bar,output,num,transition,obs_prob,pi,hidden_state,obs_state,p)
'''