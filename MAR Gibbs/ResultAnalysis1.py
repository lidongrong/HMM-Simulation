# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:44:52 2022

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
import HMM1 as HMM


# num specifies the total experiments that has been run
num=20

# Save the data
for i in range(0,num):
    os.mkdir(f'Experiment{i}')
    # Save the results
    np.save(f'Experiment{i}/Post_A.npy',out_obj[i].post_A)
    np.save(f'Experiment{i}/Post_B.npy',out_obj[i].post_B)
    np.save(f'Experiment{i}/latent_seq.npy',out_obj[i].latent_seq)
    np.savetxt(f'Experiment{i}/log_prob.txt',out_obj[i].log_prob)
    np.save(f'Experiment{i}/data.npy',out_obj[i].data)
    np.save(f'Experiment{i}/TrueHidden.npy',out_obj[i].true_hidden)


#Alternative choice of transition matrix
transition=HMM.transition

                    
# Read the data
output=[]

# define the output class of the experiments
class Out:
    def __init__(self,data,post_A,post_B,latent_seq, log_prob,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.latent_seq=latent_seq
        self.log_prob=log_prob
        self.true_hidden=true_hidden
 

for i in range(0,num):
    post_A=np.load(f'Experiment{i}/Post_A.npy')
    post_B=np.load(f'Experiment{i}/Post_B.npy')
    latent_seq=np.load(f'Experiment{i}/latent_seq.npy')
    log_prob=np.loadtxt(f'Experiment{i}/log_prob.txt')
    data=np.load(f'Experiment{i}/data.npy')
    hidden_seq=np.load(f'Experiment{i}/TrueHidden.npy')
    
    output.append(Out(data,post_A,post_B,latent_seq,log_prob,hidden_seq))





color_bar=['red','blue','green','pink','k','violet','gold','brown','c','m']


# Adjust the total number of experiments such that the plot won't be too silly
os.mkdir('ResultAnalysis')
num=8

# Paint the Trace Plot of a11
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,0,0],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[0,0],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')

plt.savefig(f'ResultAnalysis/A11.png')
plt.close('all')

# Paint the Trace Plot of a22
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,1,1],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[1,1],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/A22.png')
plt.close('all')


# Paint the Trace Plot of a33
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,2,2],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[2,2],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/A33.png')
plt.close('all')


# Paint the Trace Plot of a44
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_A)),output[i].post_A[:,3,3],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_A)),np.repeat(transition[3,3],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/A44.png')
plt.close('all')


# Paint the Trace Plot of b11
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,1,1],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[1,1],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/B11.png')
plt.close('all')

# Paint the Trace Plot of b33
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,2,2],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[2,2],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/B33.png')
plt.close('all')


# Paint the Trace Plot of b54
for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].post_B)),output[i].post_B[:,4,3],color_bar[i%len(color_bar)],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].post_B)),np.repeat(obs_prob[4,3],len(output[i].post_B)),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/B54.png')
plt.close('all')

# Paint the Trace Plot log-prob

# Acquire true hidden seq
for k in range(0,num):
    h=output[k].latent_seq.copy()
    for i in range(0,h.shape[0]):
        for j in range(0,h.shape[1]):
            if h[i,j]!='None':
                h[i,j]=output[k].true_hidden[i,j]
    output[k].true_hidden=h


num=8
# compute true log prob
true_log_p=[]

for k in range(0,num):
    h=output[k].true_hidden
    true_log_p.append(p_evaluator(transition,obs_prob,h,output[k].data))

for i in range(0,num):
    row=int(math.ceil(num/2))
    plt.subplot(row,2,i+1)
    plt.plot(np.arange(0,len(output[i].log_prob)-20),output[i].log_prob[20:],color_bar[i],label=f'Experiment{i}')
    plt.plot(np.arange(0,len(output[i].log_prob)-20),np.repeat(true_log_p[i],len(output[i].log_prob)-20),'black',label='True Value')
    plt.xlabel('iteration')
    plt.legend(loc='best')
plt.savefig(f'ResultAnalysis/log_prob.png')
plt.close('all')

# Accuracy
a=[]
for i in range(0,num):
    acc=(np.sum(output[i].latent_seq==output[i].true_hidden)-np.sum(output[i].latent_seq=='None'))/np.sum(output[i].data!='None')
    print(acc)
    a.append(acc)

    

# Calculate Mean and SD of the estimation + accuracy
num=20

# posterior mean as estimation
post_mean=[]

# calculate posterior mean
for i in range(0,num):
    post_mean.append(np.mean(output[i].post_A,axis=0))

post_mean=np.array(post_mean)
np.mean(post_mean,axis=0)
np.var(post_mean,axis=0)

# posterior mean as estimation
post_mean=[]

# calculate posterior mean
for i in range(0,num):
    post_mean.append(np.mean(output[i].post_B,axis=0))

post_mean=np.array(post_mean)
np.mean(post_mean,axis=0)
np.var(post_mean,axis=0)
    
    