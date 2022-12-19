# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:42:13 2022

@author: lidon
"""

import numpy as np
import pandas as pd
import SD_generator as sdg
import scipy.stats as stats
import time
from tqdm import tqdm
import itertools
import multiprocessing as mp
import scipy.special as ss
import logging
import torch

class HMM_Model:
    def __init__(self,data,lengths,features,feature_types):
        '''
        data: data
        lengths: length of each sequence
        feature: string list of features
        feature_types: indicate if each feature is numeric or dummy (01)
        args: additional arguments(batch size, learning rate)
        '''
        self.data=data
        self.lengths=lengths
        self.features=features
        self.feature_types=feature_types
        self.sample_size=self.data.shape[0]
        self.max_length=self.data.shape[1]
        self.feature_dim=len(features)
        self.hidden_dim=data.shape[2]-self.feature_dim
        self.data_dim=data.shape[2]
        self.x,self.y=self.split()
        
        # missing value indicator, 1 for observed, 0 for missing
        self.x_masks=~np.isnan(self.x)
        self.y_masks=~np.isnan(self.y)
    
    def split(self):
        '''
        split data to x and y
        '''
        x=self.data[:,:,:self.feature_dim]
        y=self.data[:,:,-self.hidden_dim:]
        return x,y
    

class Optimizer:
    def __init__(self):
        pass

class Random_Gibbs(Optimizer):
    '''
    random scan gibbs sampler
    '''
    def __init__(self,model,args,initial=None):
        '''
        model: a hmm model
        initial: a dictionary recording a set of initial values
        '''
        # the hmm model
        self.model=model
        self.args=args
        # posterior sample
        self.param=None
        # if initial points specified
        if isinstance(initial,dict):
            self.pi=initial['pi']
            self.transition=initial['transition']
            self.mu=initial['mu']
            self.sigma=initial['sigma']
            self.beta=initial['beta']
        else:
            # otherwise, initialize parameters by default
            self.pi=np.random.dirichlet([1]*model.hidden_dim,1)[0]
            self.transition=np.random.dirichlet([1]*model.hidden_dim,model.hidden_dim)
            self.mu=np.random.multivariate_normal([0]*model.feature_dim, np.eye(model.feature_dim),model.hidden_dim)
            # invwishart with df model.feature_dim and V=I
            self.sigma=stats.invwishart.rvs(model.feature_dim,np.eye(model.feature_dim),model.hidden_dim)
            tmp_beta=[np.random.multivariate_normal(np.zeros(self.model.feature_dim),
                                                    1*np.eye(self.model.feature_dim),
                                                    self.model.hidden_dim) 
                  for i in range(self.model.hidden_dim)]
            tmp_beta=np.array(tmp_beta)
            self.beta=tmp_beta
            #print(self.beta[0][0])
    
    
    def obs_log_likelihood(self,x,y,z,length):
        '''
        likelihood of a single observation, where x,y,z are rv from a single chain
        length: length of this chain
        
        # test code
        x,y=m.split()
        optimizer.obs_log_likelihood(x[0],y[0],y[0],optimizer.model.lengths[0])
        '''
        # calculate log likelihood
        log_likelihood=0
        # evaluate the full likelihood
        
        # first from initial probability
        start_latent=np.where(z[0]==1)[0][0]
        log_likelihood+=np.log(self.pi[start_latent])
        
        
        for i in range(0,length):
            # the transitional probability
            if i>0:
                prev=np.where(z[i-1]==1)[0][0]
                curr=np.where(z[i]==1)[0][0]
                log_likelihood+=np.log(self.transition[prev,curr])
            
            # observation probability
            latent=np.where(z[i]==1)[0][0]
            log_likelihood+=stats.multivariate_normal.logpdf(x[i],self.mu[latent],
                                                                    self.sigma[latent])
            # if y[i] is observed
            if not sum(np.isnan(y[i]))==len(y[i]):
                # observation index
                observe=np.where(y[i]==1)[0][0]
                # emission probability
                curr=np.where(z[i]==1)[0][0]
                y_prob=np.exp(-np.dot(self.beta[curr],x[i]))
                y_prob=y_prob/sum(y_prob)
                y_prob=np.log(y_prob)
                # exp(-x_ beta_i)
                log_likelihood+=y_prob[observe]
        return log_likelihood
            
            
    
    def joint_pdf(self,x,y,z,p=None):
        '''
        calculate the joint probability
        p: multiprocessor core, None by default. If want to use multiprocessing, specify the core
        # test code:
        x,y=model.split()
        p=mp.Pool(16)
        optimizer.joint_pdf(x,y,y)
        '''
        # prior for log pdf
        log_prior=0
        for i in range(len(self.mu)):
            log_prior+= stats.multivariate_normal.logpdf(self.mu[i],np.zeros(self.model.feature_dim),
                                                         np.eye(self.model.feature_dim))
            log_prior+= stats.invwishart.logpdf(self.sigma[i],self.model.feature_dim,
                                             np.eye(self.model.feature_dim))
            log_prior+= sum(stats.multivariate_normal.logpdf(self.beta[i],np.zeros(self.model.feature_dim),
                                                             np.eye(self.model.feature_dim)))
                
        
        # log likelihood
        log_pdf=0
        if not p:
            for t in range(x.shape[0]):
                log_pdf+=self.obs_log_likelihood(x[t],y[t],z[t],self.model.lengths[t])
        if p:
            total=x.shape[0]
            lengths=self.model.lengths
            log_pdf=sum(p.starmap(self.obs_log_likelihood,
                          [(x[i],y[i],z[i],lengths[i]) for i in range(total)]))
        
        return log_pdf+log_prior
            
                
            
            
        
    
    def sample_z(self,x,y,z,lengths=None, x_masks=None, y_masks=None,p=None):
        '''
        sample latent z given x,y and other paramters
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        z=optimizer.sample_z(x,y,z)
        '''
        if lengths is None:
            lengths=self.model.lengths
        if x_masks is None:
            x_masks=self.model.x_masks
        if y_masks is None:
            y_masks=self.model.y_masks
        if p is None:
            new_z=list(map(self.sample_zt,x,y,z,lengths,x_masks,
                           y_masks))
            new_z=np.array(new_z)
            return new_z
    
    def sample_zt(self,x,y,z,length,x_masks,y_masks):
        '''
        sample single z use forward backward sampling
        x_mask, y_mask: indicators of missing, 1 for observed 0 for missing
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        zt=optimizer.sample_zt(x[0],y[0],z[0],optimizer.model.lengths[0],
                            optimizer.model.x_masks[0],optimizer.model.y_masks[0])
        '''
        log_trans=np.log(self.transition)
        # step 1: forward computation
        log_alpha=[]
        # if observed
        x_logpdf=np.array([stats.multivariate_normal.logpdf(x[0],self.mu[i],self.sigma[i])
                        for i in range(self.model.hidden_dim)])
        if np.any(y_masks[0]):
            y_obs=np.where(y[0]==1)[0][0]
            
            y_logpdf=np.array([-np.dot(self.beta[i][y_obs],x[0])
                               for i in range(self.model.hidden_dim)])
            y_logpdf=y_logpdf-ss.logsumexp(y_logpdf)
            
            log_alpha.append(np.log(self.pi)+y_logpdf+x_logpdf)
        # if y1 missing
        else:
            log_alpha.append(np.log(self.pi)+x_logpdf)
        # iteration from 0 to T-1
        for t in range(1,length):
            
            # iteration from last step
            last=log_alpha[len(log_alpha)-1]
            x_logpdf=np.array([stats.multivariate_normal.logpdf(x[t],self.mu[i],self.sigma[i])
                            for i in range(self.model.hidden_dim)])
            
            
            left=[last+log_trans[:,i] for i in range(self.model.hidden_dim)]
            left=ss.logsumexp(last,axis=0)
            # y observed
            if np.any(y_masks[t]):
                y_obs=np.where(y[t]==1)[0][0]
                
                y_logpdf=np.array([-np.dot(self.beta[i][y_obs],x[0])
                                   for i in range(self.model.hidden_dim)])
                y_logpdf=y_logpdf-ss.logsumexp(y_logpdf)
                log_alpha.append(left+x_logpdf+y_logpdf)
            # y missing
            else:
                log_alpha.append(left+x_logpdf)
        
        # backward sampling: sample from z_T to z_1
        log_alpha=np.array(log_alpha)
        reverse_z=[]
        # first sample z_T
        assert len(log_alpha)==length
        prob=log_alpha[length-1]
        prob=np.exp(prob-ss.logsumexp(prob))
        # position
        latent=np.random.choice(np.arange(self.model.hidden_dim),
                                1,True,p=prob)[0]
        curr_z=np.zeros(self.model.hidden_dim)
        curr_z[latent]=1
        reverse_z.append(curr_z)
        for t in range(length-2,-1,-1):
            last_z=reverse_z[len(reverse_z)-1]
            last_latent=np.where(last_z==1)[0][0]
            left=log_trans[:,last_latent]
            prob=left+log_alpha[t]
            prob=np.exp(prob-ss.logsumexp(prob))
            
            #prob=prob/sum(prob)
            latent=np.random.choice(np.arange(self.model.hidden_dim),
                                    1,True,p=prob)[0]
            curr_z=np.zeros(self.model.hidden_dim)
            curr_z[latent]=1
            reverse_z.append(curr_z)
        reverse_z.reverse()
        reverse_z=np.array(reverse_z)
        rest=z[length:]
        
        new_z=np.concatenate((reverse_z,rest),axis=0)
        return new_z
            
        
    
    def sample_x(self,x,y,z,lengths=None, x_masks=None,p=None):
        '''
        sample missing x for imputation
        length: the lengths of each sequences
        mask: masks indicating missing values (1 for observed)
        p: mp core
        # test code:
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        x=optimizer.sample_x(optimizer.model.x,optimizer.model.y,z)
        '''
        if lengths is None:
            lengths=self.model.lengths
        if x_masks is None:
            x_masks=self.model.x_masks
            
        if p is None:
            new_x=list(map(self.sample_xt,x,y,z,lengths,x_masks))
            new_x=np.array(new_x)
            return new_x
    
    def sample_xt(self,x,y,z,length,x_masks):
        '''
        sample a single x out, not exposed to the user
        used in sample_x
        length: length of this patient
        mask: indicating missing values in x. 1 for observed, 0 for missing
        # test code:
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        x=optimizer.sample_xt(x[0],y[0],z[0],optimizer.model.lengths[0],optimizer.model.x_masks[0])
        '''
        new_x=x.copy()
        for t in range(length):
            xt=x[t]
            mskt=x_masks[t]
            # first handle two edge case: all observed or all missing
            # case 1: all missing
            if sum(mskt)==0:
                latent=np.where(z[t]==1)[0][0]
                new_x[t]=np.random.multivariate_normal(self.mu[latent], self.sigma[latent])
            # case 2: all observed, then nothing happens
            elif sum(mskt)==len(mskt):
                pass
            # case 3: partially observed
            else:
                latent=np.where(z[t]==1)[0][0]
                # observed and missing index
                obs_index=np.argwhere(mskt==True)
                mis_index=np.argwhere(mskt==False)
                obs_index=obs_index.squeeze(-1)
                mis_index=mis_index.squeeze(-1)
                
                # permute xt, mu and sigma to prepare for conditional generation
                perm=list(mis_index)+list(obs_index)
                #x_tmp=xt[perm]
                mu_tmp=self.mu[latent].copy()
                sigma_tmp=self.sigma[latent].copy()
                mu_tmp=mu_tmp[perm]
                sigma_tmp=sigma_tmp[perm,:]
                sigma_tmp=sigma_tmp[:,perm]
                
                # sample from conditional
                # mu1 for mean of missing, mu2 for mean of obs index
                mu1=mu_tmp[0:len(mis_index)]
                mu2=mu_tmp[len(mis_index):]
                sigma11=sigma_tmp[0:len(mis_index),0:len(mis_index)]
                sigma12=sigma_tmp[0:len(mis_index),len(mis_index):]
                sigma21=sigma_tmp[len(mis_index):,0:len(mis_index)]
                sigma22=sigma_tmp[len(mis_index):,len(mis_index):]
                cond_mean=mu1+np.dot(sigma12,np.dot(np.linalg.inv(sigma22),xt[obs_index]-mu2))
                cond_covariance=sigma11-np.dot(sigma12,
                                               np.dot(np.linalg.inv(sigma22),
                                                   sigma21))
                cond=np.random.multivariate_normal(cond_mean,cond_covariance)
                new_x[t][mis_index]=cond
        return new_x
                
                
            
        
            
            
    
    def sample_pi(self,x,y,z):
        '''
        sample initial distribution pi given other parameters
        return generated pi
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        optimizer.sample_pi(x,y,z)
        '''
        # count start conditions
        start=z[:,0,:]
        start=sum(start)
        new_pi=np.random.dirichlet(1+start)
        
        self.pi=new_pi
        
        return new_pi
    
    def sample_transition(self,x,y,z):
        '''
        sample transition matrix
        return sampled transition
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        optimizer.sample_transition(x,y,z)
        '''
        
        # decode the observations into int
        position=np.argwhere(z==1)
        position=position[:,2]
        
        
        lengths_sum=self.model.lengths.cumsum()
        lengths_sum=lengths_sum.astype(np.int32)
        position=np.array_split(position,lengths_sum)
        #position=np.array(position)
        
        # pad the results, fill np.nan to handle different lengths
        position=position[:-1]
        position=np.column_stack((itertools.zip_longest(*position, fillvalue=np.nan)))
        #position=position.astype(np.int32)
        
        # directly sample from posterior
        new_A=[]
        for i in range(self.model.hidden_dim):
            transform=np.array([np.sum((position[:,:-1]==i)&(position[:,1:]==j)) 
                                for j in range(self.model.hidden_dim)])
            new_A.append(np.random.dirichlet(1+transform,1)[0])
        new_A=np.array(new_A)
        self.transition=new_A
        
        return new_A
            
            
    
    def sample_mu(self,x,y,z):
        '''
        sample emission probability with mean mu
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        optimizer.sample_mu(x,y,z)
        '''
        new_mu=[]
        for i in range(self.model.hidden_dim):
            # filter x corresponding to latent state=state i
            # mask a boolean vector to find corresponding x
            mask=z[:,:,i]
            mask=(mask==1)
            # find x
            xz=x[mask]
            
            # sample size and dimension
            n=xz.shape[0]
            dim=xz.shape[1]
            mean=np.dot(
                np.linalg.inv(n * np.linalg.inv(self.sigma[i])+np.eye(dim)),
                        np.dot(np.linalg.inv(self.sigma[i]),sum(xz))
                        )
            covariance=np.linalg.inv(n*np.linalg.inv(self.sigma[i])+np.eye(dim))
            new_mu.append(np.random.multivariate_normal(mean,covariance,1)[0])
        
        new_mu=np.array(new_mu)
        self.mu=new_mu
        return new_mu
            
    
    def sample_sigma(self,x,y,z):
        '''
        sample sigma, the covariance of emission distribution
        return sampled sigma
        #test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        sig=optimizer.sample_sigma(x,y,z)
        '''
        new_sigma=[]
        for i in range(self.model.hidden_dim):
            # filter x corresponding to latent state=state i
            # mask a boolean vector to find corresponding x
            mask=z[:,:,i]
            mask=(mask==1)
            # find x
            xz=x[mask]
            
            # number of qualified samples
            n=xz.shape[0]
            # sample covariance matrix
            sample_covariance=(1/n) * np.dot((xz-self.mu[i]).T,xz-self.mu[i])
            # sample from posterior
            degree=self.model.feature_dim+n
            V=n*sample_covariance+np.eye(self.model.feature_dim)
            new_sigma.append(stats.invwishart.rvs(degree,V,1))
        new_sigma=np.array(new_sigma)
        self.sigma=new_sigma
        return new_sigma
    
    # sample beta
    def sample_beta(self,x,y,z,SGLD=True,SGLD_step=None,SGLD_batch=None):
        '''
        sample beta, the coefficients in regression
        return updated beta
        x,y,z: full data
        SGLD: if True, use Unadjusted SGLD
        SGLD_step: current iteration
        SGLD_batch: batch size
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        beta=optimizer.sample_beta(x,y,z,True,1,10)
        '''
        if SGLD:
            k=self.args.hk
            n=self.args.batch_size
            N=self.model.data.shape[0]
            # index of subsamples
            sub_index=np.random.choice(np.arange(N),n,replace=False)
            x,y,z=x[sub_index],y[sub_index],z[sub_index]
            #print(x.shape)
            new_beta=self.SGLD_beta(x,y,z,self.beta)
            self.beta=new_beta
        return self.beta
    
    def beta_grad(self,x,y,z,beta):
        '''
        x,y,z=dataset, should be numpy array
        beta: the full beta paramter, should be numpy array
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        beta=optimizer.beta
        g=optimizer.beta_grad(x[0:50],y[0:50],z[0:50],beta)
        '''
        f=0
        beta=beta.astype(x.dtype)
        x=torch.tensor(x)
        beta=torch.tensor(beta,requires_grad=True)
        for i in range(self.model.hidden_dim):
            for j in range(self.model.hidden_dim):
                maskz=z[:,:,i]
                maskz=(maskz==1)
                masky=y[:,:,j]
                masky=(masky==1)
                mask=maskz*masky
                # find x
                if mask.size!=0:
                    x_zy=x[mask]
                    f=f+self.beta_forward(x_zy,beta[i],j)
        f.backward()
        grad=beta.grad
        return grad.numpy()
    
    def beta_forward(self,x,beta,j):
        '''
        return log of p(x,y,z|\beta)p(\beta) in torch form
        test code:
        beta=np.random.normal(0,1,(7,40))
        x=np.random.normal(0,1,(50,40))
        x=torch.tensor(x,requires_grad=True)
        beta=torch.tensor(beta,requires_grad=True)
        f=optimizer.beta_forward(x,beta,0)
        '''
        #x=torch.tensor(x,requires_grad=True)
        #beta=torch.tensor(beta,requires_grad=True)
        batch=self.args.batch_size
        log_term1=-batch*torch.dot(beta[j],beta[j])/(2*self.model.data.shape[0])
        
        #term2=torch.nn.functional.softmax(-torch.matmul(x,beta.T),dim=0)
        #log_term2=torch.sum(torch.log(term2),axis=0)[j]
        
        term2=torch.matmul(-x,beta.T)
        log_term2=torch.sum(term2[:,j]-torch.logsumexp(term2,dim=1))
        return log_term1+log_term2
        
    # perform SGLD by evaluating beta on x,y and z
    #def SGLD_beta(self,x,y,z,n,k,beta):
    def SGLD_beta(self,x,y,z,beta):
        '''
        perform SGLD step
        step size h_k is adjusted to optimal as k^{-1/3}
        x are selected entries
        i, j: index identifying beta[i][j] (the sub-entry to optimize)
        n: mini batch size
        k: iteration step
        '''
        N=self.model.data.shape[0]
        n=self.args.batch_size
        self.args.hk=self.args.hk+1
        hk=self.args.learning_rate * (self.args.hk**(-1/3))
        print('hk:',hk)
        '''
        noise=np.random.multivariate_normal(np.zeros(self.model.feature_dim),
                                            hk*np.eye(self.model.feature_dim))
        '''
        noise=np.random.normal(0,np.sqrt(hk),size=beta.shape)
        
        new_beta=beta.copy()
        
        beta_grad=self.beta_grad(x,y,z,new_beta)
        
        new_beta=new_beta + (hk/2) * (N/n) * beta_grad + noise
        
        return new_beta
    
    def z_initializer(self,x,y):
        '''
        initialize latent variables x and z
        input x is partially observed x
        # test code
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        '''
        z=y.copy()
        for i in range(z.shape[0]):
            for j in range(self.model.lengths[i]):
                # if missing
                if not sum(self.model.y_masks[i][j]):
                    latent=np.random.randint(0,self.model.hidden_dim,1)[0]
                    plug=np.zeros(self.model.hidden_dim)
                    plug[latent]=1
                    z[i][j]=plug
        return z
        
    def x_initializer(self,x,y,z):
        '''
        initialize x
        # test code
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        x=optimizer.x_initializer(optimizer.model.x,optimizer.model.y,z)
        '''
        new_x=self.sample_x(x,y,z)
        return new_x
    
    def latent_initializer(self,x,y):
        '''
        initialize missing x by imputing
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        '''
        z=self.z_initializer(x,y)
        x=self.x_initializer(x,y,z)
        return x,z
                    
    
    
    def run(self,n,log_step=None,prog_bar=True,prob=0.5,SGLD=True,initial_x=None,initial_z=None):
        '''
        collect samples from n iterations
        n: total number of iterations
        log_step: step for printing results for monitoring. if None, not report
        prog_bar: if display progress bar
        prob: probability for sampling parameters (because we use random scan)
        SGLD: if use SGLD
        batch: batch size in SGLD and random scan gibbs
        initial_x,initial_z: initial values of x and z
        # test code
        param=optimizer.run(20,4,True,0.5,True,10)
        '''
        # initialize latent variables if don't pass into initial value
        if initial_x is None:
            x,y=self.model.x,self.model.y
            x,z=self.latent_initializer(x,y)
        
        # store samples
        sample_param={}
        sample_param['beta']=[]
        sample_param['mu']=[]
        sample_param['sigma']=[]
        sample_param['pi']=[]
        sample_param['transition']=[]
        batch=self.args.batch_size
        for s in tqdm(range(n)):
            if s%log_step==0:
                pass
                #print('iteration: ',s)
                
            # decide sample latent var or sample theta
            flip=np.random.choice([0,1],1,replace=True,p=[1-prob,prob])[0]
            
            # sample parameter
            if flip==1:
                new_beta=self.sample_beta(x,y,z,True,len(sample_param['beta'])+1,batch)
                new_mu=self.sample_mu(x,y,z)
                new_transition=self.sample_transition(x,y,z)
                new_pi=self.sample_pi(x,y,z)
                new_sigma=self.sample_sigma(x,y,z)
                sample_param['beta'].append(self.beta)
                print(self.beta[0][0])
                sample_param['mu'].append(self.mu)
                sample_param['sigma'].append(self.sigma)
                sample_param['pi'].append(self.pi)
                sample_param['transition'].append(self.transition)
                self.param=sample_param
            # sample latent variable
            # each time only update a small batch to accelerate computation
            elif flip==0:
                # total sample size
                N=self.model.data.shape[0]
                # number of batches
                batch_num=N//batch
                # randomly select a batch to update
                
                choose_batch=np.random.choice(np.arange(batch_num),1,True)[0]
                batch_index=np.arange(choose_batch*batch,min(N,(choose_batch+1)*batch))
                z_batch=z[batch_index]
                y_batch=y[batch_index]
                x_batch=x[batch_index]
                
                z_batch=self.sample_z(x_batch,y_batch,z_batch,self.model.lengths[batch_index],
                                      self.model.x_masks[batch_index],self.model.y_masks[batch_index])
                x_batch=self.sample_x(x_batch,y_batch,z_batch,self.model.lengths[batch_index],
                                      self.model.x_masks[batch_index])
                
                x[batch_index]=x_batch
                z[batch_index]=z_batch
                
                
                '''
                x=self.sample_x(x,y,z)
                z=self.sample_z(x,y,z)
                '''
        
        return self.param
            
                
                
        