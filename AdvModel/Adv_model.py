# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:49:26 2023

@author: lidon
"""

import numpy as np
import pandas as pd
import Adv_SD_generator as sdg
import scipy.stats as stats
import time
from tqdm import tqdm
import itertools
import multiprocessing as mp
import scipy.special as ss
import logging
import torch
import matplotlib.pyplot as plt
import utils
import numba as nb
import joblib
from joblib import Parallel,delayed
import gc
import tracemalloc
import os
import tempfile
#os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'


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
    def __init__(self,model,args=None,initial=None):
        '''
        model: a hmm model
        initial: a dictionary recording a set of initial values
        '''
        # the hmm model
        self.model=model
        self.args=args
        # posterior sample
        sample_param={}
        sample_param['beta']=[]
        sample_param['mu']=[]
        sample_param['sigma']=[]
        sample_param['pi']=[]
        sample_param['alpha']=[]
        self.param=sample_param
        # if initial points specified
        if isinstance(initial,dict):
            self.pi=initial['pi']
            self.alpha=initial['alpha']
            self.mu=initial['mu']
            self.sigma=initial['sigma']
            self.beta=initial['beta']
        else:
            # otherwise, initialize parameters by default
            self.pi=np.random.dirichlet([1]*model.hidden_dim,1)[0]
            self.mu=np.random.multivariate_normal([0]*model.feature_dim, np.eye(model.feature_dim),model.hidden_dim)
            # invwishart with df model.feature_dim and V=I
            self.sigma=stats.invwishart.rvs(model.feature_dim,np.eye(model.feature_dim),model.hidden_dim)
            tmp_beta=[np.random.multivariate_normal(np.zeros(self.model.feature_dim),
                                                    1*np.eye(self.model.feature_dim),
                                                    self.model.hidden_dim-1) 
                  for i in range(self.model.hidden_dim)]
            tmp_beta=np.array(tmp_beta)
            self.beta=tmp_beta
            self.alpha=np.random.normal(0,1,(self.model.hidden_dim,
                                             self.model.hidden_dim,
                                             self.model.feature_dim))
            self.alpha[:,-1,:]=0
            #print(self.beta[0][0])
        # real z
        self.real_z=None
        self.real_x=None
        
        # specify prior of beta and alpha, normal by default, support 'Laplace'
        self.beta_prior=self.args.beta_prior
        self.alpha_prior=self.args.alpha_prior
        #self.proposal=self.args.proposal
        # std: prior concentration term, interpreted as regularization parameter
        self.regularize=self.args.regularize
    
    # load real latent var
    def set_real(self,real_x,real_z):
        self.real_z=real_z
        self.real_x=real_x
    
    def load_true_param(self,param_path):
        '''
        load and return true parameters
        '''
        true_initial=np.load(f'{param_path}\initial.npy')
        true_alpha=np.load(f'{param_path}/alpha.npy')
        true_beta=np.load(f'{param_path}/beta.npy')
        true_mu=np.load(f'{param_path}\mu.npy')
        true_sigma=np.load(f'{param_path}\sigma.npy')
        
        # register true parameters to the optimizer
        self.true_param={}
        self.true_param['pi']=true_initial
        self.true_param['beta']=true_beta
        self.true_param['mu']=true_mu
        self.true_param['sigma']=true_sigma
        self.true_param['alpha']=true_alpha
        
        return self.true_param
    
    def set_as_true_param(self,true_param):
        '''
        set optimizer.parameter(initial, beta, etc) as true parameters
        true_param: a dict containing true parameters
        test code:
        true_param=optimizer.load_true_param(param_path)
        optimizer.set_as_true_param(true_param)
        '''
        self.pi=true_param['pi']
        self.beta=true_param['beta']
        self.mu=true_param['mu']
        self.alpha=true_param['alpha']
        self.sigma=true_param['sigma']
    
    def beta_log_prior(self,beta):
        '''
        evaluate the prior of beta
        if choice = 'Normal', prior specified as normal distribution with std
        if choice = 'Laplace', prior specified as laplace distribution: e^{-|x|/b}
        return log prob of the prior distribution
        input is a tensor, output also a tensor
        '''
        std=self.regularize
        if self.beta_prior=='Normal':
            # evaluate normal prior with std
            return -0.5 * (torch.norm(beta)**2) / (std * std)
        else:
            # evaluate laplace prior with std
            return -1* torch.norm(beta,1) / std
        
        # return a tensor
    
    def alpha_log_prior(self,alpha):
        '''
        evaluate the prior of beta
        if choice = 'Normal', prior specified as normal distribution with std
        if choice = 'Laplace', prior specified as laplace distribution: e^{-|x|/b}
        return the log prob of the prior distribution
        input a tensor, output also a tensor
        '''
        std=self.regularize
        if self.alpha_prior=='Normal':
            # evaluate normal prior with std
            return -0.5 * (torch.norm(alpha)**2) / (std * std)
        else:
            # evaluate laplace prior with std
            return -1 * torch.norm(alpha,1) / std
        
        # return a tensor
    
    
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
                y_prob=np.exp(np.dot(self.beta[curr],x[i]))
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
        # test code #2 (multicore)
        p=mp.Pool(12)
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        s=time.time()
        z=optimizer.sample_z(x,y,z,None,None,None,p)
        e=time.time()
        print('time: ',e-s)
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
        
        '''
            size=z.shape[0]
            new_z=Parallel(n_jobs=8)(delayed(self.sample_zt)(x[t],y[t],z[t],lengths[t],
                                x_masks[t],y_masks[t]) for t in range(size))
            new_z=np.array(new_z)
            return new_z
        '''
        if p:
            '''
            #print('multicore utilized')
            size=z.shape[0]
            new_z=Parallel(n_jobs=self.core_num)(delayed(self.sample_zt)(x[t],y[t],z[t],lengths[t],
                                x_masks[t],y_masks[t]) for t in range(size))
            new_z=np.array(new_z)
            return new_z
        '''
            num=self.core_num
            size=z.shape[0]
            new_z=Parallel(n_jobs=num)(delayed(self.sample_zt)(x[t],y[t],z[t],lengths[t],
                                x_masks[t],y_masks[t]) for t in range(size))
            new_z=np.array(new_z)
            #gc.collect()
            return new_z
        
    def sample_zt(self,x,y,z,length,x_masks,y_masks):
        '''
        sample single z use forward backward sampling
        x_mask, y_mask: indicators of missing, 1 for observed 0 for missing
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        start=time.time()
        zt=optimizer.sample_zt(x[0],y[0],z[0],optimizer.model.lengths[0],
                            optimizer.model.x_masks[0],optimizer.model.y_masks[0])
        end=time.time()
        print(end-start)
        '''
        # get the log of transition matrix at each state
        log_trans= self.alpha @ x.T
        log_trans=log_trans.T
        log_trans=np.transpose(log_trans,[0,2,1])
        # normalize
        log_trans=ss.softmax(log_trans,axis=-1)
        log_trans=np.log(log_trans)
        
        
        beta=np.concatenate((self.beta,np.zeros((self.beta.shape[0],1,self.beta.shape[2]))),axis=1)
        # step 1: forward computation
        log_alpha=np.zeros((length,self.model.hidden_dim))
        # if observed
        x_logpdf = np.array(list(map(stats.multivariate_normal.logpdf,np.tile(x[0],(self.model.hidden_dim,1)),
                            self.mu,self.sigma)))
        # if y1 observed
        if np.any(y_masks[0]):
            y_obs=np.where(y[0]==1)[0][0]
            #y_logpdf=np.dot(beta[:][y_obs],x[0])-ss.logsumexp(np.dot(beta,x[0]),axis=1)
            y_logpdf=np.dot(beta[:,y_obs,:],x[0])-ss.logsumexp(np.dot(beta,x[0]),axis=1)
            log_alpha[0]=np.log(self.pi)+y_logpdf+x_logpdf
        # if y1 missing
        else:
            log_alpha[0]=np.log(self.pi)+x_logpdf
        # iteration from 0 to T-1
        for t in range(1,length):
            # iteration from last step
            last=log_alpha[t-1]
            x_logpdf = np.array(list(map(stats.multivariate_normal.logpdf,np.tile(x[t],(self.model.hidden_dim,1)),
                                self.mu,self.sigma)))
            # temporary transition matrix at stage t
            left=(last+log_trans[t-1].T).T
            left=ss.logsumexp(left,axis=0)
            
            #left=[last+log_trans[:,i] for i in range(self.model.hidden_dim)]
            #left=ss.logsumexp(last,axis=0)
            # y observed
            if np.any(y_masks[t]):
                y_obs=np.where(y[t]==1)[0][0]
                y_logpdf=np.dot(beta[:,y_obs,:],x[t])-ss.logsumexp(np.dot(beta,x[t]),axis=1)
                log_alpha[t]=left + x_logpdf + y_logpdf
            # y missing
            else:
                log_alpha[t]=left + x_logpdf
        
        # backward sampling: sample from z_T to z_1
        # allocate memory in advance
        z_sample=np.zeros((length,self.model.hidden_dim))
        # first sample z_T
        assert len(log_alpha)==length
        prob=log_alpha[length-1]
        # use gumbel-max trick instead
        g=np.random.gumbel(0,1,len(prob))
        latent=np.argmax(prob+g)
        curr_z=np.zeros(self.model.hidden_dim)
        curr_z[latent]=1
        z_sample[-1]=curr_z
        # fill in values backwardly
        for t in range(length-2,-1,-1):
            last_z=z_sample[t+1]
            last_latent=np.where(last_z==1)[0][0]
            left=log_trans[t][:,last_latent]
            prob=left+log_alpha[t]
            # use gumbel max trick
            g=np.random.gumbel(0,1,len(prob))
            latent=np.argmax(prob+g)
            curr_z=np.zeros(self.model.hidden_dim)
            curr_z[latent]=1
            z_sample[t]=curr_z
        rest=z[length:]
        
        new_z=np.concatenate((z_sample,rest),axis=0)
        return new_z
            
        
    
    def sample_x(self,x,y,z,lengths=None, x_masks=None,y_masks=None,p=None):
        '''
        sample missing x for imputation
        length: the lengths of each sequences
        mask: masks indicating missing values (1 for observed)
        p: mp core
        # test code:
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        x=optimizer.sample_x(optimizer.model.x,optimizer.model.y,z)
        # test code #2(multicore):
        p=mp.Pool(12)
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        start=time.time()
        x=optimizer.sample_x(optimizer.model.x,optimizer.model.y,z,None,None,p)
        end=time.time()
        print('time: ',end-start)
        '''
        if lengths is None:
            lengths=self.model.lengths
        if x_masks is None:
            x_masks=self.model.x_masks
        if y_masks is None:
            y_masks=self.model.y_masks
            
        if p is None:
            new_x=list(map(self.sample_xt,x,y,z,lengths,x_masks,y_masks))
            new_x=np.array(new_x)
            return new_x
            
        if p:
            size=z.shape[0]
            num=self.core_num
            new_x=Parallel(n_jobs=num)(delayed(self.sample_xt)(x[t],y[t],z[t],lengths[t],
                                x_masks[t],y_masks[t]) for t in range(size))
            new_x=np.array(new_x)
            #gc.collect()
            return new_x

    
    def sample_xt(self,x,y,z,length,x_masks,y_masks):
        '''
        sample a single x out, not exposed to the user
        used in sample_x
        length: length of this patient
        mask: indicating missing values in x. 1 for observed, 0 for missing
        # test code:
        z=optimizer.z_initializer(optimizer.model.x,optimizer.model.y)
        xt=optimizer.sample_xt(x[0],y[0],z[0],optimizer.model.lengths[0],optimizer.model.x_masks[0],
                              optimizer.model.y_masks[0])
        '''
        new_x=x.copy()
        beta=np.concatenate((self.beta,np.zeros((self.beta.shape[0],1,self.beta.shape[2]))),axis=1)
        for t in range(length):
            xt=x[t]
            mskt=x_masks[t]
            # first handle two edge case: all observed or all missing
            # case 1: all missing
            if sum(mskt)==0:
                # case 1.1: yt is missing, sample from MH jump
                if not np.any(y_masks[t]):
                    latent=np.where(z[t]==1)[0][0]
                    # proposal distribution
                    prop_xt=np.random.multivariate_normal(self.mu[latent], self.sigma[latent])
                    next_latent=np.where(z[t+1]==1)[0][0]
                    # handle the edge case
                    if t < length-1:
                        next_latent=np.where(z[t+1]==1)[0][0]
                        upper=self.alpha[latent][next_latent] @ prop_xt
                        lower=self.alpha[latent][next_latent] @ x[t]
                    else:
                        upper=0
                        lower=0
                    log_u=np.log(np.random.uniform(0,1,1)[0])
                    if log_u<upper - lower :
                        new_x[t]=prop_xt
                    # otherwise: take y_t into account
                else:
                    # otherwise, yt observed, sample by metropolis jump
                    latent=np.where(z[t]==1)[0][0]
                    #next_latent=np.where(z[t]==1)[0][0]
                    prop_xt=np.random.multivariate_normal(self.mu[latent],self.sigma[latent])
                    y_obs=np.where(y[t]==1)[0][0]
                    
                    y_logpdf=np.dot(beta[latent][y_obs],x[t])-\
                        ss.logsumexp(np.dot(beta[latent],x[t]))
                    new_y_logpdf=np.dot(beta[latent][y_obs],prop_xt)-\
                        ss.logsumexp(np.dot(beta[latent],prop_xt))
                    # handle the edge case
                    if t< length -1:
                        next_latent=np.where(z[t]==1)[0][0]
                        y_logpdf += self.alpha[latent][next_latent] @ x[t]
                        new_y_logpdf += self.alpha[latent][next_latent] @ prop_xt
                    else:
                        pass
                    log_u=np.log(np.random.uniform(0,1,1)[0])
                    if log_u<new_y_logpdf-y_logpdf:
                        new_x[t]=prop_xt
                    
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
                # yt missing, directly assign
                if not np.any(y_masks[t]):
                    prop_xt=new_x[t].copy()
                    prop_xt[mis_index]=cond
                    latent=np.where(z[t]==1)[0][0]
                    # handle the edge case
                    if t < length-1:
                        next_latent=np.where(z[t+1]==1)[0][0]
                        upper=self.alpha[latent][next_latent] @ prop_xt
                        lower=self.alpha[latent][next_latent] @ x[t]
                    else:
                        upper=0
                        lower=0
                    log_u=np.log(np.random.uniform(0,1,1)[0])
                    if log_u<upper - lower :
                        new_x[t]=prop_xt
                else:
                    # handle the observed case
                    prop_xt=new_x[t].copy()
                    prop_xt[mis_index]=cond
                    latent=np.where(z[t]==1)[0][0]
                    y_obs=np.where(y[t]==1)[0][0]
                    
                    y_logpdf=np.dot(beta[latent][y_obs],x[t])-\
                        ss.logsumexp(np.dot(beta[latent],x[t]))
                    new_y_logpdf=np.dot(beta[latent][y_obs],prop_xt)-\
                        ss.logsumexp(np.dot(beta[latent],prop_xt))
                    # handle the edge case
                    if t< length -1:
                        next_latent=np.where(z[t]==1)[0][0]
                        y_logpdf += self.alpha[latent][next_latent] @ x[t]
                        new_y_logpdf += self.alpha[latent][next_latent] @ prop_xt
                    else:
                        pass
                    
                    log_u=np.log(np.random.uniform(0,1,1)[0])
                    if log_u<new_y_logpdf-y_logpdf:
                        new_x[t]=prop_xt       
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
    
    def sample_alpha(self,x,y,z):
        '''
        propose new alpha and accept it via Metropolis jump (with gradient proposal)
        if self.proposal = 'Gradient', use gradient proposal
        else use RWM
        
        TEST CODE:
            x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
            y=optimizer.model.y
            alpha=optimizer.sample_alpha(x,y,z)
        '''
        
        
        self.args.batch_size=x.shape[0]
        self.args.hk=1
        new_alpha=self.MALA_alpha(x,y,z,self.alpha)
        self.alpha=new_alpha
            
        return self.alpha
    
    def alpha_forward(self,train_x,train_y,train_z,b):
        '''
        calculate the log probability related to alpha
        tensor in, tensor out
        b a specific (i,j) elem in alpha, has shape (p,1)
        train_x are x such that the hidden states transform from i to j
        train_x has shape (n,p)
        
        TEST CODE:
            site=6
            x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
            y=optimizer.model.y
            train_x=torch.tensor(x).to(torch.float64)
            
            b=torch.tensor(beta[2])
            f=optimizer.alpha_forward(train_x,train_y,b)
        '''
        term1=self.alpha_log_prior(b)
        term2= torch.sum(train_x @ b)
        return term1 + term2
        
    def alpha_grad(self,x,y,z,alpha):
        '''
        return the log potential and its gradient with respect to alpha
        input should all be numpy array
        array in, array out
        
        TEST CODE:
            x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
            y=optimizer.model.y
            beta=optimizer.beta
            f,g=optimizer.alpha_grad(x,y,z,beta)
        '''
        f=torch.tensor(0.)
        x=x.astype(alpha.dtype)
        y=y.astype(alpha.dtype)
        x=torch.tensor(x)
        alpha=torch.tensor(alpha,requires_grad=True)
        y=torch.tensor(y)
        
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

        for i in range(self.model.hidden_dim):
            for j in range(self.model.hidden_dim):
                # acquire the state that transition from i to j happens
                transition=np.argwhere((position[:,:-1]==i)&(position[:,1:]==j))
                train_x= x[transition[:,0],transition[:,1],:]
                f=f+self.alpha_forward(train_x,y,z,alpha[i][j])
        
        f.backward()
        grad=alpha.grad
        f=f.detach().numpy()
        return f, grad.numpy()
    
    def MALA_alpha(self,x,y,z,alpha):
        '''
        perform MH step with gradient proposal
        return the sampled point
    
        TEST CODE:
            a=optimizer.MALA_alpha(x,y,z,optimizer.alpha)
        '''
        self.args.hk=1
        lr=self.args.learning_rate * 1.0
        
        # update all beta at one time
        noise= np.sqrt(2*lr) * np.random.normal(0,1,size=alpha.shape)
        
        old_alpha=alpha.copy()
        f,alpha_grad=self.alpha_grad(x,y,z,old_alpha)
        
        new_alpha = old_alpha + lr * alpha_grad +noise
        # ensure identifiability
        new_alpha[:,-1,:]=0
        f1,alpha_grad1 = self.alpha_grad(x,y,z,new_alpha)
        
        ratio=f1-f - (1/(4*lr))*np.linalg.norm(old_alpha-new_alpha-lr*alpha_grad1)**2+\
                            (1/(4*lr))*np.linalg.norm(new_alpha-old_alpha-lr*alpha_grad)**2
        u=np.random.uniform(0,1,1)[0]
        log_u=np.log(u)
        ratio=min(0,ratio)
        if log_u<=ratio:
            return new_alpha
        else:
            return old_alpha

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
    
    def check_state(self,x,y,z):
        '''
        check the states of y|z
        mx,my,mz=optimizer.check_state(x,y,z)
        '''
        mz=[]
        my=[]
        mx=[]
        for i in range(self.model.hidden_dim):
            mx.append([])
            my.append([])
            mz.append([])
            for j in range(self.model.hidden_dim):
                maskz=z[:,:,i]
                maskz=(maskz==1)
                masky=y[:,:,j]
                masky=(masky==1)
                mask=maskz*masky
                # find x
                if mask.size!=0:
                    mz[i].append(x[maskz].shape[0])
                    my[i].append(x[masky].shape[0])
                    mx[i].append(x[mask].shape[0])
        return mx,my,mz
                    
    
    # sample beta
    def sample_beta(self,x,y,z):
        '''
        sample beta, the coefficients in regression
        return updated beta
        x,y,z: full data
        SGLD: if True, use Unadjusted SGLD, otherwise, use MALA (Metropolis-Adjusted Langevin instead)
        SGLD_step: current iteration
        SGLD_batch: batch size
        # test code
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        beta=optimizer.sample_beta(x,y,z,True,1,10)
        '''
        SGLD=self.args.use_sgld
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
        # use MALA instead
        else:
            self.args.batch_size=x.shape[0]
            self.args.hk=1
            new_beta=self.MALA_beta(x,y,z,self.beta)
            self.beta=new_beta
            
        return self.beta
    
    def logistic_forward(self,x,y,b):
        '''
        This is an alternative to beta forward function
        return log p(x,y,z|\beta)p(\beta) for specific beta_i
        beta in R(6,40)
        x in R(50,40)
        y in R(50,7) (for instance)
        test code:
            site=6
            mz=z[:,:,site]==1
            train_x=x[mz]
            train_y=y[mz]
            train_x=torch.tensor(train_x).to(torch.float64)
            train_y=torch.tensor(train_y).to(torch.float64)
            b=torch.tensor(beta[6])
            f=optimizer.logistic_forward(train_x,train_y,b)
        '''
        
        #log_prior= -0.5*(self.args.batch_size/self.model.data.shape[0])*(torch.norm(b)**2)
        log_prior=self.beta_log_prior(b)
        #log_prior=0
        
        b=torch.cat((b,torch.zeros((1,b.shape[1]))),dim=0) # turn to beta in R(7,40)
        loading= y @  b # R(50,40), each row represent corresponding beta for x
        log_prod = torch.sum(loading*x,1) # each row is xt.T @ beta corresponding to xt, shape= (50,1)
        
        log_prod2=torch.logsumexp(x @ b.T,dim=1) # dominator of logistic regression, shape=(50,1)
        assert log_prod2.shape[0]==x.shape[0]
        
        return (log_prior + torch.sum(log_prod)-torch.sum(log_prod2))*(1)
        
    
    def beta_grad(self,x,y,z,beta):
        '''
        return log of energy function and its gradient
        x,y,z=dataset, should be numpy array
        beta: the full beta paramter, should be numpy array
        x,z=optimizer.latent_initializer(optimizer.model.x,optimizer.model.y)
        y=optimizer.model.y
        beta=optimizer.beta
        f,g=optimizer.beta_grad(x,y,z,beta)
        '''
        f=torch.tensor(0.)
        x=x.astype(beta.dtype)
        y=y.astype(beta.dtype)
        x=torch.tensor(x)
        beta=torch.tensor(beta,requires_grad=True)
        y=torch.tensor(y)
        
        for i in range(self.model.hidden_dim):
            
            mz=z[:,:,i]
            mz=(mz==1)
            mz=torch.tensor(mz)
            train_x=x[mz]
            train_y=y[mz]
            # remove missing rows with missing y
            train_x=train_x[~torch.any(train_y.isnan(),dim=-1)]
            train_y=train_y[~torch.any(train_y.isnan(),dim=-1)]
            f=f+self.logistic_forward(train_x,train_y,beta[i])
        # f: energy function
        f.backward()
        grad=beta.grad
        f=f.detach().numpy()
        return f, grad.numpy()

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
        hk=self.args.learning_rate * (self.args.hk**(-1))
        #print('learning rate:',hk)
        #print('hk:',hk)
        '''
        noise=np.random.multivariate_normal(np.zeros(self.model.feature_dim),
                                            hk*np.eye(self.model.feature_dim))
        '''
        noise=np.sqrt(hk)*np.random.normal(0,1,size=beta.shape)
        
        new_beta=beta.copy()
        
        f,beta_grad=self.beta_grad(x,y,z,new_beta)
        #print('objective: ',f)
        new_beta=new_beta + (hk/2) * (N/n) * beta_grad #+ noise
        
        return new_beta
    
    def MALA_beta(self,x,y,z,beta):
        N=self.model.data.shape[0]
        self.args.hk=1
        lr=self.args.learning_rate * 1.0
        
        # update all beta at one time
        noise= np.sqrt(2*lr) * np.random.normal(0,1,size=beta.shape)
        
        old_beta=beta.copy()
        f,beta_grad=self.beta_grad(x,y,z,old_beta)
        
        new_beta = old_beta + lr * beta_grad +noise
        f1,beta_grad1 = self.beta_grad(x,y,z,new_beta)
        
        ratio=f1-f - (1/(4*lr))*np.linalg.norm(old_beta-new_beta-lr*beta_grad1)**2+\
                            (1/(4*lr))*np.linalg.norm(new_beta-old_beta-lr*beta_grad)**2
        u=np.random.uniform(0,1,1)[0]
        log_u=np.log(u)
        ratio=min(0,ratio)
        
        
        #print('Log ratio: ',ratio)
        #print(f'Energy, ','f: ', f, ' new_f: ',f1)
        #print(f'grad, ', 'grad_f', np.sum(abs(beta_grad)), 'new_grad', np.sum(abs(beta_grad1)))
        
        
        if log_u<=ratio:
            return new_beta
        else:
            return old_beta
        
        
        '''
        noise=np.sqrt(2*lr)* np.random.normal(0,1,size=beta.shape)
        u=np.random.uniform(0,1,beta.shape[0])
        log_u=np.log(u)
        
        tmp_beta=beta.copy()
        
        # update beta[i] one by one for i in range(beta.shape[0])
        for i in range(beta.shape[0]):
            local_beta=np.expand_dims(tmp_beta[i],0)
            local_noise=np.expand_dims(noise[i],0)
            
            old_beta=local_beta.copy()
            local_f,local_beta_grad=self.MALA_beta_grad(x,y,z,old_beta,i)
            
            new_beta=old_beta + (lr)  * local_beta_grad + local_noise
            local_f1,local_beta_grad1=self.MALA_beta_grad(x,y,z,new_beta,i)
            
            # our implementation
            ratio1=local_f1-local_f
            ratio2=-(1/(4*lr))*np.linalg.norm(old_beta-new_beta-lr*local_beta_grad1)**2+(1/(4*lr))*np.linalg.norm(new_beta-old_beta-lr*local_beta_grad)**2
            ratio=ratio1+ratio2
            
            print('ratio: ',ratio)
            #print('ratio2: ',ratio2)
            print(f'beta{i}, ','f: ', local_f, ' new_f: ',local_f1)
            print(f'grad{i}, ', 'grad_f', np.sum(abs(local_beta_grad)), 'new_grad', np.sum(abs(local_beta_grad1)))
            ratio=min(0,ratio)
            if log_u[i]>ratio:
                old_beta=old_beta.squeeze()
                tmp_beta[i]=old_beta.copy()
            else:
                new_beta=new_beta.squeeze()
                tmp_beta[i]=new_beta.copy()
        return tmp_beta
          '''
        
        
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
        prop_x=np.random.normal(0,1,x.shape)
        new_x=x.copy()
        new_x[~self.model.x_masks]=prop_x[~self.model.x_masks]
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
    
    def check(self,s):
        '''
        check point
        s: iteration number
        '''
        b=np.array(self.param['beta'])
        plt.plot(b[:s,:,0,0])
        plt.title('beta entries')
        plt.show()
        
        sig=np.array(self.param['sigma'])
        plt.plot(sig[:s,0,0])
        plt.title('First line of sigma 0')
        plt.show()
        
        pp=np.array(self.param['pi'])
        plt.title('initial distribution')
        plt.plot(pp[:s])
        plt.show()
        
        pt=np.array(self.param['transition'])
        plt.title('first line of transition')
        plt.plot(pt[:s,0,:])
        plt.show()
        
        pm=np.array(self.param['mu'])
        plt.title('first entires of mu')
        plt.plot(pm[:s,:,0])
        plt.show()
        
        
        
    
    
    def run(self,n,log_step=None,prog_bar=True,prob=0.5,initial_x=None,initial_z=None):
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
        if not (initial_z is None):
            z=initial_z
            x=self.x_initializer(x,y,z)
        
        # register x,y,z into the model
        self.x=x
        self.y=y
        self.z=z
        
        del x
        del y
        del z
        
        
        # determine the number of core to use
        if self.args.num_core==0:
            core=None
            self.core_num=4
        else:
            #core=mp.Pool(self.args.num_core)
            core=self.args.num_core
            self.core_num=self.args.num_core
        
        # store samples
        sample_param={}
        sample_param['beta']=np.array([self.beta])
        sample_param['mu']=np.array([self.mu])
        sample_param['sigma']=np.array([self.sigma])
        sample_param['pi']=np.array([self.pi])
        sample_param['transition']=np.array([self.transition])
        
        self.param=sample_param
        
        # evaluate performance on imputing latent varianbles
        self.x_acc=[]
        self.z_acc=[]
        
        batch=self.args.batch_size
        latent_batch=self.args.latent_batch_size
        SGLD=self.args.use_sgld
        
        # monitor memory
        # tracemalloc.start()
        with Parallel(n_jobs=core,backend='multiprocessing') as parallel:
            for s in tqdm(range(n)):
                if (s+1)%log_step==0:
                    self.check(s)
                    #print('iteration: ',s)
                    
                # decide sample latent var or sample theta
                flip=np.random.choice([0,1],1,replace=True,p=[1-prob,prob])[0]
                
                # sample parameter
                if flip==1:
                    self.beta=self.sample_beta(self.x,self.y,self.z)
                    self.mu=self.sample_mu(self.x,self.y,self.z)
                    self.transition=self.sample_transition(self.x,self.y,self.z)
                    self.pi=self.sample_pi(self.x,self.y,self.z)
                    self.sigma=self.sample_sigma(self.x,self.y,self.z)
                    
                    self.param['beta']=np.vstack([self.param['beta'],[self.beta]])
                    self.param['mu']=np.vstack([self.param['mu'],[self.mu]])
                    self.param['sigma']=np.vstack([self.param['sigma'],[self.sigma]])
                    self.param['pi']=np.vstack([self.param['pi'],[self.pi]])
                    self.param['transition']=np.vstack([self.param['transition'],[self.transition]])
                    #self.param=sample_param
                    #print(self.beta[0][0])
                    
                # sample latent variable
                # each time only update a small batch to accelerate computation
                
                elif flip==0:
                    start=time.time()
                    # total sample size
                    N=self.model.data.shape[0]
                    # number of batches
                    batch_num=N//latent_batch
                    # randomly select a batch to update
                    
                    choose_batch=np.random.choice(np.arange(batch_num),1,True)[0]
                    batch_index=np.arange(choose_batch*latent_batch,min(N,(choose_batch+1)*latent_batch))
                    
                    # sample strategy 1
                    
                    
                    # sample strategy 3
                    '''
                    z=parallel(delayed(self.zt_wrapper)(t) for t in batch_index)
                    x=parallel(delayed(self.xt_wrapper)(t) for t in batch_index)
                    '''
                    z=parallel(delayed(self.sample_zt)(self.x[t],self.y[t],self.z[t],self.model.lengths[t],
                                        self.model.x_masks[t],self.model.y_masks[t]) for t in batch_index)
                    self.z[batch_index]=np.array(z)
                    x=parallel(delayed(self.sample_xt)(self.x[t],self.y[t],self.z[t],self.model.lengths[t],
                                        self.model.x_masks[t],self.model.y_masks[t]) for t in batch_index)
                    
                    self.x[batch_index]=np.array(x)
                    # if we load the real z, evaluate the performance
                    if not (self.real_z is None):
                        acc=np.sum(self.z[~np.isnan(self.z)]==self.real_z[~np.isnan(self.real_z)])
                        acc=acc/len(self.z[~np.isnan(self.z)])
                        self.z_acc.append(acc)
                    # if we load the real x, evaluate the performance
                    if not (self.real_x is None):
                        acc=np.sum((self.x[~np.isnan(self.real_x)]-self.real_x[~np.isnan(self.real_x)])**2)
                        acc=acc/np.sum(self.model.x_masks)
                        self.x_acc.append(acc)
                    
                    
                    '''
                    x=self.sample_x(x,y,z)
                    z=self.sample_z(x,y,z)
                    '''
                    end=time.time()
                    print('multicore time: ',end-start)
                    
                    #print(tracemalloc.get_traced_memory())
        return self.param
    
    def sys_scan(self,n,log_step=None,prog_bar=True,prob=0.5,initial_x=None,initial_z=None):
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
        if not (initial_z is None):
            z=initial_z
            x=self.x_initializer(x,y,z)
        
        # register x,y,z into the model
        self.x=x
        self.y=y
        self.z=z
        
        
        # determine the number of core to use
        if self.args.num_core==0:
            core=1
            self.core_num=1
        else:
            #core=mp.Pool(self.args.num_core)
            core=self.args.num_core
            self.core_num=self.args.num_core
        # store samples
        sample_param={}
        # sample_param['beta']=np.array([self.beta])
        # sample_param['mu']=np.array([self.mu])
        # sample_param['sigma']=np.array([self.sigma])
        # sample_param['pi']=np.array([self.pi])
        # sample_param['transition']=np.array([self.transition])
        sample_param['beta']=np.empty([n]+list(self.beta.shape))
        sample_param['mu']=np.empty([n]+list(self.mu.shape))
        sample_param['sigma']=np.empty([n]+list(self.sigma.shape))
        sample_param['pi']=np.empty([n]+list(self.pi.shape))
        sample_param['alpha']=np.empty([n]+list(self.alpha.shape))
    
        self.param=sample_param
        
        # evaluate performance on imputing latent varianbles
        self.x_acc=np.zeros(n)
        self.z_acc=np.zeros(n)
        
        latent_batch=self.args.latent_batch_size
        SGLD=self.args.use_sgld
        
        # total sample size
        N=self.model.data.shape[0]
        # monitor memory
        #with Parallel(n_jobs=core,backend='loky',max_nbytes='1M') as parallel:
        for new_ass in range(1):
            for s in tqdm(range(n)):
                #print(os.system('df -h'))
                if (s+1)%log_step==0:
                    self.check(s)
                
                self.beta=self.sample_beta(self.x,self.y,self.z)
                self.mu=self.sample_mu(self.x,self.y,self.z)
                self.alpha=self.sample_alpha(self.x,self.y,self.z)
                self.pi=self.sample_pi(self.x,self.y,self.z)
                self.sigma=self.sample_sigma(self.x,self.y,self.z)
                
                self.param['beta'][s]=self.beta
                self.param['mu'][s]=self.mu
                self.param['sigma'][s]=self.sigma
                self.param['pi'][s]=self.pi
                self.param['alpha'][s]=self.alpha
                              
                # sample latent variable
                #start=time.time()
                # sample strategy 1
                
                parallel=Parallel(n_jobs=core,backend='loky',max_nbytes='1M')
                z=parallel(delayed(self.sample_zt)(self.x[t],self.y[t],self.z[t],self.model.lengths[t],
                                    self.model.x_masks[t],self.model.y_masks[t]) for t in range(N))
                self.z=np.array(z)
                x=parallel(delayed(self.sample_xt)(self.x[t],self.y[t],self.z[t],self.model.lengths[t],
                                    self.model.x_masks[t],self.model.y_masks[t]) for t in range(N))
                self.x=np.array(x)
                del x
                del z
                del parallel
                
                
                # if we load the real z, evaluate the performance
                if not (self.real_z is None):
                    #self.z=z
                    acc=np.sum(self.z[~np.isnan(self.z)]==self.real_z[~np.isnan(self.real_z)])
                    acc=acc/len(self.z[~np.isnan(self.z)])
                    self.z_acc[s]=acc
                # if we load the real x, evaluate the performance
                if not (self.real_x is None):
                    #self.x=x
                    acc=np.sum((self.x[~np.isnan(self.real_x)]-self.real_x[~np.isnan(self.real_x)])**2)
                    acc=acc/np.sum(self.model.x_masks)
                    self.x_acc[s]=acc
                gc.collect()
                #end=time.time()
                #print('multicore time: ',end-start)
                
                #print(tracemalloc.get_traced_memory())
        return self.param
    
    def summary(self,param_path):
        '''
        summarize results by comparing the results to true parameters
        param_path: where true parameters lies
        '''
        true_initial=np.load(f'{param_path}/initial.npy')
        true_transition=np.load(f'{param_path}/transition.npy')
        true_beta=np.load(f'{param_path}/beta.npy')
        true_mu=np.load(f'{param_path}/mu.npy')
        true_sigma=np.load(f'{param_path}/sigma.npy')
        
        # register true parameters to the optimizer
        self.true_param={}
        self.true_param['pi']=true_initial
        self.true_param['beta']=true_beta
        self.true_param['mu']=true_mu
        self.true_param['sigma']=true_sigma
        self.true_param['transition']=true_transition
        
        # transform estimation to numpy
        keys=list(self.param.keys())
        for i in range(len(keys)):
            self.param[keys[i]]=np.array(self.param[keys[i]])
            
        # permute to the right position
        s=10
        ep=sum(self.param['pi'][-s:])/len(self.param['pi'][-s:])
        
        permute=utils.find_permute(ep,true_initial)
        permute=utils.find_permute(ep,true_initial)
        p,t,m,s,b=utils.permute_train(permute,self.param['pi'],self.param['transition'],
                                     self.param['mu'],self.param['sigma'],self.param['beta'])
        # obtain transformed parameters
        self.param['pi']=p
        self.param['transition']=t
        self.param['mu']=m
        self.param['sigma']=s
        self.param['beta']=b
        
        # start analyzing
        self.mae={}
        for i in range(len(keys)):
            error=self.param[keys[i]]-self.true_param[keys[i]]
            error=np.abs(error)
            # summation with respect to dimensions except for the first one
            mean_error=np.sum(error,axis=tuple(np.arange(1,len(error.shape))))
            # calculate average entrywise dimension
            mean_error=mean_error/np.cumprod(self.true_param[keys[i]].shape)[-1]
            #mean_error=np.abs(mean_error)
            self.mae[keys[i]]=mean_error
            assert len(mean_error.shape)==1
        
        # calculate posterior variance and mean
        post_length=int(0.3*len(self.param['pi']))
        self.mean={}
        self.variance={}
        for i in range(len(keys)):
            self.mean[keys[i]]=np.mean(self.param[keys[i]][-post_length:],axis=0)
            self.variance[keys[i]]=np.var(self.param[keys[i]][-post_length:],axis=0)
        
        return self.mae,self.mean,self.variance
        
      
    def pickle(self,path,param_path=None):
        '''
        save parameters and summary to path
        also save a config to keep important configs (lr, data size, etc)
        if param_path=None, will save parameter estimation and mean/var
        if param_path!=None, will also provide trace plots of error
        '''
        keys=list(self.param.keys())
        for i in range(len(keys)):
            np.save(f'{path}/{keys[i]}.npy',self.param[keys[i]])
        
        # save important configs
        config={}
        config['batch_size']=self.args.batch_size
        config['latent_batch_size']=self.args.latent_batch_size
        config['learning_rate']=self.args.learning_rate
        config['data_size']=self.model.x.shape[0]
        config['core']=self.args.num_core
        config['use_SGLD']=self.args.use_sgld
        
        config=pd.Series(config)
        config.to_csv(f'{path}/config.txt')
        
        if param_path is None:
            # only save mean & variance
            # calculate posterior variance and mean
            post_length=int(0.3*len(self.param['pi']))
            self.mean={}
            self.variance={}
            for i in range(len(keys)):
                self.mean[keys[i]]=np.mean(self.param[keys[i]][-post_length:],axis=0)
                self.variance[keys[i]]=np.mean(self.param[keys[i]][-post_length:],axis=0)
            m=pd.Series(self.mean)
            v=pd.Series(self.variance)
            m.to_csv(f'{path}/posterior_mean.csv')
            v.to_csv(f'{path}/posterior_var.csv')
        else:
            self.summary(param_path)
            m=pd.Series(self.mean)
            v=pd.Series(self.variance)
            mae=pd.Series(self.mae)
            m.to_csv(f'{path}/posterior_mean.csv')
            v.to_csv(f'{path}/posterior_var.csv')
            mae.to_csv(f'{path}/error.csv')
            
            # save plots
            for i in range(len(keys)):
                plt.plot(self.mae[keys[i]])
                plt.title(f'Estimation error of {keys[i]}')
                plt.savefig(f'{path}/{keys[i]}_error.png')
                plt.close()
        
        # save x accuracy
        if not (self.x_acc is None):
            plt.plot(self.x_acc)
            plt.title('mean error estimating x')
            plt.savefig(f'{path}/x_error.png')
            plt.close()
        
        if not (self.z_acc is None):
            plt.plot(self.z_acc)
            plt.title('misclassification rate for z')
            plt.savefig(f'{path}/z_rate.png')
            plt.close()
        
        
        
    
    def unpickle(self,path):
        '''
        read parameters from data
        return estimated parameters and configs
        '''
        param={}
        keys=list(self.param.keys())
        for i in range(len(keys)):
            param[keys[i]]=np.load(f'{path}/{keys[i]}.npy')
        
        config=pd.read_csv(f'{path}/config.txt')
        return param,config
        
        
        
            
                
                
        