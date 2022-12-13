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

class HMM_Model:
    def __init__(self,data,lengths,features,feature_types):
        '''
        data: data
        lengths: length of each sequence
        feature: string list of features
        feature_types: indicate if each feature is numeric or dummy (01)
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
    def __init__(self,model,initial=None):
        '''
        model: a hmm model
        initial: a dictionary recording a set of initial values
        '''
        # the hmm model
        self.model=model
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
            self.A=np.random.dirichlet([1]*model.hidden_dim,model.hidden_dim)
            self.mu=np.random.multivariate_normal([0]*model.feature_dim, np.eye(model.feature_dim),model.hidden_dim)
            # invwishart with df model.feature_dim and V=I
            self.sigma=stats.invwishart.rvs(model.feature_dim,np.eye(model.feature_dim),model.hidden_dim)
            tmp_beta=[np.random.multivariate_normal(np.zeros(self.model.feature_dim),
                                                    5*np.eye(self.model.feature_dim),
                                                    self.model.hidden_dim) 
                  for i in range(self.model.hidden_dim)]
            tmp_beta=np.array(tmp_beta)
            self.beta=tmp_beta
    
    
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
                log_likelihood+=np.log(self.A[prev,curr])
            
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
            
                
            
            
        
    
    def sample_z(self,x,y,z):
        '''
        sample latent z given x,y and other paramters
        '''
        pass
    
    def sample_x(self,x,y,z):
        '''
        sample missing x for imputation
        '''
        pass
    
    def sample_pi(self,x,y,z):
        '''
        sample initial distribution pi given other parameters
        return generated pi
        # test code
        optimizer.sample_pi(x,y,y)
        '''
        # count start conditions
        start=y[:,0,:]
        start=sum(start)
        new_pi=np.random.dirichlet(1+start)
        
        self.pi=new_pi
        
        return new_pi
    
    def sample_transition(self,x,y,z):
        '''
        sample transition matrix
        return sampled transition
        # test code
        optimizer.sample_transition(x,y,y)
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
        optimizer.sample_mu(x,y,y)
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
        sig=optimizer.sample_sigma(x,y,y)
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
    
    def sample_beta(self):
        '''
        sample beta, the coefficients in regression
        return updated beta
        '''
        pass
    
    
    
    def run(self,n,log_step=None,prog_bar=True):
        '''
        collect samples from n iterations
        n: total number of iterations
        log_step: step for printing results for monitoring. if None, not report
        prog_bar: if display progress bar
        '''
        for i in tqdm(range(n)):
            if i%log_step==0:
                print(i)
                time.sleep(0.1)
                
        