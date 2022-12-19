# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:27:20 2022

@author: lidon
"""
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import os

# patient record
class Record:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z


class Synthesize:
    """
    generate synthetic data
    """
    def __init__(self,covariates=None,pi=None,transition=None,beta=None,mu=None,sigma=None,num=None,lbd=100):
        '''
        covariates: demographic covariates of patients
        pi: initial distribution
        transition: transition matrix
        beta: regression coefficients
        mu: mean of conditional distribution of y|z
        sigma: covariance matrix of conditional distribution of y|z. i.e. y|z ~ N(mu_z,sigma_z)
        num: patient number
        lbd: poisson distribution parameter, used to control average length of each seq
        '''
        self.covariates=covariates
        self.pi=pi
        self.transition=transition
        self.beta=beta
        self.mu=mu
        self.sigma=sigma
        self.num=num
        self.lbd=lbd
        # record and censored record
        self.emr=None
        self.partial_emr=None
    
    def generate_sequences(self):
        '''
        generate fully observed sequences
        '''
        full_patient=[]
        latent_size=len(self.pi)
        # generate sequences one by one
        
        # hidden state
        for i in range(self.num):
            length=np.random.poisson(self.lbd,1)[0]
            assert length>0
            # hidden state
            z=[]
            x=[]
            y=[]
            for j in range(length):
                # first generate hidden states
                if j==0:
                    state=np.random.choice(latent_size,1,True,p=self.pi)[0]
                    new_z=[0]*latent_size
                    new_z[state]=1
                    z.append(new_z)
                else:
                    state=np.random.choice(latent_size,1,True,p=self.transition[state])[0]
                    new_z=[0]*latent_size
                    new_z[state]=1
                    z.append(new_z)
                    
                # Then generate x from z:
                # acquire corresponding mu and sigma
                tmp_mu=self.mu[state]
                tmp_sigma=self.sigma[state]
                new_x=np.random.multivariate_normal(tmp_mu,tmp_sigma,1)[0]
                x.append(new_x)
                
                # finally generate y from x and z
                # beta_z
                tmp_beta=self.beta[state]
                # logistic regression probability
                prob=np.exp(-np.dot(self.beta,new_x))
                prob=prob/sum(prob)
                y_state=np.random.choice(latent_size,1,True,p=prob)[0]
                new_y=[0]*latent_size
                new_y[y_state]=1
                y.append(new_y)
            x=np.array(x)
            #x=x.swapaxes(1,0)
            #x=x.T
            y=np.array(y)
            z=np.array(z)
            y=y.astype(np.float32)
            z=z.astype(np.float32)
            new_record=Record(x,y,z)
            full_patient.append(new_record)
        self.emr=full_patient
        return full_patient
    
    # p is the missing rate
    def generate_partial_sequences(self,p):
        '''
        generate partially observed sequences
        with missing rate p
        observations will be set as np.nan with prob p
        '''
        assert self.emr
        full_patient=self.emr
        partial_obs=[]
        for i in range(self.num):
            pat=full_patient[i]
            padding=np.random.choice([np.nan,1],pat.x.shape,True,[p,1-p])
            partial_x=pat.x*padding
            partial_y=pat.y.copy()
            for j in range(partial_y.shape[0]):
                if np.random.choice([1,0],1,True,[p,1-p])[0]:
                    partial_y[j]=[np.nan]*len(self.pi)
            partial_obs.append(Record(partial_x,partial_y,pat.z))
        self.partial_emr=partial_obs
        return partial_obs
    
    def save_data(self,path=''):
        '''
        save data to path specified
        3 folders will be constructed: the first folder stores true states z
        The second folder stores full data 
        The third folder stores partial data
        '''
        os.mkdir(f'{path}/HiddenStates')
        os.mkdir(f'{path}/FullData')
        os.mkdir(f'{path}/PartialData')
        hidden_path=f'{path}/HiddenStates'
        full_path=f'{path}/FullData'
        partial_path=f'{path}/PartialData'
        # make sure exists
        assert self.partial_emr and self.emr
        for i in range(self.num):
            # first store hidden data
            hidden_data=self.emr[i].z
            hidden_data=pd.DataFrame(hidden_data)
            hidden_data.to_csv(f'{hidden_path}/{i}_path.csv')
            
            # then store full data
            x=self.emr[i].x
            y=self.emr[i].y
            d={self.covariates[i]:x[:,i] for i in range(len(self.covariates))}
            for j in range(len(self.pi)):
                d[j]=y[:,j]
            
            full_d=pd.DataFrame(d)
            full_d.to_csv(f'{full_path}/{i}.csv')
            
            # finally store partial data
            x=self.partial_emr[i].x
            y=self.partial_emr[i].y
            d={self.covariates[i]:x[:,i] for i in range(len(self.covariates))}
            for j in range(len(self.pi)):
                d[j]=y[:,j]
            
            partial_d=pd.DataFrame(d)
            partial_d.to_csv(f'{partial_path}/{i}.csv')
            
            
            
        
        
        
    
    