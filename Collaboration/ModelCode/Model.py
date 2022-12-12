# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:42:13 2022

@author: lidon
"""

import numpy as np
import pandas as pd
import SD_generator as sdg
import scipy.stats as stats

class hmm_model:
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
        
    

class optimizer:
    def __init__(self):
        pass

class gibbs(optimizer):
    def __init__(self,model,initial=None):
        '''
        model: a hmm model
        initial: a dictionary recording a set of initial values
        '''
        self.model=model
        # if initial specified
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
            self.mu=np.random.multivariate_normal([0]*model.feature_dim, np.eye(model.feature_dim))
            self.sigma=stats.wishart.rvs(model.feature_dim,np.eye(model.feature_dim))
            self.beta=np.array([np.random.multivariate_normal([0]*model.feature_dim, np.eye(model.feature_dim))
                                for i in range(model.hidden_dim)])
        