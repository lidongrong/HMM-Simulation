# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:42:18 2022

@author: lidon
"""
import matplotlib.pyplot as plt
import SD_generator as sdg
import Model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path='D:\Object\PROJECTS\HMM\SynData\Rate0.5\PartialData'
param_path='D:\Object\PROJECTS\HMM\SynData'
hidden_data_path='D:\Object\PROJECTS\HMM\SynData\Rate0.5\HiddenStates'
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide'])

d=3
covariates=covariates[0:d]

types=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))
sample_size=1000
data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)  
z=sdg.hidden_data_reader(hidden_data_path,sample_size)
beta=np.load('D:\Object\PROJECTS\HMM\SynData/beta.npy')

if __name__ == '__main__':
    
    m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    x,y=m.split()
    
    parser=argparse.ArgumentParser(description="tunning paramters")
    parser.add_argument("-batch","--batch-size",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
    parser.add_argument("-hk","--hk",default=1,type=float)
    parser.add_argument("-lbatch","--latent-batch-size",default=1000,type=int)
    parser.add_argument("-SGLD","--use-sgld",default=True,type=bool)
    parser.add_argument("-core","--num-core",default=0,type=int)
    args=parser.parse_args(args=[])
    
    # adjust argparser
    args.batch_size=sample_size
    args.hk=1
    args.latent_batch_size=50
    args.use_sgld=False
    # learning rate suggested by Wainwright, have a try
    args.learning_rate = (1/126)*(1/(0.15*sample_size))
    args.num_core=8
    
    
    optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
    true_param=optimizer.load_true_param(param_path)
    optimizer.set_as_true_param(true_param)
    #mx,my,mz=optimizer.check_state(x,y,z)
    #optimizer.beta=beta
    param=optimizer.run(n=30000,log_step=250,prog_bar=True,prob=1,initial_x=None,initial_z=z)
    
    