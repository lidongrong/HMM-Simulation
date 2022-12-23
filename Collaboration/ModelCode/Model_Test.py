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

path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\SynData\Rate0.5\PartialData'
hidden_data_path='D:\Files\CUHK_Material\Research_MakeThisObservable\EMR\Data\SynData\Rate0.5\HiddenStates'
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
types=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))
sample_size=500
data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)  
z=sdg.hidden_data_reader(hidden_data_path,sample_size)

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
    args.batch_size=1000
    args.hk=1
    args.latent_batch_size=50
    args.use_sgld=False
    args.learning_rate=0.0005
    args.num_core=0
    
    optimizer=Model.Random_Gibbs(model=m,args=args,initial=None)
    
    
    
    param=optimizer.run(n=2500,log_step=4,prog_bar=True,prob=0.6,initial_x=None,initial_z=None)
    
    pb=np.array(param['beta'])
    pp=np.array(param['pi'])
    plt.plot(pb[:,0,0,:])
    plt.plot(pp[:])
    