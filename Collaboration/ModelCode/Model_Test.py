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
covariates=np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
       'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
       'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC', 'ACEI',
       'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
       'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
       'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
       'Tenofovir Disoproxil Fumarate', 'Thiazide']) 
types=np.array(['numeric','dummy'])
covariate_types=np.array(['numeric']*len(covariates))
sample_size=10000
data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)  

if __name__ == '__main__':
    
    m=Model.HMM_Model(data,lengths,covariates,covariate_types)
    x,y=m.split()
    
    parser=argparse.ArgumentParser(description="training paramters")
    parser.add_argument("-batch","--batch-size",default=40,type=int)
    parser.add_argument("-lr","--learning-rate",default=0.01,type=float)
    parser.add_argument("-hk","--hk",default=1,type=float)
    args=parser.parse_args(args=[])
    
    optimizer=Model.Random_Gibbs(model=m,args=args)
    
    param=optimizer.run(n=4000,log_step=4,prog_bar=True,prob=0.7,SGLD=True)
    