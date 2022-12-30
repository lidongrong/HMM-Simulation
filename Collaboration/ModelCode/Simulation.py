# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:03:22 2022

@author: lidon
"""

import SD_generator as sdg
import Model as Model
import numpy as np
import pandas as pd
import multiprocessing as mp
import time

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
sample_size=5000

#data,lengths=sdg.data_loader(path,sample_size,covariates,covariate_types)

rate=[0.5,0.6,0.8,0.9,0.95,0.97]

def main(path,covariates,covariate_types,sample_size,nums,n):
    '''
    path: the path where dataset stores
    covariates: list/array of names of covariates
    covariate_types: list/array indicating if each covariate is numeric/dummy
    sample_size: sample size
    nums: number of simulations to run
    n: number of iterations in each simulation
    '''
    
  