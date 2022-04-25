# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:10:08 2022

@author: lidon
"""

# Make sure that this script is under HMMSimulation directory
import pandas as pd
import numpy as np
import HMM
import SeqSampling
import EMHMM
import ZMARGibbs
import multiprocessing as mp
import os

# read the data
data=pd.read_csv('SCHIZREP.txt',header=None,sep='  ')
data.columns=['No','Severity','Week','Drug1','Drug2']

# filter the drug group & control group
placebo=data.loc[data.Drug1==0,:]
drug=data.loc[data.Drug1!=0,:]

# construct sequence data
placebo_no=placebo['No'].unique()
drug_no=drug['No'].unique()

placebo_data=[]
drug_data=[]

#first construct placebo sequences
for no in placebo_no:
    patient_data=placebo.loc[placebo.No==no]
    
    week=patient_data['Week']
    week=week.array
    seve=patient_data['Severity']
    seve=seve.array
    
    seq=np.arange(0,7)
    seq=seq.astype(np.float32)
    seq[:]=None
    seq[week]=seve
    placebo_data.append(seq)
    

# Then construct drug sequences
for no in drug_no:
    patient_data=drug.loc[drug.No==no]
    
    week=patient_data['Week']
    week=week.array
    seve=patient_data['Severity']
    seve=seve.array
    
    seq=np.arange(0,7)
    seq=seq.astype(np.float32)
    seq[:]=None
    seq[week]=seve
    drug_data.append(seq)
    
drug_data=np.array(drug_data)
placebo_data=np.array(placebo_data)   

# change ill type 
drug_data=drug_data.astype(str)
placebo_data=placebo_data.astype(str)


def change_type(data):
    data=data.copy()
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if data[i,j]=='nan':
                data[i,j] ='None'
                
            elif float(data[i,j])>=1 and float(data[i,j])<=2:
                data[i,j]='normal'
                
            elif float(data[i,j])>=2 and float(data[i,j])<4.5:
                data[i,j]='mild'
            elif float(data[i,j])>=4.5 and float(data[i,j])<5.5:
                data[i,j]='moderate'
            elif float(data[i,j])>=5.5:
                data[i,j]='severe'
    return data

drug_data=change_type(drug_data)
placebo_data=change_type(placebo_data)

hidden_state=np.array(['normal','mild','moderate','severe'])
obs_state=np.array(['normal','mild','moderate','severe'])


A=np.random.dirichlet((8,8,8,8),4)
B=np.random.dirichlet((8,8,8,8),4)
pi=np.random.dirichlet((8,8,8,8),4)[0]

if __name__=='__main__':
    p=mp.Pool(16)
    
    os.mkdir('RealData')
    os.mkdir('RealData/SimulationResult')
    os.mkdir('RealData/NaiveResult')
    
    for i in range(0,20):
        A=np.random.dirichlet((8,8,8,8),4)
        B=np.random.dirichlet((8,8,8,8),4)
        pi=np.random.dirichlet((8,8,8,8),4)[0]
        
        drug_data1=[]
        for j in range(0,len(drug_data)-1):
            drug_data1.append(drug_data[j][drug_data[j]!='None'])
        drug_data1=np.array(drug_data1)
        
        #HMM
        a,b,pt,func=EMHMM.EMTrain(A,B,pi,drug_data,0.0001,hidden_state,obs_state,p)
        # Naive HMM
        a1,b1,p1,func1=EMHMM.EMTrain(A,B,pi,drug_data1,0.0001,hidden_state,obs_state,p)
        
        os.chdir('RealData/SimulationResult')
        os.mkdir(f'Experiment{i}')
        # Save the results
        np.save(f'Experiment{i}/at.npy',a)
        np.save(f'Experiment{i}/bt.npy',b)
        np.save(f'Experiment{i}/pit.npy',pt)
        np.save(f'Experiment{i}/data.npy',drug_data)
        #np.save(f'Experiment{i}/TrueHidden.npy',hidden_data)
        np.save(f'Experiment{i}/ObjFunc.npy',func)
        os.chdir('..')
        os.chdir('..')
        
        #Save the Naive Model
        os.chdir('RealData/NaiveResult')
        os.mkdir(f'Experiment{i}')
        np.save(f'Experiment{i}/at.npy',a1)
        np.save(f'Experiment{i}/bt.npy',b1)
        np.save(f'Experiment{i}/pit.npy',p1)
        np.save(f'Experiment{i}/data.npy',drug_data1)
        #np.save(f'Experiment{i}/TrueHidden.npy',hidden_data)
        np.save(f'Experiment{i}/ObjFunc.npy',func1)
        os.chdir('..')
        os.chdir('..')
        
        



