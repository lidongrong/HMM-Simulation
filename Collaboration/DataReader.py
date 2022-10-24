# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:52:56 2022

@author: lidon
"""

# read all data
# script should be put outside of the ETVTDF TimeSeries data file
# data list:
# HBV, TimeSeries Combined
# cirrhosis_date, Demographics, DM_date
# Dyslipidaemia_date, ETVTDF first date
# HCC_date, hypertension_date

import pandas as pd
import numpy as np
import os

path='ETVTDF timeseries'

# read data into the environment
def read_data(path):
    cirrhosis=pd.read_csv(f'{path}/cirrhosis_date.txt',sep='\t')
    dm=pd.read_csv(f'{path}/DM_date.txt',sep='\t')
    dys=pd.read_csv(f'{path}/Dyslipidaemia_date.txt',sep='\t')
    etv=pd.read_csv(f'{path}/ETVTDF firstDate.txt',sep=',')
    hcc=pd.read_csv(f'{path}/HCC_date.txt',sep='\t')
    hypertension=pd.read_csv(f'{path}/HCC_date.txt',sep='\t')
    demographic=pd.read_csv(f'{path}/Demographics.txt',sep=',',encoding='ISO-8859-1')
    
    dic={'cirrhosis':cirrhosis,'dm':dm,'dys':dys,
         'etv':etv,'hcc':hcc,'hypertension':hypertension,'demographic':demographic}
    
    return dic
#dic=read_data(path)

# get all patient id
def get_id(path,dic):
    
    # get ids from Time series
    os.chdir(f'{path}/TimeSeries Combined')
    patient_id=os.listdir()
    for i in range(0,len(patient_id)):
        patient_id[i]=int(patient_id[i][:-4])
    os.chdir('..')
    #pd.to_csv(patient_id,'patient_id.txt')
    return patient_id
#patient_id=get_id(path)


class patient:
    # pid represents patient id
    def __init__(self,pid,dic):
        self.pid=pid
        # read the time series
        self.ts=pd.read_csv(f'{path}/TimeSeries Combined/{self.pid}.txt',sep='\t')
        try:
            self.hbv=pd.read_csv(f'{path}/HBV/{self.pid}.txt',sep='\t')
        except FileNotFoundError:
            hbv_frame={'class':[np.nan],'Item':[np.nan],'Date':[np.nan],'Result':[np.nan]}
            hbv_frame=pd.DataFrame(hbv_frame)
            self.hbv=hbv_frame
        
        self.cirrhosis=dic['cirrhosis'][dic['cirrhosis']['ReferenceKey']==self.pid]
        # fill with nan if empty
        if self.cirrhosis.empty==True:
            self.cirrhosis.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.cirrhosis.columns))]
        
        self.dm=dic['dm'][dic['dm']['ReferenceKey']==self.pid]
        if self.dm.empty==True:
            self.dm.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.dm.columns))]

                
        self.dys=dic['dys'][dic['dys']['ReferenceKey']==self.pid]
        if self.dys.empty==True:
            self.dys.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.dys.columns))]

                
        self.etv=dic['etv'][dic['etv']['ReferenceKey']==self.pid]
        # change date type-> from dd/mm/yy to yy-mm-dd
        if self.etv.empty!=True:
            date=self.etv['ETVTDF_firstdate'].values[0]
            date=date.split('/')
            date=date[2]+'-'+date[1]+'-'+date[0]
            self.etv['ETVTDF_firstdate'].values[0]=date
            
        if self.etv.empty==True:
            self.etv.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.etv.columns))]

                
        self.hcc=dic['hcc'][dic['hcc']['ReferenceKey']==self.pid]
        if self.hcc.empty==True:
            self.hcc.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.hcc.columns))]

                
        self.hypertension=dic['hypertension'][dic['hypertension']['ReferenceKey']==self.pid]
        if self.hypertension.empty==True:
            self.hypertension.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.hypertension.columns))]
        
        self.demographic=dic['demographic'][dic['demographic']['ReferenceKey']==self.pid]
        if self.demographic.empty==True:
            self.hypertension.loc[0]=[np.int32(self.pid)]+[np.nan for k in range(1,len(self.hypertension.columns))]


# cocunt the number of variates
def variate(pid,dic):
    variates={'HBV':[],'Lab':[],'Med':[]}
    for i in pid:
        print('patient id: ', i)
        p=patient(i,dic)
        items=p.ts['Item'].values
        
        # append medical items
        med_item=p.ts['Item'][p.ts['class']=='Med'].values
        variates['Med']=np.union1d(variates['Med'],med_item)
        
        lab_item=p.ts['Item'][p.ts['class']=='Lab'].values
        variates['Lab']=np.union1d(variates['Lab'],med_item)
        if not pd.isnull(p.hbv['Result'].values[0]):
            items=p.hbv['Result'].values
            variates['HBV']=np.union1d(variates['HBV'],items)
    return variates


# test code
dic=read_data(path) 
pid=get_id(path,dic)
variates=variate(pid,dic)
#covariates_number=57+len(p.cirrhosis.columns)+len(p.dm.columns)+len(p.dys.columns)+len(p.etv.columns)+len(p.hcc.columns)+len(p.hypertension.columns)+len(p.demographic.columns)-7