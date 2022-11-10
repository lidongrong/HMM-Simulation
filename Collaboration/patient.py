# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:55:28 2022

@author: lidon
"""

import pandas as pd
import numpy as np
import os

# get all patents' id
# get all patents' data

# path can be user-defined
path='ETVTDF timeseries'

# all variables
variates={'HBV': np.array(['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
        'HBeAg-ALT>1ULN'], dtype=object),
 'Lab': np.array(['AFP', 'ALB', 'ALT', 'AST', 'Anti-HBe', 'Anti-HBs', 'Cr', 'FBS',
        'GGT', 'HBVDNA', 'HBeAg', 'HBsAg', 'HCVRNA', 'HDL', 'Hb', 'HbA1c',
        'INR', 'LDL', 'PLT', 'PT', 'TBili', 'TC', 'TG', 'WCC'],
       dtype=object),
 'Med': np.array(['ACEI', 'ARB', 'Anticoagulant', 'Antiplatelet', 'BetaBlocker',
        'CalciumChannelBlocker', 'Cytotoxic', 'Entecavir', 'IS', 'Insulin',
        'Interferon', 'LipidLoweringAgent', 'OHA', 'Tenofovir Alafenamide',
        'Tenofovir Disoproxil Fumarate', 'Thiazide'], dtype=object)}


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


# get the demographic frame
def get_demo(path):
    demo=pd.read_csv(f'{path}/Demographics.txt',sep=',',encoding='ISO-8859-1')
    return demo


class patient:
    def __init__(self,patient_id,path=path,variates=variates,read_demo=True):
        # get patient id
        # get patient time series
        # decide if read the demographic (False as default to avoid frequent IO)
        self.path=path
        self.pid=patient_id
        # read longitudinal data into a pandas dataframe
        self.ts=pd.read_csv(f'{path}/TimeSeries Combined/{self.pid}.txt',sep='\t')
        
        # read demographics
        if read_demo:
            demo=pd.read_csv(f'{path}/Demographics.txt',sep=',',encoding='ISO-8859-1')
            self.demo=demo[demo['ReferenceKey']==self.pid]
        else:
            self.demo=None
        self.variates=variates
        self.data=None
        
        # read hbv related dataset
        try:
            self.hbv=pd.read_csv(f'{path}/HBV/{self.pid}.txt',sep='\t')
        except FileNotFoundError:
            self.hbv=None
        
        # read cirrhosis
        cirr=pd.read_csv(f'{path}/cirrhosis_date.txt',sep='\t')
        self.cirrhosis=cirr[cirr['ReferenceKey']==self.pid]
        
        # read HCC
        hcc=pd.read_csv(f'{path}/HCC_date.txt',sep='\t')
        self.hcc=hcc[hcc['ReferenceKey']==self.pid]
        
        
    
    # align the dates in self.ts
    # let the beginning date as 0
    # align by month by default (therefore neglect days effect)
    # alined date will be stored in a new columns in self.ts
    # named 'AlignedDate(time)'
    def align_date(self,time='month'):
        date=pd.to_datetime(self.ts['Date'])
        
        if time=='month':
            aligned_date=[]
            aligned_date.append(0)
            for i in range(1,len(date)):
                # number of months measured from the start
                diff=12*(date[i].year-date[0].year)+(date[i].month-date[0].month)
                aligned_date.append(diff)
            self.ts['AlignedDate(month)']=aligned_date
        if time=='year':
            aligned_date=[]
            aligned_date.append(0)
            for i in range(1,len(date)):
                # number of months measured from the start
                diff=date[i].year-date[0].year
                aligned_date.append(diff)
            self.ts['AlignedDate(year)']=aligned_date
        if time=='day':
            aligned_date=[]
            aligned_date.append(0)
            for i in range(1,len(date)):
                # number of months measured from the start
                diff=(date[i]-date[0]).days
                aligned_date.append(diff)
            self.ts['AlignedDate(day)']=aligned_date
    
    # construct observational matrix for patient p
    def construct_matrix(self):
        # obtain total number of dates
        all_dates=self.ts[self.ts.columns[len(self.ts.columns)-1]]
        final_day=max(all_dates)
        # obtain all covariates in time series
        covariates=np.concatenate((self.variates['Lab'],self.variates['Med']))
        
        data=[]
        # record values of each covariates at time t
        for t in range(0,final_day):
            #print(t)
            
            # FIRST STEP: ADD time specific features
            # Missing
            if t not in all_dates.array:
                obs=[np.nan for k in range(0,len(covariates)+3)]
                
            # date recorded, record the observations
            else:
                # acquire data in this month
                ts_now=self.ts[all_dates==t]
                # observation at this time point
                obs=[]
                # select the feature
                for feature in covariates:
                    # feature at t timepoint 
                    ts_feature=ts_now[ts_now['Item']==feature]
                    # if this feature missing at t
                    if ts_feature.empty:
                        obs.append(np.nan)
                    # not empty, randomly select the first observed, otherwise also missing
                    else:
                        results=ts_feature['Result']
                        # if all nan, still missing
                        if pd.isnull(results).all():
                            obs.append(np.nan)
                        # otherwise select one that is not nan
                        else:
                            temp=np.random.choice(results[~np.isnan(results)])
                            obs.append(temp)
            
                # SECOND STEP: ADD demographic data (sex and age)
                # Sex: male 1 female 0
                if self.demo['Sex'].array[0]=='M':
                    obs.append(1)
                else:
                    obs.append(0)
                # add age
                birth=pd.to_datetime(self.demo['DateofBirthyyyymmdd'].array[0])
                ts_now=self.ts[all_dates==t]
                now=pd.to_datetime(ts_now['Date'].array[0])
                obs.append(now.year-birth.year)
                
                #covariates=np.concatenate((covariates,['Sex','Age']))
                
                # Third STEP: Add stages
                stages=['HBeAg+ALT<=1ULN', 'HBeAg+ALT>1ULN', 'HBeAg-ALT<=1ULN',
                       'HBeAg-ALT>1ULN','Cirr','HCC','Death']
                
                # read HBV status
                dead=self.demo['DateofRegisteredDeath'].array[0]
                #print(dead)
                # not dead
                status=np.nan
                if pd.isnull(dead):
                    # first check HBV status
                    if (self.hbv is not None) and (not self.hbv.empty):
                        for day in range(0,len(self.hbv['Date'])):
                            date=self.hbv['Date'][day]
                            date=pd.to_datetime(date)
                            if date.year==now.year and date.month==now.month:
                                status=self.hbv['Result'][day]
                                
                            
                    #and check cirrhosis status
                    elif (self.cirrhosis is not None) and (not self.cirrhosis.empty):
                        date=self.cirrhosis['Cirrhosis_comp_date'].array[0]
                        date=pd.to_datetime(date)
                        if date.year==now.year and date.month==now.month:
                            status='Cirr'
                        
                    # finally check HCC status
                    elif (self.hcc is not None) and (not self.hcc.empty):
                        temp_hcc=self.hcc[self.hcc.columns[1:4]].to_numpy()
                        temp_hcc=temp_hcc.ravel()
                        for i in range(0,len(temp_hcc)):
                            date=pd.to_datetime(temp_hcc[i])
                            if date.year==now.year and date.month==now.month:
                                status='HCC'
                                
                    
                # death happens
                else:
                    dead=pd.to_datetime(dead)
                    if dead<=now:
                        status='Death'
                    else:
                        # first check HBV status
                        if (self.hbv is not None) and (not self.hbv.empty):
                            for day in range(0,len(self.hbv['Date'])):
                                date=self.hbv['Date'][day]
                                date=pd.to_datetime(date)
                                if date.year==now.year and date.month==now.month:
                                    status=self.hbv['Result'][day]
                                    
                                
                                #obs.append(np.nan)
                        #and check cirrhosis status
                        elif (self.cirrhosis is not None) and (not self.cirrhosis.empty):
                            date=self.cirrhosis['Cirrhosis_comp_date'].array[0]
                            date=pd.to_datetime(date)
                            if date.year==now.year and date.month==now.month:
                                #obs.append('Cirr')
                                status='Cirr'
                            
                        # finally check HCC status
                        elif (self.hcc is not None) and (not self.hcc.empty):
                            temp_hcc=self.hcc[self.hcc.columns[1:4]].to_numpy()
                            temp_hcc=temp_hcc.ravel()
                            for i in range(0,len(temp_hcc)):
                                date=pd.to_datetime(temp_hcc[i])
                                if date.year==now.year and date.month==now.month:
                                    status='HCC'
                                
                                
                obs.append(status)
            data.append(obs)
        covariates=np.concatenate((covariates,['Sex','Age','Stage']))
        data=pd.DataFrame(data,columns=covariates)
        self.data=data
        
        return covariates
                        
                        
                    
            
        
        
        
# read all patients' id into storage      
pid=get_id(path,None)

# test code
p=patient(pid[0])
p.align_date()       
        