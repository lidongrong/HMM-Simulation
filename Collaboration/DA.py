# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:25:28 2022

@author: lidon
"""

import OpenData as OD
import pandas as pd
import os

# Data Analysis on Vicky's Data
Px,Dx,Lab,Med=OD.data_reader(OD.path)


# ID of patients
Dx_id=Dx['SubjectID'].drop_duplicates().array
Px_id=Px['SubjectID'].drop_duplicates().array
Lab_id=Lab['SubjectID'].drop_duplicates().array
Med_id=Med['SubjectID'].drop_duplicates().array

# Result Analysis shows that the data consists of the records
# of 50 patients with different IDs. 
# Med id contains all patients' IDs.

# Med_id contains all ids of patients
id_book=Med_id

# Return the medical record in Dx, Px, Lab & Med when questing patient ID
def record_requester(patient_id):
    Px_record=Px[Px['SubjectID']==patient_id]
    Dx_record=Dx[Dx['SubjectID']==patient_id]
    Med_record=Med[Med['SubjectID']==patient_id]
    Lab_record=Lab[Lab['SubjectID']==patient_id]
    return Px_record,Dx_record,Lab_record,Med_record

# Construct a merged frame of a patient, and sort it with time index according to aligned date
# For dates that Px(or Dx, Med, Lab) that have no records, fill in NaN
def merge_record(patient_id):
    Px_record,Dx_record,Lab_record,Med_record=record_requester(patient_id)
    # Merge according to the date
    # First, change the name of the date columns to the same name to merge
    Px_record=Px_record.rename(columns={Px_record.columns[2]:'Aligned Date'})
    Dx_record=Dx_record.rename(columns={Dx_record.columns[2]:'Aligned Date'})
    Lab_record=Lab_record.rename(columns={Lab_record.columns[2]:'Aligned Date'})
    Med_record=Med_record.rename(columns={Med_record.columns[4]:'Aligned Date'})
    
    record=pd.concat([Px_record,Dx_record,Lab_record,Med_record])
    record=record.sort_values('Aligned Date')
    return record

# Generate Personal Health Record
# input is a list of all patients' ids
# For each patient, generate a electronic health record and save them
# under the directory PersonalEHR
def EHR_generator(id_book):
    os.mkdir('PersonalEHR')
    for pid in id_book:
        # file name cannot consist of ':'
        file_name=pid[10:]
        record=merge_record(pid)
        record.to_csv(f'PersonalEHR/{file_name}.csv')
    return 1
    