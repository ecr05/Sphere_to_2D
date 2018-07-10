#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:52:16 2018

@author: er17
"""
import pandas as pd
import copy as cp

folder='TRAIN_ga_regression'

df=pd.read_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'+folder+'/featuresets/projectedcentre_16/'+folder+'_data_csv_left.pk1')
dfR=pd.read_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'+folder+'/featuresets/projectedcentre_15/'+folder+'_data_csv_right.pk1')
dfLandR=pd.concat((df,dfR))
dfLandR_gpuserver=cp.deepcopy(dfLandR)
dfLandR_gpuserver['data']=dfLandR_gpuserver['data'].str.replace('/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'+folder+'/featuresets/projectedcentre', '/home/er17/DATA/DHCP/'+folder+'/featuresets/projectedcentre')

dfLandR.to_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'+folder+'/featuresets/projectedcentre_16/'+folder+'_data_csv_left_and_right.pk1')
dfLandR_gpuserver.to_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'+folder+'/featuresets/projectedcentre_16/'+folder+'_data_csv_left_and_right_gpuserver.pk1')

print(dfLandR['data'] == dfLandR_gpuserver['data'])