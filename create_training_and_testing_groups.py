#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:37:53 2018

@author: er17
"""

import pandas as pd
import numpy as np
import os

demographics=pd.read_pickle('/data/PROJECTS/dHCP/demographics/dHCP_demographics_filtered30-01-18_days.pk1')
outdir='/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/groups'
# label premature babies
demographics['is_prem']=(demographics['birth_ga'] < 34).astype(int)


# remove preterm subject's first scans (using rather ad hoc range which seeks to exclude twins born around 34 -35 weeks as premies)
prems_firstscans=demographics.loc[(demographics['birth_ga'] < 34) & (demographics['scan_ga'] < 36)].drop_duplicates(subset='id', keep='first')
prems_second_scans=demographics.loc[((demographics['birth_ga'] < 34) & (demographics['scan_ga'] > 36)) ].drop_duplicates(subset='id', keep='last')
demographics_terms=demographics.loc[((demographics['birth_ga'] >36))]
demographics_no_preterm_second_scans=demographics.loc[demographics.isin(prems_second_scans)['session']==False]

################### FOR PREMATURITY CLASSIFICATION ############################

# now randomly sample from the prems group and the term subjects

TEST_prems=prems_second_scans.sample(frac=0.15)
TRAIN_prems=prems_second_scans.loc[prems_second_scans.isin(TEST_prems)['session']==False]

TEST_terms=demographics_terms.sample(frac=0.05)
TRAIN_terms=demographics_terms.loc[demographics_terms.isin(TEST_terms)['session']==False]

# merge prem and term components together to give one test and train list

TEST_prem_classification=pd.concat([TEST_prems, TEST_terms]).reset_index()
TRAIN_prem_classification=pd.concat([TRAIN_prems, TRAIN_terms]).reset_index()

# create subject lists 

TESTlist_premclass=[]
for index, row in TEST_prem_classification.iterrows():
    TESTlist_premclass.append('sub-'+ str(row['id'])+'-ses-' + str(row['session']) )

TRAINlist_premclass=[]
for index, row in TRAIN_prem_classification.iterrows():
    TRAINlist_premclass.append('sub-'+ str(row['id'])+'-ses-' + str(row['session']) )
    
np.savetxt(os.path.join(outdir,'TEST_prem_vs_term.txt'),TESTlist_premclass,fmt='%s')
np.savetxt(os.path.join(outdir,'TRAIN_prem_vs_term.txt'),TRAINlist_premclass,fmt='%s')

TEST_prem_classification.to_pickle(os.path.join(outdir,'TEST_prem_vs_term.pk1'))
TRAIN_prem_classification.to_pickle(os.path.join(outdir,'TRAIN_prem_vs_term.pk1'))

################### FOR GA REGRESSION ############################

# merge prem first scans and term 
# now randomly sample from the prems group and the term subjects

TEST_ga_regression=demographics_no_preterm_second_scans.sample(frac=0.07)
TRAIN_ga_regression=demographics_no_preterm_second_scans.loc[demographics_no_preterm_second_scans.isin(TEST_ga_regression)['session']==False]

# create subject lists 

TESTlist_ga_regression=[]
for index, row in TEST_ga_regression.iterrows():
    TESTlist_ga_regression.append('sub-'+ str(row['id'])+'-ses-' + str(row['session']) )

TRAINlist_ga_regression=[]
for index, row in TRAIN_ga_regression.iterrows():
    TRAINlist_ga_regression.append('sub-'+ str(row['id'])+'-ses-' + str(row['session']) )

np.savetxt(os.path.join(outdir,'TEST_ga_regression.txt'),TESTlist_ga_regression,fmt='%s')
np.savetxt(os.path.join(outdir,'TRAIN_ga_regression.txt'),TRAINlist_ga_regression,fmt='%s')

TEST_ga_regression.to_pickle(os.path.join(outdir,'TEST_ga_regression.pk1'))
TRAIN_ga_regression.to_pickle(os.path.join(outdir,'TRAIN_ga_regression.pk1'))