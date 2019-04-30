#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:37:30 2016

@author: ecr05
"""

from collections import namedtuple
import mapping as cm
import numpy as np
import os
import nibabel
import glob

def get_idlist(regularexpression):
    
    """
        get subject ids from files

        Parameters
        ----------
        regularexpression : regular expression for data files

        Returns
        -------
        idlist : list of subject ids
        
    """
    
    filenames=glob.glob(regularexpression)
    ids=[idn.replace('/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/featuresets/', '') for idn in filenames]
    ids=[idn.replace('.L.MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.32k_fs_LR.func.gii', '') for idn in ids]
    
    return ids;

def get_datalists(use_labels, paths):


    """
            load all original gifti data

            Parameters
            ----------
            paths : dictionary with data paths

            Returns
            -------
            dataset : named tuple containing feature data
            labelset: named tuple containing label data
        
    """
    
   
    datalist=paths['list']

    DataMatrix = namedtuple('DataMatrix','DATA,ids,samples,features') # namedtupele for holding and indexing all the data
    trainingfunc = nibabel.load(paths['fname'].replace('%subjid%',datalist[1]))
    numfeatures = trainingfunc.numDA
    numdatapoints = trainingfunc.darrays[0].data.shape[0];

    # create named tuple for data files
    dataset = DataMatrix(DATA=np.zeros((numdatapoints,(len(datalist)*numfeatures))),ids=[],samples=len(datalist),features=numfeatures)
    
    
    start=0
    if use_labels:
        # create datamatrix for all label files
        labelset = np.zeros((numdatapoints,len(datalist)))
        
    for ind,name in enumerate(datalist):
        print(name,ind,len(datalist))
        func = nibabel.load(paths['fname'].replace('%subjid%',name))
        if use_labels:
            # fill label array with single subject labels
            label = nibabel.load(paths['lname'].replace('%subjid%',name))
            labelset[:,ind] = label.darrays[0].data
        
            
        for d in range(0,func.numDA):
            #fill data array
            dataset.DATA[:,start+d] = func.darrays[d].data
        dataset.ids.append(name)    
        start += numfeatures


    ALLDATA={}
    ALLDATA['data']=dataset
    
        
    if use_labels:
        ALLDATA['labels']=labelset

    return ALLDATA
    
    
def project_data(DATA, interpolator,data_paths,resampleH,resampleW,use_labels,normalise):
    """
               project data from sphere to plane


               Parameters
               ----------
               data_set         : struct containing single subject featuresets
               interpolator     : links spherical mesh grid points to coordinates on 2D project
                                  (currently only nearest neighbour available)
               paths            : dictionary with data paths

               Returns
               -------
       """
    
    
    cm.project_data(DATA, interpolator, data_paths, resampleH, resampleW,normalise)
    
    if normalise:
        csv_outname=os.path.join(data_paths['outdir'],data_paths['abr']+'normalised.pk1')
    else:
        csv_outname=os.path.join(data_paths['outdir'],data_paths['abr']+'.pk1')


# =============================================================================
#     if paths.usegrouplabels:
#         filename = os.path.join(Odir, 'projections'+ aug+n_end)
#     else:
#         filename = os.path.join(Odir, 'projections'+ aug+'Nature' +n_end)
# 
# =============================================================================
    cm.write_projection_paths(DATA['data'], csv_outname, data_paths,use_labels,normalise)


