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


def get_datalists(datalist,dirname):


    """
            load all original gifti data

            Parameters
            ----------
            datalist : list of subject ids
            dirname : path to input directory

            Returns
            -------
            dataset : named tuple containing feature data
            labelset: named tuple containing label data
        
    """


    DataMatrix = namedtuple('DataMatrix','DATA,ids,samples,features') # namedtupele for holding and indexing all the data
    trainingfunc = nibabel.load(os.path.join(Fdirname,datalist[1]+ paths.featuretype))
    numfeatures = trainingfunc.numDA
    numdatapoints = trainingfunc.darrays[0].data.shape[0];

    dataset = DataMatrix(DATA=np.zeros((numdatapoints,(len(datalist)*numfeatures))),ids=[],samples=len(datalist),features=numfeatures)
    
    
    start=0
    if paths.use_labels:
        if paths.usegrouplabels:
            # use just one label for all subjects - projected onto subject featurespace through registration
            label=nibabel.load(paths.labelname);
            labelset = DataMatrix(DATA=np.zeros((numdatapoints,1)),samples=1,features=1)
            labelset.DATA[:,0]=label.darrays[0].data
            print('usegrouplabels',labelset.samples)
    
        else:
            # use single subject label files derived from Nature paper classifier
            labelset = np.zeros((numdatapoints,len(datalist)))
            print('use Nature labels')
        
    for ind,name in enumerate(datalist):
        print(name,ind,len(datalist))
        func = nibabel.load(os.path.join(Fdirname, name + paths.featuretype))
        if paths.use_labels and not paths.usegrouplabels:
            # fill label array with single subject labels
            label = nibabel.load(os.path.join(Ldirname,name+paths.subjlabelname))
            labelset[:,ind] = label.darrays[0].data
        
            
        for d in range(0,func.numDA):
            
            if paths.remove_outliers:
                # this is really hacky - not recommended - need better way
                func.darrays[d].data=cm.remove_outliers(d, func.darrays[d].data)
            # fill data array
            dataset.DATA[:,start+d] = func.darrays[d].data
        dataset.ids.append(name)    
        start += numfeatures


    ALLDATA={}
    ALLDATA['data']=dataset
    
        
    if paths.use_labels:
        ALLDATA['labels']=labelset

    return ALLDATA
    
    
def project_data(DATA, interpolator, Odir, abr, aug,meta_csv,data_csv):
    """
               project data from sphere to plane


               Parameters
               ----------
               data_set         : struct containing single subject featuresets
               interpolator     : links spherical mesh grid points to coordinates on 2D project
                                  (currently only nearest neighbour available)
               Ofile            : output file
               abr              : file naming convention
               aug              : numerical indexing of files by augmentation

               Returns
               -------
       """
    
    cm.project_data(DATA, interpolator, Odir, paths.resampleH, paths.resampleW,paths.lons, abr, aug, paths.usegrouplabels,paths.normalise)
    
    if paths.normalise:
        data_csv.replace('.pk1', 'normalised.pk1')


# =============================================================================
#     if paths.usegrouplabels:
#         filename = os.path.join(Odir, 'projections'+ aug+n_end)
#     else:
#         filename = os.path.join(Odir, 'projections'+ aug+'Nature' +n_end)
# 
# =============================================================================
    cm.write_projection_paths(DATA['data'], data_csv, Odir, abr, aug,paths.use_labels,paths.usegrouplabels, paths.getFeatureCorrelations,paths.normalise,meta_csv)


