#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:37:30 2016

@author: ecr05
"""

from collections import namedtuple
import mapping as cm
import matplotlib.pyplot as plt 
import numpy as np
import os
import nibabel
import copy
import HCPmultimodal_paths as paths


def get_datalists(datalist,Ldirname,Fdirname):


    """
            load all original gifti data

            Parameters
            ----------
            datalist : list of subject ids
            Ldirname : path to labels directory
            Fdirname : path to features
            getcorr : estimate correlations between subject feature maps and the group average

            Returns
            -------
            dataset : named tuple containing feature data
            labelset: named tuple containing label data
            correlationset: named tuple containing correlation maps
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

        if paths.getFeatureCorrelations:
            correlationset =np.zeros((numdatapoints,(len(datalist)+1)))
            # estimate correlation maps showing agreement of individual subject data with group
            corrdata = nibabel.load(os.path.join(Fdirname,name+paths.correlationtype))

            correlationset[:,ind] = corrdata.darrays[0].data
        
        for d in range(0,func.numDA):
            # fill data array
            dataset.DATA[:,start+d] = func.darrays[d].data
        dataset.ids.append(name)    
        start += numfeatures

    if paths.getFeatureCorrelations:
        corrdata = nibabel.load(os.path.join(Fdirname, paths.average_feature_correlations))
        correlationset[:, len(datalist)] = corrdata.darrays[0].data

    ALLDATA={}
    ALLDATA['data']=dataset
    
    if paths.getFeatureCorrelations:
        ALLDATA['correlations']=correlationset
        
    if paths.use_labels:
        ALLDATA['labels']=labelset

    return ALLDATA
    
    
def project_data(DATA, interpolator, Odir, abr, aug):
    """
               project data from sphere to plane


               Parameters
               ----------
               data_set         : struct containing single subject featuresets
               label_set        : struct containing label files
               correlation_set  : struct containing correlation maps
                                  (showing agreement between subject data and that of group)
               interpolator     : links spherical mesh grid points to coordinates on 2D project
                                  (currently only nearest neighbour available)
               Odir             : output directory
               abr              : file naming convention
               aug              : numerical indexing of files by augmentation

               Returns
               -------
       """
    n_end = '.txt'
    cm.project_data(DATA, interpolator, Odir, paths.resampleH, paths.resampleW,paths.lons, abr, aug, paths.usegrouplabels,paths.normalise)
    if paths.normalise:
        n_end='normalised.txt'


    if paths.usegrouplabels:
        filename = os.path.join(Odir, 'projections'+ aug+n_end)
    else:
        filename = os.path.join(Odir, 'projections'+ aug+'Nature' +n_end)

    cm.write_projection_paths(DATA['data'], filename, Odir, abr, aug,paths.usegrouplabels, paths.getFeatureCorrelations,paths.normalise)


