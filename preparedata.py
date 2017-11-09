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


    DataMatrix = namedtuple('DataMatrix','DATA,samples,features') # namedtupele for holding and indexing all the data
    trainingfunc = nibabel.load(os.path.join(Fdirname,datalist[1]+ paths.featuretype))
    numfeatures = trainingfunc.numDA
    numdatapoints = trainingfunc.darrays[0].data.shape[0];

    dataset = DataMatrix(DATA=np.zeros((numdatapoints,(len(datalist)*numfeatures))),samples=len(datalist),features=numfeatures)
    correlationset = DataMatrix(DATA=np.zeros((numdatapoints,(len(datalist)+1))),samples=len(datalist)+1,features=1)
    
    start=0
    if paths.usegrouplabels:
        # use just one label for all subjects - projected onto subject featurespace through registration
        label=nibabel.load(paths.labelname);
        labelset = DataMatrix(DATA=np.zeros((numdatapoints,1)),samples=1,features=1)
        labelset.DATA[:,0]=label.darrays[0].data
        print('usegrouplabels',labelset.samples)

    else:
        # use single subject label files derived from Nature paper classifier
        labelset = DataMatrix(DATA=np.zeros((numdatapoints,len(datalist))),samples=len(datalist),features=1)
        print('use Nature labels')
        
    for ind,name in enumerate(datalist):
        #print(name,ind,len(datalist))
        func = nibabel.load(os.path.join(Fdirname, name + paths.featuretype))
        if not paths.usegrouplabels:
            # fill label array with single subject labels
            label = nibabel.load(os.path.join(Ldirname,name+paths.subjlabelname))
            labelset.DATA[:,ind] = label.darrays[0].data

        if paths.getFeatureCorrelations:
            # estimate correlation maps showing agreement of individual subject data with group
            corrdata = nibabel.load(os.path.join(Fdirname,name+paths.correlationtype))

            correlationset.DATA[:,ind] = corrdata.darrays[0].data
        
        for d in range(0,func.numDA):
            # fill data array
            dataset.DATA[:,start+d] = func.darrays[d].data

        start += numfeatures

    if paths.getFeatureCorrelations:
        corrdata = nibabel.load(os.path.join(Fdirname, paths.average_feature_correlations))
        correlationset.DATA[:, len(datalist)] = corrdata.darrays[0].data

    return dataset, labelset, correlationset
    
    
def project_data(data_set, label_set, correlation_set, interpolator, Odir, abr, aug):
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
    cm.project_data(label_set, data_set, correlation_set, interpolator, Odir, paths.resampleH, paths.resampleW,paths.lons, abr, aug, paths.usegrouplabels, paths.getFeatureCorrelations,paths.normalise)
    if paths.normalise:
        n_end='normalised.txt'


    if paths.usegrouplabels:
        filename = os.path.join(Odir, 'projections'+ aug+n_end)
    else:
        filename = os.path.join(Odir, 'projections'+ aug+'Nature' +n_end)

    cm.write_projection_paths(data_set.samples, filename, Odir, abr, aug,paths.usegrouplabels, paths.getFeatureCorrelations,paths.normalise)


