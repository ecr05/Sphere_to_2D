#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:54:51 2016

@author: ecr05
"""

import numpy as np
import nibabel
import mapping as cm
import os
import Interpolation as intp
import HCPmultimodal_paths_dHCP as paths
import matplotlib.pyplot as plt
import pandas as pd

normalised=False

num_test=len(os.listdir(paths.outputdir))

# load template data
surf=nibabel.load(paths.surfname)
template=nibabel.load(paths.templategiftiname)

coordinates=surf.darrays[0].data

# get cartesion coordinates for point at zero longitude 90 degrees latitude
zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2)
#test_filenames=pd.read_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/TRAINING_exclprembatch/featuresets/projectedvisAFFINE/TRAININGoutname-regressionlookup.pk1')
#outpath=paths.training_paths['Odir']

#if normalised:
#    test_filenames =pd.read_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/TESTINGDATA/featuresets/projectedvisL/TESTINGoutname-lookupnormalised.pk1')
#else:
test_filenames =pd.read_pickle('/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/TESTING_controls_nopreterms/featuresets/projectedMOTORAFFINE/TESTING_controlsoutname-regressionlookupL.pk1')
# re-project data back from plane to sphere
# do this for every projection view and merge results (**hack!**)
ptinds=[]#zerocoord]

# then rotate sphere to centre on different label regions
for i in paths.projection_centres:
    print('projection centre:', i)
    ptinds.append(coordinates[i])

# create array for each subject to save results of project


for aug in np.arange(len(ptinds)):
    
    print('aug',aug)
    ROT = cm.rodrigues_rotation(ptinds[aug], zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT,coordinates)
    # nearest neighbour interpolator for mapping spherical data to 2D plane
    NN=intp.Interpolation(paths.resampleH, paths.resampleW, 20, 10)
    NN.get_neighbourhoods(coordinates2)
    NN.get_inv_neighbourhoods(coordinates2)

    for index, row in test_filenames.iterrows(): 
    
        uniquevals = []
        fname =row['data']
   #     fname = os.path.join( paths.testing_paths_2['Odir'],row['data']) # reproject normalised testing data
        fname = os.path.join(paths.outputdir,paths.outputname + str(row['id'])+'_'+str(row['session']) + '.npy') # reproject autoencoder output
       
        print(row['id'],fname)
        img = np.load(fname)
        plt.imshow(img[0,:,:,0])
        plt.show()
        #img = cm.unpadd(img,[paths.resampleH, paths.resampleW, 1])

        projection = cm.invert_patch_categories_full(img[0,:,:,:], coordinates, NN, paths.resampleH, paths.resampleW, paths.lons)

        newgifti=nibabel.gifti.gifti.GiftiImage(header=None)
        
        for i in range(projection.shape[1]):
            newgifti.add_gifti_data_array(nibabel.gifti.gifti.GiftiDataArray(projection[:,i].astype('float32')))
        print(os.path.join(paths.outputdir, paths.outputname + 'subj-' + row['id']+ '-aug-' + str(aug) +'.func.gii'))
        #nibabel.save(newgifti,os.path.join(paths.testing_paths_2['Odir'], 'TESTING' + str(row['id'])+'_'+str(row['session']) +'.func.gii'))
       
        nibabel.save(newgifti, os.path.join(paths.outputdir, paths.outputname + str(row['id'])+'_'+str(row['session']) +'.func.gii'))
        
    
    


