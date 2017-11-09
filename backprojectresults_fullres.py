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
import HCPmultimodal_paths as paths

# set label ids
if len(paths.labels) == 0:
    # if no label shortlist provided assume projection contains all 180 left hemisphere labels (from label id 181-360)
    labels = np.arange(181, 361) # list of label ids
else:
    labels = paths.labels

num_labels = len(labels)

# load template data
surf=nibabel.load(paths.surfname)
template=nibabel.load(paths.templategiftiname)
labels=nibabel.load(paths.labelname)

coordinates=surf.darrays[0].data

# get cartesion coordinates for point at zero longitude 90 degrees latitude
zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2)

# re-project data back from plane to sphere
# do this for every projection view and merge results (**hack!**)
ptinds = [zerocoord]

aug = 0
for i in paths.projection_centres:
    x=np.where(labels.darrays[0].data == i)[0]
    ptinds.append(coordinates[x[0]])

# create array for each subject to save results of project
augmentations=[]
for subj in range(0, paths.num_test):
    newlabels = np.zeros((coordinates.shape[0],len(ptinds)))
    augmentations.append(newlabels)

for aug in np.arange(len(ptinds)):
    print('aug',aug)
    ROT = cm.rodrigues_rotation(ptinds[aug], zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT,coordinates)
    # nearest neighbour interpolator for mapping spherical data to 2D plane
    NN=intp.Interpolation(paths.resampleH, paths.resampleW, 20, 10)
    NN.get_neighbourhoods(coordinates2)
    NN.get_inv_neighbourhoods(coordinates2)

    for subj in range(0, paths.num_test):
    
        uniquevals = []
    
        fname = os.path.join(paths.outputdir, 'TESTINGGrayScaleLabels-subj-' + str(subj) + '-aug-' + str(aug) + '-Nature.npy')

        print(subj,fname)
        img = np.load(fname)
        img = cm.unpadd(img,[paths.resampleH, paths.resampleW, 1])

        projection = cm.invert_patch_categories_full(img, coordinates, NN, paths.resampleH, paths.resampleW, paths.lons,labels)

        augmentations[subj][:,aug] = projection[:,0]

# now merge results from different projections into one single view
final_labels = np.zeros((coordinates.shape[0], 29))

labels.remove_gifti_data_array(0)
for subj in range(0, paths.num_test):

    final_labels[:, subj] = cm.label_fusion(augmentations[subj])
    labels.add_gifti_data_array(nibabel.gifti.GiftiDataArray(final_labels[:, subj].astype('float32')))

nibabel.save(labels, os.path.join(paths.outputdir, paths.outputname))
