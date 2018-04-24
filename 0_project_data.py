import nibabel
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
import mapping as cm
import preparedata as pd

import Interpolation as intp
import HCPmultimodal_paths as paths
import os
import copy


# load surf - constant for all subjects
surf=nibabel.load(paths.surfname)
coordinates=surf.darrays[0].data

# load  labels - constant for all subjects - group parcellation
labels=nibabel.load(paths.labelname)

# datasets
datasets=[paths.training_paths]#, paths.validation_paths, paths.testing_paths]

# To ameliorate the effect of the projection distorting the feature space,
# and to simulate the effect of data augmentation on the labels
# optionally rotate the sphere to centre on a number of different label regions prior to projection
# Therebye obtaining projections from different 'views'

zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2) # starting point of rotation
ptinds=[]#zerocoord]

# then rotate sphere to centre on different label regions
for i in paths.projection_centres:
    print('projection centre:', i)
    x=np.where(labels.darrays[0].data==i)[0]
    ptinds.append(coordinates[x[0]])

print('len ptinds',len(ptinds))
# and obtain projections for all subjects from each projection centre
for aug, pt in enumerate(ptinds):

    print('aug', aug)
    coordinates2 = copy.deepcopy(coordinates)

    ROT = cm.rodrigues_rotation(pt, zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT, coordinates2)

    # estimate nearest neighbour interpolator for projections from this view
    NN = intp.Interpolation(paths.resampleH, paths.resampleW, 20,10)
    NN.get_neighbourhoods(coordinates2)

    for i in datasets:

        if not os.path.exists(i['Odir']):
             print('mkdir')
             os.mkdir(i['Odir'])

        print('get lists')
        if paths.usegrouplabels == True:
             # use group average labels for each subject (as used in Glasser et al. Nature 2016)
             DATA = pd.get_datalists(i['list'], i['Ldir'], i['Fdir'])
        else:
             # training on output of Nature paper classifier - learnt labels for each individual
             DATA = pd.get_datalists(i['list'], i['Ldir'], i['Fdir'])

        DATAbefore=copy.deepcopy(DATA['data'].DATA)
        if paths.group_normalise:
            cm.group_normalise(DATA['data'])
            
        print('project data')
        #pd.project_data(DATA, NN, i['Odir'], i['abr'], str(paths.projection_centres[aug]))