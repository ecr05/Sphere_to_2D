import nibabel
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
import mapping as cm
import preparedata as pd

import Interpolation as intp
#import HCPmultimodal_paths as paths
import os
import copy
import argparse

def project_data(args, data_paths):
    # load surf - constant for all subjects
    surf=nibabel.load(args.surfname)
    coordinates=surf.darrays[0].data
    
    # load  labels - constant for all subjects - group parcellation
    labels=nibabel.load(args.labelname)
    
    # datasets
    datasets=[paths.validation_paths,paths.training_paths,paths.testing_paths]#,
    
    # To ameliorate the effect of the projection distorting the feature space project data from the same vertex
    # all data should be pre-aligned
    
    zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2) # starting point of rotation
    x=np.where(labels.darrays[0].data==args.projection_centre)[0]
    rotation_centre=coordinates[x[0]]     

    ROT = cm.rodrigues_rotation(rotation_centre, zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT, copy.deepcopy(coordinates))

    # estimate nearest neighbour interpolator for projections from this view
    NN = intp.Interpolation(paths.resampleH, paths.resampleW, 20,10)
    NN.get_neighbourhoods(coordinates2)
            
    if not os.path.exists(paths['Odir']):
         print('create output directory')
         os.mkdir(paths['Odir'])

    DATA = pd.get_datalists(paths['list'], paths['dir'])

    if args.group_normalise:
         DATA['data']=DATA['data']._replace(DATA=cm.group_normalise(DATA['data']))

    print('project data')
    pd.project_data(DATA, NN, paths['Odir'], paths['abr'], paths['meta_csv'])

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: DHCP convolutional autoencoder training script')
    parser.add_argument('surfname',  help='template surface compatible with all func.gii (data) files')
    parser.add_argument('labelname',  help='template regional label file (for definition of projection center)')
    parser.add_argument('datadir',  help='top level of directory where data is located')
    parser.add_argument('outdir',  help='output directory')
    parser.add_argument('outname',  help='output filename')
    parser.add_argument('idlist',  help='list of file identifiers (to allow easy mapping between input and output)')
    parser.add_argument('meta',  help='meta data file (e.g. matching ids with behavioural/cognitive markers')
    parser.add_argument('--projection_centre',  help='region to be used as project centre', default='0',type=int)
    parser.add_argument('--use_labels', action='store_true')
    #parser.add_argument('--use_grouplabels', action='store_true')
    parser.add_argument('--group_normalise', action='store_true',help='normalise across entire data set to set all features with mean zero and standard deviation 1')
    parser.add_argument('--normalise', action='store_true',help='normalise each file separately')

# =============================================================================
#     parser.add_argument('--restart', action='store_true')
#     parser.add_argument('--verbose', action='store_true')
#     parser.add_argument('--cuda_devices', '-c', default='0')
#     parser.add_argument('--save_path', '-p', default='/tmp/dchp_cae_lr0.01_feat2/')
#     parser.add_argument('--input_path')
#     parser.add_argument('--data_csv', default='train.csv')
#     parser.add_argument('--val_index', default=None)
#     parser.add_argument('--config', default='')
#     parser.add_argument('--id', default='id')
#     parser.add_argument('--label', default='is_prem')
#     parser.add_argument('--classification', action='store_true')
#     parser.add_argument('--run_decoding', action='store_true')
#     parser.add_argument('--rescale', action='store_true')
#     parser.add_argument('--random_state', default=42)
# =============================================================================
    
    args = parser.parse_args()
    
    paths = {'dir': args.datadir,
             'Odir': args.outdir,
              'list': np.genfromtxt(args.idlist, dtype=str),
              'meta_csv': args.meta,
              #'data_csv': args.outname +'.pk1',
              'abr': args.outname}

    project_data(args,paths)