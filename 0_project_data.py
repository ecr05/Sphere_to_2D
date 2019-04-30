import nibabel
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
import mapping as cm
import preparedata as pr

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
    if args.templatelabel:
        labels=nibabel.load(args.templatelabel)
        zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2) # starting point of rotation
        x=np.where(labels.darrays[0].data==args.projection_centre)[0]
        print('projection centre',)
    else:
        x=args.projection_centre
    
    rotation_centre=coordinates[x[0]]    
    
    # To ameliorate the effect of the projection distorting the feature space project data from the same vertex
    # all data should be pre-aligned
    
     

    ROT = cm.rodrigues_rotation(rotation_centre, zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT, copy.deepcopy(coordinates))

    # estimate nearest neighbour interpolator for projections from this view
    NN = intp.Interpolation(args.resampleH, args.resampleW, 20,10)
    NN.get_neighbourhoods(coordinates2)
            
    if not os.path.exists(paths['Odirname']):
         print('create output directory')
         os.mkdir(paths['Odirname'])

    DATA = pr.get_datalists(args.use_labels,data_paths)

    if args.group_normalise:
         DATA['data']=DATA['data']._replace(DATA=cm.group_normalise(DATA['data']))

    print('project data')
    pr.project_data(DATA, NN, paths,args.resampleH, args.resampleW,args.use_labels,args.normalise)

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: DHCP convolutional autoencoder training script')
    parser.add_argument('surfname',  help='template surface compatible with all func.gii (data) files')
    parser.add_argument('featname',  help='generic file path (with %subjid% in place of subject identifier e.g. /path/to/data/%subjid%.L.data.func.fii')
    parser.add_argument('outdir',  help='output directory')
    parser.add_argument('--templatelabel',  help='template regional label file (for definition of projection center)')
    parser.add_argument('--subjlabel',  help='subject label generic file path (for segmentation projections, using %subjid% widlcard as above)')
    parser.add_argument('--idlist',  help='list of file identifiers (to allow easy mapping between input and output)')    
    parser.add_argument('--outname',  help='output filename', default = 'DATA')      
    parser.add_argument('--meta',  help='meta data file (e.g. matching ids with behavioural/cognitive markers')
    parser.add_argument('--projection_centre',  help='region to be used as project centre', default='0',type=int)
    parser.add_argument('--resampleH', help='height of projection',default=240)
    parser.add_argument('--resampleW', help='width of projection',default=320)
    parser.add_argument('--use_labels', action='store_true')
    parser.add_argument('--use_grouplabels', action='store_true')
    parser.add_argument('--group_normalise', action='store_true',help='normalise across entire data set to set all features with mean zero and standard deviation 1')
    parser.add_argument('--normalise', action='store_true',help='normalise each file separately')

    args = parser.parse_args()
    
    if args.idlist:
        print('read idlist',args.idlist)
        idlist=np.genfromtxt(args.idlist, dtype=str)
    else:
        print('get id list')
        idlist=pr.get_idlist( args.featname.replace('%subjid%','*'))
    print('len', len(idlist),idlist[0],idlist[len(idlist)-1])
        
    if args.use_labels:
        if args.subjlabel==None:
            raise ValueError('if using subj labels you must supply a value to the --subjlabel argument') 
        else:
            labelname=args.subjlabel
      
    print(args.surfname)

    paths = { 'Odirname': args.outdir,
             'fname': args.featname,
             'lname': labelname,
             'list': idlist,
             'meta_csv': args.meta,
             'abr': args.outname}

    project_data(args,paths)