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
import argparse
from PIL import Image

def load_image_file(fname):
    
    if fname.find('npy') != -1: 
        img = np.load(args.fname)
    elif fname.find('jpg') != -1 or  fname.find('png') != -1:
        img=np.asarray(Image.open(fname).convert('L'))
    else:
        f_list=np.loadtxt(fname,dtype='str')
        img=load_image_file(f_list[0])
        for f in f_list[1:]:
            img=np.dstack((img,load_image_file(f)))
            
    if img.ndim < 3:
        img=np.expand_dims(img,axis=2)
        
    print('img size', img.shape,img.ndim)    
    
    return img
def reproject_data(args):
    
    # load file
    
    img=load_image_file(args.fname)
    
    # load template data
    surf=nibabel.load(args.surfname)
    func=nibabel.load(args.funcname)
    coordinates=surf.darrays[0].data

    if args.templatelabel:
        # find projection centre from group label file
        labels=nibabel.load(args.templatelabel)
        zerocoord=intp.spherical_to_cartesian(100,0,np.pi/2) # starting point of rotation
        x=np.where(labels.darrays[0].data==args.projection_centre)[0]
    else:
        x=args.projection_centre
        
    # re-project data back from plane to sphere
    rotation_centre=coordinates[x[0]]
      
    ROT = cm.rodrigues_rotation(rotation_centre, zerocoord)  # get rotation matrix from current centre to 0 longitude
    coordinates2 = cm.rotate(ROT,coordinates)
    # nearest neighbour interpolator for mapping spherical data to 2D plane
    NN=intp.Interpolation(args.resampleH, args.resampleW, 20, 10)
    NN.get_neighbourhoods(coordinates2)
    NN.get_inv_neighbourhoods(coordinates2)

    
    #img = cm.unpadd(img,[paths.resampleH, paths.resampleW, 1])

    projection = cm.invert_patch_categories_full(img, coordinates, NN, args.resampleH, args.resampleW)
    if projection.shape[1]==1:
        func.darrays[0].data=projection.astype('float32')
    else:
        func.darrays[0].data=projection[:,0].astype('float32')
        for i in range(1,projection.shape[-1]):
            func.add_gifti_data_array(nibabel.gifti.GiftiDataArray(projection[:,i].astype('float32')))
            
    
    print('save')
    nibabel.save(func, args.oname)

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: project labels back from 2D to surface')
    parser.add_argument('surfname',  help='template surface compatible with metric file')
    parser.add_argument('funcname',  help='template metric file or list of files')
    parser.add_argument('fname',  help='name of file to back_project')
    parser.add_argument('oname',  help='output name')
    parser.add_argument('--templatelabel',  help='template regional label file (for definition of projection center)')
    parser.add_argument('--projection_centre',  help='region to be used as project centre', default='0',type=int)
    parser.add_argument('--resampleH', help='height of projection',default=240)
    parser.add_argument('--resampleW', help='width of projection',default=320)
    
    args = parser.parse_args()

    reproject_data(args)