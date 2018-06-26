# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:43:36 2016

Functions to project cortical data onto a 2D plane via the sphere and back again;
select patches and augment data/labels
@author: ecr05
"""

import sys
import os



import numpy as np
import copy
import matplotlib.pyplot as plt 
import Interpolation as intp
import pandas as pd

def sliceplane(plane, start_height, start_width, x_size, y_size):
    # slice patch from 3D array dim (x,y,channels)
    newdata = plane[int(start_height):int(start_height+x_size), int(start_width):int(start_width+y_size),:]
    return newdata


def normalize(data,norm_axis):
    """
        normalise feature maps

        Parameters
        ----------
        data : multivariate feature maps as 3d np.arrays dims (x,y,channels)

        Returns
        -------
        datanormed : normalised data
    """

    datastd = np.std(data,axis=norm_axis)
    print('in normalise', norm_axis, datastd.shape )
    print('mean',np.mean(data,axis=norm_axis).shape )
    if np.where(datastd == 0)[0].shape != (0,):
        print('isnan')
        datastd[np.where(datastd==0)]=1;

    datanormed = (data - np.mean(data,axis=norm_axis)) / datastd
    if np.where(np.isnan(datanormed))[0].shape != (0,):
         print('isnan2')

    print('mean normalised',np.mean(datanormed,axis=norm_axis))
    print('std normalised',np.std(datanormed,axis=norm_axis))

    
    return datanormed


def rodrigues_rotation(origin,newpt):
    """
            rotate sphere to move longitude 0, latitude 45 degrees to a new position centred on the newpt

            Parameters
            ----------
            origin: current projection centre
            newpt: coordinate intended as new projection centre

            Returns
            -------
            Rot: (3x3) array rotation matrix
    """
    #
    #http://electroncastle.com/wp/?p=39
    #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p1=intp.normalize(origin)
    p2=intp.normalize(newpt)

    angle = np.arccos(np.dot(p1,p2)) # get angle of rotation
    
    if angle> np.pi:
        raise ValueError('rodrigues rotation error: the specified rotation angle exceeds pi')
    else:
        print("rodrigues rotation: angle of rotation is %f"%angle)    
        
    ID = np.identity(3)

    # get axis of rotation
    cross = np.cross(p1,p2)
    cross = intp.normalize(cross)
    # get rotation matrix
    if angle == 0:
        Rot = ID
    elif intp.normalize(cross) == 0:
        Rot = np.negative(ID);
    else:
       
       u = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0],0]])
       Rot = ID+np.sin(angle)*u+(1-np.cos(angle))*(u.dot(u))
    
    return Rot


def rotate(rotation, coords):
    """
           rotate coordinates

           Parameters
           ----------
           rotation: rotation matrix
           coords: coordinate matrix

           Returns
           -------
           Rot: (nx3) array of rotated n points
    """
    coordsrot = np.dot(rotation,np.transpose(coords));
    return np.transpose(coordsrot)

def rescale_labels(labels,labelfunc):
    # rest original label indexing
    func=copy.deepcopy(labelfunc)
    for l,label in enumerate(labels):
       
        x = np.where(labelfunc == l+1)#[0]
        func[(x)]=label
    
    return func

def zero_labels(labels,labelfunc):
    # labels must index concurrently from 0
    func=copy.deepcopy(labelfunc)
    func = func * 0
    newlabel=[]
    for l,label in enumerate(labels):
       
        x = np.where(labelfunc == label)#[0]

        func[(x)]=l+1
        newlabel.append(l+2)

    
    return func,newlabel

def invert_projection_test(data2D,coords,interp,nlats,nlons,lons):
 
    
    DATA=np.zeros((coords.shape[0],data2D.shape[2]))
    
    # fill data with cortical features
    for x,index in enumerate(coords):
        flatindex=interp.invcorrespondence[x];
        latind=flatindex[0]
        lonind=flatindex[1]
    # fill data with cortical features
           
        DATA[x,:]=data2D[latind,lonind,:]

    
    return DATA

def invert_patch_categories_full(img,coords,interp,newH,newW,lons,labels=[]):



    #plt.imshow(img[0,:,:,0])
    #plt.show()
    #zeropadded=inv_slice(croppedrescale2,newH,newW,h,w)
    SURFdata=invert_projection_test(img,coords,interp,newH,newW,lons)
    #SURFdata=invert_projection_test(img[0,:,:,:],coords,interp,newH,newW,lons)
    if len(labels) > 0:
        rescaledlabels = rescale_labels(labels, SURFdata)
        print('rescaled shape', rescaledlabels.shape)
        return rescaledlabels
    else:
        return SURFdata

def remove_outliers(d,data):
    # quick hack to get rid of xtreme outliers (before there was some really high values)
    if  (np.max(data)>np.mean(data)+10*np.std(data)) or (np.min(data)<np.mean(data)-10*np.std(data)):
        print(d,  'data stats:', np.mean(data), np.std(data), np.max(data), np.min(data),np.mean(data)+10*np.std(data),np.mean(data)-10*np.std(data))

        data[np.where(data>np.mean(data)+10*np.std(data))]= np.mean(data)
    

def reformat_datamat_rows_to_columns(ALLDATA):
    newmat=np.zeros((ALLDATA.samples*ALLDATA.DATA.shape[0],ALLDATA.features))
    
    for f in range(ALLDATA.features):
        for s in range(ALLDATA.samples):
            #print(f,s,'ALLDATA.DATA[:,f*s] shape',s*ALLDATA.DATA.shape[0],(s+1)*ALLDATA.DATA.shape[0],newmat.shape,ALLDATA.DATA.shape[0],ALLDATA.DATA[:,f*s].shape,newmat[s*ALLDATA.DATA.shape[0]:(s+1)*ALLDATA.DATA.shape[0],f].shape)
            newmat[s*ALLDATA.DATA.shape[0]:(s+1)*ALLDATA.DATA.shape[0],f]= ALLDATA.DATA[:,s*ALLDATA.features+f]       
    
    return newmat

def reformat_datamat_columns_to_rows(datamat,ALLDATA):
    newmat=np.zeros(ALLDATA.DATA.shape)
    
    for f in range(ALLDATA.features):
        for s in range(ALLDATA.samples):
            #print(f,s,'datamat[s*ALLDATA.DATA.shape[0]:,f]',datamat[s*ALLDATA.DATA.shape[0]:(s+1)*ALLDATA.DATA.shape[0],f].shape,newmat[:,f*s].shape)
            newmat[:,s*ALLDATA.features+f]= datamat[s*ALLDATA.DATA.shape[0]:(s+1)*ALLDATA.DATA.shape[0]:,f]       
    
    return newmat

def group_normalise(ALLDATA):
    """
          normalise data across group; first restructures data s.t. all examples for each feature fit into one column
          i.e. (n_subjects*n_vertices,n_features)
          then it normalises and returns data to (n_vertices,n_subjects*n_features) format

          Parameters
          ----------
          ALLDATA: data for all subjects in (n_vertices,n_subjects*n_features) format

          Returns
          -------
          
    """
    #alldatamean=np.mean(ALLDATA.DATA,axis=0)
    newmat=reformat_datamat_rows_to_columns(ALLDATA)
    newmat=normalize(newmat,0) # normalise along each column
    newmat2=reformat_datamat_columns_to_rows(newmat,ALLDATA)
# =============================================================================
#     diff=ALLDATA.DATA-newmat2
#     print('diff',np.sum(diff))
# =============================================================================
    #print('group_normalise_mean',np.mean(newmat,axis=0))
    #ALLDATA._replace(DATA=newmat)
    return newmat2

def project(DATAset, interp, nlats, nlons, lons):
    """
          project points from sphere onto 2D grid

          Parameters
          ----------
          DATAset: labels/features to be projected
          interp: saved correspondences between the sphere and 2D grid
          nlats: number of bins in x direction
          nlons: number of bins in y direction
          lons: edges of longitude bins

          Returns
          -------
          Rot: (nx3) array of rotated n points
    """

    data = np.zeros((nlats, nlons, DATAset.shape[1]))

    # fill data with cortical features
    for index, x in np.ndenumerate(data[:, :, 0]):
        latind = index[0]
        lonind = index[1]

        data[latind, lonind, :] = DATAset[interp.correspondence[index], :]

    return data

def project_data(DATA,interp,outdir,newH,newW,lons,abr,aug,usegroup,use_normalisation):
    """
         project labels, featuremaps and optionally feature-correlation maps onto 2D plan

         Parameters
         ----------
         alllabels: np.array containing label surface data for all subjects within a group
         alldata: np.array containing multivariate feature maps for all subjects in a group
         allcorr:  np.array containing correlation (of feature) maps for all subjects in a group
         interp: saved correspondences between the sphere and 2D grid
         outdir: output directory
         newH: y dimensions of 2d projection
         newW: x dimensions of 2d projection
         lons: edges of longitude bins
         abr: group type
         aug: numerical index relating to the number of projection centres
         usegroup: use group labels
         usecorrelations: if defined use feature correlation maps and project these to 2D plane

         Returns
         -------
   """
   
    # project data to 2D
    alldata=DATA['data']
    
    twoD = project(alldata.DATA, interp, newH, newW, lons)

    if 'labels' in DATA: 
        print('in project data, labels== true')
        # if working with label files then also project these to 2D
        twoDL = project(DATA['labels'],interp,newH,newW,lons)
        if usegroup==True:
            np.save(os.path.join(outdir,abr +'GrayScaleLabels-group-aug-' + aug), twoDL[:,:,0])
            
    if  'correlations'  in DATA: 
        print('in project data, correlations== true')
        # if using correlations with group maps then also project these to 2D
        twoDcorr = project(DATA['correlations'],interp,newH,newW,lons)
        
    start=0    
    for subj in range(0,alldata.samples):
        #print('subj', subj, DATA['data'].ids[subj] )
        if 'labels' in DATA and usegroup==False:
            np.save(os.path.join(outdir,abr +'GrayScaleLabels-subj-'+ DATA['data'].ids[subj] + '-aug-' + aug+'-Nature'), twoDL[:,:,subj])

        if 'correlations'  in DATA:
            np.save(os.path.join(outdir,abr +'featurecorrelations-subj-'+ DATA['data'].ids[subj] + '-aug-' + aug), twoDcorr[:,:,subj])

        if use_normalisation:
            normalised_data = normalize(twoD[:, :, start:start + alldata.features])
            np.save(os.path.join(outdir, abr + 'data_-subj-' + DATA['data'].ids[subj]  + '-aug-' + aug + 'normalised'),normalised_data)
        else:
# =============================================================================
#             data=twoD[:, :, start:start + alldata.features].reshape((320x240,3))
#             if np.sum(data,axis=0)
# =============================================================================
            
            np.save(os.path.join(outdir, abr + 'data_-subj-' + DATA['data'].ids[subj]  + '-aug-' + aug), twoD[:, :, start:start + alldata.features])
            
        start = start+alldata.features

    if 'correlations'  in DATA:
        np.save(os.path.join(outdir,abr +'meanfeaturecorrelations-aug-' + aug), twoDcorr[:,:,alldata.samples])


def write_projection_paths(DATA, filename, indir, abr, aug, use_labels, use_grouplabels, use_correlations,use_normalisation,meta_csv):
    """
        write paths out to file

         Parameters
         ----------
         samples: list of subj ids
         filename: output filename
         indir: path to data files
         aug: indexing projection centres
         use_grouplabels: ue group labels
         use_correlations: save featuremap correlations

         Returns
         -------
       """
    meta_data=pd.read_pickle(meta_csv)
    
    df = pd.DataFrame() 
    print('write projection paths')
    for subj in range(0,DATA.samples):
        #print('subj', subj, DATA.ids[subj],type(DATA.ids[subj]) )        
        if use_labels:
            if use_grouplabels == True:
                label = os.path.join(indir, abr + 'GrayScaleLabels-group-aug-' + aug +'.npy')
            else:
                label = os.path.join(indir, abr + 'GrayScaleLabels-subj-' + DATA.ids[subj]+ '-aug-' + aug + '-Nature.npy')

        if use_normalisation:
            print(os.path.join(indir, abr + 'data_-subj-' + DATA.ids[subj] + '-aug-' + aug+ 'normalised.npy'))
            data=os.path.join(indir, abr + 'data_-subj-' + DATA.ids[subj] + '-aug-' + aug+ 'normalised.npy')
        else:
            data = os.path.join(indir, abr + 'data_-subj-' +DATA.ids[subj]+ '-aug-' + aug + '.npy')

        row={};
        row['fileid']=  DATA.ids[subj] 
        row['data']= data
        if use_labels:
            row['labels']= label
        
        df = df.append(row, ignore_index=True)
        
    output=df.merge(meta_data, on=['fileid'])
    output=output.drop(['fileid','index'], axis=1)
    output.to_pickle(os.path.join(indir,filename))
        
    
# =============================================================================
#         if use_correlations:
#             corrdata=os.path.join(indir, abr + 'featurecorrelations-subj-' +DATA.ids[subj] + '-aug-' + aug + '.npy')
#             meancorrdata=os.path.join(indir, abr + 'meanfeaturecorrelations-aug-' + aug+'.npy')
#             #target.write(data + ' ' + label + ' ' + corrdata + ' ' + meancorrdata + '\n')
#         else:
#             target.write(data + ' ' + label + '\n')
# 
# =============================================================================
