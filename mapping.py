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
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
import Interpolation as intp


def sliceplane(plane, start_height, start_width, x_size, y_size):
    # slice patch from 3D array dim (x,y,channels)
    newdata = plane[int(start_height):int(start_height+x_size), int(start_width):int(start_width+y_size),:]
    return newdata


def normalize(data):
    """
        normalise feature maps

        Parameters
        ----------
        data : multivariate feature maps as 3d np.arrays dims (x,y,channels)

        Returns
        -------
        datanormed : normalised data
    """

    datastd = np.std(data,axis=(0,1))

    if np.where(datastd == 0)[0].shape != (0,):
        print('isnan')
        datastd[np.where(datastd==0)]=1;

    datanormed = (data - np.mean(data,axis=(0,1))) / datastd
    if np.where(np.isnan(datanormed))[0].shape != (0,):
         print('isnan2')

    return datanormed


# def zero_labels(label_image, label_list):
#     """
#         Filter hi-dimensional label image to leave subset of labels to test/train on
#
#         Parameters
#         ----------
#         label_image : the label file as np array
#         label_list  : the list of labels to test/train on
#
#         Returns
#         -------
#         float32 np.array
#     """
#     new_image = copy.deepcopy(label_image)*0
#
#     new_label = []
#     for l,label in enumerate(label_list):
#         x = np.where(label_image == label)
#
#         new_image[x] = l+1
#         new_label.append(l+2)
#
#     return new_image, new_label


# def elastic_transform(labels, features, alpha, sigma, random_state=None):
#     """Elastic deformation of images as described in [Simard2003]_.
#
#
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#        Convolutional Neural Networks applied to Visual Document Analysis", in
#        Proc. of the International Conference on Document Analysis and
#        Recognition, 2003.
#
#        Parameters
#        ----------
#        labels : the label file as np.array
#        features  : the multivariate features as np.array
#        alpha : scaling
#        sigma : smoothing
#        random_state : initialisation of random number generator
#
#        Returns
#         -------
#        newlabels: deformed labels
#        newdata: deformed features
#
#     """
#
#     assert len(labels.shape) == 2
#
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#
#     shape = labels.shape
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
#     x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#     indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
#     indicesorig = np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))
#
#     newdata = np.zeros((features.shape[0], features.shape[1], features.shape[2]))
#
#     # interpolate data using spline interpolation
#     for i in np.arange(features.shape[2]):
#         newdata[:, :, i] = map_coordinates(features[:, :, i], indices, order=1).reshape(shape)
#
#     indexorigarray = np.transpose(np.asarray(indicesorig)[:, :, 0])
#     indexarray = np.transpose(np.asarray(indices)[:, :, 0])
#     imageflat = np.reshape(labels, (shape[0] * shape[1], 1))
#
#     slicemap = newdata[:, :, 0]
#     # interpolate labels using spline interpolation
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(indexorigarray)
#     distances, NNindices = nbrs.kneighbors(indexarray)
#     newlabels = np.zeros((len(indexarray), 1))
#     for i in np.arange(len(indexarray)):
#         newlabels[i] = imageflat[NNindices[i]]
#
#     newlabels2 = newlabels.reshape(shape)
#     newlabels2[np.where(slicemap == 0)] = 0
#
#     return newlabels2, newdata


# def get_patch_centres(group_labels, examples, image_size, patch_size, mean_sim=None, sim_instance=None):
#     """
#        select centres for patches randomly
#
#        Parameters
#        ----------
#        group_labels : use group average labels
#        examples:  number of patch examples to be drawn
#        image_size: original image size
#        patch_size: dimensions of patch to be extracted
#        mean_sim : correlation of single subject features with that of the group
#        sim_instance  : mean correlation across the group
#
#        Returns
#        -------
#        float32 np.array
#    """
#
#     if group_labels is True:
#         if mean_sim is None or sim_instance is None:
#             sys.exit("use of group labels requires feature correlation maps")
#
#     randh = np.arange(image_size[0] - patch_size[0])
#     randw = np.arange(image_size[1] - patch_size[1])
#     np.random.shuffle(randh)
#     np.random.shuffle(randw)
#
#     randh2 = []
#     randw2 = []
#
#     i = 0
#     # print('num_examples',n_examples, _comparetogroup,_switch)
#     while len(randh2) < examples:
#         h = randh[i]
#         w = randw[i]
#
#         if group_labels == True:
#             # if comparing to group then only pick training samples where the featurespace is close to that of group mean - as described in Nature paper
#             corrpatch = sliceplane(np.expand_dims(sim_instance, axis=2), h, w, patch_size[0],patch_size[1])
#             meancorrpatch = sliceplane(np.expand_dims(mean_sim, axis=2), h, w, patch_size[0],patch_size[1])
#             if np.mean(corrpatch) >= 0.9 * np.mean(meancorrpatch):
#                 if h not in randh2:
#                     if w not in randw2:
#                         randh2.append(h)
#                         randw2.append(w)
#             if i == examples * 2:
#                 randh2 = randh[:examples]
#                 randw2 = randw[:examples]
#
#             i += 1
#         else:
#             randh2 = randh[:examples]
#             randw2 = randw[:examples]
#
#     return np.column_stack((randh2, randw2))


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

def project_data(alllabels,alldata,allcorr,interp,outdir,newH,newW,lons,abr,aug,usegroup,use_correlations,use_normalisation):
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
    twoDL = project(alllabels.DATA,interp,newH,newW,lons)
    twoD = project(alldata.DATA, interp, newH, newW, lons)
    if use_correlations:
        twoDcorr = project(allcorr.DATA,interp,newH,newW,lons)

    #plt.imshow(twoDL[:,:,0])
    #plt.show()
    
    
    if usegroup==True:
        np.save(os.path.join(outdir,abr +'GrayScaleLabels-group-aug-' + aug), twoDL[:,:,0])
        
    start=0    
    for subj in range(0,alldata.samples):
        
        if usegroup==False:
            np.save(os.path.join(outdir,abr +'GrayScaleLabels-subj-'+ str(subj)+ '-aug-' + aug+'-Nature'), twoDL[:,:,subj])

        if use_correlations:
            np.save(os.path.join(outdir,abr +'featurecorrelations-subj-'+ str(subj)+ '-aug-' + aug), twoDcorr[:,:,subj])

        if use_normalisation:
            normalised_data = normalize(twoD[:, :, start:start + alldata.features])
            np.save(os.path.join(outdir, abr + 'data_-subj-' + str(subj) + '-aug-' + aug + 'normalised'),normalised_data)
        else:
            np.save(os.path.join(outdir, abr + 'data_-subj-' + str(subj) + '-aug-' + aug), twoD[:, :, start:start + alldata.features])
        start = start+alldata.features

    if use_correlations:
        np.save(os.path.join(outdir,abr +'meanfeaturecorrelations-aug-' + aug), twoDcorr[:,:,alldata.samples])


def write_projection_paths(samples, filename, indir, abr, aug, use_grouplabels, use_correlations,use_normalisation):
    """
        write paths out to file

         Parameters
         ----------
         samples: list of subj ids
         filename: output filename
         indir: path to data files
         aug: indexing projection centres
         use_grouplabels: use group labels
         use_correlations: save featuremap correlations

         Returns
         -------
       """
    target = open(filename, 'w')
    
    for subj in range(0,samples):
                
        if use_grouplabels == True:
            label = os.path.join(indir, abr + 'GrayScaleLabels-group-aug-' + aug +'.npy')
        else:
            label = os.path.join(indir, abr + 'GrayScaleLabels-subj-' + str(subj)+ '-aug-' + aug + '-Nature.npy')

        if use_normalisation:
            data=os.path.join(indir, abr + 'data_-subj-' + str(subj)+ '-aug-' + aug+ 'normalised.npy')
        else:
            data = os.path.join(indir, abr + 'data_-subj-' + str(subj) + '-aug-' + aug + '.npy')

        if use_correlations:
            corrdata=os.path.join(indir, abr + 'featurecorrelations-subj-' + str(subj)+ '-aug-' + aug + '.npy')
            meancorrdata=os.path.join(indir, abr + 'meanfeaturecorrelations-aug-' + aug+'.npy')
            target.write(data + ' ' + label + ' ' + corrdata + ' ' + meancorrdata + '\n')
        else:
            target.write(data + ' ' + label + '\n')

