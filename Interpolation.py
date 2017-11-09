# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:10:14 2016

Contains functions for interpolating between cartesian and spherical grids
Only nearest neighbour interpolator included so far
@author: ecr05
"""

import numpy as np
import math
import collections as col


def magnitude(v):
    # estimate magnitude of vector v
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))  


def normalize(v):
    # normalize vector v
    vmag = magnitude(v)

    if vmag == 0:
        return v
    else:
        return [v[i]/vmag for i in range(len(v))]

    
def unit_sphere(sphere):
    # rescale all mesh coordinates to unit magnitude
    for ind,i in enumerate(sphere):
        inew = normalize(i)
        sphere[ind,:] = inew
        
    return sphere


def cartesian_to_spherical(coord):
    # convert from cartesian to spherical coords
    r=math.sqrt(math.pow(coord[0],2)+math.pow(coord[1],2)+math.pow(coord[2],2))
  
    theta = np.arctan2(coord[1],coord[0])
    phi=np.arccos(coord[2]/r)

    return theta, phi


def spherical_to_cartesian(r,theta,phi):
    # convert from cartesian to spherical coords
    x=r*np.sin(phi)*np.cos(theta)
    y=r*np.sin(phi)*np.sin(theta)
    z=r*np.cos(phi)
  
    return x, y, z
    
    
class Relations:
    # bins spherical grid into a coarse 2D histogram grid to speed up neighbourhood search
    
    def __init__(self,nth,nph):
        self.Grid=dict()
        self.nth=nth
        self.nph=nph
        self.delta_th=2.*np.pi/nth; 
        self.delta_ph=np.pi/nph
        self.count=np.zeros((self.nph,self.nth))
        self.relations=col.defaultdict(list)
        self.inrelations=col.defaultdict(list)
        
    def get_relations_for_loc(self,s_th,s_ph):
        # bin spherical coordinates into regularly space long/latitude grid (currently no conformal projection used)

        longind=int(s_th/self.delta_th)
        latind=int(s_ph/self.delta_ph)
        if longind==self.nth:
            longind-=1
        if latind==self.nph:
            latind-=1
        
        if(len(self.relations[(latind,longind)])==0):
            print(s_th,s_ph,latind,longind)
            print(self.count) 

        totallist=[];

       
        for j in range(-1,1,1):
            
            if longind+j==self.nth:
                longind=-1
            if longind-j==-1:
                longind=self.nth
                    
            totallist.extend(self.relations[(latind,longind+j)])
        
        return self.relations[(latind,longind)],longind*self.delta_th+0.5*self.delta_th,latind*self.delta_ph+0.5*self.delta_ph
        
    def get_relations(self,spherecoords):
        
        for ind,i in enumerate(spherecoords):
            theta,phi=cartesian_to_spherical(i)
            
            longind=int((theta+np.pi)/self.delta_th)
            latind=int((phi)/self.delta_ph)
            if longind == self.nth:
                longind -= 1
            if latind == self.nph:
                latind -= 1

            self.count[latind,longind] += 1
            self.relations[(latind, longind)].append(ind)
        

class Interpolation:
    """A class for interpolation between triangulated and rectangular image grids"""
    
    def __init__(self, nlats, nlons, cth, cph):
        self.delta_x = 2.*np.pi/(nlons-1)
        self.delta_y = np.pi/(nlats-1)
        self.nlons = nlons
        self.nlats = nlats
        self.coarsecorrespondence = Relations(cth,cph)
        self.correspondence = dict()
        self.invcorrespondence = dict()
        self.flatcoords = np.zeros((nlats, nlons))

    def get_neighbourhoods(self, spherecoords):
        # base class is the nearest neighbour method

        self.coarsecorrespondence.get_relations(spherecoords)
     
        for index, x in np.ndenumerate(self.flatcoords):

            phi = self.delta_y*index[0]
            theta = self.delta_x*index[1]
            
            nearestindices, lon, lat = self.coarsecorrespondence.get_relations_for_loc(theta,phi)
            
            coord=spherical_to_cartesian(100, theta-np.pi, phi)
            mindist=magnitude(coord)
            mincoord=spherecoords.shape[0]
            
            for sindex in nearestindices:
                dist = np.array([spherecoords[sindex][0]-coord[0],spherecoords[sindex][1]-coord[1],spherecoords[sindex][2]-coord[2]]);
                if magnitude(dist) < mindist:
                    
                    mindist = magnitude(dist)
                    mincoord = sindex
            
            if mincoord < spherecoords.shape[0]:
                self.correspondence[index] = mincoord
            else:
                print('no neighbour',x)

    def get_inv_neighbourhoods(self, spherecoords):
        # get inverse neighbourhoods - grid coordinates that overlap with spherical coordinates
     
        for x,index in enumerate(spherecoords):

            s_th,s_ph = cartesian_to_spherical(index)
            longind = int((s_th+np.pi)/self.delta_x)
            latind = int(s_ph/self.delta_y)

            self.invcorrespondence[x] = (latind,longind)
            

                