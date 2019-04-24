import os
import numpy as np

filetag=''

hemi='left'
if hemi== 'left':
    hemi_template='L'
else :
     hemi_template='R'

basedirname = '/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS_new'
trainingdir = os.path.join(basedirname,'TRAIN_prem_vs_term'+filetag)
testingdir = os.path.join(basedirname,'TEST_prem_vs_term'+filetag)
outputdir='/data/PROCESSED_DATA/PROJECTS/dHCP/prem_classification_050718/output_wholeimage' # output of deep learning 

# define path to group data
dirname = '/data/PROJECTS/dHCP/PROCESSED_DATA/TEMPLATES/new_surface_template/'
#dirname = '/data/PROJECTS/dHCP/PROCESSED_DATA/TEMPLATES/original_surface_template/' # old atlas

# template label file
labelname = os.path.join(dirname,'labels/40.'+hemi_template+'.label.gii')

# common surface for all label files
surfname = os.path.join(dirname,'week40.iter30.sphere.'+hemi_template+'.dedrift.AVERAGE_removedAffine.surf.gii')
#surfname = os.path.join(dirname,'week40.iter9.ico.sphere.'+hemi_template+'.dedrift.AVERAGE.surf.gii') old atlas

#template gifti - used to project output onto
#templategiftiname= os.path.join(dirname,'sulc/week40.iter30.sulc.'+hemi_template+'.AVERAGE.shape.gii')


# features - feature files should have form subjID+hemi+featuretype
featuretype =  '_'+ hemi+ '_myelin_thickness_curvature.32k_fs_LR.func.gii'
#featuretype =  '_'+ hemi+ '_thickness_curvature_dummy_myelin.32k_fs_LR.func.gii'


# name for the backprojected output from the deep learning
outputname = 'output'

# define training, validation and test directories and paths
TRAINING = os.path.join(trainingdir,'TRAIN_prem_vs_term'+filetag+'.txt')
TESTING = os.path.join(testingdir,'TEST_prem_vs_term'+filetag+'.txt')

# optionally define additional label regions on which to centre the sphere prior to projection
#Occipital = 16 (L) and 15 (R)
# Frontal =  30 (L) and 29 (R)
# Parietal = 32 (L) and 31 (R)

if hemi == 'left':
    projection_centres=[16]#,262, 311, 331,240]  # regions 55b IFSa TGd  IP1 PGs p32pr
else:
    projection_centres=[15]
    
    
training_paths = {'Ldir': os.path.normpath(trainingdir+ '/labels/'),
                  'Fdir': os.path.normpath(trainingdir+ '/featuresets/'),
                  'Odir': os.path.normpath(trainingdir+ '/featuresets/projectedcentre_'+ str(projection_centres[0])),
                  'list': np.genfromtxt(TRAINING , dtype=str),
                  'meta_csv': os.path.join(trainingdir, 'TRAIN_prem_vs_term'+filetag+'.pk1'),
                  'data_csv': 'TRAIN_prem_vs_term_data_csv_'+hemi+'.pk1',
                  'abr': 'TRAINING_'+hemi}

testing_paths = { 'Ldir': os.path.normpath(testingdir+ '/labels/'),
                  'Fdir': os.path.normpath(testingdir+ '/featuresets/'),
                  'Odir': os.path.normpath(testingdir+ '/featuresets/projectedcentre_'+ str(projection_centres[0])),
                  'list': np.genfromtxt(TESTING , dtype=str),
                  'meta_csv': os.path.join(testingdir,'TEST_prem_vs_term'+filetag+'.pk1'),
                  'data_csv':'TEST_prem_vs_term_data_csv_'+hemi+'.pk1',
                  'abr': 'TESTING_'+hemi}

# save projection centres
if not os.path.exists(training_paths['Odir']):
    os.mkdir(training_paths['Odir'])
    
if not os.path.exists(testing_paths['Odir']):    
    os.mkdir(testing_paths['Odir'])


np.savetxt(os.path.join(training_paths['Odir'],'projection_centres.txt'),projection_centres)
np.savetxt(os.path.join(testing_paths['Odir'],'projection_centres.txt'),projection_centres)


# for back projection debugging
bp_paths=testing_paths


# tuning parameters


    
# define dimensions of projection plane
resampleH=240
resampleW=320

delta_d=2.*np.pi/(resampleW-1); # width of longitude bin if w not in  randw2:s
lons = (delta_d*np.indices((resampleW,1))[0,:,:]) #edges of longitude bins

use_labels=False
usegrouplabels = False # use the group average labels for all subjects (projected onto each subjects feature space through registration as per Glasser et al, Nature 2016)
getFeatureCorrelations = False #useful when using group labels as it can be used to filter training data (see Glasser et al, Nature 2016)
normalise = False # necessary for old version
group_normalise=True
remove_outliers=False
