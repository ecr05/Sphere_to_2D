import os
import numpy as np

basedirname = '/home/er17/Documents/PROJECTS/HCP//HCP_PARCELLATION/'
trainingdir = os.path.join(basedirname,'TRAININGDATA')
testingdir = os.path.join(basedirname,'TESTINGDATA')
validationdir = os.path.join(basedirname,'VALIDATIONSET')
outputdir=os.path.join(basedirname,'HCPmultimodal_180labels_corrected_labels/17_05_17_15_23/output') #--outpath in multimodal_infer.py

# define path to group data
dirname = os.path.join(basedirname,'Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210'
                                   '/MNINonLinear/fsaverage_LR32k/')
# common surface for all label files
surfname = os.path.join(dirname,'Q1-Q6_RelatedParcellation210.L.sphere.32k_fs_LR.surf.gii')

# group labels
labelname = os.path.join(dirname,'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LRtest.L.label.gii')

#template gifti - used to project output onto
templategiftiname= os.path.join(dirname,'Q1-Q6_RelatedParcellation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.L.func.gii')

#individual subject label file names should have form subjID+featuretype
subjlabelname='.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii'

# features - feature files should have form subjID+featuretype
featuretype = '.L.MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.32k_fs_LR.func.gii'

# correlation maps - files that summarise correlation of single subject features with that of the group average
# files should have form subjID+correlationtype
correlationtype = '.L.MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO_sigma_3_groupcorrelation.func.gii'

#average feature correlation map - correlations for all subjects averaged across groups
#allows training data to be filtered to only use training datasets who's features resemble that of the group average
average_feature_correlations = 'MEAN_groupcorrelation.32k_fs_LR.func.gii'

# name for N-D output file containg prediction segmentations for each of the N test subjects
outputname = 'HCPmultimodal_180labels_corrected_labels_17_05_17_15_23.func.gii'

# define training, validation and test directories and paths
TRAINING = os.path.join(trainingdir,'TRAININGlist')
TESTING = os.path.join(testingdir,'TESTINGlist')
VALIDATION = os.path.join(validationdir,'VALIDATIONlist')


training_paths = {'Fdir': os.path.normpath(trainingdir+ '/featuresets/'),
                  'Ldir': os.path.normpath(trainingdir+'/classifiedlabels/'),
                  'Odir': os.path.normpath(trainingdir+ '/featuresets/projected'),
                  'list': np.genfromtxt(TRAINING , dtype=str),
                  'abr': 'TRAINING'}

testing_paths = { 'Fdir': os.path.normpath(testingdir+ '/featuresets/'),
                  'Ldir': os.path.normpath(testingdir+'/classifiedlabels/'),
                  'Odir': os.path.normpath(testingdir+ '/featuresets/projected'),
                  'list': np.genfromtxt(TESTING , dtype=str),
                  'abr': 'TESTING'}

validation_paths = {'Fdir': os.path.normpath(validationdir + '/featuresets/'),
                    'Ldir': os.path.normpath(validationdir + '/classifiedlabels/'),
                    'Odir': os.path.normpath(validationdir + '/featuresets/projected'),
                    'list': np.genfromtxt(VALIDATION , dtype=str),
                    'abr': 'VALIDATION'}


# tuning parameters

# optionally define additional label regions on which to centre the sphere prior to projection
projection_centres=[192,262, 311, 331,240]  # regions 55b IFSa TGd  IP1 PGs p32pr

# define dimensions of projection plane
resampleH=240
resampleW=320

delta_d=2.*np.pi/(resampleW-1); # width of longitude bin if w not in  randw2:s
lons = (delta_d*np.indices((resampleW,1))[0,:,:]) #edges of longitude bins

usegrouplabels = False # use the group average labels for all subjects (projected onto each subjects feature space through registration as per Glasser et al, Nature 2016)
getFeatureCorrelations = False #useful when using group labels as it can be used to filter training data (see Glasser et al, Nature 2016)
normalise = True # necessary for old version

