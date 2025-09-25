import os

################## Directories ##################
# contains the original CoW multiclass masks
cow_mlt_seg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media/cow_mlt_seg') 

# folder where all the algorithm outputs are stored. Take the parent directory of cow_mlt_seg_dir
media_dir = os.path.dirname(cow_mlt_seg_dir)

# path to voreen tool binaries
voreen_tool_path = "/home/fmusio/projects/voreen-src-unix-nightly/build/bin" 

# folder containing model weights for nnUNet prediction
nnunet_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predict_skeleton/nnUNet_results') 

# export the nnUNet_results path as an environment variable
os.environ["nnUNet_results"] = nnunet_results

################## Modality ##################
modality = 'mr' # 'ct' or 'mr'

################## Main Pipeline ##################
# Bool args for main pipeline
# Set to True to run the corresponding step
crop_and_resample_mlt_mask = True
do_nnunet_prediction = True
connect_skeleton = True
extract_graph = True
postprocess_graph = True
compute_radius_along_graph = True
extract_features = True

### Args Step 1: Cropping, mask correction and resampling
do_mask_corrections = True # perform mask corrections or not
# NOTE: For skipping individual mask correction steps, set variables below to None
min_overall_segment_size = 35 # minimum overall segment size (#voxels) to keep after cropping (before resampling)
threshold_for_component_removal = 15 # threshold for removing small disconnected components (#voxels) (before resampling)
max_path_length_mask = None # maximum path length for connecting segments in the mask (after resampling)

### Args Step 3: Connect skeleton
max_path_length = 25 # maximum path length for connecting skeletons
remove_floating_segments = True # whether to remove floating segments (after connecting skeleton)
n_jobs = 12 # number of jobs for parallel processing

### Args Step 7: Extract features
radius_attribute = 'ce_radius' # attribute name for radius computation
## segment features
median_p1=7.2 # if pcom absent, define p1 as segment with constant length
median_a1=15.6 # if acom absent, define a1 as segment with constant length
median_c7=7.1 # if pcom absent, define c7 as segment with constant length
margin_from_cow=10 # marginal length from cow for BA/P2/C6/MCA/A2/3rd-A2 segments
## bifurcation features
dist_angle = 1 # distance [mm] to sample points around bifurcation for angle computation
nr_pts_angle_average = 1 # number of points to average for angle computation
nr_edges_rad_avg = 3 # number of edges to average for radius computation
fixed_dist_radius = False # whether to use fixed distance for radius computation or dynamic distance (max dist from bifurcation to boundary points)
## fetal pca
fetal_percentile = 25 # percentile of radius values at which to compare P1 and Pcom
fetal_factor = 1.05 # factor by which Pcom radius has to be bigger than P1 radius for fetal variant to be present 
