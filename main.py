import os
from shutil import copy
import time
from tqdm import tqdm 

from logger import logger, error_logger

from process_mask.resample_and_crop_seg import run_cropping_and_resampling
from predict_skeleton.run_nnunet_inference import nnunet_predict_skeleton
from connect_skeleton.connect_and_label_predicted_cnt import connect_skeleton_mask
from extract_voreen_graph.extract_graph import run_graph_extraction
from extract_voreen_graph.generate_dataset import generate_vtp
from postprocess_graph.postprocess_connected_cnt_graphs import run_postprocessing
from compute_radius.compute_radius_along_cnt import run_radius_computation
from extract_features.compute_bifurcation_geometry import extract_bifurcation_geometry
from extract_features.compute_segment_geometry import extract_segment_geometry
from extract_features.fetal_pca_detection import run_fetal_detection

# import list of filenames for the pipeline
from configs import cow_mlt_seg_dir

# import image modality
from configs import modality

# import directories
from configs import cow_mlt_seg_dir

# import voreen workspace dirs/files
from configs import voreen_tool_path

# import pipeline bool args
from configs import crop_and_resample_mlt_mask, do_nnunet_prediction, connect_skeleton, extract_graph
from configs import postprocess_graph, compute_radius_along_graph, extract_features

# import pipeline parameters
from configs import do_mask_corrections, min_overall_segment_size, threshold_for_component_removal, max_path_length_mask
from configs import max_path_length, remove_floating_segments, n_jobs
from configs import radius_attribute, median_p1, median_a1, median_c7, margin_from_cow
from configs import dist_angle, nr_pts_angle_average, nr_edges_rad_avg, fixed_dist_radius
from configs import fetal_percentile, fetal_factor


################## Directories ##################
work_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where this main.py file is located

# folder where all the algorithm outputs are stored
media_dir = os.path.join(work_dir, 'media/topcow_preds/UZH/') 
if not os.path.exists(media_dir):
    os.makedirs(media_dir)

# define all the other directories
mlt_mask_dir = os.path.join(media_dir, 'mlt_mask_resampled')  # contains resampled and cropped mlt masks (with trailing _0000)
nnunet_dir = os.path.join(work_dir, 'predict_skeleton') # nnUnet parent directory
predicted_skeleton_dir = os.path.join(media_dir, 'predicted_skeletons') # contains nnUnet binary predicted skeletons
voreen_output_dir = os.path.join(media_dir, 'voreen_output') # voreen raw output (centerline graphs)

intermediate_dir = os.path.join(media_dir, 'intermediate') # non-smoothed nodes and graphs (for debugging)

output_dir = os.path.join(media_dir, 'output_final') # output directory for final cnt graphs and additional files
graph_dir = os.path.join(output_dir, 'cow_graphs') # postprocessed cow graphs
node_dir = os.path.join(output_dir, 'cow_nodes') # cow node files
variant_dir = os.path.join(output_dir, 'cow_variants') # cnt cow files 
feature_dir = os.path.join(output_dir, 'cow_features') # cow feature files
mesh_dir = os.path.join(output_dir, 'cow_meshes') # cow mesh files
skeleton_dir = os.path.join(output_dir, 'cow_skeletons') # cow connected skeleton files

voreen_dir = os.path.join(work_dir, 'extract_voreen_graph')
voreen_workspace_file = os.path.join(voreen_dir, "feature-vesselgraphextraction_customized_command_line.vws")
voreen_tmp_dir = os.path.join(voreen_dir, "voreen_tmpdir")
voreen_cache_dir = os.path.join(work_dir, "voreen_cachedir")

################## Pipeline ##################
def run_pipeline_for_single_case(cow_seg_filename): 
    """
    Run pipeline for single case including:
    - Cropping and resampling of segmentation mask
    - nnUNet prediction for binary skeleton
    - Connecting and labeling of predicted skeleton
    - Graph extraction using voreen
    - Graph postprocessing
    - Radius computation
    - Feature extraction

    Args:
        cow_seg_filename (str): The filename of the cow segmentation mask.

    Returns:
        float: Runtime in seconds for this case
    """
    # Setup start time
    id_start = time.perf_counter()

    # Setup tqdm progress bar for tracking pipeline steps
    steps = []
    if crop_and_resample_mlt_mask:
        steps.append("Cropping and resampling")
    if do_nnunet_prediction:
        steps.append("nnUNet prediction")
    if connect_skeleton:
        steps.append("Connecting skeleton")
    if extract_graph:
        steps.append("Graph extraction")
    if postprocess_graph:
        steps.append("Graph postprocessing")
    if compute_radius_along_graph:
        steps.append("Radius computation")
    if extract_features:
        steps.append("Feature extraction")
    if not steps:
        steps = ["No steps"] # placeholder, if there are no steps
    
    # Create progress bar for this case
    progress_bar = tqdm(steps, desc=f"Pipeline for {cow_seg_filename}")
    
    # define file names and paths
    output_filname_without_suffix = cow_seg_filename.replace('.nii.gz', '')
    # The filenames of the resampled mlt masks contain the trailing _0000 as is nnUNet's convention!
    mlt_mask_filepath = os.path.join(mlt_mask_dir, f'{output_filname_without_suffix}_0000.nii.gz')
    connected_maskname = f'{output_filname_without_suffix}_connected_0000.nii.gz'
    
    # Voreen output folder is named after the input segmentation file
    voreen_output_dir_case = os.path.join(voreen_output_dir, output_filname_without_suffix)

    # Initialize variables that might be referenced before assignment
    skeleton_filepath = None
    input_file_graph_extraction = None
    voreen_graph_file = None
    mesh_source_file = None
    postprocessed_graph_file = None
    node_dict_file = None

    assert cow_seg_filename.endswith('.nii.gz'), 'Filename must end with .nii.gz'
    logger.info(f'\nProcessing {cow_seg_filename}...')

    for step in progress_bar:

        step_start_time = time.perf_counter()
        
        # 1: Cropping and resampling cow segmentation
        if step == 'Cropping and resampling':
            logger.info(f'\nCropping and resampling {cow_seg_filename}...')
            # NOTE: The filenames of the resampled mlt masks contain the trailing _0000 as is nnUNet's convention!
            cow_seg_filepath = os.path.join(cow_mlt_seg_dir, cow_seg_filename)
            run_cropping_and_resampling(cow_seg_filepath, 
                                        mlt_mask_dir,
                                        correct_mask = do_mask_corrections,
                                        min_segment_size=min_overall_segment_size, 
                                        threshold_for_component_removal=threshold_for_component_removal, 
                                        max_path_length=max_path_length_mask,
                                        crop_buffer=1, # buffer for cropping to foreground mask
                                        resamp_spacing=[0.25, 0.25, 0.25] # spacing for resampling
                                        )

        # 2: Run nnUNet prediction
        elif step == 'nnUNet prediction':
            logger.info('\nRunning nnUNet prediction...')
            if os.path.exists(os.path.join(mlt_mask_dir, connected_maskname)):
                logger.info(f'Using resampled and CONNECTED mlt mask {connected_maskname} for skeleton prediction...')
                mlt_mask_filepath_for_skel_pred = os.path.join(mlt_mask_dir, connected_maskname)
            else:
                logger.info(f'Using resampled mlt mask {mlt_mask_filepath} for skeleton prediction...')
                mlt_mask_filepath_for_skel_pred = mlt_mask_filepath
            
            skeleton_filepath = nnunet_predict_skeleton(mlt_mask_filepath_for_skel_pred, 
                                                        nnunet_dir, 
                                                        predicted_skeleton_dir,
                                                        dataset_id='113', # nnUNet dataset id
                                                        folds='all', # Ensembling all folds. Change if you only want to use a single fold (0...4)
                                                        trainer='nnUNetTrainer',
                                                        configuration='3d_fullres',
                                                        plans='nnUNetPlans')
    
        # 3: Connecting and labeling predicted binary skeleton
        elif step == 'Connecting skeleton':
            logger.info('\nConnecting and labeling predicted binary skeleton...')
            if os.path.exists(os.path.join(mlt_mask_dir, connected_maskname)) and do_mask_corrections:
                logger.info(f'Using resampled and CONNECTED mlt mask {connected_maskname} for skeleton refinement...')
                mlt_mask_filepath_for_skel_ref = os.path.join(mlt_mask_dir, connected_maskname)
            else:
                logger.info(f'Using resampled mlt mask {mlt_mask_filepath} for skeleton refinement...')
                mlt_mask_filepath_for_skel_ref = mlt_mask_filepath
            if skeleton_filepath is None:
                skeleton_filepath = os.path.join(predicted_skeleton_dir, cow_seg_filename)
            
            input_file_graph_extraction = connect_skeleton_mask(mlt_mask_filepath_for_skel_ref, 
                                                                skeleton_filepath,
                                                                skeleton_dir,
                                                                intermediate_dir,
                                                                max_path_length,
                                                                remove_floating_segments,
                                                                n_jobs)

        # 4: Extracting graph using voreen
        elif step == 'Graph extraction':
            logger.info('\nExtracting graph from predicted and connected CoW skeletons using the Voreen tool...')
            if input_file_graph_extraction is None:
                input_file_graph_extraction = os.path.join(skeleton_dir, cow_seg_filename)

            if os.path.exists(input_file_graph_extraction.replace('.nii.gz', '_noFloating.nii.gz')) and remove_floating_segments:
                logger.info(f'Using connected skeleton with FLOATING SEGMENTS REMOVED for graph extraction...')
                input_file_graph_extraction = input_file_graph_extraction.replace('.nii.gz', '_noFloating.nii.gz')
            
            if not os.path.exists(voreen_output_dir_case):
                os.makedirs(voreen_output_dir_case)
            
            run_graph_extraction(input_file_graph_extraction, 
                                 voreen_output_dir_case, 
                                 voreen_tool_path,
                                 voreen_tmp_dir,
                                 voreen_cache_dir,
                                 voreen_workspace_file,
                                 bulge_size=0.5, # Voreen bulge size (any small value will do, not very important for skeletons)
                                 remove_connections=True # Preprocessing: remove invalid connections on the mask level
                                 )
            
            # Generating .vtp datasets of centerlinegraphs
            logger.info('\nGenerating .vtp dataset of centerline graph...')
            voreen_graph_file, mesh_source_file = generate_vtp(voreen_output_dir_case, 
                                                               mlt_mask_filepath,
                                                               do_mesh_smoothing=True, # Smooth the mesh
                                                               pass_band=0.25, # pass band for Taubin smoothing
                                                               add_mesh_labels=True # adding class labels to mesh
                                                               )

        # 5: Postprocessing graph
        elif step == 'Graph postprocessing':
            if voreen_graph_file is None:
                voreen_graph_file = os.path.join(voreen_output_dir_case, f'{output_filname_without_suffix}_multi_cnt_graph.vtp')
            # copy mesh to output folder
            if mesh_source_file is None:
                mesh_source_file = os.path.join(voreen_output_dir_case, f'{output_filname_without_suffix}_multi_mesh.vtp')
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)
            mesh_dest_file = os.path.join(mesh_dir, f'{output_filname_without_suffix}.vtp')
            copy(mesh_source_file, mesh_dest_file)

            logger.info('\nPostprocessing extracted graph...')
            postprocessed_graph_file, node_dict_file = run_postprocessing(voreen_graph_file, 
                                                                          output_filname_without_suffix,
                                                                          intermediate_dir,
                                                                          graph_dir,
                                                                          variant_dir,
                                                                          node_dir,
                                                                          window_size_smoothing=5 # window size for moving average filter
                                                                          )

        # 6: compute the radius for each edge in the graph and
        elif step == 'Radius computation':
            if postprocessed_graph_file is None:
                postprocessed_graph_file = os.path.join(graph_dir, f'{output_filname_without_suffix}.vtp')
            if node_dict_file is None:
                node_dict_file = os.path.join(node_dir, f'{output_filname_without_suffix}.json')

            logger.info('\nComputing radius along graph...')
            run_radius_computation(postprocessed_graph_file, 
                                   mlt_mask_filepath, 
                                   node_dict_file, 
                                   n_jobs,
                                   use_meanpoint=False, # if True, the mean of the surface mesh points at a given cross section is used instead of the actual centerline point
                                   use_ce_rad_cap=False # if True, CE radius near bifurcations is capped at 1.33 x MIS radius
                                   )

        # 7: Extract features
        elif step == 'Feature extraction':
            if postprocessed_graph_file is None:
                postprocessed_graph_file = os.path.join(graph_dir, f'{output_filname_without_suffix}.vtp')

            logger.info('\nExtracting features from graph...')
            logger.info('1) Segment features:')
            extract_segment_geometry(postprocessed_graph_file, 
                                    variant_dir,
                                    node_dir,
                                    feature_dir,
                                    modality, 
                                    radius_attribute, 
                                    median_p1, 
                                    median_a1, 
                                    median_c7, 
                                    margin_from_cow,
                                    smooth_curve=True, # If True, the (splipy) curve is smoothed before geometry computation.
                                    factor_num_points=2, # Sampling factor_num_points x actual number of segment centerline points
                                    threshold_nan_radius=0.5,  # threshold for the fraction of NaN radius values along a segment path to consider the radius computation failed
                                    threshold_broken_segment_removal=0.66  # threshold for the fraction of the median length of a broken segment (A1 or P1) to remove it from the feature dict
                                    )
            
            logger.info('2) Bifurcation features:')
            # NOTE: We're using spheres around bifurcation points for geometry computation
            extract_bifurcation_geometry(postprocessed_graph_file, 
                                         variant_dir,
                                         node_dir,
                                         feature_dir,
                                         radius_attribute,
                                         dist_angle=dist_angle,
                                         angle_average=nr_pts_angle_average,
                                         radius_average=nr_edges_rad_avg,
                                         use_fixed_dist_radius=fixed_dist_radius
                                         )
    
            logger.info('3) Fetal PCA detection:')
            run_fetal_detection(postprocessed_graph_file, 
                                variant_dir, 
                                feature_dir, 
                                percentile=fetal_percentile, 
                                factor=fetal_factor
                                )
            
        elif step == "No steps":
            logger.info('No steps for processing the files...')
            break

        # Calculate elapsed time for this step
        step_end_time = time.perf_counter()
        step_elapsed_time = step_end_time - step_start_time
        
        logger.info(f'Finished step: {step} for {cow_seg_filename} in {step_elapsed_time:.2f} seconds!')
    
    id_end = time.perf_counter()
    runtime = id_end - id_start
    logger.info(f"\nFinished pipeline! Took {runtime:.2f} seconds")

    return runtime



if __name__ == "__main__":
    import numpy as np
    import traceback
    list_of_errors = []
    list_runtimes = []
    
    # get list of input files
    cow_seg_files = os.listdir(cow_mlt_seg_dir)
    cow_seg_files.sort()  # sort files to ensure consistent order
    ids = ['ct_180']
    cow_seg_files = [f for f in cow_seg_files if '_ct_' in f]
    logger.info(f'\nProcessing {len(cow_seg_files)} cases...')
    logger.debug(f'Cases: {cow_seg_files}')

    # log pipeline steps and parameters
    logger.info(f'\nPipeline is running with the following steps:'
                 f'\n- Crop and resample mlt mask: {crop_and_resample_mlt_mask}'
                 f'\n\t-args: do_mask_corrections={do_mask_corrections}, min_overall_segment_size={min_overall_segment_size}, threshold_for_component_removal={threshold_for_component_removal}, max_path_length_mask={max_path_length_mask}'
                 f'\n- nnUNet prediction: {do_nnunet_prediction}'
                 f'\n- Connect skeleton: {connect_skeleton}'
                 f'\n\t-args: max_path_length={max_path_length}, n_jobs={n_jobs}'
                 f'\n- Extract graph: {extract_graph}'
                 f'\n- Postprocess graph: {postprocess_graph}'
                 f'\n- Compute radius along graph: {compute_radius_along_graph}'
                 f'\n- Extract features: {extract_features}'
                 f'\n\t- segment features args: radius_attribute={radius_attribute}, median_p1={median_p1}, median_a1={median_a1}, median_c7={median_c7}, margin_from_cow={margin_from_cow}'
                 f'\n\t- bifurcation features args: dist_angle={dist_angle}, nr_pts_angle_average={nr_pts_angle_average}, nr_edges_rad_avg={nr_edges_rad_avg}, fixed_dist_radius={fixed_dist_radius}'
                 f'\n\t- fetal pca args: fetal_percentile={fetal_percentile}, fetal_factor={fetal_factor}'
                )
    
    # log paths
    logger.debug(f'\nwork_dir: {work_dir}'
                 f'\nmedia_dir: {media_dir}'
                 f'\nmlt_mask_dir: {mlt_mask_dir}'
                 f'\nnnunet_dir: {nnunet_dir}'
                 f'\npredicted_skeleton_dir: {predicted_skeleton_dir}'
                 f'\nvoreen_output_dir: {voreen_output_dir}'
                 f'\nintermediate_dir: {intermediate_dir}'
                 f'\noutput_dir: {output_dir}'
                 f'\ngraph_dir: {graph_dir}'
                 f'\nnode_dir: {node_dir}'
                 f'\nvariant_dir: {variant_dir}'
                 f'\nfeature_dir: {feature_dir}'
                 f'\nmesh_dir: {mesh_dir}'
                 f'\nskeleton_dir: {skeleton_dir}'
                 f'\nvoreen_workspace_file: {voreen_workspace_file}'
                 f'\nvoreen_tmp_dir: {voreen_tmp_dir}'
                 f'\nvoreen_cache_dir: {voreen_cache_dir}'
                 f'\nvoreen_tool_path: {voreen_tool_path}'
                 )

    ###### Catching errors ########'
    # setup error logger
    error_logger.error(f"Error log created at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    error_logger.error("="*50 + "\n")
    
    # Loop through input files
    for cow_seg_input in cow_seg_files:
        try:
            runtime = run_pipeline_for_single_case(cow_seg_input)
            list_runtimes.append(runtime)
        except Exception as e:
            error_msg = f"Error processing {cow_seg_input}: {e}"
            error_traceback = traceback.format_exc()
            logger.info(f"Error processing {cow_seg_input}: {e}")
            logger.error(f"{error_msg}\n{error_traceback}")
            
            # Log error message and traceback separately to error log file
            error_logger.error(f"{error_msg}\n{error_traceback}\n")
            
            # append id to list of errors
            list_of_errors.append(cow_seg_input)

    logger.info(f'\nDone with all cases')
    logger.info(f'Errors occurred in the following cases:')
    logger.info(list_of_errors)
    logger.info(f'\nDone with all cases')

    mean_runtime = np.mean(list_runtimes)
    std_runtime = np.std(list_runtimes)
    logger.info(f'Mean runtime: {mean_runtime:.3f} seconds')
    logger.info(f'Standard deviation of runtimes: {std_runtime:.3f} seconds')


    

    
    
