import sys
import os
# add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import nibabel as nib
import copy
from joblib import Parallel, delayed
from scipy.ndimage import label as nd_label
from connect_skeleton.astar_utilities import astar_3d, find_closest_components, find_closest_component_ngh
from utils.utils_connecting_skeleton import *

from logger import logger

def connect_broken_segment(label, labels, mlt_skeleton, mlt_mask_arr, max_path_length, keep_largest_only=False):
    """
    Connect broken segments in a skeletonized image for a specific label.
    This function attempts to connect disconnected components of a labeled segment
    in a skeleton image using A* pathfinding. Special handling is applied for
    certain anatomical structures like ACA (labels 11, 12) and Acom (label 10).
    Parameters
    ----------
    label : int
        The label of the segment to process.
    labels : list or array
        Collection of all labels present in the image.
    mlt_skeleton : ndarray
        Labeled skeleton image where each segment has a unique label value.
    mlt_mask_arr : ndarray
        Mask array where regions are labeled.
    max_path_length : int
        Maximum allowed path length for connections. Paths exceeding this length
        will not be connected to prevent erroneous connections.
    keep_largest_only : bool, optional
        If True, only the largest connected component of the segment will be retained.
    Returns
    -------
    ndarray
        Connected skeleton mask for the specified segment with the original label value.
        Only the largest connected component is retained.
    Notes
    -----
    - For ACA labels (11 and 12), processing includes Acom (label 10) to prevent loops
      at Acom bifurcations.
    - Acom (label 10) is skipped when processed directly as it's included with ACA.
    - The function uses A* pathfinding to connect broken components and stops if the
      path length exceeds max_path_length.
    """

    if 10 in labels and (label == 11 or label == 12):
        # For ACA labels 11 and 12, we need to treat them together with Acom
        # Do this to prevent loops at Acom bifurcations! 
        segment_skeleton_mask = ((mlt_skeleton == label) | (mlt_skeleton == 10)).astype(np.uint8)
        segment_mask = ((mlt_mask_arr == label) | (mlt_mask_arr == 10)).astype(np.uint8)
    elif label == 10:
        # Skip Acom since it's already included in the previous step
        return np.zeros_like(mlt_skeleton)

    else:
        segment_skeleton_mask = (mlt_skeleton == label).astype(np.uint8)
        segment_mask = (mlt_mask_arr == label).astype(np.uint8)

    closest_pair, num_features = find_closest_components(segment_skeleton_mask)
    if closest_pair == None:
        logger.info(f'Segment {label} is already connected')
    else:
        logger.info(f'Segment {label} is broken with {num_features} components. Connecting...')
        while closest_pair != None:
            path = astar_3d(tuple(closest_pair[0]), tuple(closest_pair[1]), segment_mask)
            logger.debug(f'\tA* path: {path} with length {len(path)}')
            if len(path) > max_path_length:
                logger.warning(f'\tALERT: path length is {len(path)}. Not connecting...')
                break
            else:
                logger.info(f'\tConnecting path of length {len(path)}...')
                for point in path:
                    segment_skeleton_mask[point] = 1
                closest_pair, num_features = find_closest_components(segment_skeleton_mask)
    
    segment_skeleton_mask *= label

    # Keeping only largest component does not work for imperfect predictions anymore
    if keep_largest_only:
        segment_skeleton_mask = keep_largest_component(segment_skeleton_mask)

    return segment_skeleton_mask

def connect_neighboring_segments(ngh_segments, nghs_skeleton, connected_skeleton, mlt_mask_arr, 
                                 allowed_neighbors, max_path_length=30):
    """
    Connects neighboring segments in a skeletonized image based on allowed_neighbors.
    Returns a new array with the connections only.
    
    Parameters
    ----------
    ngh_segments : tuple
        A tuple containing two labels representing the segments to connect.
    nghs_skeleton : list
        List of tuples representing the neighboring segments in the skeleton.
    connected_skeleton : ndarray
        Labeled skeleton image.
    mlt_mask_arr : ndarray
        Mask array.
    allowed_neighbors : dict
        Dictionary mapping each label to its allowed neighboring labels.
    max_path_length : int, optional
        Maximum allowed path length for connections. Paths exceeding this length
        will not be connected to prevent erroneous connections. Default is 30.
    
    Returns
    -------
    ndarray
        A new array containing only the connecting paths between the segments.
    """
    connections = np.zeros_like(connected_skeleton)
    label, ngh = ngh_segments[0], ngh_segments[1]
    if ngh_segments in nghs_skeleton:
        logger.info(f'Neighboring segments {ngh_segments} already connected')
   
    elif ngh_segments not in nghs_skeleton and ngh in allowed_neighbors[label]:
        logger.info(f'Neighboring segments {ngh_segments} NOT connected')
        logger.info(f'\tConnecting {label} with neighbor {ngh}')
        segment_skeleton_mask = (connected_skeleton == label).astype(np.uint8)
        segment_skeleton_mask_ngh = (connected_skeleton == ngh).astype(np.uint8)
        ngh_segment_mask = ((mlt_mask_arr == label) | (mlt_mask_arr == ngh)).astype(np.uint8)
        closest_pair = find_closest_component_ngh(segment_skeleton_mask, segment_skeleton_mask_ngh)
        
        path = astar_3d(tuple(closest_pair[0]), tuple(closest_pair[1]), ngh_segment_mask)
        logger.debug(f'\tA* path: {path} with length {len(path)}')
        if len(path) > max_path_length:
            logger.warning(f'\tALERT: path length is {len(path)}. Not connecting...')
        else:
            for point in path:
                connections[point] = 1
            logger.info(f'\tConnected segments {label}-{ngh} with path length {len(path)}')
    return connections

def remove_floating_segments_func(connected_skeleton):
    """
    Removes floating segments from the connected skeleton. We only consider Pcoms, Acom and 3rd-A2 here.
    Floating segments are defined as those that are not connected to their corresponding neighbors.
    
    Parameters
    ----------
    connected_skeleton : ndarray
        Labeled and connected skeleton image.
    
    Returns
    -------
    ndarray
        Skeleton image with floating segments removed.
    """
    labels = np.unique(connected_skeleton)
    labels = labels[labels != 0]  # Exclude background
    labels_removed = []
    unique_neighbors = compute_all_unique_neighbors(connected_skeleton)
    if 8 in labels:
        if not ((2, 8) in unique_neighbors or (4, 8) in unique_neighbors):
            logger.warning('ALERT: Removing floating Pcom (8)!')
            connected_skeleton[connected_skeleton == 8] = 0
            labels_removed.append(8)
    if 9 in labels:
        if not ((3, 9) in unique_neighbors or (6, 9) in unique_neighbors):
            logger.warning('ALERT: Removing floating Pcom (9)!')
            connected_skeleton[connected_skeleton == 9] = 0
            labels_removed.append(9)
    if 10 in labels:
        if not ((10, 11) in unique_neighbors or (10, 12) in unique_neighbors):
            logger.warning('ALERT: Removing floating Acom (10)!')
            connected_skeleton[connected_skeleton == 10] = 0
            labels_removed.append(10)
    if 15 in labels:
        if not ((10, 15) in unique_neighbors):
            logger.warning('ALERT: Removing floating 3rd-A2 (15)!')
            connected_skeleton[connected_skeleton == 15] = 0
            labels_removed.append(15)

    return connected_skeleton, labels_removed


def connect_skeleton_mask(mlt_mask_filepath: str, skeleton_filepath: str, connected_skeleton_dir: str, intermediate_dir: str,
                          max_path_length: int, remove_floating_segments: bool = False, n_jobs: int = 12):
    """
    Connects fragmented skeleton and assigns anatomical labels based on a multi-label mask.
    
    This function takes a predicted skeleton and a multi-label mask, connects broken segments
    within each label, connects neighboring segments according to anatomical rules, and ensures
    a fully connected vessel network. It uses A* pathfinding to create paths between disconnected
    components, prioritizing paths that stay within the vessel mask.
    
    Parameters:
    -----------
    mlt_mask_filepath : str
        Path to the multi-label mask file (.nii.gz)
    skeleton_filepath : str
        Path to the binary skeleton file (.nii.gz)
    connected_skeleton_dir : str
        Directory to save the connected skeleton
    intermediate_dir : str
        Directory to save intermediate files in case of floating segment removal
    max_path_length : int
        Maximum length of paths to connect components
    remove_floating_segments : bool, optional
        Whether to remove floating segments after connecting (default: False)
    n_jobs : int, optional
        Number of parallel jobs to run (default: 12)
        
    Returns:
    --------
    str
        Path to the saved connected skeleton file
    """

    logger.info('Connecting skeleton mask with args:'
                f'\n\t- mlt mask input={mlt_mask_filepath}'
                f'\n\t- skeleton input={skeleton_filepath}'
                f'\n\t- connected skeleton output={connected_skeleton_dir}'
                f'\n\t- max_path_length={max_path_length}'
                f'\n\t- remove_floating_segments={remove_floating_segments}'
                f'\n\t- n_jobs={n_jobs}')

    # load masks
    try:
        logger.info(f"Loading mask file {mlt_mask_filepath} and skeleton file {skeleton_filepath}")
        mlt_mask = nib.load(mlt_mask_filepath)
        skeleton = nib.load(skeleton_filepath)
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return None
    
    mlt_mask_arr = mlt_mask.get_fdata().astype(np.uint8)
    skeleton_arr = skeleton.get_fdata().astype(np.uint8)

    # get unique labels (without background)
    labels = np.unique(mlt_mask_arr)
    labels = labels[labels != 0]

    # label skeleton
    logger.info(f'Assigning mask labels {labels} to skeleton')
    # TODO: Check if correcting acom is needed!
    mlt_skeleton = assign_labels_to_skeleton(mlt_mask_arr, skeleton_arr, correct_acom=False)

    # insert missing labels
    mlt_skeleton = insert_missing_labels(mlt_skeleton, mlt_mask_arr, labels)

    connected_skeleton = np.zeros_like(skeleton_arr)
    
    # 1: connect broken segments
    logger.info(f'Connecting broken segments...')
    # reorder labels to connect ACAs before Acom
    if 10 in labels:
        labels = np.concatenate([labels[labels != 10], [10]])
    
    for label in labels:
        connected_segment = connect_broken_segment(label, labels, mlt_skeleton, mlt_mask_arr, max_path_length, keep_largest_only=False)
        if connected_segment.sum() == 0:
            pass
        else:
            mlt_skeleton[connected_segment == label] = 0
            connected_segment = assign_labels_to_skeleton(mlt_mask_arr, connected_segment, correct_acom=True, label=label)
            # Add the newly connected segment to mlt_skeleton
            mlt_skeleton += connected_segment
            connected_skeleton += connected_segment

    connected_skeleton = assign_labels_to_skeleton(mlt_mask_arr, connected_skeleton)

    # remove isolated Acom voxels 
    if 10 in labels:
        relabel_isolated_voxels(connected_skeleton, label=10)
    
    # 2: connect neighboring components
    logger.info(f'Connecting neighboring segments...')
    allowed_neighbors = {1: {2,3}, 2: {1,8}, 3: {1, 9}, 4: {5, 8, 11}, 5: {4}, 6: {7, 9, 12}, 7: {6}, 8: {2, 4}, 9: {3, 6}, 10: {11, 12, 15}, 11: {4, 10}, 12: {6, 10}, 15: {10}}
    nghs_skeleton = compute_all_unique_neighbors(connected_skeleton)
    logger.debug(f'Connected skeleton has unique neighbors: {nghs_skeleton}')
    nghs_mask = compute_all_unique_neighbors(mlt_mask_arr)
    # add obligatory neighbors for imperfect mask predictions
    nghs_mask = add_obligatory_neighbors(nghs_mask)
    logger.debug(f'Multi-label mask has unique neighbors: {nghs_mask}')
    results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(connect_neighboring_segments)(ngh_segments, nghs_skeleton, connected_skeleton, mlt_mask_arr, allowed_neighbors, max_path_length) 
            for ngh_segments in nghs_mask)
    for result in results:
        connected_skeleton += result

    
    # Getting num_features
    binary_skel = (connected_skeleton > 0).astype(np.uint8)
    _, num_features = nd_label(binary_skel, structure=np.ones((3, 3, 3)))

    connected_skeleton = assign_labels_to_skeleton(mlt_mask_arr, connected_skeleton)

    # remove isolated Acom voxels 
    if 10 in labels:
        relabel_isolated_voxels(connected_skeleton, label=10)

    logger.info(f'Connecting skeleton done! Skeleton has {num_features} components')
    segment_skeleton_img = nib.Nifti1Image(connected_skeleton, skeleton.affine, skeleton.header)
    connected_skeleton_savepath = os.path.join(connected_skeleton_dir, os.path.basename(skeleton_filepath))
    if not os.path.exists(connected_skeleton_dir):
        logger.debug(f'Creating directory: {connected_skeleton_dir}')
        os.makedirs(connected_skeleton_dir)
    logger.info(f'Saving connected skeleton to {connected_skeleton_savepath}')
    nib.save(segment_skeleton_img, connected_skeleton_savepath)

    if remove_floating_segments:
        connected_skeleton_floating_removed = copy.deepcopy(connected_skeleton)
        connected_skeleton_floating_removed, labels_removed = remove_floating_segments_func(connected_skeleton_floating_removed)
        if not np.array_equal(connected_skeleton, connected_skeleton_floating_removed):
            logger.warning(f'ALERT: Floating segments removed! Labels removed: {labels_removed}')
            # Getting num_features
            binary_skel = (connected_skeleton_floating_removed > 0).astype(np.uint8)
            _, num_features = nd_label(binary_skel, structure=np.ones((3, 3, 3)))
            logger.info(f'After removing floating segments, skeleton has {num_features} components')
            segment_skeleton_img = nib.Nifti1Image(connected_skeleton_floating_removed, skeleton.affine, skeleton.header)
            connected_skeleton_nofloat_savepath = connected_skeleton_savepath.replace('.nii.gz', '_noFloating.nii.gz')
            logger.info(f'Saving connected skeleton with floating segments removed to {connected_skeleton_nofloat_savepath}')
            nib.save(segment_skeleton_img, connected_skeleton_nofloat_savepath)
            # Also save .txt file in intermediate dir with removed labels
            removed_labels_dir = os.path.join(intermediate_dir, 'cow_removed_labels')
            if not os.path.exists(removed_labels_dir):
                os.makedirs(removed_labels_dir)
            removed_labels_savepath = os.path.join(removed_labels_dir, os.path.basename(skeleton_filepath).replace('.nii.gz', f'.txt'))
            with open(removed_labels_savepath, 'w') as f:
                f.writelines([f"removed labels: {labels_removed}\n"])

    return connected_skeleton_savepath

    

