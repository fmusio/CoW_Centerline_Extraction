import sys
import os
# add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import nibabel as nib
import heapq
from scipy import ndimage
import nibabel.processing as nip
from connect_skeleton.astar_utilities import get_neighbors, find_closest_components
from utils.utils_connecting_skeleton import keep_largest_component

from logger import logger

def resample_nib(img, voxel_spacing=[0.25, 0.25, 0.25], order=0):
    """
    Resamples the nifti image (mask) from its original spacing to another specified spacing.
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    """

    logger.debug(f"Resampling image from {img.header.get_zooms()} to {voxel_spacing}")

    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()

    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=0)
    
    return new_img

def crop_images(mask, buffer=1):
    """
    Crop the mask to the bounding box of the foreground region including a buffer.

    Parameters:
    ----------
    mask: nibabel image, resampled mlt mask
    buffer: int, buffer to add to the bounding box

    Returns:
    ----------
    mask_cropped: nibabel image, cropped mlt mask
    """

    logger.debug(f"Cropping mask with buffer={buffer}")

    nii_array = mask.get_fdata()
    
    # Identify the foreground labels (assuming non-zero values represent foreground)
    foreground_indices = np.where(nii_array > 0)

    if len(foreground_indices[0]) == 0:
        logger.error("No foreground pixels found in mask!")
        raise ValueError("No foreground pixels found in mask!")

    # Determine the bounding box for the foreground region
    min_x, max_x = foreground_indices[0].min(), foreground_indices[0].max()
    min_y, max_y = foreground_indices[1].min(), foreground_indices[1].max()
    min_z, max_z = foreground_indices[2].min(), foreground_indices[2].max()
    
    # Add buffer to the bounding box and ensure boundaries are within image dimensions
    min_x = max(0, min_x - buffer)
    max_x = min(nii_array.shape[0] - 1, max_x + buffer)
    min_y = max(0, min_y - buffer)
    max_y = min(nii_array.shape[1] - 1, max_y + buffer)
    min_z = max(0, min_z - buffer)
    max_z = min(nii_array.shape[2] - 1, max_z + buffer)

    crop = (
        slice(min_x, max_x + 1),
        slice(min_y, max_y + 1),
        slice(min_z, max_z + 1),
    )

    logger.debug(f"Cropped dimensions: x={min_x}-{max_x}, y={min_y}-{max_y}, z={min_z}-{max_z}")

    return mask.slicer[crop]

def heuristic_cost(current, goal):
    """
    Heuristic cost function for A* algorithm is just the Euclidean distance
    
    Parameters:
    -----------
    current (tuple): Current point (x, y, z).
    goal (tuple): Goal point (x, y, z).
    
    Returns:
    --------
    float: Heuristic cost.
    """
    euclidean_distance = np.linalg.norm(np.array(current) - np.array(goal))
    cost = euclidean_distance
    logger.debug(f'Heuristic cost from {current} to {goal}: {cost}')
    return cost

def astar_3d(start, goal, segment_mask):
    """
    A* pathfinding algorithm for 3D grids.
    This implementation finds the optimal path between two points in a 3D mask segment
    The algorithm uses:
    - A heuristic that is a combination of minimum Euclidean distance
    - A priority queue (heap) to efficiently select the most promising path
    - A full 26-neighborhood connectivity for smooth 3D paths
    
    Parameters:
    -----------
    start (tuple): Start point (x, y, z).
    goal (tuple): Goal point (x, y, z).
    segment_mask (np.ndarray): Binary mask where non-zero values represent the object.
    
    Returns:
    --------
    list: Path from start to goal.
    """

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    cost = heuristic_cost(start, goal)
    f_score = {start: cost}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        neighbors = get_neighbors(current, segment_mask.shape)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_cost(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

def connect_broken_mask(label, segment_mask, max_path_length, keep_largest_only=True):
    """
    Connect broken segments in a mask for a specific label.
    This function attempts to connect disconnected components of a labeled segment
    using A* pathfinding. 

    Parameters
    ----------
    label : int
        The label of the segment to process.
    segment_mask : ndarray
        Binary mask of the segment where non-zero values represent the segment.
    max_path_length : int
        Maximum allowed path length for connections. Paths exceeding this length
        will not be connected to prevent erroneous connections.
    keep_largest_only : bool, optional
        If True, only the largest connected component of the segment will be retained.
    Returns
    -------
    ndarray
        Connected mask mask for the specified segment with the original label value.
    """
    closest_pair, num_features = find_closest_components(segment_mask)
    if closest_pair is not None:
        while closest_pair != None:
            path = astar_3d(tuple(closest_pair[0]), tuple(closest_pair[1]), segment_mask)
            logger.debug(f'A* path: {path} with length {len(path)}')
            if len(path) > max_path_length:
                logger.warning(f'ALERT: path length is {len(path)}. Too long! Not connecting...')
                break
            else:
                logger.debug(f'path length is {len(path)}. Connecting...')
                for point in path:
                    x, y, z = point
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                nx, ny, nz = x + dx, y + dy, z + dz
                                if (0 <= nx < segment_mask.shape[0] and 
                                    0 <= ny < segment_mask.shape[1] and 
                                    0 <= nz < segment_mask.shape[2]):
                                    segment_mask[nx, ny, nz] = 1
                closest_pair, num_features = find_closest_components(segment_mask)
    
    segment_mask *= label

    # Keeping only largest component does not work for imperfect predictions anymore
    if keep_largest_only:
        logger.debug(f'Keeping largest component only for segment {label}')
        segment_mask = keep_largest_component(segment_mask)
    else:
        logger.debug(f'Keeping all components for segment {label}')

    return segment_mask

def run_cropping_and_resampling(cow_seg_filepath: str, save_dir: str, correct_mask: bool, min_segment_size: int, 
                                threshold_for_component_removal: int, max_path_length: int, crop_buffer: int = 1, 
                                resamp_spacing: list = [0.25, 0.25, 0.25]):
    """
    Run cropping, mask correction and resampling for the CoW segmentation mask. 
    Save the resulting mask to corresponding save_dir

    Parameters:
    ----------
    cow_seg_filepath: str, path to the segmentation file
    save_dir: str, directory to save the resulting mask
    buffer: int, buffer to add to the foreground bounding box for cropping
    spacing: list, target spacing for resampling [x, y, z]
    correct_mask: bool, whether to perform mask corrections or not
    min_segment_size: int, minimum size of the overall segment to keep after cropping (before resampling)
    threshold_for_component_removal: int, threshold for removing small disconnected components of a segment (before resampling)
    max_path_length: int, maximum path length for connecting components

    Returns:
    ----------
    output_filename: str, path to the saved output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        logger.debug(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    # Add trailing _0000 to the filename for nnUNet prediction
    output_filename = os.path.join(save_dir, os.path.basename(cow_seg_filepath).replace('.nii.gz', '_0000.nii.gz'))

    logger.info(f'Running cropping and resampling with args:'
                f'\n\t- input file: {cow_seg_filepath}'
                f'\n\t- output file: {output_filename}'
                f'\n\t- correct_mask: {correct_mask}'
                f'\n\t- min_segment_size: {min_segment_size}'
                f'\n\t- threshold_for_component_removal: {threshold_for_component_removal}'
                f'\n\t- max_path_length: {max_path_length}'
                f'\n\t- crop_buffer: {crop_buffer}'
                f'\n\t- resamp_spacing: {resamp_spacing}'
                )
    
    # Load the segmentation file
    try:
        logger.info(f"Loading file {cow_seg_filepath}")
        mask_nii = nib.load(cow_seg_filepath)
    except Exception as e:
        logger.error(f"Error loading file {cow_seg_filepath}: {e}")
        return None
    
    if not correct_mask:
        min_segment_size = None
        threshold_for_component_removal = None
        max_path_length = None

    # Crop to bounding box
    logger.info(f'Cropping mask to foreground with buffer of {crop_buffer}...')
    cropped_mask = crop_images(mask_nii, buffer=crop_buffer)
    nii_array = cropped_mask.get_fdata().astype(np.uint8) 
    
    # if correc_mask is true, mask will be corrected by:
    #   - Removing whole segments that are too small
    #   - Removing small disconnected components withing a segment 
    if correct_mask:
        logger.info(f'Checking for small segments with #voxels < {min_segment_size}...')
        logger.info(f'...and checking for small disconnected components within segments with #voxels < {threshold_for_component_removal}...')
        unique_labels, counts = np.unique(nii_array, return_counts=True)
        # Remove background (label 0)
        non_zero_indices = unique_labels != 0
        unique_labels = unique_labels[non_zero_indices]
        counts = counts[non_zero_indices]
        
        for label, count in zip(unique_labels, counts):
            # Special case for label 10 
            if min_segment_size is not None:
                current_min_size = 20 if label == 10 else min_segment_size
                if count < current_min_size:
                    logger.info(f"Removing small segment with label {label} (size: {count}, threshold: {current_min_size})")
                    nii_array[nii_array == label] = 0

            if threshold_for_component_removal is not None:
                # Create binary mask for this label
                binary_mask = (nii_array == label)
                
                # Find connected components
                labeled_array, num_features = ndimage.label(binary_mask)
                
                # Get sizes of each component
                component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
                
                # Log information about components
                logger.debug(f"...Label {label} has {num_features} connected components with sizes: {component_sizes}")

                # for all the labels except pcoms (8, 9), acom (10) and 3rd-A2 (15), we remove components smaller than some threshold
                if label not in [8, 9, 10, 15]:  # Exclude pcom and acom
                    for i, size in enumerate(component_sizes):
                        if size < threshold_for_component_removal:
                            logger.info(f"Removing small component {i+1} of label {label} (size: {size}, threshold: {threshold_for_component_removal})")
                            nii_array[labeled_array == (i + 1)] = 0

    # Create a new nibabel image from the modified array
    cropped_mask = nib.Nifti1Image(nii_array, cropped_mask.affine, header=cropped_mask.header)

    # Resample to desired spacing
    logger.info(f'Resampling to spacing {resamp_spacing}')
    resampled_mask = resample_nib(cropped_mask, resamp_spacing)
    nii_array = resampled_mask.get_fdata().astype(np.uint8)

    # if correc_mask is true, mask will be corrected by:
    #   - Connecting disconnected segments
    if correct_mask and max_path_length is not None:
        logger.info(f'Checking mask segments for more than one component after resampling...')
        unique_labels, counts = np.unique(nii_array, return_counts=True)
        # Remove background (label 0)
        non_zero_indices = unique_labels != 0
        unique_labels = unique_labels[non_zero_indices]
        for label in unique_labels:
            # if some segment (except acom) still has more than one component, we try to connect (like we do for the skeleton), with a max path length
            if label not in [10]:
                segment_mask = (nii_array == label).astype(np.uint8)
                labeled_array, num_features = ndimage.label(segment_mask)
                if num_features > 1:
                    logger.info(f"Label {label} has more than one component ({num_features}). Attempting to connect them.")
                    
                    # Find connected components
                    labeled_array, num_features = ndimage.label(segment_mask)
                    
                    # Get sizes of each component
                    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
                    
                    # Log information about components
                    logger.info(f"Label {label} has {num_features} connected components with sizes: {component_sizes}")
                    
                    # If there are multiple components, we can try to connect them using a simple approach
                    if num_features > 1:
                        logger.info(f"Connecting components of label {label} with max path length of {max_path_length}.")
                        connected_segment_mask = connect_broken_mask(label, segment_mask, max_path_length, keep_largest_only=False)

                        # Update the nii_array with the connected segment mask
                        nii_array[connected_segment_mask > 0] = label
    
    # Create a new nibabel image from the modified array
    resampled_mask_connected = nib.Nifti1Image(nii_array, resampled_mask.affine, header=resampled_mask.header)
    
    # Compare the number of foreground voxels in both masks
    resampled_array = resampled_mask.get_fdata()
    connected_array = resampled_mask_connected.get_fdata()
    resampled_foreground = np.sum(resampled_array > 0)
    connected_foreground = np.sum(connected_array > 0)

    # If the number of foreground voxels is different, save the connected mask separately
    if resampled_foreground != connected_foreground:
        logger.info(f"Number of foreground voxels changed from {resampled_foreground} to {connected_foreground}")
        connected_output_filename = output_filename.replace('_0000.nii.gz', '_connected_0000.nii.gz')
        logger.info(f'Saving connected mask to {connected_output_filename}')
        nib.save(resampled_mask_connected, connected_output_filename)
        logger.info(f'Saving original resampled mask to {output_filename}')
        nib.save(resampled_mask, output_filename)
    
    else:
        logger.info(f'Saving resampled mask to {output_filename}')
        nib.save(resampled_mask_connected, output_filename)
    
    return output_filename

