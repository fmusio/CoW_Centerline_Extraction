import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from logger import logger

def assign_labels_to_skeleton(mask_data, skeleton_data, correct_acom=False, label=None):
    """
    This function overlays a binary skeleton on a labeled mask image and
    transfers the labels from the mask to the skeleton pixels. 
    It also has an option to correct Acom voxels (label 10): if True, it will relabel isolated acom 
    skeleton voxels (26-connected neighborhood of the skeleton) that neighbor aca labels 11 or 12 
    (6-connected neighborhood of mask).

    Parameters:
    ----------
    mask_data: numpy.ndarray, multi-class segmentation mask 
    skeleton_data: numpy.ndarray, binary image representing the skeleton
    correct_acom: bool, if True, corrects Acom voxels (label 10) that neighbor labels 11 or 12
    label: int or None, if specified, assigns this label to skeleton pixels where the mask is zero.
           If None, assigns the label of the nearest non-zero mask pixel.
    
    Returns:
    ----------
    numpy.ndarray: A new array with the same shape as the skeleton, where
                   skeleton pixels have been assigned their class labels
                   from the mask. 
    """
    
    # Ensure the skeleton is binary
    skeleton_data = (skeleton_data > 0).astype(np.uint8)
    if np.sum(skeleton_data) == 0:
        logger.error("Skeleton data is empty. Please provide a valid skeleton.")
        raise ValueError("Skeleton data is empty. Please provide a valid skeleton.")
    
    # Create an output array to store the labeled skeleton
    mlt_skeleton = np.zeros_like(skeleton_data, dtype=mask_data.dtype)
    
    # First assign labels from the mask to the skeleton
    mlt_skeleton[skeleton_data > 0] = mask_data[skeleton_data > 0]

    # Find skeleton points where the mask is zero
    zero_mask_skeleton = (skeleton_data > 0) & (mask_data == 0)
    zero_mask_coords = np.array(np.nonzero(zero_mask_skeleton)).T

    # If there are any skeleton points with zero mask values
    if len(zero_mask_coords) > 0:
        if label is None:
            logger.info(f"Found {len(zero_mask_coords)} skeleton points with zero mask value. Assigning closest non-zero label.")
            logger.debug(f'\tZero-mask skeleton coordinates: {zero_mask_coords}')

            # Find all non-zero mask points
            non_zero_mask = mask_data > 0
            non_zero_coords = np.array(np.nonzero(non_zero_mask)).T
            non_zero_labels = mask_data[non_zero_mask]
            
            # Use KDTree for efficient nearest neighbor search
            mask_tree = cKDTree(non_zero_coords)
            
            # For each zero-mask skeleton point, find the nearest non-zero mask point
            distances, indices = mask_tree.query(zero_mask_coords, k=1)
            
            # Assign the label of the nearest non-zero mask point to the zero-mask skeleton point
            for i, coord in enumerate(zero_mask_coords):
                mlt_skeleton[tuple(coord)] = non_zero_labels[indices[i]]
        else:
            logger.info(f"Found {len(zero_mask_coords)} skeleton points with zero mask value. Assigning label {label}.")
            # Assign the specified label to all zero-mask skeleton points
            mlt_skeleton[zero_mask_skeleton] = label

    if correct_acom:
        # Find all voxels labeled as 10 (Acom)
        acom_voxels = np.where(mlt_skeleton == 10)
        if len(acom_voxels[0]) > 2:
            acom_coords = np.array(acom_voxels).T
            # Correct Acom voxels (label 10) that neighbor labels 11 or 12
            logger.debug("Correcting Acom voxels that neighbor labels 11 or 12 (6-ngh)")

            # For each  voxel
            for x, y, z in acom_coords:
                # Define 6-connected neighborhood structure element
                structure_element = np.zeros((3, 3, 3), dtype=bool)
                # Set the center and the 6 face-connected neighbors to True
                structure_element[0, 1, 1] = True # Left
                structure_element[2, 1, 1] = True # Right
                structure_element[1, 0, 1] = True # Top
                structure_element[1, 2, 1] = True # Bottom
                structure_element[1, 1, 0] = True # Front
                structure_element[1, 1, 2] = True # Back

                unique_neighbors_mask = get_unique_neighbors_of_voxel((x, y, z), mask_data, structure=structure_element)

                # Check if any of the neighbors are 11 or 12
                if 11 in unique_neighbors_mask:
                    logger.debug(f"Assigning label {11} to Acom voxel at ({x}, {y}, {z})")
                    mlt_skeleton[x, y, z] = 11
                if 12 in unique_neighbors_mask:
                    logger.debug(f"Assigning label {12} to Acom voxel at ({x}, {y}, {z})")
                    mlt_skeleton[x, y, z] = 12

    return mlt_skeleton

def assign_labels_to_skeleton_using_nearest_ngh(mask_data, skeleton_data):
    """
    Assigns labels from a multi-class mask to a binary skeleton based on nearest neighbor.
    This function takes a binary skeleton and a multi-class mask, and assigns labels to 
    each skeleton pixel by finding the nearest labeled pixel in the mask. It uses a KD-tree 
    for efficient nearest neighbor search.
    Parameters
    ----------
    mask_data : numpy.ndarray, multi-class mask array
    skeleton_data : numpy.ndarray, binary array representing the skeleton. 

    Returns
    -------
    numpy.ndarray: Labeled skeleton array, where each skeleton pixel is assigned the 
                   label of its nearest neighbor from the mask.

    Notes
    -----
    - The function uses scipy's cKDTree for efficient nearest neighbor search
    - Each skeleton pixel is assigned exactly one label from the mask
    - If multiple mask pixels are equidistant, the first one encountered is used
    """
    
    # Ensure the skeleton is binary
    skeleton_data = (skeleton_data > 0).astype(np.uint8)
    if np.sum(skeleton_data) == 0:
        logger.error("Skeleton data is empty. Please provide a valid skeleton.")
        raise ValueError("Skeleton data is empty. Please provide a valid skeleton.")
    
    # Create an output array to store the labeled skeleton
    mlt_skeleton = np.zeros_like(skeleton_data, dtype=mask_data.dtype)
    
    # For each index where skeleton_data > 0, find the nearest neighbor in mask_data with label > 0
    skeleton_indices = np.array(np.nonzero(skeleton_data)).T
    mask_indices = np.array(np.nonzero(mask_data)).T
    mask_labels = mask_data[mask_data > 0]
    logger.debug(f'Using KD-Tree for nearest neighbor search')
    mask_tree = cKDTree(mask_indices)
    distances, indices = mask_tree.query(skeleton_indices, k=1)
    for i, idx in enumerate(skeleton_indices):
        mlt_skeleton[tuple(idx)] = mask_labels[indices[i]]

    return mlt_skeleton

def keep_largest_component(segment_skeleton_mask):
    """
    This function keeps the largest connected component of a binary mask.
    It is useful for segment skeletons that may have multiple disconnected parts, 
    which might be the case when distance between disconnected parts exceeds max_path_length.

    Parameters:
    ----------
    segment_skeleton_mask (np.ndarray): Binary mask of the segment skeleton.

    Returns:
    -------
    np.ndarray: Binary mask of the largest connected component
    """

    labeled_skeleton, num_features = nd_label(segment_skeleton_mask, structure=np.ones((3, 3, 3)))
    if num_features == 1:
        return segment_skeleton_mask
    else:
        logger.warning(f'Segment skeleton has {num_features} components. Keeping largest component only...')
        component_sizes = []
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_skeleton == i)
            component_sizes.append(component_size)
        
        largest_component_idx = np.argmax(component_sizes) + 1
        largest_component = (labeled_skeleton == largest_component_idx).astype(np.uint8)
        return largest_component

def get_unique_neighbors_of_voxel(voxel, mask, structure=None):
    """
    This function returns the unique neighboring labels of a given voxel in a 3D mask.
    
    Parameters:
    voxel (tuple): Coordinates of the voxel (x, y, z).
    mask (np.ndarray): 3D array with different labels.
    structure (np.ndarray): Structure element for neighborhood definition. 
                            If None, uses a 3D structure element.
    
    Returns:
    list: List of unique neighboring labels.
    """
    mask_voxel = np.zeros_like(mask, dtype=bool)
    mask_voxel[voxel] = True
    dilated_mask = binary_dilation(mask_voxel, structure=structure)
    border_mask = dilated_mask & ~mask_voxel
    neighbor_labels = np.unique(mask[border_mask])
    neighbor_labels = neighbor_labels[neighbor_labels != 0]  # Exclude background
    return neighbor_labels.tolist()

def compute_all_unique_neighbors(mask):
    """
    This function identifies all unique pairs of neighboring segments in a 3D mask.
    It uses a 3D structure element and binary dilation to find neighboring segments, 
    and returns a list of tuples representing the unique pairs of neighboring segments.
    
    Parameters:
    mask (np.ndarray): 3D array with different labels.
    
    Returns:
    list: List of tuples (label1, label2) representing unique neighboring segments.
    """
    structure = np.ones((3, 3, 3))
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    neighbor_pairs = set()

    for label in unique_labels:
        segment_mask = (mask == label)
        dilated_mask = binary_dilation(segment_mask, structure=structure)
        border_mask = dilated_mask & ~segment_mask
        neighbor_labels = np.unique(mask[border_mask])
        neighbor_labels = neighbor_labels[(neighbor_labels != 0) & (neighbor_labels != label)]
        for ngh in neighbor_labels:
            pair = tuple(sorted((label, ngh)))
            neighbor_pairs.add(pair)

    return list(neighbor_pairs)

def insert_missing_labels(mlt_skeleton, mlt_mask_array, labels):
    """
    Insert missing segment labels into the skeleton by placing label points at the maximum distance
    from each segment's boundary.
    This function checks if any labels from the multi-class segmentation mask are missing in the 
    skeleton. For each missing label, it uses a distance transform to find the point furthest from 
    the segment boundary (typically the center of the segment) and assigns the label to that point.
    
    Parameters
    ----------
    mlt_skeleton : np.ndarray, skeleton array labeled with segment labels. 
    mlt_mask_array : np.ndarray, multi-class segmentation mask
    labels : list, list of unique segment labels present in the multi-class mask.
    
    Returns
    -------
    np.ndarray: Modified skeleton with missing labels inserted at appropriate locations.
    """
   
    for label in labels:
        segment_skeleton_mask = (mlt_skeleton == label).astype(np.uint8)
        segment_mask = (mlt_mask_array == label).astype(np.uint8)
        # If label not present (2 or less voxels) for skeleton, add a points at the maximum distance of segment boundary
        if np.sum(segment_skeleton_mask) < 3:
            assert np.sum(segment_mask) != 0, 'Segment mask should not be empty!'
            logger.warning(f'Segment {label} is missing in the skeleton. Inserting missing label using distance field...')
            # Check each voxel in the segment mask
            structure_element = np.ones((3, 3, 3), dtype=bool)
            voxel_coords = np.where(segment_skeleton_mask > 0)
            for x, y, z in zip(*voxel_coords):
                # Get unique neighbors of this voxel
                unique_neighbors = get_unique_neighbors_of_voxel((x, y, z), mlt_skeleton, structure=structure_element)
                # If there's more than one unique neighbor
                unique_neighbors = [ngh for ngh in unique_neighbors if ngh != label]  # Exclude label itself
                if len(unique_neighbors) > 0:
                    mlt_skeleton[x, y, z] = unique_neighbors[0]  # Assign the first unique neighbor label
                
            df = distance_transform_edt(segment_mask)
            max_distance = np.max(df)
            mlt_skeleton[np.where(df == max_distance)] = label
    
    return mlt_skeleton

def relabel_isolated_voxels(skeleton_array, label=10):
    """
    Remove isolated voxels from the connected skeleton segment of the given label.
    An isolated voxel is defined as a voxel that does not have any 26-connected neighbors of the same label.

    Args:
        skeleton_array (np.ndarray): The input skeleton array.
        label (int): The label of the segment to process.

    Returns:
        np.ndarray: The modified skeleton array with isolated voxels removed.
    """
    # Create a binary mask for the specified label
    segment_mask = (skeleton_array == label).astype(np.uint8)
    
    # Label connected components in the segment mask
    labeled_skeleton, num_features = nd_label(segment_mask, structure=np.ones((3, 3, 3)))
    
    # If there is only one component, return the original skeleton
    if num_features == 1:
        return skeleton_array
    
    # Find isolated voxels (those that are of size 1 only)
    isolated_voxels = np.zeros_like(skeleton_array, dtype=bool)
    for i in range(1, num_features + 1):
        component_mask = (labeled_skeleton == i)
        if np.sum(component_mask) == 1:  # Isolated voxel
            isolated_voxels |= component_mask
    # Convert isolated voxels to indices
    isolated_voxels = np.where(isolated_voxels)
    if len(isolated_voxels[0]) == 0:
        logger.debug('No isolated voxels found in the skeleton.')
        return skeleton_array
    else:
        # Log the number of isolated voxels found
        logger.debug(f'Found {len(isolated_voxels[0])} isolated voxels in the skeleton for label {label}.')
        
        # find neighboring labels of isolated voxels in the skeleton
        # Create a structure element for 26-neighborhood (all adjacent voxels: face, edge, and corner)
        structure_element = np.ones((3, 3, 3), dtype=bool)
        # Exclude the center voxel itself
        structure_element[1, 1, 1] = False
        for x, y, z in zip(*isolated_voxels):
            # Get the unique neighboring labels
            unique_neighbors_skeleton = get_unique_neighbors_of_voxel((x, y, z), skeleton_array, structure=structure_element)
            if len(unique_neighbors_skeleton) > 0:
                # Assign the label of the first neighbor that is not 10
                for ngh in unique_neighbors_skeleton:
                    if ngh != label:
                        logger.debug(f'Replacing isolated voxel at ({x}, {y}, {z}) with label {ngh}')
                        skeleton_array[x, y, z] = ngh
                        break
            else:
                logger.debug(f'No neighbors found for isolated voxel at ({x}, {y}, {z}). Keeping it as is.')

def add_obligatory_neighbors(nghs_mask):
    """
    This function adds obligatory neighbors to the mask based on already existing connectivities.
    It ensures that the obligatory neighbors are present, even if they are not 
    connected in the mask (imperfect predictions).

    Parameters:
    ----------
    nghs_mask : np.ndarray, binary mask of neighboring segments.

    Returns:
    -------
    np.ndarray: Updated mask with obligatory neighbors added.
    """
    labels = list(set([ngh[0] for ngh in nghs_mask] + [ngh[1] for ngh in nghs_mask])) 
    # Acom
    if 10 in labels:
        if 11 in labels:
            if not (10, 11) in nghs_mask and not (11, 10) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (10, 11)')
                nghs_mask.append((10, 11))
        if 12 in labels:
            if not (10, 12) in nghs_mask and not (12, 10) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (10, 12)')
                nghs_mask.append((10, 12))
    # R-Pcom
    if 8 in labels:
        if 2 in labels:
            if not (2, 8) in nghs_mask and not (8, 2) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (2, 8)')
                nghs_mask.append((2, 8))
        if 4 in labels:
            if not (4, 8) in nghs_mask and not (8, 4) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (4, 8)')
                nghs_mask.append((4, 8))
    
    # L-Pcom
    if 9 in labels:
        if 3 in labels:
            if not (3, 9) in nghs_mask and not (9, 3) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (3, 9)')
                nghs_mask.append((3, 9))
        if 6 in labels:
            if not (6, 9) in nghs_mask and not (9, 6) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (6, 9)')
                nghs_mask.append((6, 9))

    # R-PCA
    if 2 in labels:
        if 8 not in labels:
            if (1, 2) not in nghs_mask and (2, 1) not in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (1, 2)')
                nghs_mask.append((1, 2))
    
    # L-PCA
    if 3 in labels:
        if 9 not in labels:
            if (1, 3) not in nghs_mask and (3, 1) not in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (1, 3)')
                nghs_mask.append((1, 3))
    
    # R-MCA
    if 5 in labels:
        if 4 in labels:
            if not (4, 5) in nghs_mask and not (5, 4) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (4, 5)')
                nghs_mask.append((4, 5))
    
    # L-MCA
    if 7 in labels:
        if 6 in labels:
            if not (6, 7) in nghs_mask and not (7, 6) in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (6, 7)')
                nghs_mask.append((6, 7))
    
    # R-ACA
    if 11 in labels:
        if 10 not in labels:
            assert 4 in labels, 'R-ACA (11) should be connected to R-ICA (4)!'
            if (4, 11) not in nghs_mask and (11, 4) not in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (4, 11)')
                nghs_mask.append((4, 11))
    
    # L-ACA
    if 12 in labels:
        if 10 not in labels:
            assert 6 in labels, 'L-ACA (12) should be connected to L-ICA (6)!'
            if (6, 12) not in nghs_mask and (12, 6) not in nghs_mask:
                logger.warning('ALERT: Adding obligatory neighbor (6, 12)')
                nghs_mask.append((6, 12))
    
    # 3rd-A2
    if 15 in labels:
        assert 10 in labels, '3rd-A2 (15) should be connected to Acom (10)!'
        if (10, 15) not in nghs_mask and (15, 10) not in nghs_mask:
            logger.warning('ALERT: Adding obligatory neighbor (10, 15)')
            nghs_mask.append((10, 15))
    
    return nghs_mask
   