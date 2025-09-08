import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label
from scipy.spatial import cKDTree

from logger import logger

def get_neighbors(point, shape):
    """
    Get neighbors of a point in a 3D grid.
    
    Parameters:
    point (tuple): Point (x, y, z).
    shape (tuple): Shape of the 3D grid.
    
    Returns:
    list: List of neighbor points.
    """
    # Pre-compute all possible neighbor offsets
    neighbor_offsets = np.array([
        [dx, dy, dz] for dx in [-1, 0, 1] 
        for dy in [-1, 0, 1] 
        for dz in [-1, 0, 1] 
        if not (dx == 0 and dy == 0 and dz == 0)
    ])
    # Generate neighbors by adding offsets to the point
    neighbors = point + neighbor_offsets
    # Filter out neighbors that are outside the grid bounds
    valid_neighbors = neighbors[
        (neighbors[:, 0] >= 0) & (neighbors[:, 0] < shape[0]) & 
        (neighbors[:, 1] >= 0) & (neighbors[:, 1] < shape[1]) &
        (neighbors[:, 2] >= 0) & (neighbors[:, 2] < shape[2])
    ]

    # Convert valid neighbors to a list of tuples
    valid_neighbors = [tuple(neighbor) for neighbor in valid_neighbors]
    
    return valid_neighbors

def find_closest_components(skeleton_segment):
    """
    This function identifies connected components in a 3D skeleton segment and finds the
    closest pair of points between any two components. It's useful for reconnecting broken
    segments in skeletal structures.
    
    Parameters:
    ----------
    skeleton_segment : ndarray, 3D binary array representing a skeleton segment.
    
    Returns:
    -------
    tuple or (None, None)
        If multiple components exist:
            Returns a tuple (closest_pair, num_features) where:
            - closest_pair: A list [coords1, coords2, distance] containing the coordinates
              of the closest points between any two components and their Euclidean distance.
            - num_features: The total number of connected components found.
        If only one component exists:
            Returns (None, None).
    Notes:
    -----
    The function uses KD-trees for efficient nearest neighbor searches between components.
    Each component is compared with all others to find the minimum distance between any pair.
    """

    # Label connected components
    labeled_skeleton, num_features = nd_label(skeleton_segment, structure=np.ones((3, 3, 3)))
    if num_features == 1:
        return None, None
    else:
        # Get the coordinates of each component
        component_coords = []
        closest_pairs = []
        for i in range(1, num_features + 1):
            coords = np.array(np.nonzero(labeled_skeleton == i)).T
            component_coords.append(coords)
        
        # Connect each component to its nearest neighbor component
        for i in range(num_features):
            coords1 = component_coords[i]
            tree1 = cKDTree(coords1)
            
            min_distance = float('inf')
            closest_pair = None
            
            for j in range(num_features):
                if i == j:
                    continue
                
                coords2 = component_coords[j]
                tree2 = cKDTree(coords2)
                
                distances, indices = tree1.query(tree2.data, k=1)
                min_idx = np.argmin(distances)
                
                if distances[min_idx] < min_distance:
                    min_distance = distances[min_idx]
                    closest_pair = [coords1[indices[min_idx]], coords2[min_idx], distances[min_idx]]
            
            closest_pairs.append(closest_pair)
        
        # sort by distance
        closest_pairs = sorted(closest_pairs, key=lambda x: x[2])
        
        return closest_pairs[0], num_features
    
def find_closest_component_ngh(skeleton_segment1, skeleton_segment2):
    """
    Find the closest pair of points between two neighboring skeleton segments.
    This function processes two 3D numpy arrays that represent skeleton segments. It first
    labels connected components in each segment to extract all non-zero coordinates. It then
    constructs KD-trees for these coordinates in order to efficiently query the nearest neighbors.
    Finally, the function identifies the pair of points—one from each segment—that have the smallest
    Euclidean distance between them.
    TODO: Catch the case for missing A1/P1 segments.

    Parameters:
    ----------
    skeleton_segment1 (np.ndarray): first skeleton segment.
    skeleton_segment2 (np.ndarray): second skeleton segment.

    Returns:
    -------
    list: A list containing three elements:
            - The coordinates (as an array) of the closest point from skeleton_segment1.
            - The coordinates (as an array) of the closest point from skeleton_segment2.
            - The Euclidean distance between these two points.
    """
    
    # Label connected components
    labeled_skeleton1, num_features1 = nd_label(skeleton_segment1, structure=np.ones((3, 3, 3)))
    labeled_skeleton2, num_features2 = nd_label(skeleton_segment2, structure=np.ones((3, 3, 3)))
    assert num_features1 > 0 and num_features2 > 0, 'Both skeleton segments must have at least one connected component'
    
    component_coords = []
    closest_pairs = []
    if num_features1 == 1:
        coords = np.array(np.nonzero(labeled_skeleton1 == 1)).T
    else:
        for i in range(1, num_features1 + 1):
            if i == 1:
                concat_coords = np.array(np.nonzero(labeled_skeleton1 == i)).T
            else:
                concat_coords = np.concatenate((concat_coords, np.array(np.nonzero(labeled_skeleton1 == i)).T), axis=0)
        coords = concat_coords
    component_coords.append(coords)
    if num_features2 == 1:
        coords = np.array(np.nonzero(labeled_skeleton2 == 1)).T
    else:
        for i in range(1, num_features2 + 1):
            if i == 1:
                concat_coords = np.array(np.nonzero(labeled_skeleton2 == i)).T
            else:
                concat_coords = np.concatenate((concat_coords, np.array(np.nonzero(labeled_skeleton2 == i)).T), axis=0)
        coords = concat_coords
    component_coords.append(coords)
    assert len(component_coords) == 2, 'Only two components are allowed'

    coords1 = component_coords[0]
    tree1 = cKDTree(coords1)
    
    min_distance = float('inf')
    closest_pair = None
            
    coords2 = component_coords[1]
    tree2 = cKDTree(coords2)
                
    distances, indices = tree1.query(tree2.data, k=1)
    min_idx = np.argmin(distances)

    if distances[min_idx] < min_distance:
        min_distance = distances[min_idx]
        closest_pair = [coords1[indices[min_idx]], coords2[min_idx], distances[min_idx]]
            
    closest_pairs.append(closest_pair)
    assert len(closest_pairs) == 1, 'Only one pair is allowed'
    
    return closest_pairs[0]

def heuristic_cost(current, goal, distance_field, weight_euclid=1, weight_distance=2):
    """
    Heuristic cost function for A* algorithm. 
    Weighted sum of Euclidean distance and distance from mask boundary.
    
    Parameters:
    -----------
    current (tuple): Current point (x, y, z).
    goal (tuple): Goal point (x, y, z).
    distance_field (np.ndarray): Distance field of the mask.
    
    Returns:
    --------
    float: Heuristic cost.
    """
    euclidean_distance = np.linalg.norm(np.array(current) - np.array(goal))
    distance_from_obstacles = distance_field[current]
    cost = weight_euclid*euclidean_distance - weight_distance*distance_from_obstacles
    # logger.debug(f'Heuristic cost from {current} to {goal}: {cost}')
    return cost

def astar_3d(start, goal, segment_mask):
    """
    A* pathfinding algorithm for 3D grids.
    This implementation finds the optimal path between two points in a 3D volume
    while trying to stay within a segment mask and away from boundaries.
    The algorithm uses:
    - A distance transform (EDT) to prefer paths away from segment boundaries
    - A combined heuristic that balances Euclidean distance with boundary distance
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
    # takes on values 0 outside of mask (i.e. for background voxels)
    distance_field = distance_transform_edt(segment_mask)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    cost = heuristic_cost(start, goal, distance_field)
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
                f_score[neighbor] = g_score[neighbor] + heuristic_cost(neighbor, goal, distance_field)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []