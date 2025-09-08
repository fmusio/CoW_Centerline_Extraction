import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vtk
import numpy as np
import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import copy
import json
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from utils.utils_graph_processing import *
from utils.utils_meshing import get_vtk_mesh
from postprocess_graph.node_extraction import find_acom_bif_for_3rd_a2

from logger import logger


def get_boundary_and_bif_cell_ids(node_dict, polydata):
    """
    Get cell IDs that are adjoining a boundary point and cell IDs of vessels near bifurcation points.
    These IDs can optionally be treated separately for radius computation.

    Args:
    node_dict: dict, dictionary containing the node information
    polydata: vtkPolyData, centerline graph

    Returns:
    boundary_ids: list, cell IDs of boundary points
    boundary_ngh_labels: dict, dictionary containing the neighboring labels for each boundary cell
    bif_near_cells: list, cell IDs of child vessels near bifurcation points
    bif_point_ids: list, point IDs of bifurcation points
    """

    bif_point_ids = []
    bif_near_cells = []
    boundary_ids = []
    boundary_ngh_labels = {}
    if '1' in node_dict:
        ba_dict = node_dict['1']
        if 'R-PCA boundary' in ba_dict:
            rpca_boundary = ba_dict['R-PCA boundary']
            for i in range(len(rpca_boundary)):
                rpca_boundary_id = rpca_boundary[i]['id']
                boundary_cells = get_cellIds_for_point(rpca_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'BA bifurcation' in ba_dict:
                    bif_id = ba_dict['BA bifurcation'][0]['id']
                    bif_point_ids.append(bif_id)
                    path = find_shortest_path(bif_id, rpca_boundary_id, polydata, 1)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
        if 'L-PCA boundary' in ba_dict:
            lpca_boundary = ba_dict['L-PCA boundary']
            for i in range(len(lpca_boundary)):
                lpca_boundary_id = lpca_boundary[i]['id']
                boundary_cells = get_cellIds_for_point(lpca_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'BA bifurcation' in ba_dict:
                    bif_id = ba_dict['BA bifurcation'][0]['id']
                    bif_point_ids.append(bif_id)
                    path = find_shortest_path(bif_id, lpca_boundary_id, polydata, 1)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
    for label in [2, 3]:
        if str(label) in node_dict:
            pca_dict = node_dict[str(label)]
            if 'Pcom boundary' in pca_dict:
                pcom_boundary_id = pca_dict['Pcom boundary'][0]['id']
                boundary_cells = get_cellIds_for_point(pcom_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'Pcom bifurcation' in pca_dict:
                    pcom_bif_id = pca_dict['Pcom bifurcation'][0]['id']
                    bif_point_ids.append(pcom_bif_id)
                    path = find_shortest_path(pcom_bif_id, pcom_boundary_id, polydata, label)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
    
    for label in [4, 6]:
        if str(label) in node_dict:
            ica_dict = node_dict[str(label)]
            if 'Pcom boundary' in ica_dict:
                pcom_boundary_id = ica_dict['Pcom boundary'][0]['id']
                boundary_cells = get_cellIds_for_point(pcom_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'Pcom bifurcation' in ica_dict:
                    pcom_bif_id = ica_dict['Pcom bifurcation'][0]['id']
                    bif_point_ids.append(pcom_bif_id)
                    path = find_shortest_path(pcom_bif_id, pcom_boundary_id, polydata, label)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
            if 'MCA boundary' in ica_dict:
                mca_boundary_id = ica_dict['MCA boundary'][0]['id']
                boundary_cells = get_cellIds_for_point(mca_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'ICA bifurcation' in ica_dict:
                    ica_bif_id = ica_dict['ICA bifurcation'][0]['id']
                    bif_point_ids.append(ica_bif_id)
                    path = find_shortest_path(ica_bif_id, mca_boundary_id, polydata, label)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
            if 'ACA boundary' in ica_dict:
                aca_boundary_id = ica_dict['ACA boundary'][0]['id']
                boundary_cells = get_cellIds_for_point(aca_boundary_id, polydata)
                boundary_ids += boundary_cells
                labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                for cell in boundary_cells:
                    ngh = copy.deepcopy(labels_ngh)
                    ngh.remove(get_label_for_cell(cell, polydata))
                    boundary_ngh_labels[cell] = ngh
                if 'ICA bifurcation' in ica_dict:
                    ica_bif_id = ica_dict['ICA bifurcation'][0]['id']
                    bif_point_ids.append(ica_bif_id)
                    path = find_shortest_path(ica_bif_id, aca_boundary_id, polydata, label)['path']
                    for edge in path:
                        bif_near_cells.append(get_cellId_for_edge(edge, polydata))
    
    for label in [11, 12]:
        if str(label) in node_dict:
            aca_dict = node_dict[str(label)]
            if 'Acom boundary' in aca_dict:
                acom_boundary = aca_dict['Acom boundary']
                for i in range(len(acom_boundary)):
                    acom_boundary_id = acom_boundary[i]['id']
                    boundary_cells = get_cellIds_for_point(acom_boundary_id, polydata)
                    boundary_ids += boundary_cells
                    labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
                    for cell in boundary_cells:
                        ngh = copy.deepcopy(labels_ngh)
                        ngh.remove(get_label_for_cell(cell, polydata))
                        boundary_ngh_labels[cell] = ngh
                    if 'Acom bifurcation' in aca_dict:
                        acom_bif_id = aca_dict['Acom bifurcation'][i]['id']
                        bif_point_ids.append(acom_bif_id)
                        path = find_shortest_path(acom_bif_id, acom_boundary_id, polydata, label)['path']
                        for edge in path:
                            bif_near_cells.append(get_cellId_for_edge(edge, polydata))
    
    if '15' in node_dict:
        a2_dict = node_dict['15']
        if 'Acom boundary' in a2_dict:
            acom_boundary_id = a2_dict['Acom boundary'][0]['id']
            boundary_cells = get_cellIds_for_point(acom_boundary_id, polydata)
            boundary_ids += boundary_cells
            labels_ngh = list(set([get_label_for_cell(cell, polydata) for cell in boundary_cells]))
            for cell in boundary_cells:
                ngh = copy.deepcopy(labels_ngh)
                ngh.remove(get_label_for_cell(cell, polydata))
                boundary_ngh_labels[cell] = ngh
            acom_bif_id = find_acom_bif_for_3rd_a2(acom_boundary_id, polydata)
            bif_point_ids.append(acom_bif_id)
            path = find_shortest_path(acom_bif_id, acom_boundary_id, polydata, 10)['path']
            for edge in path:
                bif_near_cells.append(get_cellId_for_edge(edge, polydata))

    bif_point_ids = list(set(bif_point_ids))

    return boundary_ids, boundary_ngh_labels, bif_near_cells, bif_point_ids

def get_remaining_bif_ids(bif_near_cells, bif_point_ids, polydata, nr_edges=6):
    """
    Identify the remaining bifurcation points and add +/- nr_edges edges to bif_branch_edges.

    Args:
    bif_near_cells: list, list of cell IDs of child vessels near bifurcation points
    bif_point_ids: list, list of point IDs of bifurcation points
    polydata: vtkPolyData, centerline graph
    nr_edges: int, number of edges to append to bif_branch_edges

    Returns:
    bif_near_cells: list, updated list of cell IDs vessels near bifurcation points
    bif_point_ids: list, updated list of point IDs of bifurcation points
    """
    unique_labels = np.unique(get_label_array(polydata))
    check_labels = [2, 3, 5, 6, 11, 12]
    if '15' in unique_labels:
        check_labels.append(10)
    degree_array = polydata.GetPointData().GetArray("degree")
    for label in check_labels:
        if label in unique_labels:
            edge_list, cell_ids = get_edge_list(polydata, label)
            cell_ids = list(cell_ids)
            
            points_segment = get_pointIds_for_label(label, polydata)
            for point in points_segment:
                deg = degree_array.GetValue(point)
                if deg >= 3:
                    if point in bif_point_ids:
                        pass
                    else:
                        bif_point_ids.append(point)
                        ngh_cells_bif = get_cellIds_for_point(point, polydata)
                        for cell in ngh_cells_bif:
                            if cell not in bif_near_cells:
                                try:
                                    idx = cell_ids.index(cell)
                                    edge = edge_list[idx]
                                    if point == edge[0]:
                                        bif_near_cells.append(cell)
                                        for i in range(1, nr_edges):
                                            edge_0 = edge_list[idx+i-1]
                                            edge_1 = edge_list[idx+i]
                                            cell_1 = cell_ids[idx+i]
                                            if degree_array.GetValue(edge_0[1]) == 2 and edge_1[1] == edge_0[1] + 1:
                                                bif_near_cells.append(cell_1)
                                            
                                    elif point == edge[1]:
                                        bif_near_cells.append(cell)
                                        for i in range(1, nr_edges):
                                            edge_0 = edge_list[idx-i+1]
                                            edge_1 = edge_list[idx-i]
                                            cell_1 = cell_ids[idx-i]
                                            if degree_array.GetValue(edge_0[0]) == 2 and edge_1[0] == edge_0[0] - 1:
                                                bif_near_cells.append(cell_1)
                                except:
                                    pass

    bif_near_cells = list(set(bif_near_cells))
    bif_point_ids = list(set(bif_point_ids))

    return bif_near_cells, bif_point_ids


    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=10, color='blue', label='Surface Points')
    ax.scatter(midpoint_2d[0], midpoint_2d[1], color='red', s=50, marker='x', label='Midpoint')

    # Draw circle with min_rad
    min_circle = plt.Circle(midpoint_2d, min_rad, fill=False, color='green', linestyle='--', label='MIS Radius')
    ax.add_patch(min_circle)

    # Draw circle with ce_radius
    ce_circle = plt.Circle(midpoint_2d, ce_radius, fill=False, color='orange', linestyle='-', label='CE Radius')
    ax.add_patch(ce_circle)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    # ax.set_title(f'Cross-Section Analysis for Radius Estimation')
    ax.legend(fontsize=14, loc='upper left')
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.show()

def get_seg_meshes(nii_mask, label_masks, distance_transforms, n_jobs=12):
    """
    Generate meshes from segments in a labeled volume.
    This function creates meshes for each labeled segment in a NIfTI image. It modifies
    each segment's mask by including voxels from neighboring segments that are within
    a specified distance threshold (1.5 units), before generating the mesh.
    Parameters:
    -----------
    nii_mask : nibabel.Nifti1Image
        The NIfTI image containing labeled segments.
    label_masks : dict
        Dictionary mapping label IDs to corresponding binary masks.
    distance_transforms : dict
        Dictionary mapping label IDs to distance transform images.
    n_jobs : int
        Number of parallel jobs to run.
    Returns:
    --------
    dict
        Dictionary mapping label IDs to VTK mesh objects representing the segments.
    """

    ngh_segments = {
        '1': [2, 3],
        '2': [1, 8],
        '3': [1, 9],
        '4': [5, 8, 11],
        '5': [4],
        '6': [7, 9, 12],
        '7': [6],
        '8': [2, 4],
        '9': [3, 6],
        '10': [11, 12, 15],
        '11': [4, 10],
        '12': [6, 10],
        '15': [10]
    }

    affine = nii_mask.affine
    nii_data = nii_mask.get_fdata().astype(np.uint8)
    labels_array = np.unique(nii_data)

    def mesh_process_single_label(label, nii_data, ngh_segments, distance_transforms, label_masks, affine):
        """Process a single label to create a mesh"""
        logger.info(f'\t...mesh for label {label}')
        if str(label) in ngh_segments:
            # Get the list of neighboring labels for this label
            ngh_labels_list = ngh_segments[str(label)]
            distance_map = distance_transforms[label][0]
            label_mask = copy.deepcopy(label_masks[label][0])  # Make a deep copy to avoid modifying the original
            
            # Create a binary mask where any of these labels are present
            ngh_mask = np.isin(nii_data, ngh_labels_list).astype(np.uint8)
            # Include neighboring segment voxels within the distance threshold
            label_mask[np.where(ngh_mask & (distance_map < 1.5))] = 1

            segment_mesh = get_vtk_mesh(nib.Nifti1Image(label_mask, affine), do_smoothing=True, pass_band=0.25)
            return label, segment_mesh
        return label, None

    # For each label and its neighboring labels, process in parallel
    segments_meshes = {}
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(mesh_process_single_label)(
            label, nii_data, ngh_segments, distance_transforms, label_masks, affine
        ) for label in labels_array
    )

    # Collect results
    for label, mesh in results:
        if mesh is not None:
            segments_meshes[label] = mesh

    return segments_meshes

def process_cell_surface_meshes(id, centerline, surface_meshes, labels_array, bif_near_cells=[], use_meanpoint=False,
                                use_ce_rad_cap=False, bif_min_rad_factor=1.33, z_score_cutoff=3, eps_value=0.4):
    """
    Computes radius measurements at a cross-section of a cell in a centerline.
    This function creates a plane at the midpoint of a cell (edge) and intersects it with the corresponding
    surface mesh to determine radius metrics. It employs DBSCAN clustering and outlier removal to ensure accurate radius estimation.
    Optionally, the function handles cells close to bifurcations differently.
    Parameters:
    ----------
    id : int
        Cell identifier in the centerline.
    centerline : vtkPolyData
        The centerline data structure containing cells.
    surface_meshes : dict
        Dictionary mapping labels to surface mesh objects.
    labels_array : array-like
        Array containing labels for each cell.
    bif_near_cells : list or set
        Collection of cell IDs that are near bifurcations.
    use_meanpoint : bool, default=False
        If True, use the mean of the surface points of the cross-section for radius calculations.
    use_ce_rad_cap : bool, default=False
        If True, apply a cap on the circle-equivalent radius close to bifurcations (based on the minimum radius).
    bif_min_rad_factor : float, default=1.33
        Factor to adjust radius calculations near bifurcations for the ce radius (if use_ce_rad_cap=True).
    z_score_cutoff : float, default=3
        Threshold for outlier removal based on z-score.
    eps_value : float, default=0.4
        The maximum distance between points for DBSCAN clustering.
    Returns:
    -------
    tuple or (nan, nan, nan)
        A tuple containing (min_radius, circle_equivalent_radius).
        Returns NaN values if insufficient points are found for accurate estimation.
    Notes:
    -----
    The function performs multiple stages of computation:
    1. Creates a cross-section plane at the cell midpoint
    2. Extracts intersection points with the surface mesh
    3. Projects points to 2D for analysis
    4. Applies clustering to identify the proper vessel contour
    5. Removes outliers based on statistical properties
    6. Calculates MIS and CE radii
    7. (Optionally) Adjusts calculations for cells near bifurcations
    """
    
    cell = centerline.GetCell(id)
    p1 = np.array(centerline.GetPoint(cell.GetPointId(0)))
    p2 = np.array(centerline.GetPoint(cell.GetPointId(1)))
    
    # cell label
    label = labels_array[id]

    # Distinguish between edges close to bifurcations and 'normal' edges
    is_close_to_bif = id in bif_near_cells
    
    # Compute the midpoint of the cell
    midpoint = (p1 + p2) / 2.0

    # Compute the tangent vector of the cell (= normal vector for cross section)
    tangent = p2 - p1
    tangent /= np.linalg.norm(tangent)

    # Create a VTK plane for intersection
    plane = vtk.vtkPlane()
    plane.SetOrigin(midpoint)
    plane.SetNormal(tangent)

    mesh = surface_meshes[label]
    
    # Extract the cross-section using the plane
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    
    cross_section = cutter.GetOutput()

    # Extract points from the cross-section
    points = []
    for j in range(cross_section.GetNumberOfPoints()):
        points.append(cross_section.GetPoint(j))
    points = np.array(points)

    if len(points) >= 10:  # Ensure enough points for meaningful clustering
        # Project points to 2D
        if abs(tangent[2]) > 1e-8:
            u = np.array([1.0, 0.0, -tangent[0]/tangent[2]])
        elif abs(tangent[1]) > 1e-8:
            u = np.array([1.0, -tangent[0]/tangent[1], 0.0])
        else:
            u = np.array([0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        v /= np.linalg.norm(v)

        # Convert 3D intersection points to 2D coordinates on the plane
        points_2d = np.zeros((len(points), 2))
        for i, point in enumerate(points):
            points_2d[i, 0] = np.dot(point - midpoint, u)
            points_2d[i, 1] = np.dot(point - midpoint, v)
        
        midpoint_2d = [np.dot(midpoint - midpoint, u), np.dot(midpoint - midpoint, v)]

        # Compute distances for clustering
        distances_midpoint = np.linalg.norm(points_2d - midpoint_2d, axis=1)
        distances_reshaped = distances_midpoint.reshape(-1, 1)
        
        # clustering on distances
        # choose 0.3 <= eps <= 0.5
        dbscan = DBSCAN(eps=eps_value, min_samples=5, algorithm='ball_tree')
        cluster_labels = dbscan.fit_predict(distances_reshaped)
        valid_clusters = np.unique(cluster_labels[cluster_labels != -1])
        if len(valid_clusters) > 1:
            
            # Find distances from each point to midpoint
            closest_idx = np.argmin(distances_midpoint)
            selected_cluster = cluster_labels[closest_idx]
            
            # Filter points to keep only those in the selected cluster
            mask = cluster_labels == selected_cluster
            points_2d = points_2d[mask]

        # cut outliers based on z-score
        if z_score_cutoff is not None:
            std_val = np.std(points_2d, axis=0)
            # Prevent division by zero
            std_val[std_val == 0] = 1
            # Compute z-scores in one pass
            z_score = np.abs((points_2d - midpoint_2d) / std_val)
            mask = np.all(z_score < z_score_cutoff, axis=1)
            points_2d = points_2d[mask]
        
        if len(points_2d) >= 10: # Ensure enough points for meaningful radius estimation
            # Sort points to form a simple polygon (clockwise or counterclockwise)
            center = np.mean(points_2d, axis=0)
            angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_points = points_2d[sorted_indices]

            # Apply the shoelace formula to calculate the area
            x = sorted_points[:, 0]
            y = sorted_points[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # Calculate the circle-equivalent radius
            ce_radius = np.sqrt(area / np.pi)
            
            # Compute midpoint in 2D coordinates
            if not is_close_to_bif and use_meanpoint:
                meanpoint_2d = np.mean(points_2d, axis=0)
            else:
                meanpoint_2d = midpoint_2d

            distances = np.linalg.norm(points_2d - meanpoint_2d, axis=1)
            
            # Sort distances and take the median of the n closest points
            sorted_distances = np.sort(distances)
            n_points_all = len(points_2d)

            # dynamically determine n_points_min_rad based on the number of points
            n_points_min_rad = int(np.floor(n_points_all / 10))
            n_points_min_rad = max(4, n_points_min_rad) # Ensure at least 4 points
            n_points_min_rad = min(10, n_points_min_rad)  # Ensure not more than 10
            
            n_pts = min(n_points_min_rad*2, len(sorted_distances))  
            min_rad = np.median(sorted_distances[:n_pts]) # compute min radius as median of n closest points
            
            # Combine conditions to avoid redundant checks
            if is_close_to_bif and use_ce_rad_cap:
                if ce_radius > bif_min_rad_factor * min_rad:
                    ce_radius = bif_min_rad_factor * min_rad
                elif use_meanpoint:
                    # Only recompute when necessary
                    meanpoint_2d = np.mean(points_2d, axis=0)
                    distances = np.linalg.norm(points_2d - meanpoint_2d, axis=1)
                    sorted_distances = np.sort(distances)
                    n_pts = min(n_points_min_rad*2, len(sorted_distances))  # In case there are fewer than 10 points
                    min_rad = np.median(sorted_distances[:n_pts])

            return min_rad, ce_radius
            
        else:
            return np.nan, np.nan
    else:
        # Fallback in case of insufficient points
        return np.nan, np.nan

          
def cross_section_analysis(centerline, nii_mask, node_dict, n_jobs=12, use_meanpoint=False, 
                           use_ce_rad_cap=False, bif_min_rad_factor=1.33):
    """
    Radius computation based on cross-sectional analysis (CSA) along the centerline edges. 
    This function calculates various radius measurements (MIS, CE) for each
    edge along a centerline based using the surface mesh extracted for each segment. 
    
    Parameters
    ----------
    centerline : vtkPolyData
        The centerline structure for which to compute radius measurements.
    nii_mask : nibabel.Nifti1Image
        The segmentation mask corresponding to the centerline.
    node_dict : dict
        Dictionary containing information about nodes in the centerline.
    n_jobs : int, optional
        Number of parallel jobs to run for computation. Default is 12.
    use_meanpoint : bool, optional
        If True, uses the mean of the surface points for radius calculations. Default is False.
    use_ce_rad_cap : bool, optional
        If True, applies a cap on the circle-equivalent radius close to bifurcations. Default is False.
    bif_min_rad_factor : float, optional
        Factor to adjust radius measurements at bifurcation points. Default is 1.25.
    
    Returns
    -------
    vtkPolyData
        The input centerline with added cell data arrays for radius measurements:
        - 'mis_radius': Maximally inscribed sphere radius
        - 'ce_radius': circle equivalent radius
    """
    # Prepare arrays for storing radius measurements
    min_array = vtk.vtkDoubleArray()
    min_array.SetName('mis_radius')
    ce_array = vtk.vtkDoubleArray()
    ce_array.SetName('ce_radius')
    # NOTE: Only needed for voreen original graphs
    # avg_array = vtk.vtkDoubleArray()
    # avg_array.SetName('voreen_radius')

    nii_data_orig = nii_mask.get_fdata().astype(np.uint8)

    labels_array = get_label_array(centerline)
    unique_labels = np.unique(labels_array)
    label_masks = {}
    distance_transforms = {}
    logger.info(f'Preparing binary masks and distance fields for each segment separately...')

    def df_process_single_label(label, nii_data_orig):
        """Process a single label to create mask and distance transform"""
        # Create binary mask for this label
        mask = (nii_data_orig == label).astype(np.uint8)
        # Compute distance transform
        distance_transform = ndimage.distance_transform_edt(1 - mask)
        return label, mask, distance_transform

    # Process all labels in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(df_process_single_label)(label, nii_data_orig) 
        for label in unique_labels
    )
    
    # Assign results to appropriate dictionaries
    for label, mask, distance_transform in results:
        label_masks[label] = [mask]  # Wrap in list to keep consistent format
        distance_transforms[label] = [distance_transform]  # Wrap in list to keep consistent format

    # Optionally, close to bifurcations the radius estimate can be capped for the CE radius (not done by default)
    if use_meanpoint or use_ce_rad_cap:
        logger.info(f'Extracting cells near bifurcations for radius correction (use_meanpoint={use_meanpoint}, use_ce_rad_cap={use_ce_rad_cap})')
        boundary_ids, ngh_labels, bif_near_cells, bif_point_ids = get_boundary_and_bif_cell_ids(node_dict, centerline)
        bif_near_cells, _ = get_remaining_bif_ids(bif_near_cells, bif_point_ids, centerline)  
    else:
        bif_near_cells = []

    logger.info(f'Creating meshes for all segments separately...')
    segments_meshes = get_seg_meshes(nii_mask, label_masks, distance_transforms, n_jobs=n_jobs)

    logger.info(f'Computing MIS and CE radius values for each centerline cell using {n_jobs} parallel jobs...')
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_cell_surface_meshes)(
            i, centerline, segments_meshes, labels_array, bif_near_cells, use_meanpoint, use_ce_rad_cap, bif_min_rad_factor
            ) 
            for i in range(centerline.GetNumberOfCells())
    )
    
    logger.info(f'...done! Checking for NaN values now.')
    # NOTE: Isolated single cell nan values are interpolated. 
    # Two or more consecutive nan values are kept for broken segmentations! 
    has_nans = any(np.isnan(val) for result in results for val in result)
    if has_nans:
        for i in range(len(results)):
            result_values = list(results[i])
            
            for j in range(2):  # For each of the two values (min_rad, ce_rad)
                if np.isnan(result_values[j]):
                    # Check if this is an isolated NaN value (both neighbors are non-NaN)
                    is_isolated_nan = (i > 0 and i < len(results) - 1 and 
                                     not np.isnan(results[i-1][j]) and 
                                     not np.isnan(results[i+1][j]))
                    
                    if is_isolated_nan:
                        # For isolated NaN values, use the average of neighbors
                        logger.info(f"\tIsolated NaN radius value found! Interpolating NaN value at index {i}, value {j}")
                        # Get the label of the current cell
                        current_label = labels_array[i]

                        # Check if neighbors exist and have the same label
                        valid_neighbors = []
                        if i > 0 and labels_array[i-1] == current_label:
                            valid_neighbors.append(results[i-1][j])
                        if i < len(results) - 1 and labels_array[i+1] == current_label:
                            valid_neighbors.append(results[i+1][j])

                        # Only interpolate if we have valid neighbors with the same label
                        if len(valid_neighbors) == 2:
                            result_values[j] = (valid_neighbors[0] + valid_neighbors[1]) / 2.0
                        elif len(valid_neighbors) == 1:
                            result_values[j] = valid_neighbors[0]
                        else:
                            raise ValueError(
                                f"No valid neighbors found for cell {i} with label {current_label}. "
                                "Cannot interpolate NaN value."
                            )
                            # No valid neighbors with the same label
                            result_values[j] = 0.0  # Default value when no valid neighbors found
            
            results[i] = tuple(result_values)

    for min_rad, ce_rad in results:
        min_array.InsertNextValue(min_rad)
        ce_array.InsertNextValue(ce_rad)

    # NOTE: Only needed for the original voreen centerline graphs!
    # avg_radii = centerline.GetCellData().GetArray('voreen_radius') 
    # for i in range(centerline.GetNumberOfCells()): 
    #     avg_array.InsertNextValue(avg_radii.GetValue(i))

    # NOTE: if there are some unwanted radius attributes, you can remove them like shown below
    # if centerline.GetCellData().HasArray('avg_radius'):
    #     centerline.GetCellData().RemoveArray('avg_radius')
    
    centerline.GetCellData().AddArray(ce_array)
    centerline.GetCellData().AddArray(min_array)

    return centerline

def run_radius_computation(cnt_vtp_file: str, mlt_mask_nii_file: str, node_dict_file: str, n_jobs: int = 12, 
                           use_meanpoint: bool = False, use_ce_rad_cap: bool = False):
    """
    Compute and update the radius along a centerline based on a mask and node dictionary.
    This function reads a centerline from a VTP file, a mask from a NII file, and a node dictionary
    from a JSON file. It then computes the radius along the centerline using Cross-Sectional Area (CSA)
    analysis and updates the original centerline file with the radius information.
    Args:
        cnt_vtp_file (str): Path to the VTP file containing the centerline data.
        mlt_mask_nii_file (str): Path to the NII file containing the mask data.
        node_dict_file (str): Path to the JSON file containing the node dictionary.
        n_jobs (int, optional): Number of parallel jobs for processing. Defaults to 12.
        use_meanpoint (bool, optional): Whether to use the mean of the surface mesh intersection for radius calculation. Defaults to False.
        use_ce_rad_cap (bool, optional): Whether to apply a cap on the circle-equivalent radius at bifurcations. Defaults to False.
                                        -> if True, a cap of 1.33 is used in the CSA() function below!
    Returns:
        None: The function modifies the input centerline file directly.
    """
    
    # Read the centerline and surface mesh
    centerline = get_vtk_polydata_from_file(cnt_vtp_file)
    nii_mask = nib.load(mlt_mask_nii_file)
    with open(node_dict_file, 'r') as f:
        node_dict = json.load(f)

    logger.info(f'Running radius computation along the centerline with args:'
                f'\n\t- cnt_vtp_file={cnt_vtp_file}'
                f'\n\t- mlt_mask_nii_file={mlt_mask_nii_file}'
                f'\n\t- node_dict_file={node_dict_file}'
                f'\n\t- n_jobs={n_jobs}'
                f'\n\t- use_meanpoint={use_meanpoint}'
                f'\n\t- use_ce_rad_cap={use_ce_rad_cap}')

    centerline = cross_section_analysis(centerline, nii_mask, node_dict, n_jobs=n_jobs, use_meanpoint=use_meanpoint, 
                                        use_ce_rad_cap=use_ce_rad_cap, bif_min_rad_factor=1.33)
    
    # save centerline to file
    logger.info(f"Saving centerline with updated radius information to {cnt_vtp_file}")
    write_vtk_polydata_to_file(centerline, cnt_vtp_file)



