import os
import json
import numpy as np
from utils.utils_graph_processing import *
from utils.utils_feature_extraction import *

from logger import logger

def extract_ba_ica_bif_geometry(nodes_dict, variant_dict, polydata, radius_attribute='ce_radius', bif_name='BA',
                                use_fixed_dist_angle=False, dist_angle=3, angle_average=3, 
                                use_fixed_dist_radius=False, dist_radius=3, radius_average=3):
    """
    Extract bifurcation geometry: angles, radius ratios, bifurcation exponents for ICA and BA bifurcations

    Args:
    nodes_dict: dict, dictionary containing
    variant_dict: dict, dictionary containing CoW variants
    polydata: vtkPolyData, polydata of the vessel
    radius_attribute: str, attribute name for radius computation. ('ce_radius', 'voreen_radius', 'mis_radius', 'max_radius')
    bif_name: str, bifurcation name ['BA', 'R-ICA', 'L-ICA']
    use_fixed_dist_angle: bool, if True use fixed distance for angle computation (else use dynamic distance based on segment boundary points)
    dist_angle: float, distance from bifurcation point for angle computation
    angle_average: int, number of points to average for angle computation
    use_fixed_dist_radius: bool, if True use fixed distance for radius computation
    dist_radius: float, distance from bifurcation point for radius computation (if use_fixed_dist_radius is True)
    radius_average: int, number of variants to average for radius computation

    Returns:
    bif_geom: list, list containing dictionary of bifurcation geometry
    """
    bif_geom = []
    bif, parent_start = None, None

    if bif_name == 'BA':
        if variant_dict['posterior']['R-P1'] and variant_dict['posterior']['L-P1']:
            if '1' in nodes_dict:
                parent_label, child1_label, child2_label = 1, 2, 3
                bif, parent_start = extract_node_entries(nodes_dict, 1, ['BA bifurcation', 'BA start'])
                child1_boundary, child1_end = extract_node_entries(nodes_dict, 2, ['BA boundary', 'PCA end'])
                child2_boundary, child2_end = extract_node_entries(nodes_dict, 3, ['BA boundary', 'PCA end'])
                name_parent_child1 = 'BA/R-PCA'
                name_parent_child2 = 'BA/L-PCA'
                name_child1_child2 = 'R-PCA/L-PCA'
                name_child1_parent = 'R-PCA/BA'
                name_child2_parent = 'L-PCA/BA'
                parent_name, child1_name, child2_name = 'BA', 'R-PCA', 'L-PCA'

    elif bif_name == 'R-ICA':
        if variant_dict['anterior']['R-A1']:
            if '4' in nodes_dict:
                parent_label, child1_label, child2_label = 4, 5, 11
                bif, parent_start = extract_node_entries(nodes_dict, 4, ['ICA bifurcation', 'ICA start'])
                if parent_start is None:
                    parent_start = extract_node_entries(nodes_dict, 4, ['Pcom boundary'])[0]
                    assert parent_start is not None
                child1_boundary, child1_end = extract_node_entries(nodes_dict, 5, ['ICA boundary', 'MCA end'])
                child2_boundary, child2_end = extract_node_entries(nodes_dict, 11, ['ICA boundary', 'ACA end'])
                name_parent_child1 = 'ICA/MCA'
                name_parent_child2 = 'ICA/ACA'
                name_child1_child2 = 'MCA/ACA'
                name_child1_parent = 'MCA/ICA'
                name_child2_parent = 'ACA/ICA'
                parent_name, child1_name, child2_name = 'ICA', 'MCA', 'ACA'
    
    elif bif_name == 'L-ICA':
        if variant_dict['anterior']['L-A1']:
            if '6' in nodes_dict:
                parent_label, child1_label, child2_label = 6, 7, 12
                bif, parent_start = extract_node_entries(nodes_dict, 6, ['ICA bifurcation', 'ICA start'])
                if parent_start is None:
                    parent_start = extract_node_entries(nodes_dict, 6, ['Pcom boundary'])[0]
                    assert parent_start is not None
                child1_boundary, child1_end = extract_node_entries(nodes_dict, 7, ['ICA boundary', 'MCA end'])
                child2_boundary, child2_end = extract_node_entries(nodes_dict, 12, ['ICA boundary', 'ACA end'])
                name_parent_child1 = 'ICA/MCA'
                name_parent_child2 = 'ICA/ACA'
                name_child1_child2 = 'MCA/ACA'
                name_child1_parent = 'MCA/ICA'
                name_child2_parent = 'ACA/ICA'
                parent_name, child1_name, child2_name = 'ICA', 'MCA', 'ACA'
        
    else:
        raise ValueError("Wrong bifurcation name!")
    
    if bif is None or parent_start is None:
        # No bifurcation to extract
        pass

    else:
        bif_id = bif[0]['id']
        has_nan_rad = False
        has_nan_angle = False
        # get points at dist from bifurcation for parent and children vessels
        parent_path = find_shortest_path(bif_id, parent_start[0]['id'], polydata, parent_label)['path']
        if use_fixed_dist_angle:
            pointId_parent_angle = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_angle)
        if use_fixed_dist_radius:
            pointId_parent_rad = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_radius)
        
        if (child1_boundary is not None and len(child1_boundary) > 1):
            assert len(child2_boundary) == 1
            assert len(child1_boundary) == 2
            child1_paths = find_all_paths(bif_id, child1_end[0]['id'], polydata, [parent_label, child1_label])
            pointId_child1_angle = []
            pointId_child1_rad = []
            child1_path = []
            for path in child1_paths:
                path = path['path']
                id_angle = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist_angle)
                if id_angle not in pointId_child1_angle:
                    pointId_child1_angle.append(id_angle)
                id_rad = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist_radius)
                if id_rad not in pointId_child1_rad:
                    pointId_child1_rad.append()
                    child1_path.append(path)
            assert len(pointId_child1_rad) == 2
            assert len(pointId_child1_angle) == 2
            assert len(child1_boundary) == len(pointId_child1_rad) == len(pointId_child1_angle) == 2        

            child2_path = find_shortest_path(bif_id, child2_end[0]['id'], polydata, [parent_label, child2_label])['path']
            if use_fixed_dist_angle:
                pointId_child2_angle = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_angle)
            if use_fixed_dist_radius:
                pointId_child2_rad = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_radius)

            assert len(child1_boundary) == len(pointId_child1_rad) == len(pointId_child1_angle) == 2
            for i in range(len(child1_boundary)):

                if not use_fixed_dist_angle:
                    # we use the boundaries as references for the child vessels
                    pointId_child1_angle[i] = child1_boundary[i]['id']
                    pointId_child2_angle = child2_boundary[0]['id']
                    distances = get_distances_to_boundaries(bif, [child1_boundary[i]], child2_boundary)
                    # We take the average distance to get a reference point for the parent vessel
                    dist_angle_p = float(np.mean(distances))
                    pointId_parent_angle = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_angle_p)

                if not use_fixed_dist_radius:
                    # we use the boundaries as references for the child vessels
                    pointId_child1_rad[i] = child1_boundary[i]['id']
                    pointId_child2_rad = child2_boundary[0]['id']
                    distances = get_distances_to_boundaries(bif, [child1_boundary[i]], child2_boundary)
                    # We take the average distance to get a reference point for the parent vessel
                    dist_radius_p = float(np.mean(distances)) 
                    pointId_parent_rad = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_radius_p)
                    # id_rad = find_endpoint_for_fixed_length(child1_path[i], polydata, length_threshold=dist_radius)
                    # pointId_child1_rad[i] = id_rad
                    # pointId_child2_rad = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_radius)

                # check for nan
                for path, id in zip([parent_path, child1_path[i], child2_path], [pointId_parent_rad, pointId_child1_rad[i], pointId_child2_rad]):
                    if check_for_nan(path, id, polydata):
                        has_nan_rad = True
                        logger.warning(f'\tWarning: NaN found in bifurcation path for radius estimation for bifurcation {bif_name}!')
                        break
                if has_nan_rad:
                    for path, id in zip([parent_path, child1_path[i], child2_path], [pointId_parent_angle, pointId_child1_angle[i], pointId_child2_angle]):
                        if check_for_nan(path, id, polydata):
                            has_nan_angle = True
                            logger.warning(f'\tWarning: NaN found in bifurcation path for angle estimation for bifurcation {bif_name}!')
                            break
                
                if has_nan_angle:
                    angle_parent_child1, angle_parent_child2, angle_child1_child2 = np.nan, np.nan, np.nan
                else:
                    angle_parent_child1, angle_parent_child2, angle_child1_child2 = compute_angles(bif_id, pointId_parent_angle, pointId_child1_angle[i], pointId_child2_angle, 
                                                                                                   parent_label, child1_label, child2_label,
                                                                                                   polydata, nr_of_points_for_avg=angle_average)
                
                if has_nan_rad:
                    radius_parent, radius_child1, radius_child2 = np.nan, np.nan, np.nan
                else:
                    radius_parent, radius_child1, radius_child2 = compute_radii(pointId_parent_rad, pointId_child1_rad[i], pointId_child2_rad, polydata, radius_attribute=radius_attribute,
                                                                                nr_of_edges_for_avg=radius_average, start_parent=parent_start[0]['id'], end_child1=child1_end[0]['id'], end_child2=child2_end[0]['id'])
                            
                
                # compute ratios
                ratio_finet, ratio_pc1, ratio_pc2, ratio_c1c2, ratio_area_sum = compute_ratios(radius_parent, radius_child1, radius_child2)
                bif_exp = compute_bifurcation_exponent(radius_parent, radius_child1, radius_child2)
                
                bif_geom.append({
                    'bifurcation': {
                        'midpoint': bif_id,
                        'points_angle': [pointId_parent_angle, pointId_child1_angle[i], pointId_child2_angle],
                        'points_radius': [pointId_parent_rad, pointId_child1_rad[i], pointId_child2_rad],
                        'radius_parent': np.round(radius_parent,3),
                        'radius_child1': np.round(radius_child1,3),
                        'radius_child2': np.round(radius_child2,3),
                    },
                    'angles': {
                        name_parent_child1: np.round(angle_parent_child1,3),
                        name_parent_child2: np.round(angle_parent_child2,3),
                        name_child1_child2: np.round(angle_child1_child2,3)
                    },
                    'ratios': {
                        'radius sum': np.round(ratio_finet,3),
                        'area sum':  np.round(ratio_area_sum,3),
                        name_child1_child2: np.round(ratio_c1c2,3),
                        name_child1_parent: np.round(ratio_pc1,3),
                        name_child2_parent: np.round(ratio_pc2,3),
                    },
                    'bifurcation exponent': np.round(bif_exp,3)
                })


        elif (child2_boundary is not None and len(child2_boundary) > 1):
            assert len(child1_boundary) == 1
            assert len(child2_boundary) == 2

            child1_path = find_shortest_path(bif_id, child1_end[0]['id'], polydata, [parent_label, child1_label])['path']
            if use_fixed_dist_angle:
                pointId_child1_angle = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_angle)
            if use_fixed_dist_radius:
                pointId_child1_rad = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_radius)

            child2_paths = find_all_paths(bif_id, child2_end[0]['id'], polydata, [parent_label, child2_label])
            pointId_child2_angle = []
            pointId_child2_rad = []
            child2_path = []
            for path in child2_paths:
                path = path['path']
                id_angle = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist_angle)
                if id_angle not in pointId_child2_angle:
                    pointId_child2_angle.append(id_angle)
                id_rad = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist_radius)
                if id_rad not in pointId_child2_rad:
                    pointId_child2_rad.append(id_rad)
                    child2_path.append(path)

            assert len(pointId_child2_rad) == 2
            assert len(pointId_child2_angle) == 2
            assert len(child2_boundary) == len(pointId_child2_rad) == len(pointId_child2_angle) == 2
            

            for i in range(len(child2_boundary)):

                if not use_fixed_dist_angle:
                    # we use the boundaries as references for the child vessels
                    pointId_child1_angle = child1_boundary[0]['id']
                    pointId_child2_angle[i] = child2_boundary[i]['id']  
                    distances = get_distances_to_boundaries(bif, child1_boundary, [child2_boundary[i]])
                    # We take the average distance to get a reference point for the parent vessel
                    dist_angle_p = float(np.mean(distances))
                    pointId_parent_angle = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_angle_p)

                if not use_fixed_dist_radius:
                    # we use the boundaries as references for the child vessels
                    pointId_child1_rad = child1_boundary[0]['id']
                    pointId_child2_rad[i] = child2_boundary[i]['id']
                    distances = get_distances_to_boundaries(bif, child1_boundary, [child2_boundary[i]])
                    # We take the average distance to get a reference point for the parent vessel
                    dist_radius_p = float(np.mean(distances)) 
                    pointId_parent_rad = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_radius_p)
                    # pointId_child1_rad = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_radius)
                    # id_rad = find_endpoint_for_fixed_length(child2_path[i], polydata, length_threshold=dist_radius)
                    # pointId_child2_rad[i] = id_rad

                # check for nan
                for path, id in zip([parent_path, child1_path, child2_path[i]], [pointId_parent_rad, pointId_child1_rad, pointId_child2_rad[i]]):
                    if check_for_nan(path, id, polydata):
                        has_nan_rad = True
                        logger.warning(f'\tWarning: NaN found in bifurcation path for radius estimation for bifurcation {bif_name}!')
                        break 
                if has_nan_rad:
                    for path, id in zip([parent_path, child1_path, child2_path[i]], [pointId_parent_angle, pointId_child1_angle, pointId_child2_angle[i]]):
                        if check_for_nan(path, id, polydata):
                            has_nan_angle = True
                            logger.warning(f'\tWarning: NaN found in bifurcation path for angle estimation for bifurcation {bif_name}!')
                            break
                
                if has_nan_angle:
                    angle_parent_child1, angle_parent_child2, angle_child1_child2 = np.nan, np.nan, np.nan
                else:
                    angle_parent_child1, angle_parent_child2, angle_child1_child2 = compute_angles(bif_id, pointId_parent_angle, pointId_child1_angle, pointId_child2_angle[i], 
                                                                                                   parent_label, child1_label, child2_label, 
                                                                                                   polydata, nr_of_points_for_avg=angle_average)
                
                if has_nan_rad:
                    radius_parent, radius_child1, radius_child2 = np.nan, np.nan, np.nan
                else:
                    radius_parent, radius_child1, radius_child2 = compute_radii(pointId_parent_rad, pointId_child1_rad, pointId_child2_rad[i], polydata, radius_attribute=radius_attribute,
                                                                                nr_of_edges_for_avg=radius_average, start_parent=parent_start[0]['id'], end_child1=child1_end[0]['id'], end_child2=child2_end[0]['id'])
                            
                
                # compute ratios
                ratio_finet, ratio_pc1, ratio_pc2, ratio_c1c2, ratio_area_sum = compute_ratios(radius_parent, radius_child1, radius_child2)
                bif_exp = compute_bifurcation_exponent(radius_parent, radius_child1, radius_child2)
                
                bif_geom.append({
                    'bifurcation': {
                        'midpoint': bif_id,
                        'points_angle': [pointId_parent_angle, pointId_child1_angle, pointId_child2_angle[i]],
                        'points_radius': [pointId_parent_rad, pointId_child1_rad, pointId_child2_rad[i]],
                        'radius_parent': np.round(radius_parent,3),
                        'radius_child1': np.round(radius_child1,3),
                        'radius_child2': np.round(radius_child2,3),
                    },
                    'angles': {
                        name_parent_child1: np.round(angle_parent_child1,3),
                        name_parent_child2: np.round(angle_parent_child2,3),
                        name_child1_child2: np.round(angle_child1_child2,3)
                    },
                    'ratios': {
                        'radius sum': np.round(ratio_finet,3),
                        'area sum':  np.round(ratio_area_sum,3),
                        name_child1_child2: np.round(ratio_c1c2,3),
                        name_child1_parent: np.round(ratio_pc1,3),
                        name_child2_parent: np.round(ratio_pc2,3)
                    },
                    'bifurcation exponent': np.round(bif_exp,3)
                })
        
        else:
            child1_path = find_shortest_path(bif_id, child1_end[0]['id'], polydata, [parent_label, child1_label])['path']
            if use_fixed_dist_angle:
                pointId_child1_angle = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_angle)
            if use_fixed_dist_radius:
                pointId_child1_rad = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_radius)

            child2_path = find_shortest_path(bif_id, child2_end[0]['id'], polydata, [parent_label, child2_label])['path']
            if use_fixed_dist_angle:
                pointId_child2_angle = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_angle)
            if use_fixed_dist_radius:
                pointId_child2_rad = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_radius)

            if not use_fixed_dist_angle:
                # We use the boundaries as references for the child vessels
                pointId_child1_angle = child1_boundary[0]['id']
                pointId_child2_angle = child2_boundary[0]['id']
                distances = get_distances_to_boundaries(bif, child1_boundary, child2_boundary)
                # We take the average distance to get a reference point for the parent vessel
                dist_angle_p = float(np.mean(distances))
                pointId_parent_angle = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_angle_p)

            if not use_fixed_dist_radius:
                # We use the boundaries as references for the child vessels
                pointId_child1_rad = child1_boundary[0]['id']
                pointId_child2_rad = child2_boundary[0]['id']
                distances = get_distances_to_boundaries(bif, child1_boundary, child2_boundary)
                # We take the average distance to get a reference point for the parent vessel
                dist_radius_p = float(np.mean(distances))
                pointId_parent_rad = find_endpoint_for_fixed_length(parent_path, polydata, length_threshold=dist_radius_p)
                # pointId_child1_rad = find_endpoint_for_fixed_length(child1_path, polydata, length_threshold=dist_radius_p)
                # pointId_child2_rad = find_endpoint_for_fixed_length(child2_path, polydata, length_threshold=dist_radius_p)
            
            for path, id in zip([parent_path, child1_path, child2_path], [pointId_parent_rad, pointId_child1_rad, pointId_child2_rad]):
                if check_for_nan(path, id, polydata):
                    has_nan_rad = True
                    logger.warning(f'\tWarning: NaN found in bifurcation path for radius estimation for bifurcation {bif_name}!')
                    break 
            if has_nan_rad:
                for path, id in zip([parent_path, child1_path, child2_path], [pointId_parent_angle, pointId_child1_angle, pointId_child2_angle]):
                    if check_for_nan(path, id, polydata):
                        has_nan_angle = True
                        logger.warning(f'\tWarning: NaN found in bifurcation path for angle estimation for bifurcation {bif_name}!')
                        break

            if has_nan_angle:
                angle_parent_child1, angle_parent_child2, angle_child1_child2 = np.nan, np.nan, np.nan
            else:            
                angle_parent_child1, angle_parent_child2, angle_child1_child2 = compute_angles(bif_id, pointId_parent_angle, pointId_child1_angle, pointId_child2_angle, 
                                                                                               parent_label, child1_label, child2_label,
                                                                                               polydata, nr_of_points_for_avg=angle_average)
                                                                                               
            
            if has_nan_rad:
                radius_parent, radius_child1, radius_child2 = np.nan, np.nan, np.nan
            else:
                radius_parent, radius_child1, radius_child2 = compute_radii(pointId_parent_rad, pointId_child1_rad, pointId_child2_rad, polydata, radius_attribute=radius_attribute,
                                                                            nr_of_edges_for_avg=radius_average, start_parent=parent_start[0]['id'], end_child1=child1_end[0]['id'], end_child2=child2_end[0]['id'])      
            
            # compute ratios
            ratio_finet, ratio_pc1, ratio_pc2, ratio_c1c2, ratio_area_sum = compute_ratios(radius_parent, radius_child1, radius_child2)
            bif_exp = compute_bifurcation_exponent(radius_parent, radius_child1, radius_child2)
            
            bif_geom.append({
                'bifurcation': {
                    'midpoint': bif_id,
                    'points_angle': [pointId_parent_angle, pointId_child1_angle, pointId_child2_angle],
                    'points_radius': [pointId_parent_rad, pointId_child1_rad, pointId_child2_rad],
                    'radius_parent': np.round(radius_parent,3),
                    'radius_child1': np.round(radius_child1,3),
                    'radius_child2': np.round(radius_child2,3),
                },
                'angles': {
                    name_parent_child1: np.round(angle_parent_child1,3),
                    name_parent_child2: np.round(angle_parent_child2,3),
                    name_child1_child2: np.round(angle_child1_child2,3)
                },
                'ratios': {
                    'radius sum': np.round(ratio_finet,3),
                    'area sum':  np.round(ratio_area_sum,3),
                    name_child1_child2: np.round(ratio_c1c2,3),
                    name_child1_parent: np.round(ratio_pc1,3),
                    name_child2_parent: np.round(ratio_pc2,3)  
                },
                'bifurcation exponent': np.round(bif_exp,3)
            })

    return bif_geom
    
def extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, bif_location='R-ICA', dist=1.5, angle_average=3):
    """
    Extract Acom/Pcom bifurcation angles

    Args:
    nodes_dict: dict, dictionary containing
    variant_dict: dict, dictionary containing CoW variants
    polydata: vtkPolyData, polydata of the vessel
    bif_location: str, bifurcation name ['R-ICA', 'L-ICA', 'R-PCA', 'L-PCA', 'R-ACA', 'L-ACA']
    dist: float, distance from bifurcation point for angle computation
    angle_average: int, number of points to average for angle computation

    Returns:
    bif_geom: list, list containing dictionary of bifurcation angles
    """
    bif_geom = []
    bif, parent_start, parent_end, com_end = None, None, None, None

    if bif_location == 'R-ICA':
        if variant_dict['posterior']['R-Pcom']:
            parent_label, com_label = 4, 8
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 4, ['Pcom bifurcation', 'ICA start', 'ICA bifurcation'])
            if parent_end is None:
                parent_end = extract_node_entries(nodes_dict, 4, ['MCA boundary'])[0]
                if parent_end is None:
                    parent_end = extract_node_entries(nodes_dict, 4, ['ACA boundary'])[0]
            assert parent_end is not None
            com_boundary, com_end = extract_node_entries(nodes_dict, 8, ['ICA boundary', 'PCA boundary'])
            name_parent_start_end = 'C6/C7'
            name_parent_start_com = 'C6/Pcom'
            name_parent_end_com = 'C7/Pcom'
            parent_name_start, parent_name_end, com_name = 'C6', 'C7', 'Pcom'
    
    elif bif_location == 'L-ICA':
        if variant_dict['posterior']['L-Pcom']:
            parent_label, com_label = 6, 9
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 6, ['Pcom bifurcation', 'ICA start', 'ICA bifurcation'])
            if parent_end is None:
                parent_end = extract_node_entries(nodes_dict, 6, ['MCA boundary'])[0]
                if parent_end is None:
                    parent_end = extract_node_entries(nodes_dict, 6, ['ACA boundary'])[0]
            assert parent_end is not None
            com_boundary, com_end = extract_node_entries(nodes_dict, 9, ['ICA boundary', 'PCA boundary'])
            name_parent_start_end = 'C6/C7'
            name_parent_start_com = 'C6/Pcom'
            name_parent_end_com = 'C7/Pcom'
            parent_name_start, parent_name_end, com_name = 'C6', 'C7', 'Pcom'
    
    elif bif_location == 'R-PCA':
        if variant_dict['posterior']['R-Pcom']:
            parent_label, com_label = 2, 8
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 2, ['Pcom bifurcation', 'BA boundary', 'PCA end'])
            com_boundary, com_end = extract_node_entries(nodes_dict, 8, ['PCA boundary', 'ICA boundary'])
            name_parent_start_end = 'P1/P2'
            name_parent_start_com = 'P1/Pcom'
            name_parent_end_com = 'P2/Pcom'
            parent_name_start, parent_name_end, com_name = 'P1', 'P2', 'Pcom'
    
    elif bif_location == 'L-PCA':
        if variant_dict['posterior']['L-Pcom']:
            parent_label, com_label = 3, 9
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 3, ['Pcom bifurcation', 'BA boundary', 'PCA end'])
            com_boundary, com_end = extract_node_entries(nodes_dict, 9, ['PCA boundary', 'ICA boundary'])
            name_parent_start_end = 'P1/P2'
            name_parent_start_com = 'P1/Pcom'
            name_parent_end_com = 'P2/Pcom'
            parent_name_start, parent_name_end, com_name = 'P1', 'P2', 'Pcom'
    
    elif bif_location == 'R-ACA':
        if variant_dict['anterior']['Acom']:
            parent_label, com_label = 11, 10
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 11, ['Acom bifurcation', 'ICA boundary', 'ACA end'])
            com_boundary, com_end = extract_node_entries(nodes_dict, 10, ['R-ACA boundary', 'L-ACA boundary'])
            name_parent_start_end = 'A1/A2'
            name_parent_start_com = 'A1/Acom'
            name_parent_end_com = 'A2/Acom'
            parent_name_start, parent_name_end, com_name = 'A1', 'A2', 'Acom'

    elif bif_location == 'L-ACA':
        if variant_dict['anterior']['Acom']:
            parent_label, com_label = 12, 10
            bif, parent_start, parent_end = extract_node_entries(nodes_dict, 12, ['Acom bifurcation', 'ICA boundary', 'ACA end'])
            com_boundary, com_end = extract_node_entries(nodes_dict, 10, ['L-ACA boundary', 'R-ACA boundary'])
            name_parent_start_end = 'A1/A2'
            name_parent_start_com = 'A1/Acom'
            name_parent_end_com = 'A2/Acom'
            parent_name_start, parent_name_end, com_name = 'A1', 'A2', 'Acom'
    
    else:
        raise ValueError("Wrong bifurcation name!")

    if bif is None or parent_start is None or parent_end is None or com_end is None:
        # No bifurcation to extract
        pass
    
    else:
        # NOTE: bif_id, parent_start or com_end can have more than one entry!
        has_nan_angle = False
        if len(bif) == 1:
            bif_id = bif[0]['id']
            if len(parent_start) > 1: # P1 fenestration with two BA/PCA boundary points
                assert len(parent_start) == 2
                assert len(com_end) == 1
                assert parent_label == 2 or parent_label == 3
                parent_start_path = []
                pointId_parent_start = []
                for i in range(len(parent_start)):
                    parent_start_paths = find_all_paths(bif_id, parent_start[i]['id'], polydata, [parent_label, com_label])
                    for path in parent_start_paths:
                        path = path['path']
                        id_start = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist)
                        if id_start not in pointId_parent_start:
                            pointId_parent_start.append(id_start)
                            parent_start_path.append(path)
                
                parent_end_path = find_shortest_path(bif_id, parent_end[0]['id'], polydata, parent_label)['path']
                pointId_parent_end = find_endpoint_for_fixed_length(parent_end_path, polydata, length_threshold=dist)

                com_path = find_shortest_path(bif_id, com_end[0]['id'], polydata, [parent_label, com_label])['path']
                pointId_com = find_endpoint_for_fixed_length(com_path, polydata, length_threshold=dist)

                for i in range(len(pointId_parent_start)):
                    if pointId_parent_start[i] == pointId_parent_end:
                        continue
                    # check for nan
                    for path, id in zip([parent_start_path[i], parent_end_path, com_path], [pointId_parent_start[i], pointId_parent_end, pointId_com]):
                        if check_for_nan(path, id, polydata):
                            logger.warning(f'\tWarning: NaN found in bifurcation path for angle estimation for communicating bifurcation at {bif_location}!')
                            has_nan_angle = True
                            break
                    if has_nan_angle:
                        angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = np.nan, np.nan, np.nan
                    else:
                        angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = compute_angles(bif_id, pointId_parent_start[i], pointId_parent_end, pointId_com,
                                                                                                              parent_label, parent_label, com_label, 
                                                                                                              polydata, nr_of_points_for_avg=angle_average)
                    bif_geom.append({
                        'bifurcation': {
                            'midpoint': bif_id,
                            'dist_angle': dist,
                            'points_angle': [pointId_parent_start[i], pointId_parent_end, pointId_com]
                        },
                        'angles': {
                            name_parent_start_end: np.round(angle_parent_start_end,3),
                            name_parent_start_com: np.round(angle_parent_start_com,3),
                            name_parent_end_com: np.round(angle_parent_end_com,3)
                        }
                    })
            else:
                assert len(parent_start) == 1
                if len(com_end) == 1: # usual case
                    # Do we need to consider multiple paths? Maybe for A1 fenestrations...?
                    parent_start_path = find_shortest_path(bif_id, parent_start[0]['id'], polydata, parent_label)['path']
                    pointId_parent_start = find_endpoint_for_fixed_length(parent_start_path, polydata, length_threshold=dist)

                    parent_end_path = find_shortest_path(bif_id, parent_end[0]['id'], polydata, parent_label)['path']
                    pointId_parent_end = find_endpoint_for_fixed_length(parent_end_path, polydata, length_threshold=dist)

                    com_path = find_shortest_path(bif_id, com_end[0]['id'], polydata, [parent_label, com_label])['path']
                    pointId_com = find_endpoint_for_fixed_length(com_path, polydata, length_threshold=dist)
                    
                    if pointId_parent_start == pointId_parent_end: # might happen for P1 fenestrations
                        parent_start_paths = find_all_paths(bif_id, parent_start[0]['id'], polydata, parent_label)
                        if len(parent_start_paths) == 2:
                            for path in parent_start_paths:
                                path = path['path']
                                id_start = find_endpoint_for_fixed_length(path, polydata, length_threshold=dist)
                                if id_start != pointId_parent_end:
                                    pointId_parent_start = id_start
                                    break

                    # check for nan
                    for path, id in zip([parent_start_path, parent_end_path, com_path], [pointId_parent_start, pointId_parent_end, pointId_com]):
                        if check_for_nan(path, id, polydata):
                            logger.warning(f'\tWarning: NaN found in bifurcation path for angle estimation for communicating bifurcation at {bif_location}!')
                            has_nan_angle = True
                            break
                    if has_nan_angle:
                        angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = np.nan, np.nan, np.nan
                    else:
                        angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = compute_angles(bif_id, pointId_parent_start, pointId_parent_end, pointId_com, 
                                                                                                              parent_label, parent_label, com_label, 
                                                                                                              polydata, nr_of_points_for_avg=angle_average)

                    bif_geom.append({
                        'bifurcation': {
                            'midpoint': bif_id,
                            'dist_angle': dist,
                            'points_angle': [pointId_parent_start, pointId_parent_end, pointId_com]
                        },
                        'angles': {
                            name_parent_start_end: np.round(angle_parent_start_end,3),
                            name_parent_start_com: np.round(angle_parent_start_com,3),
                            name_parent_end_com: np.round(angle_parent_end_com,3)
                        }
                    })
                
                else: # One-sided Acom fenestration
                    assert len(com_end) == 2
                    assert com_label == 10

                    # Do we need to consider multiple paths? Maybe for A1 fenestrations...?
                    # TODO: Check (A1) fenestrations and reconsider...
                    parent_start_path = find_shortest_path(bif_id, parent_start[0]['id'], polydata, parent_label)['path']
                    pointId_parent_start = find_endpoint_for_fixed_length(parent_start_path, polydata, length_threshold=dist)

                    parent_end_path = find_shortest_path(bif_id, parent_end[0]['id'], polydata, parent_label)['path']
                    pointId_parent_end = find_endpoint_for_fixed_length(parent_end_path, polydata, length_threshold=dist)

                    pointId_com = []
                    for i in range(len(com_end)):
                        com_path = find_shortest_path(bif_id, com_end[i]['id'], polydata, [parent_label, com_label])['path']
                        id_com = find_endpoint_for_fixed_length(com_path, polydata, length_threshold=dist)
                        if id_com not in pointId_com:
                            pointId_com.append(id_com)
                    
                    for i in range(len(pointId_com)):
                        angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = compute_angles(bif_id, pointId_parent_start, pointId_parent_end, pointId_com[i],
                                                                                                              parent_label, parent_label, com_label, 
                                                                                                              polydata, nr_of_points_for_avg=angle_average)

                        bif_geom.append({
                            'bifurcation': {
                                'midpoint': bif_id,
                                'dist_angle': dist,
                                'points_angle': [pointId_parent_start, pointId_parent_end, pointId_com[i]]
                            },
                            'angles': {
                                name_parent_start_end: np.round(angle_parent_start_end,3),
                                name_parent_start_com: np.round(angle_parent_start_com,3),
                                name_parent_end_com: np.round(angle_parent_end_com,3)
                            }
                        })

        elif len(bif) > 1: # Acom fenestration
            assert com_label == 10
            assert len(bif) == 2
            for i in range(len(bif)):
                bif_id = bif[i]['id']
                parent_start_path = find_shortest_path(bif_id, parent_start[0]['id'], polydata, parent_label)['path']
                pointId_parent_start = find_endpoint_for_fixed_length(parent_start_path, polydata, length_threshold=dist)

                parent_end_path = find_shortest_path(bif_id, parent_end[0]['id'], polydata, parent_label)['path']
                pointId_parent_end = find_endpoint_for_fixed_length(parent_end_path, polydata, length_threshold=dist)

                if len(com_end) == 1:
                    com_end_id = com_end[0]['id']
                elif len(com_end) == 2:
                    # Only works if the ordering of bif and com_end is the same! 
                    com_end_id = com_end[i]['id']
                else:
                    raise ValueError("Wrong number of com ends!")
                com_path = find_shortest_path(bif_id, com_end_id, polydata, [parent_label, com_label])['path']
                # Acom fenestration: make sure that path com_path doesn't cross the other bifurcation
                pointIds_com_path = [p[1] for p in com_path]
                if bif[(i+1)%2]['id'] in pointIds_com_path:
                    com_paths = find_all_paths(bif_id, com_end_id, polydata, [parent_label, com_label])
                    for path in com_paths:
                        path = path['path']
                        if path != com_path:
                            com_path = path
                            break
                pointId_com = find_endpoint_for_fixed_length(com_path, polydata, length_threshold=dist)

                angle_parent_start_end, angle_parent_start_com, angle_parent_end_com = compute_angles(bif_id, pointId_parent_start, pointId_parent_end, pointId_com, 
                                                                                                      parent_label, parent_label, com_label,
                                                                                                      polydata, nr_of_points_for_avg=angle_average)

                bif_geom.append({
                    'bifurcation': {
                        'midpoint': bif_id,
                        'dist_angle': dist,
                        'points_angle': [pointId_parent_start, pointId_parent_end, pointId_com]
                    },
                    'angles': {
                        name_parent_start_end: np.round(angle_parent_start_end,3),
                        name_parent_start_com: np.round(angle_parent_start_com,3),
                        name_parent_end_com: np.round(angle_parent_end_com,3)
                    }
                })
    
    return bif_geom

def extract_bifurcation_geometry(cnt_vtp_file: str, variant_dir: str, node_dir: str, feature_dir: str, 
                                 radius_attribute: str = 'ce_radius', use_fixed_dist_angle=False, dist_angle: float = 1.5, 
                                 angle_average: float = 3, radius_average: float = 3, use_fixed_dist_radius: bool = False, 
                                 dist_radius: float = 2.5):
    """
    Extract bifurcation geometry: 
        - angles, radius ratios, bifurcation exponents for ICA and BA bifurcations
        - angles for Acom and Pcom bifurcations
    
    Args:
    cnt_vtp_file: str, path to centerline vtp file
    variant_dir: str, path to directory containing variant json files
    node_dir: str, path to directory containing node json files
    feature_dir: str, path to directory to save feature json files
    radius_attribute: str, attribute name for radius computation. ('ce_radius', 'mis_radius')
    use_fixed_dist_angle: bool, whether to use fixed distance for angle computation or dynamic distance (based on segment boundary points)
    dist_angle: float, distance [mm] to sample points around bifurcation for angle computation
    angle_average: float, number of points to average for angle computation
    radius_average: float, number of points to average for radius computation
    use_fixed_dist_radius: bool, whether to use fixed distance for radius computation or dynamic distance
                                 (max dist from bifurcation to boundary points)
    dist_radius: float, distance [mm] from bifurcation point for radius computation (only if use_fixed_dist_radius is True)

    Returns:
    savepath: str, path to the saved feature json file
    """

    filename = os.path.basename(cnt_vtp_file)
    json_filename = filename.replace('.vtp', '.json')

    node_file = os.path.join(node_dir, json_filename)
    variant_file = os.path.join(variant_dir, json_filename)
    feature_file = os.path.join(feature_dir, json_filename)

    logger.info(f'Computings bifurcation features with args:'
                f'\n\t- cnt_vtp_file={cnt_vtp_file}'
                f'\n\t- variant_dir={variant_dir}'
                f'\n\t- node_dir={node_dir}'
                f'\n\t- feature_dir={feature_dir}'
                f'\n\t- radius_attribute={radius_attribute}'
                f'\n\t- dist_angle={dist_angle}'
                f'\n\t- angle_average={angle_average}'
                f'\n\t- radius_average={radius_average}'
                f'\n\t- use_fixed_dist_radius={use_fixed_dist_radius}'
                f'\n\t- dist_radius={dist_radius}'
                )

    polydata = get_vtk_polydata_from_file(cnt_vtp_file)
    with open(node_file, 'r') as f:
        nodes_dict = json.load(f)
    with open(variant_file, 'r') as f:
        variant_dict = json.load(f)
    with open(feature_file, 'r') as f:
        feature_dict = json.load(f)

    logger.info(f'Extracting bifurcation geometry (angles, radius ratios, bifurcation exponent) for major bifurcations...')
    logger.info('...BA bifurcation')
    ba_bif = extract_ba_ica_bif_geometry(nodes_dict, variant_dict, polydata, radius_attribute, 'BA', use_fixed_dist_angle, dist_angle, angle_average, use_fixed_dist_radius, dist_radius, radius_average)
    logger.info(f'...R-ICA bifurcation')
    rica_bif = extract_ba_ica_bif_geometry(nodes_dict, variant_dict, polydata, radius_attribute, 'R-ICA', use_fixed_dist_angle, dist_angle, angle_average, use_fixed_dist_radius, dist_radius, radius_average)
    logger.info(f'...L-ICA bifurcation')
    lica_bif = extract_ba_ica_bif_geometry(nodes_dict, variant_dict, polydata, radius_attribute, 'L-ICA', use_fixed_dist_angle, dist_angle, angle_average, use_fixed_dist_radius, dist_radius, radius_average)
    logger.info(f'Extracting bifurcation geometry (angles) for minor bifurcations...')
    logger.info(f'...Pcom/PCA bifurcations')
    rpca_pcom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'R-PCA', dist_angle, angle_average)
    lpca_pcom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'L-PCA', dist_angle, angle_average)
    logger.info(f'...Pcom/ICA bifurcations')
    rica_pcom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'R-ICA', dist_angle, angle_average)
    lica_pcom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'L-ICA', dist_angle, angle_average)
    logger.info(f'...Acom bifurcations')
    raca_acom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'R-ACA', dist_angle, angle_average)
    laca_acom_bif = extract_acom_pcom_bif_geometry(nodes_dict, variant_dict, polydata, 'L-ACA', dist_angle, angle_average)
    # Combine all bifurcation features into a single dictionary for logging
    all_bif_features = {
        'BA bifurcation': ba_bif,
        'R-ICA bifurcation': rica_bif,
        'L-ICA bifurcation': lica_bif,
        'R-PCA/Pcom bifurcation': rpca_pcom_bif,
        'L-PCA/Pcom bifurcation': lpca_pcom_bif,
        'R-ICA/Pcom bifurcation': rica_pcom_bif,
        'L-ICA/Pcom bifurcation': lica_pcom_bif,
        'R-ACA/Acom bifurcation': raca_acom_bif,
        'L-ACA/Acom bifurcation': laca_acom_bif
    }

    # Log the combined dictionary
    logger.debug(f"bifurcation features: {all_bif_features}")

    logger.info(f'Updating feature dict with bifurcation features and saving to {feature_file}')
    if len(ba_bif) > 0:
        feature_dict['1']['BA bifurcation'] = ba_bif
    if len(rica_bif) > 0:
        feature_dict['4']['ICA bifurcation'] = rica_bif
    if len(lica_bif) > 0:
        feature_dict['6']['ICA bifurcation'] = lica_bif
    if len(rpca_pcom_bif) > 0:
        feature_dict['2']['Pcom bifurcation'] = rpca_pcom_bif
    if len(lpca_pcom_bif) > 0:
        feature_dict['3']['Pcom bifurcation'] = lpca_pcom_bif
    if len(rica_pcom_bif) > 0:
        feature_dict['4']['Pcom bifurcation'] = rica_pcom_bif
    if len(lica_pcom_bif) > 0:
        feature_dict['6']['Pcom bifurcation'] = lica_pcom_bif
    if len(raca_acom_bif) > 0:
        feature_dict['11']['Acom bifurcation'] = raca_acom_bif
    if len(laca_acom_bif) > 0:
        feature_dict['12']['Acom bifurcation'] = laca_acom_bif

    with open(feature_file, 'w') as f:
        json.dump(feature_dict, f, indent=4)

    return feature_file
