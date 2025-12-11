import vtk
import numpy as np
import copy
from utils.utils_graph_processing import *
from postprocess_graph.extract_segments_for_smoothing import extract_segments

from logger import logger

def merge_polydatas(polydatas):
    """
    Merge polydata objects together

    Args:
    polydatas: dict, dictionary of polydata objects to merge

    Returns:
    vtkPolyData, merged polydata object
    """
    poly = vtk.vtkAppendPolyData()
    segments_ordered = ['BA', 'R-PCA', 'L-PCA', 'R-ICA', 'R-MCA', 'L-ICA', 'L-MCA', 'R-Pcom', 'L-Pcom', 'R-ACA', 'L-ACA', 'Acom', '3rd-A2']
    for segment in segments_ordered:
        if segment in polydatas:
            for i in range(len(polydatas[segment])):
                poly.AddInputData(polydatas[segment][i])
    poly.Update()
    return poly.GetOutput()

def find_fixpoints(paths):
    """
    Find fixpoints for smoothing for segments with two paths (e.g., A1/P1/Acom fenestration).
    The fixpoints are the last and first common points in the two paths before divergins and after converging again.

    Args:
    paths: list, list of paths

    Returns:
    fixpoint1: int, first fixpoint
    fixpoint2: int, second fixpoint
    """
    assert len(paths) == 2, 'More than two paths found!'
    pointIds_paths = []
    for path in paths:
        # pointIds = [p[0] for p in path['path'][1:]]
        pointIds = [p[0] for p in path['path']]
        pointIds_paths.append(pointIds)
    # Find the last common number before the lists diverge -> fixpoint for beginning of fenestration
    fixpoint1 = None
    list1, list2 = pointIds_paths[0], pointIds_paths[1]
    for i in range(min(len(list1), len(list2))):
        if list1[i] == list2[i]:
            fixpoint1 = list1[i]
        else:
            break

    # Find the first common number after lists converge -> fixpoint for end of fenestration
    fixpoint2 = None
    for i in range(1, min(len(list1), len(list2)) + 1):
        if list1[-i] == list2[-i]:
            fixpoint2 = list1[-i]
        else:
            break
    
    return fixpoint1, fixpoint2

def get_ordered_fixpoints(fixpoints, all_paths):
    """
    Get the ordered fixpoints for fenestration with more than 2 paths and fixpoints.

    Args:
    fixpoints: list, list of fixpoints
    all_paths: list, list of all paths (for ACA/PCA segment)

    Returns:
    ordered_fixpoints: tuple, ordered fixpoints
    """
    for path in all_paths:
        pointIds = [p[0] for p in path['path']]
        # if all fixpoints are in pointIds, return them in order
        if all(fp in pointIds for fp in fixpoints):
            ordered_fixpoints = tuple(sorted(fixpoints, key=lambda x: pointIds.index(x)))
            return ordered_fixpoints

def get_points_for_aca_pca_fenestration(all_paths, segment, polydata):
    """
    Get the ordered points for the A1/P1 fenestration with multiple paths and fixpoints.
    The function finds the fixpoints of the fenestration and combines the paths accordingly.

    Args:
    all_paths: list, list of all paths (for ACA/PCA segment)
    segment: list, segment (start, end, labels)
    polydata: vtkPolyData, polydata object

    Returns:
    pts: list, list of points
    """
    if len(all_paths) == 2: # almost all fenestrations contain just 2 paths
        # find fixpoints for fenestration (beginning and end of fenestration)
        fixpoint1, fixpoint2 = find_fixpoints(all_paths)
        assert fixpoint1 is not None and fixpoint2 is not None, 'Fixpoints not found for ACA/PCA fenestration!'

        # combine 4 segment paths in case of ACA/PCA fenestration
        path1, path2, path3, path4 = [], [], [], []

        if fixpoint1 != segment[0]:
            paths = find_all_paths(segment[0], fixpoint1, polydata, segment[2])
            assert len(paths) == 1
            path1 = paths[0]['path']
        
        paths = find_all_paths(fixpoint1, fixpoint2, polydata, segment[2])
        assert len(paths) == 2
        path2, path3 = paths[0]['path'], paths[1]['path']

        paths = find_all_paths(fixpoint2, segment[1], polydata, segment[2])
        assert len(paths) == 1
        path4 = paths[0]['path']
        
        pts = [[p[0] for p in path] + [path[-1][1]] for path in [path1, path2, path3, path4] if len(path) > 0]
    
    elif len(all_paths) == 3: # rare fenestrations with 3 paths
        # loop over all combinations of 2 paths to find all fixpoints
        fixpoints = []
        for i in range(len(all_paths)):
            for j in range(i+1, len(all_paths)):
                path_a = all_paths[i]
                path_b = all_paths[j]
                fixpoint1, fixpoint2 = find_fixpoints([path_a, path_b])
                if fixpoint1 is not None and fixpoint2 is not None:
                    fixpoints.append(fixpoint1)
                    fixpoints.append(fixpoint2)
        fixpoints = list(set(fixpoints))
        assert len(fixpoints) > 2, 'Wrong number of fixpoints found for ACA/PCA fenestration with 3 paths!'
        if len(fixpoints) == 4 or len(fixpoints) == 3:
            path1, path2, path3, path4, path5, path6, path7 = [], [], [], [], [], [], []
            ordered_fixpoints = get_ordered_fixpoints(fixpoints, all_paths)
            if ordered_fixpoints[0] != segment[0]:
                paths = find_all_paths(segment[0], ordered_fixpoints[0], polydata, segment[2])
                assert len(paths) == 1
                path1 = paths[0]['path']
            else:
                path1 = []
            # take the path with only two fixpoints (first and last) as path2
            for path in all_paths:
                pointIds = [p[0] for p in path['path']]
                # check how many fixpoints are in pointIds
                if not all(fp in pointIds for fp in ordered_fixpoints):
                    assert len([fp for fp in ordered_fixpoints if fp in pointIds]) == 2, 'Path with more than 2 fixpoints found!'
                    assert ordered_fixpoints[0] in pointIds and ordered_fixpoints[-1] in pointIds, 'Wrong fixpoints in path!'
                    id1 = pointIds.index(ordered_fixpoints[0])
                    pointIds_end = [p[1] for p in path['path']]
                    id2 = pointIds_end.index(ordered_fixpoints[-1])
                    path2 = path['path'][id1:id2+1]
                    all_paths.remove(path)
                    break
            
            assert path2 != [], 'Path2 not found!'
            assert len(all_paths) == 2, 'Wrong number of remaining paths!'

            if len(fixpoints) == 3:
                nodes_deg_5 = get_nodes_of_degree_n(5, segment[2], polydata)
                assert len(nodes_deg_5) == 1, 'No node of degree 5 node found for fenestration with only 3 fixpoints!'
                fixpoint1, fixpoint2, fixpoint3 = ordered_fixpoints[0], ordered_fixpoints[1], ordered_fixpoints[2]

            else:
                fixpoint1, fixpoint2, fixpoint3, fixpoint4 = ordered_fixpoints[0], ordered_fixpoints[1], ordered_fixpoints[2], ordered_fixpoints[3]
           
            paths = find_all_paths(fixpoint1, fixpoint2, polydata, segment[2])
            # find path that contains no other fixpoints
            for path in paths:
                pointIds = [p[0] for p in path['path'][1:]]
                if all(fp not in pointIds for fp in ordered_fixpoints):
                    path3 = path['path']
                    break
            
            paths = find_all_paths(fixpoint2, fixpoint3, polydata, segment[2])
            for path in paths:
                pointIds = [p[0] for p in path['path'][1:]]
                if all(fp not in pointIds for fp in ordered_fixpoints):
                    path4 = path['path']
                    paths.remove(path)
                    break
            for path in paths:
                pointIds = [p[0] for p in path['path'][1:]]
                if all(fp not in pointIds for fp in ordered_fixpoints):
                    path5 = path['path']
                    break
            
            if len(fixpoints) == 4:
                paths = find_all_paths(fixpoint3, fixpoint4, polydata, segment[2])
                for path in paths:
                    pointIds = [p[0] for p in path['path'][1:]]
                    if all(fp not in pointIds for fp in ordered_fixpoints):
                        path6 = path['path']
                        break

                paths = find_all_paths(fixpoint4, segment[1], polydata, segment[2])
                assert len(paths) == 1
                path7 = paths[0]['path']
            else:
                path6 = []
                paths = find_all_paths(fixpoint3, segment[1], polydata, segment[2])
                assert len(paths) == 1
                path7 = paths[0]['path']

            pts = [[p[0] for p in path] + [path[-1][1]] for path in [path1, path2, path3, path4, path5, path6, path7] if len(path) > 0]
            
        else:
            raise NotImplementedError('Other than 4 fixpoints not implemented yet!')

    return pts
    

def get_points_for_acom_fenestration(acom_segments, polydata):
    """
    Get the ordered points for the Acom fenestration with multiple segments and fixpoints.
    The function finds the fixpoints of the fenestration and combines the paths accordingly.

    Args:
    acom_segments: list, list of Acom segments
    polydata: vtkPolyData, polydata object

    Returns:
    pts: list, list of points
    """
    
    assert len(acom_segments) == 2, 'Acom has more than 2 segments!'
    segment1, segment2 = acom_segments[0], acom_segments[1]
    paths1 = find_all_paths(segment1[0], segment1[1], polydata, segment1[2])
    paths2 = find_all_paths(segment2[0], segment2[1], polydata, segment2[2])

    # make sure the paths do not cross the wrong acom bifurcations
    if segment1[0] == segment2[0]:
        path1 = paths1[0] if segment2[1] not in [p[1] for p in paths1[0]['path']] else paths1[1]
        path2 = paths2[0] if segment1[1] not in [p[1] for p in paths2[0]['path']] else paths2[1]
    else:
        path1 = paths1[0] if segment2[0] not in [p[0] for p in paths1[0]['path']] else paths1[1]
        path2 = paths2[0] if segment1[0] not in [p[0] for p in paths2[0]['path']] else paths2[1]
    
    # Find fixpoints for fenestration (beginning and end of fenestration)
    all_paths = [path1, path2]
    fixpoint1, fixpoint2 = find_fixpoints(all_paths)
    path1 = path1['path']
    path2 = path2['path']
    if fixpoint1 == None and fixpoint2 == None:
        p1, p2 = path1, path2
        p3 = []
    elif fixpoint1 == None:
        # combine 3 segment paths
        index_fixpoint2_path1 = [p[1] for p in path1].index(fixpoint2) 
        index_fixpoint2_path2 = [p[1] for p in path2].index(fixpoint2)
        p1 = path1[:index_fixpoint2_path1+1]
        p2 = path2[:index_fixpoint2_path2+1]
        p3 = []
        if index_fixpoint2_path1 + 1 < len(path1):
            p3 = path1[index_fixpoint2_path1+1:]
        
    elif fixpoint2 == None:
        index_fixpoint1_path1 = [p[0] for p in path1].index(fixpoint1) 
        index_fixpoint1_path2 = [p[0] for p in path2].index(fixpoint1)
        p1 = []
        if index_fixpoint1_path1 > 0:
            p1 = path1[:index_fixpoint1_path1]
        p2 = path1[index_fixpoint1_path1:]
        p3 = path2[index_fixpoint1_path2:]
    
    pts = [[p[0] for p in path] + [path[-1][1]] for path in [p1, p2, p3] if len(path) > 0]
    return pts

def reorder_acom_fenestration_nodes(nodes_dict, segments, polydata):
    """
    Reorder the Acom fenestration nodes such that the nodes closest to the ICA bifurcation are first.
    This function also updates the nodes_dict with the newly ordered Acom nodes.

    Args:
    nodes_dict: dict, dictionary containing the CoW nodes
    segments: list, list of Acom segments
    polydata: vtkPolyData, polydata object

    Returns:
    nodes_dict: dict, updated dictionary containing the newly ordered Acom nodes
    segments: list, updated list of segments
    """
    assert len(segments) == 2, 'Acom doesnt have 2 segments!'
    segment1, segment2 = segments[0], segments[1]
    raca_first_node, raca_second_node = segment1[0], segment2[0]
    laca_first_node, laca_second_node = segment1[1], segment2[1]
    if segment1[0] != segment2[0]:
        aca_start = nodes_dict['11']['ICA boundary'][0]['id']
        aca_end = nodes_dict['11']['ACA end'][0]['id']
        paths = find_all_paths(aca_start, aca_end, polydata, 11)
        assert len(paths) == 1
        path = paths[0]['path']
        index1, index2 = None, None
        for idx, tup in enumerate(path):
            if tup[0] == raca_first_node:
                index1 = idx
            if tup[0] == raca_second_node:
                index2 = idx
            if index1 is not None and index2 is not None:
                break

        if index1 is not None and index2 is not None and index2 < index1:
            raca_first_node, raca_second_node = raca_second_node, raca_first_node
            segment1 = (raca_first_node, segment1[1], segment1[2])
            segment2 = (raca_second_node, segment2[1], segment2[2])
            nodes_dict['11']['Acom bifurcation'][0] = get_node_dict_entry(raca_first_node, 3, 11, polydata)[0]
            nodes_dict['11']['Acom bifurcation'][1] = get_node_dict_entry(raca_second_node, 3, 11, polydata)[0]

    if segment1[1] != segment2[1]:
        aca_start = nodes_dict['12']['ICA boundary'][0]['id']
        aca_end = nodes_dict['12']['ACA end'][0]['id']
        paths = find_all_paths(aca_start, aca_end, polydata, 12)
        assert len(paths) == 1
        path = paths[0]['path']
        index1, index2 = None, None
        for idx, tup in enumerate(path):
            if tup[0] == laca_first_node:
                index1 = idx
            if tup[0] == laca_second_node:
                index2 = idx
            if index1 is not None and index2 is not None:
                break

        if index1 is not None and index2 is not None and index2 < index1:
            laca_first_node, laca_second_node = laca_second_node, laca_first_node
            segment1 = (segment1[0], laca_first_node, segment1[2])
            segment2 = (segment2[0], laca_second_node, segment2[2])
            nodes_dict['12']['Acom bifurcation'][0] = get_node_dict_entry(laca_first_node, 3, 12, polydata)[0]
            nodes_dict['12']['Acom bifurcation'][1] = get_node_dict_entry(laca_second_node, 3, 12, polydata)[0]

    r_aca_boundaries = [n['id'] for n in nodes_dict['11']['Acom boundary']]
    if len(r_aca_boundaries) > 1:
        assert len(r_aca_boundaries) == 2, 'More than 2 Acom boundaries found!'
        r_aca_first_boundary, _ = find_closest_node_to_point(r_aca_boundaries, raca_first_node, 11, polydata)
        r_aca_second_boundary = r_aca_boundaries[1] if r_aca_boundaries[0] == r_aca_first_boundary else r_aca_boundaries[0]
        nodes_dict['11']['Acom boundary'][0] = get_node_dict_entry(r_aca_first_boundary, 2, 11, polydata)[0]
        nodes_dict['11']['Acom boundary'][1] = get_node_dict_entry(r_aca_second_boundary, 2, 11, polydata)[0]
        nodes_dict['10']['R-ACA boundary'][0] = get_node_dict_entry(r_aca_first_boundary, 2, 10, polydata)[0]
        nodes_dict['10']['R-ACA boundary'][1] = get_node_dict_entry(r_aca_second_boundary, 2, 10, polydata)[0]
    
    l_aca_boundaries = [n['id'] for n in nodes_dict['12']['Acom boundary']]
    if len(l_aca_boundaries) > 1:
        assert len(l_aca_boundaries) == 2, 'More than 2 Acom boundaries found!'
        l_aca_first_boundary, _ = find_closest_node_to_point(l_aca_boundaries, laca_first_node, 12, polydata)
        l_aca_second_boundary = l_aca_boundaries[1] if l_aca_boundaries[0] == l_aca_first_boundary else l_aca_boundaries[0]
        nodes_dict['12']['Acom boundary'][0] = get_node_dict_entry(l_aca_first_boundary, 2, 12, polydata)[0]
        nodes_dict['12']['Acom boundary'][1] = get_node_dict_entry(l_aca_second_boundary, 2, 12, polydata)[0]
        nodes_dict['10']['L-ACA boundary'][0] = get_node_dict_entry(l_aca_first_boundary, 2, 10, polydata)[0]
        nodes_dict['10']['L-ACA boundary'][1] = get_node_dict_entry(l_aca_second_boundary, 2, 10, polydata)[0]
    
    
    return nodes_dict, [segment1, segment2]

def extract_point_ids_in_order(nodes_dict, variant_dict, polydata):
    """
    This function processes a CoW centerline graph graph to:
    1. Extract anatomical segments from the graph
    2. Reorder them according to a predefined anatomical sequence
    3. Extract point IDs in the correct order along each segment's path
    The function handles various anatomical configurations, including:
    - Standard vessel paths
    - Multiple segment vessels (e.g., MCA branches)
    - Fenestration variants in Acom, A1, and P1 segments
    - Special ordering cases based on the presence/absence of certain vessels
    For each segment, the function finds the appropriate path(s) between the defining 
    nodes and extracts the point IDs along these paths. The ordering of segments is 
    critical for subsequent smoothing operations.

    Args:
    nodes_dict (dict): Dictionary containing the CoW nodes
    variant_dict (dict): Dictionary containing the CoW variants
    polydata (vtkPolyData): Polydata representation of the centerline graph

    Returns:
    point_id_lists (list): Nested list of point IDs for each segment
    segments_ordered (dict): Dictionary of segments in the correct anatomical order
    nodes_dict (dict): Updated dictionary containing the CoW nodes
    """
    # Extract segments for smoothing (start, end, labels)
    segments = extract_segments(nodes_dict, variant_dict, polydata)
    # logger.debug(f'\tExtracted {len(segments)} segments for smoothing: {segments}')

    point_id_lists = []

    # Specify ordering or segments
    ant_dict = variant_dict['anterior']
    keys_ordered = ['R-ICA', 'R-MCA', 'L-ICA', 'L-MCA', 'BA', 'R-PCA', 'L-PCA', 'R-ACA', 'L-ACA', 'Acom', '3rd-A2', 'R-Pcom', 'L-Pcom']
    if not ant_dict['R-A1']:
        keys_ordered = ['R-ICA', 'R-MCA', 'L-ICA', 'L-MCA', 'BA', 'R-PCA', 'L-PCA', 'L-ACA', 'R-ACA', 'Acom', '3rd-A2', 'R-Pcom', 'L-Pcom']
    if ('6' in nodes_dict and not 'ICA start' in nodes_dict['6']) or ('4' in nodes_dict and not 'ICA start' in nodes_dict['4']):
        keys_ordered = ['BA', 'R-PCA', 'L-PCA', 'R-ICA', 'L-ICA', 'R-MCA', 'L-MCA', 'R-ACA', 'L-ACA', 'Acom', '3rd-A2', 'R-Pcom', 'L-Pcom']
    
    segments_ordered = {}

    for key in keys_ordered:
        if key in segments.keys():
            segments_ordered[key] = segments[key]
            pts = []
            if len(segments[key]) == 2:
                if key == 'Acom': # Acom fenestration
                    # reorder Acom nodes with nodes closer to ICA bifurcation coming first
                    nodes_dict, segments[key] = reorder_acom_fenestration_nodes(nodes_dict, segments[key], polydata)
                    # Get ordered points for Acom fenestration with multiple segments and fixpoints
                    pts = get_points_for_acom_fenestration(segments[key], polydata)
                    logger.warning(f'\tALERT: Acom fenestration found! Segments: {segments[key]}')
                    logger.warning(f'\tPoints for Acom fenestration: {pts}')
                else: # 2 segments (e.g. for BA branch with missing P1)
                    for i in range(len(segments[key])):
                        segment = segments[key][i]
                        path = find_shortest_path(segment[0], segment[1], polydata, segment[2])['path']
                        point_ids = [p[0] for p in path] + [path[-1][1]]
                        pts.append(point_ids)

            elif len(segments[key]) == 1:
                segment = segments[key][0]
                all_paths = find_all_paths(segment[0], segment[1], polydata, segment[2])
                if len(all_paths) == 1: # usual case
                    path = all_paths[0]['path']
                    point_ids = [p[0] for p in path] + [path[-1][1]]
                    pts.append(point_ids)
                else: # 2 paths (e.g. for A1/P1 fenestration)
                    if len(all_paths) == 2:
                        logger.warning(f'\tALERT: {key} has multiple paths! Possible fenestration detected!')
                        # Check if A1/P1 fenestration is present
                        if key == 'R-PCA':
                            assert variant_dict['fenestration']['R-P1'], 'R-P1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        elif key == 'L-PCA':
                            assert variant_dict['fenestration']['L-P1'], 'L-P1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        elif key == 'BA':
                            assert variant_dict['fenestration']['R-P1'] or variant_dict['fenestration']['L-P1'], 'P1 fenestration not found'
                            assert not variant_dict['posterior']['R-P1'] or not variant_dict['posterior']['L-P1'], 'P1 present, fenestration not possible'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        elif key == 'R-ACA':
                            assert variant_dict['fenestration']['R-A1'], 'R-A1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        elif key == 'L-ACA':
                            assert variant_dict['fenestration']['L-A1'], 'L-A1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        else:
                            logger.warning(f'\tALERT: {key} has multiple paths!')
                            path = find_shortest_path(segment[0], segment[1], polydata, segment[2])['path']
                            point_ids = [p[0] for p in path] + [path[-1][1]]
                            pts.append(point_ids)

                        logger.warning(f'\tPoints for {key} fenestration: {pts}')
                    
                    elif len(all_paths) == 3:
                        logger.warning(f'\tALERT: {key} has multiple paths! Possible fenestration detected!')
                        if key == 'R-PCA':
                            assert variant_dict['fenestration']['R-P1'], 'R-P1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        elif key == 'L-PCA':
                            assert variant_dict['fenestration']['L-P1'], 'L-P1 fenestration not found'
                            pts = get_points_for_aca_pca_fenestration(all_paths, segment, polydata)
                        else: 
                            logger.warning(f'\tALERT: {key} has 3 paths!')
                            raise NotImplementedError(f'Fenestration with 3 paths not implemented for segment {key}?!')
                    else:
                        logger.warning(f'\tALERT: {key} has {len(all_paths)} paths!')
                        raise ValueError(f'More than 2 or 3 paths found for segment {key}?!')
                
            else:
                # 3 or more segments (e.g. for MCA branches)
                for i in range(len(segments[key])):
                    segment = segments[key][i]
                    path = find_shortest_path(segment[0], segment[1], polydata, segment[2])['path']
                    point_ids = [p[0] for p in path] + [path[-1][1]]
                    pts.append(point_ids)
      
            point_id_lists.append(pts)

    return point_id_lists, segments_ordered, nodes_dict


def get_ids_and_fixpoints_from_nodes_dict(nodes_dict):
    """
    Extract node ids from nodes_dict. All bifurcation points are considered fixpoints for the smoothing.

    Args:
    nodes_dict: dict, dictionary containing the CoW nodes

    Returns:
    ids: list, list of all node ids
    points_for_fixing: list, list of all bifurcation points
    """
    ids, fix_points = [], []
    for key, data in nodes_dict.items():
        for sub_key, items in data.items():
            ids.extend(item['id'] for item in items)
            if 'bifurcation' in sub_key:
                fix_points.extend(item['id'] for item in items)
    return ids, fix_points


def match_and_update_nodes(node_id_orig, coords_orig, node_id_new, coords_new, nodes_dict, nodes_dict_new, segment_keys):
    """
    Match and update nodes between the original and the reordered & smoothed polydata.
    This function updates the node id and coordinates in the nodes_dict_new based on the original node id and coordinates.

    Args:
    node_id_orig: int, original node id
    coords_orig: list, original node coordinates
    node_id_new: int, new node id (after reorder and smoothing)
    coords_new: list, new node coordinates (after reorder and smoothing)
    nodes_dict: dict, dictionary containing the CoW nodes
    nodes_dict_new: dict, dictionary containing the CoW nodes after reorder and smoothing
    segment_keys: list, list of segment keys

    Returns:
    nodes_dict_new: dict, updated dictionary containing the CoW nodes after reorder and smoothing
    """
    coords_orig = [round(c, 5) for c in coords_orig]
    coords_new = [round(c, 5) for c in coords_new]

    for key in segment_keys:
        key1 = str(key)
        value1 = nodes_dict[key1]
        assert isinstance(value1, dict)
        for key2, value2 in value1.items():
            assert isinstance(value2, list)
            for i in range(len(value2)):
                if value2[i]['id'] == node_id_orig:
                    assert value2[i]['coords'] == coords_orig
                    nodes_dict_new[key1][key2][i]['id'] = node_id_new
                    nodes_dict_new[key1][key2][i]['coords'] = coords_new
    return nodes_dict_new

def reorder_and_smooth(node_dict, variant_dict, polydata, window_size=5):
    """
    Reorders and smooths a polydata object representing the CoW centerline graph.
    This function applies a moving average filter to smooth the polydata while preserving
    the connectivity and topology of the vascular network. It processes each segment of
    the vascular network individually, maintaining fixed points at vessel junctions and
    bifurcations. The function handles special cases for different anatomical segments
    (Acom, Pcom, ICA, etc.) and ensures proper connectivity between segments.

    Args:
    node_dict (dict): Dictionary containing the CoW nodes
    variant_dict (dict): Dictionary containing the CoW variants
    polydata (vtkPolyData): polydata centerline graph
    window_size (int, optional): Size of the window for the moving average filter. Defaults to 5.

    Returns:
    polydata_new (vtkPolyData): Smoothed polydata object with reordered points
    node_dict_new (dict): Updated node dictionary with new coordinates
    """
    # Extract the point IDs from the node_dict json file
    node_ids, fix_points = get_ids_and_fixpoints_from_nodes_dict(node_dict)
    points_fixed = []

    points = polydata.GetPoints()

    edge_list, cell_ids = get_edge_list(polydata)

    label_array = polydata.GetCellData().GetArray('labels')

    point_id_lists, segments, node_dict = extract_point_ids_in_order(node_dict, variant_dict, polydata)
    logger.debug(f'\tExtracted {len(point_id_lists)} segments for smoothing: {segments}')
    assert len(point_id_lists) == len(segments)

    if '3rd-A2' in segments.keys():
        fix_points.append(segments['3rd-A2'][0][0])
        node_ids.append(segments['3rd-A2'][0][0])

    polydatas = {}
    points_to_fix = {}

    node_dict_new = copy.deepcopy(node_dict)

    for segment_key, point_ids_seg in zip(segments.keys(), point_id_lists):
        logger.info(f'\t...segment {segment_key}')
        logger.debug(f'\tPoint IDs for segment {segment_key}: {point_ids_seg}')
        if segment_key == 'Acom' and len(point_ids_seg) == 3:
            fps = point_ids_seg[0][-1]
            points_fixed.append(fps)
            points_to_fix[fps] = points.GetPoint(fps)
        polydatas_list = []
        segment_labels = segments[segment_key][0][2]
        for j in range(len(point_ids_seg)):
            point_ids = point_ids_seg[j]

            label_array_new = vtk.vtkIntArray()
            label_array_new.SetName("labels")

            # create segments separately and merge them in the end
            # Create a vtkPoints object to store the points
            points_new = vtk.vtkPoints()
            # Create a vtkIntArray to store degree attributes for the points
            pt_degrees = vtk.vtkIntArray()
            pt_degrees.SetName("degree")
            # Create a vtkCellArray object to store the line cells
            lines = vtk.vtkCellArray()

            n_cell = 0
            num_points = len(point_ids)

            # Convert points to a numpy array for easier manipulation
            points_array_orig = np.array([points.GetPoint(i) for i in point_ids])
            points_array = np.array([points.GetPoint(i) for i in point_ids])

            # Apply the moving average filter
            coords_new = points_array[0]
            deg = get_point_degree(point_ids[0], edge_list)
            pt_degrees.InsertNextValue(deg)
            if segment_key in ['Acom', 'R-Pcom', 'L-Pcom', '3rd-A2']:
                assert point_ids[0] in points_fixed
                coords_new = points_to_fix[point_ids[0]]
                points_array[0] = coords_new
            if not variant_dict['posterior']['L-P1'] and 'BA bifurcation'in node_dict['1'] and segment_key == 'BA' and j > 0:
                assert point_ids[0] in points_fixed
                coords_new = points_to_fix[point_ids[0]]
                points_array[0] = coords_new
            if not variant_dict['posterior']['R-P1'] and 'BA bifurcation'in node_dict['1'] and segment_key == 'BA' and j > 0:
                assert point_ids[0] in points_fixed
                coords_new = points_to_fix[point_ids[0]]
                points_array[0] = coords_new
            if '4' in node_dict and not 'ICA start' in node_dict['4'] and segment_key == 'R-ICA':
                assert point_ids[0] in points_fixed
                coords_new = points_to_fix[point_ids[0]]
                points_array[0] = coords_new
            if '6' in node_dict and not 'ICA start' in node_dict['6'] and segment_key == 'L-ICA':
                assert point_ids[0] in points_fixed
                coords_new = points_to_fix[point_ids[0]]
                points_array[0] = coords_new
            
            points_new.InsertNextPoint(coords_new)

            # preemptively insert last point
            coords_new_last = points_array[-1]
            if (segment_key == 'Acom' and (variant_dict['anterior']['L-A1'] and variant_dict['anterior']['R-A1'])) \
            or (segment_key == 'Acom' and ('Acom bifurcation' in node_dict['12'] and 'ICA boundary' in node_dict['12'] and variant_dict['anterior']['R-A1'])) \
            or (segment_key == 'Acom' and ('Acom bifurcation' in node_dict['11'] and 'ICA boundary' in node_dict['11'] and variant_dict['anterior']['L-A1'])):
                assert point_ids[-1] in points_fixed
                coords_new_last = points_to_fix[point_ids[-1]]
                points_array[-1] = coords_new_last
            if (segment_key == 'R-Pcom' and variant_dict['posterior']['R-P1']) \
                or (segment_key == 'R-Pcom' and ('Pcom bifurcation' in node_dict['2'] and 'BA boundary' in node_dict['2'])):
                assert point_ids[-1] in points_fixed
                coords_new_last = points_to_fix[point_ids[-1]]
                points_array[-1] = coords_new_last
            if (segment_key == 'L-Pcom' and variant_dict['posterior']['L-P1']) \
                or (segment_key == 'L-Pcom' and ('Pcom bifurcation' in node_dict['3'] and 'BA boundary' in node_dict['3'])):
                assert point_ids[-1] in points_fixed
                coords_new_last = points_to_fix[point_ids[-1]]
                points_array[-1] = coords_new_last

            if point_ids[0] in node_ids:
                node_id_orig = point_ids[0]
                coords_orig = points_array_orig[0]
                node_id_new = n_cell
                points_to_fix[point_ids[0]] = coords_new
                points_fixed.append(point_ids[0])
                node_dict_new = match_and_update_nodes(node_id_orig, list(coords_orig), node_id_new, list(coords_new), node_dict, node_dict_new, segment_labels)
                
            for i in range(1, num_points-1):
                deg = get_point_degree(point_ids[i], edge_list)
                pt_degrees.InsertNextValue(deg)
                # take moving average
                start = max(0, i - window_size // 2)
                end = min(num_points, i + window_size // 2 + 1)
                mean = np.mean(points_array[start:end], axis=0)
                coords_new = mean
                if point_ids[i] in node_ids:
                    node_id_orig = point_ids[i]
                    coords_orig = points_array_orig[i]
                    node_id_new = n_cell
                    if point_ids[i] in fix_points and point_ids[i] not in points_fixed:
                        points_to_fix[point_ids[i]] = mean
                        points_fixed.append(point_ids[i])

                    node_dict_new = match_and_update_nodes(node_id_orig, list(coords_orig), node_id_new, list(coords_new), node_dict, node_dict_new, segment_labels)
                
                points_new.InsertNextPoint(coords_new)
                # Add line cells to the vtkCellArray object
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, n_cell)
                line.GetPointIds().SetId(1, n_cell + 1)
                lines.InsertNextCell(line)
                # Get cell id of original polydata edge
                try:
                    cell_index = edge_list.index((point_ids[i-1], point_ids[i]))
                except ValueError:
                    cell_index = edge_list.index((point_ids[i], point_ids[i-1]))
                cell_id_orig = cell_ids[cell_index]
                label_array_new.InsertNextValue(int(label_array.GetValue(cell_id_orig)))
                n_cell += 1

            deg = get_point_degree(point_ids[-1], edge_list)
            pt_degrees.InsertNextValue(deg)
            points_new.InsertNextPoint(coords_new_last)
            if point_ids[-1] in node_ids:
                    node_id_orig = point_ids[-1]
                    coords_orig = points_array_orig[-1]
                    node_id_new = n_cell
                    points_to_fix[point_ids[-1]] = coords_new_last
                    points_fixed.append(point_ids[-1])
                    node_dict_new = match_and_update_nodes(node_id_orig, list(coords_orig), node_id_new, list(coords_new_last), node_dict, node_dict_new, segment_labels)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, n_cell)
            line.GetPointIds().SetId(1, n_cell+1)
            lines.InsertNextCell(line)

            # Get cell id of original polydata edge
            try:
                cell_index = edge_list.index((point_ids[-2], point_ids[-1]))
            except ValueError:
                cell_index = edge_list.index((point_ids[-1], point_ids[-2]))
            cell_id_orig = cell_ids[cell_index]
            label_array_new.InsertNextValue(int(label_array.GetValue(cell_id_orig)))

            # Create a vtkPolyData object and set the points and lines
            polydata_new = vtk.vtkPolyData()
            polydata_new.SetPoints(points_new)
            polydata_new.SetLines(lines)
            # Set the array as the active scalars
            polydata_new.GetCellData().AddArray(label_array_new)
            # Set the array as the active scalars
            polydata_new.GetCellData().SetScalars(label_array_new)

            # Add the degree array to the point data of the vtkPolyData object
            polydata_new.GetPointData().AddArray(pt_degrees)

            # Optionally, mark the polydata as modified
            polydata_new.Modified()

            polydatas_list.append(polydata_new)

        polydatas[segment_key] = polydatas_list

    # Merge all segments into one polydata object
    polydata_new = merge_polydatas(polydatas)
    polydata_new = clean_polydata(polydata_new)

    return polydata_new, node_dict_new


    