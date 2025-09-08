import vtk
import os
import json
from utils.utils_graph_processing import *
from postprocess_graph.node_extraction import *
from postprocess_graph.extract_cow_variant_from_graph import get_cow_variant
from postprocess_graph.reorder_and_smooth_centerline import reorder_and_smooth
from postprocess_graph.sanity_check_graphs_and_nodes import run_sanity_check

from logger import logger

def change_background_label(polydata, background_label=0, max_iterations=20):
    """
    This function identifies all cells in the polydata that have a label of 0 (considered
    as background), finds the neighboring cells for each, and changes the background label 
    to the maximum label value among those neighbors.
    
    Args:
    polydata : vtkPolyData, polydata centerline graph
    background_label : int (optional), background label value (default is 0)
    max_iterations : int (optional), maximum number of iterations (default is 20)

    Returns:
    polydata : vtkPolyData, polydata centerline graph with updated labels
    """
    labels = get_label_array(polydata)
    if background_label not in labels:
        return polydata
        
    iteration = 0
    
    while background_label in labels and iteration < max_iterations:
        background_cell_id = np.where(labels == background_label)[0]
        logger.info(f'Cells with background label found. Background cell ids: {background_cell_id}. Changing labels...')
        for id in background_cell_id:
            neighbors = get_neighbors(id, polydata)
            ngh_labels = [labels[ngh] for ngh in neighbors]
            label = np.max(ngh_labels).astype(np.uint8)
            if label == background_label:
                logger.debug(f'All neighboring cells of background cell {id} also have background label. Skipping...')
                continue
            logger.debug(f'Changing label from {background_label} to {label} for cell {id}')
            polydata = set_label(label, id, polydata)
        
        labels = get_label_array(polydata)
        iteration += 1

    if background_label in labels:
        logger.warning(f'Could not remove all background labels after {max_iterations} iterations.')

    return polydata
    
def relabel_aca_pca_segments(variant_dict, polydata):
    """
    Relabel ACA and PCA segments to correct for Acom and Pcom label spilling.
    This function uses the variant_dict to determine which segments are present and relabels
    the segments close to the acom/pcom bifurcations if necessary.
    
    Args:
    variant_dict: dict, variant dict
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with updated labels
    """
    label_array = get_label_array(polydata)

    # ACA segments
    if variant_dict['anterior']['Acom']:
        raca_nodes_1 = get_nodes_of_degree_n(1, 11, polydata)
        raca_ica_boundary = find_boundary_points(11, 4, polydata)
        laca_nodes_1 = get_nodes_of_degree_n(1, 12, polydata)
        laca_ica_boundary = find_boundary_points(12, 6, polydata)

        # Relabel R-ACA 
        # if label spilling happens at Acom bifurcation, the number of R-ACA 1-nodes 
        # should be larger than 2
        if len(raca_nodes_1) > 2 and len(raca_ica_boundary) > 0:
            # find path that runs through the acom bifurcation
            longest_shortest_path = find_longest_shortest_path(raca_ica_boundary, raca_nodes_1, polydata, [11, 10])['path']
            for edge in longest_shortest_path:
                cell_id = get_cellId_for_edge(edge, polydata)
                if label_array[cell_id] != 11:
                    logger.debug(f'\tChanging label of cell {cell_id} from {label_array[cell_id]} to 11')
                    polydata = set_label(11, cell_id, polydata)
        
        # Relabel L-ACA 
        # if label spilling happens at Acom bifurcation, the number of L-ACA 1-nodes 
        # should be larger than 2
        if len(laca_nodes_1) > 2 and len(laca_ica_boundary) > 0:
            # find path that runs through the acom bifurcation
            longest_shortest_path = find_longest_shortest_path(laca_ica_boundary, laca_nodes_1, polydata, [12, 10])['path']
            for edge in longest_shortest_path:
                cell_id = get_cellId_for_edge(edge, polydata)
                if label_array[cell_id] != 12:
                    logger.debug(f'\tChanging label of cell {cell_id} from {label_array[cell_id]} to 12')
                    polydata = set_label(12, cell_id, polydata)

    # 3rd-A2 segment
    # Set edge labels between 3rd-A2 origin and 3rd-A2 boundary to 10
    if variant_dict['anterior']['3rd-A2']:
        acom_boundary = find_boundary_points(10, 15, polydata) + find_boundary_points(11, 15, polydata) + find_boundary_points(12, 15, polydata)
        assert len(acom_boundary) == 1, 'More than one boundary found for 3rd-A2'
        acom_bif_a2 = find_acom_bif_for_3rd_a2(acom_boundary, polydata, labels=[10,11,12,15])
        path = find_shortest_path(acom_bif_a2, acom_boundary, polydata, [10, 11, 12])
        edge_labels_to_change = []
        for edge in path['path']:
            cell_id = get_cellId_for_edge(edge, polydata)
            edge_label = label_array[cell_id]
            if edge_label != 10:
                if edge_label not in edge_labels_to_change:
                    edge_labels_to_change.append(edge_label)
                logger.debug(f'\tChanging label of cell {cell_id} from {edge_label} to 10')
                polydata = set_label(10, cell_id, polydata)   

    # R-PCA segment
    if variant_dict['posterior']['R-Pcom']:
        rpca_nodes_1 = get_nodes_of_degree_n(1, 2, polydata)
        rpca_ba_boundary = find_boundary_points(2, 1, polydata)
        
        # if label spilling happens at R-Pcom bifurcation, the number of R-PCA 1-nodes 
        # should be larger than 2
        if len(rpca_nodes_1) > 2 and len(rpca_ba_boundary) > 0:
             # find path that runs through the pcom bifurcation
            longest_shortest_path = find_longest_shortest_path(rpca_ba_boundary, rpca_nodes_1, polydata, [2, 8])['path']
            for edge in longest_shortest_path:
                cell_id = get_cellId_for_edge(edge, polydata)
                if label_array[cell_id] != 2:
                    logger.debug(f'\tChanging label of cell {cell_id} from {label_array[cell_id]} to 2')
                    polydata = set_label(2, cell_id, polydata)

    # L-PCA segment
    if variant_dict['posterior']['L-Pcom']:
        lpca_nodes_1 = get_nodes_of_degree_n(1, 3, polydata)
        lpca_ba_boundary = find_boundary_points(3, 1, polydata)

        # if label spilling happens at L-Pcom bifurcation, the number of L-PCA 1-nodes
        # should be larger than 2
        if len(lpca_nodes_1) > 2 and len(lpca_ba_boundary) > 0:
            # find path that runs through the pcom bifurcation
            longest_shortest_path = find_longest_shortest_path(lpca_ba_boundary, lpca_nodes_1, polydata, [3, 9])['path']
            for edge in longest_shortest_path:
                cell_id = get_cellId_for_edge(edge, polydata)
                if label_array[cell_id] != 3:
                    logger.debug(f'\tChanging label of cell {cell_id} from {label_array[cell_id]} to 3')
                    polydata = set_label(3, cell_id, polydata)
    
    return polydata

def remove_self_loop_edges(polydata):
    """
    Find self-loop edges in polydata object and remove them.
    Self-Loop edges are edges that connect a point to itself or edges that are duplicated in the edge list.

    Args:
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with loop/duplicate edges removed
    """
    edge_list, cell_ids = get_edge_list(polydata)
    assert len(edge_list) == len(cell_ids)
    cells_to_remove = []
    # Use a dictionary to track seen edges for efficient duplicate detection
    seen_edges = {}
    for i, (edge, cell) in enumerate(zip(edge_list, cell_ids)):
        if edge in seen_edges:
            logger.debug(f'\tDetected duplicate edge {edge} with ID {cell}')
            cells_to_remove.append(cell)
        else:
            seen_edges[edge] = i
    
        if edge[0] == edge[1]:
            logger.debug(f'\tDetected loop edge {edge} with ID {cell}')
            cells_to_remove.append(cell)
        if (edge[1], edge[0]) in edge_list:
            cell_reversed = cell_ids[edge_list.index((edge[1], edge[0]))]
            if cell_reversed not in cells_to_remove:
                cells_to_remove.append(cell)

    if len(cells_to_remove) > 0:
        logger.debug(f'\tRemoving {len(cells_to_remove)} loop edges')
        polydata = delete_cells(cells_to_remove, polydata)

    return polydata

def node_is_bifurcation(ref_node, label, polydata):
    """
    Check if ref_node is actually a bifurcation node. To do this, the path between boundary points
    and the ref_node is checked. If there are no higher degree nodes in between, the ref_node is a 
    bifurcation node. ref_node must have degree 3 or higher.

    Args:
    ref_node: int, reference node (of degree 3 or higher)
    label: int, label
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    closest_node_to_bif: bool, True if ref_node is closest to bifurcation node
    """
    ref_node_is_bif = False
    if label == 4:
        labels = [8, 4]
        boundary_label = 8
        boundary_pts = find_boundary_points(label, boundary_label, polydata)
    elif label == 6:
        labels = [9, 6]
        boundary_label = 9
        boundary_pts = find_boundary_points(label, boundary_label, polydata)
    elif label == 11 or label == 12:
        labels = [10, label]
        boundary_label = 10
        boundary_pts = find_boundary_points(label, boundary_label, polydata)
    else:
        raise ValueError('Label not supported!')

    high_nodes = get_nodes_of_degree_n(3, labels, polydata) + get_nodes_of_degree_n(4, labels, polydata) + get_nodes_of_degree_n(5, labels, polydata)
    high_nodes.remove(ref_node)
    for pt in boundary_pts:
        # if pt in high_nodes:
        #     high_nodes.remove(pt)
        if pt == ref_node:
            ref_node_is_bif = True
            return ref_node_is_bif
        else:
            path = find_shortest_path(ref_node, pt, polydata, label)['path']
            path_ids = [p[0] for p in path] + [path[-1][1]]
            nodes_inbetween = [node for node in high_nodes if node in path_ids]
            if len(nodes_inbetween) == 0:
                ref_node_is_bif = True
                return ref_node_is_bif

    return ref_node_is_bif
    
def remove_short_branches(polydata, labellist=[4, 6, 11, 12], dist_threshold=10):
    """
    Removing short branches if they are not connected to a bifurcation node.
    This function checks for nodes of degree 1 and finds the closest node of higher degree.
    If the distance is smaller than dist_threshold and the higher degree node is no bifurcation node, 
    the path between the two nodes is removed.

    Args:
    polydata: vtkPolyData, polydata centerline graph
    labellist: list, list of labels to remove short branches from

    Returns:
    polydata: vtkPolyData, polydata centerline graph with short branches removed
    """
    nodes_1 = get_nodes_of_degree_n(1, None, polydata)
    closest_node_is_bif = False
    cells_to_remove = []
    for label in labellist:
        if label == 11:
            labels = [10, 11]
        elif label == 12:
            labels = [10, 12]
        elif label == 4:
            labels = [8, 4]
        elif label == 6:
            labels = [9, 6]
        else:
            labels = [label]
        nodes_3 = get_nodes_of_degree_n(3, labels, polydata)
        nodes_4 = get_nodes_of_degree_n(4, labels, polydata)
        nodes_5 = get_nodes_of_degree_n(5, labels, polydata)
        higher_nodes = nodes_3 + nodes_4 + nodes_5
        points = get_pointIds_for_label(label, polydata)
        for node in nodes_1:
            if node in points:
                closest_3_node, dist = find_closest_node_to_point(higher_nodes, node, label, polydata)
                if dist < dist_threshold:
                    if label == 4 or label == 6 or label == 11 or label == 12:
                        # checking for bifurcation node is only done for ICAs and ACAs
                        closest_node_is_bif = node_is_bifurcation(closest_3_node, label, polydata)
                    if not closest_node_is_bif:
                        path = find_shortest_path(node, closest_3_node, polydata, label)['path']
                        logger.debug(f'\tPath for segment {label} between {node} and {closest_3_node}: {path}')
                        logger.warning(f'\tALERT: Short branch found! Removing branch with {len(path)} cells for label {label}...')
                        for edge in path:
                            cells_to_remove.append(get_cellId_for_edge(edge, polydata))
    
    if len(cells_to_remove) > 0:
        # remove cells
        polydata = delete_cells(cells_to_remove, polydata)

    return polydata
    
def update_ids_of_node_dict(node_dict, polydata):
    """
    Update ids of all nodes in node_dict.
    This function takes the coordinates of each node in nodes_dict and finds the corresponding point id
    of the polydata object by calling the function find_id_for_coord. The id is then updated in the node_dict.

    Args:
    node_dict: dict, node dict
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    node_dict: dict, updated node dict
    """
    point_coords_dict = get_point_coord_lookup(polydata)

    for key1, value1 in node_dict.items():
        assert isinstance(value1, dict)

        for key2, value2 in value1.items():
            if isinstance(value2, dict):
                node_coord = value2['coords']
                new_id = find_id_for_coord(node_coord, point_coords_dict)
                value2['id'] = new_id
            elif isinstance(value2, list):
                for item in value2:
                    node_coord = item['coords']
                    new_id = find_id_for_coord(node_coord, point_coords_dict)
                    item['id'] = new_id
    
    return node_dict
    
def update_degrees_of_node_dict(node_dict, polydata):
    """
    Update degree of all nodes in node_dict.

    Args:
    node_dict: dict, node dict
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    node_dict_updated: dict, updated node dict
    """
    degree_lookup_table = get_degree_lookup_table(polydata)
    for key1, value1 in node_dict.items():
        assert isinstance(value1, dict)
        
        for key2, value2 in value1.items():
            if isinstance(value2, dict):
                value2['degree'] = degree_lookup_table[value2['id']]
            elif isinstance(value2, list):
                for item in value2:
                    item['degree'] = degree_lookup_table[item['id']]
    return node_dict

def move_3rd_a2_origin(polydata):
    """
    If the 3rd-A2 originates from the ACA, then move the origin to the Acom bifurcation.

    Args:
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with updated 3rd-A2 origin
    """
    label_array = get_label_array(polydata)
    if 15 in label_array:
        # Path between 3rd-A2 origin and 3rd-A2 boundary already relabeled to 10. 
        # If the 3rd-A2 origin is attached to ACA, then there are more than 1 Acom components.
        if get_number_of_connected_components(10, polydata) >= 2:
            for label in [11, 12]:
                aca_acom_nodes_3 = get_nodes_of_degree_n(3, [label, 10], polydata)
                aca_acom_nodes_4 = get_nodes_of_degree_n(4, [label, 10], polydata)
                aca_acom_nodes_high = aca_acom_nodes_3 + aca_acom_nodes_4
                a2_acom_boundary = find_boundary_points(10, 15, polydata)
                aca_acom_boundary = find_boundary_points(label, 10, polydata)
                a2_origin, _ = find_closest_node_to_point(aca_acom_nodes_high, a2_acom_boundary[0], [label, 10], polydata)
                acom_bif_id, _ = find_closest_node_to_point(aca_acom_nodes_high, aca_acom_boundary[0], [label, 10], polydata)
                logger.debug(f'\t3rd-A2 origin: {a2_origin}, Acom bifurcation: {acom_bif_id}')
                if a2_origin is not None and acom_bif_id is not None:
                    if a2_origin == acom_bif_id:
                        return polydata
                    else:
                        try:
                            find_shortest_path(a2_origin, acom_bif_id, polydata, [10])
                            return polydata
                        except: # if no path within segment 10 can be found, we have to move the origin
                            logger.warning('\tALERT: 3rd-A2 origin attached to ACA! Moving origin to Acom bifurcation...')
                            path = find_shortest_path(a2_origin, a2_acom_boundary[0], polydata, [10])['path']
                            pt = path[0][1]
                            cell_id = get_cellId_for_edge(path[0], polydata)
                            points = polydata.GetPoints()
                            points.SetPoint(pt, points.GetPoint(acom_bif_id))
                            logger.debug(f'\tChanging point {pt} to {acom_bif_id}')
                            polydata = delete_cells([cell_id], polydata)

                            polydata.Modified()

                            # clean filter
                            clean_filter = vtk.vtkCleanPolyData()
                            clean_filter.SetInputData(polydata)
                            clean_filter.SetTolerance(0.001)  # Set your desired tolerance
                            clean_filter.Update()
                            polydata = clean_filter.GetOutput()
                            
                            return polydata
    
    return polydata

def remove_self_loop(polydata, length_threshold=5):
    """
    This function finds and removes small self-loops: 
    -> The loop consists of 1 higher degree node and one loop path between them
    
    Args:
    polydata: vtkPolyData, polydata centerline graph
    length_threshold: int, threshold for the length of the loop path (default is 5)

    Returns:
    polydata: vtkPolyData, polydata centerline graph with self-loops removed
    """
    # identify nodes of degree 4 or higher
    nodes_4 = get_nodes_of_degree_n(4, None, polydata)
    nodes_5 = get_nodes_of_degree_n(5, None, polydata)
    nodes_higher = nodes_4 + nodes_5
    if len(nodes_higher) == 0:
        return polydata
    else:
        for node in nodes_higher:
            cell_ids = get_cellIds_for_point(node, polydata)
            labels = [int(get_label_for_cell(cell_id, polydata)) for cell_id in cell_ids]
            labels = list(set(labels))
            # find direct neighbors of these nodes
            nghs = get_neighbors_for_point(node, polydata)
            # for each neighbor, check if there are more than 1 paths to the higher degree node
            loop_paths = []
            for ngh in nghs:
                paths = find_all_paths(ngh, node, polydata, labels)
                if len(paths) > 1:
                    if len(paths) == 2:
                        path1, path2 = paths[0]['path'], paths[1]['path']
                        if len(path1) == 1:
                            loop_paths.append([(node, ngh)] + path2)
                        elif len(path2) == 1:
                            loop_paths.append([(node, ngh)] + path1)
                        else:
                            logger.warning(f'\tALERT: no path of length 1 found for node {node} and neighbor {ngh}. Paths: {paths}')
                            raise ValueError(f'\tALERT: no path of length 1 found for node {node} and neighbor {ngh}. Paths: {paths}')
                    else:
                        logger.warning(f'\tALERT: More than 2 paths found for node {node} and neighbor {ngh}. Paths: {paths}')
                        raise ValueError(f'\tALERT: More than 2 paths found for node {node} and neighbor {ngh}. Paths: {paths}')
            if len(loop_paths) > 0:
                assert len(loop_paths) == 2, 'More than 2 loop paths found for node {}!'.format(node)
                assert len(loop_paths[0]) == len(loop_paths[1]), 'Loop paths have different lengths for node {}!'.format(node)
                nodeIds_path1 = set([edge[0] for edge in loop_paths[0]])
                nodeIds_path2 = set([edge[0] for edge in loop_paths[1]])
                assert nodeIds_path1 == nodeIds_path2, 'Self-loop paths have different node ids for node {}!'.format(node)
                logger.warning(f'\tALERT: Small self-loop detected for node {node}! Removing path...')
                # remove one of the identical loop paths
                if len(loop_paths[0]) < length_threshold:
                    logger.debug(f'\tRemoving loop path: {loop_paths[0]}')
                    cells_to_remove = []
                    for edge in loop_paths[0]:
                        cell_id = get_cellId_for_edge(edge, polydata)
                        cells_to_remove.append(cell_id)
                    polydata = delete_cells(cells_to_remove, polydata)
                    logger.debug(f'\tRemoved cells: {cells_to_remove}')
                else:
                    logger.warning(f'\tALERT: Loop path {loop_paths[0]} is longer than threshold {length_threshold}. Not removing it.')
                    raise ValueError(f'\tALERT: Loop path {loop_paths[0]} is longer than threshold {length_threshold}. Not removing it.')

        return polydata


def remove_aca_acom_small_loops(polydata, loop_length_threshold=10):
    """
    This function finds and removes small loops in the ACA and Acom segments.
    We distinguish between two cases:
    1. The loop consists of two higher degree nodes and two loop paths between them.
       In that case we remove the longer of the two paths.
    2. The loop consists of three higher degree nodes and three loop paths between them.
       In that case we find the closest node to the opposite Acom boundary and remove the longer of 
       the two paths that are not connected to the closest node.

    Args:
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with ACA/Acom small loops removed
    """
    labels_array = get_label_array(polydata)
 
    if 10 in labels_array:
        for aca_label in [11, 12]:
            # find all small loops
            small_loops, loop_nodes = check_for_small_loops(polydata, [aca_label, 10], max_length=loop_length_threshold)
            if len(small_loops) > 0:
                assert len(small_loops) > 1, 'Too few loop paths!'
                logger.warning(f'\tALERT: Small ACA/Acom loop detected! Deleting longer part...')
                if len(small_loops) == 2: # in case of 2 loop paths, we can simply remove the longest one
                    logger.debug(f'\tLoop consists of two paths: {small_loops}')
                    path1, path2 = small_loops[0], small_loops[1]
                    path_to_remove = path1 if len(path1) > len(path2) else path2
                    logger.debug(f'\tPath to remove: {path_to_remove}')
                    cells_to_remove = []
                    for edge in path_to_remove:
                        cells_to_remove.append(get_cellId_for_edge(edge, polydata))
                elif len(small_loops) == 3: # in case of 3 loop paths, we cannot simply remove the longest one
                    logger.debug(f'\tLoop consists of three paths: {small_loops}')
                    if aca_label == 11:
                        # NOTE: Simply taking the first boundary point might go wrong for double Acom!
                        opposite_boundary = find_boundary_points(12, 10, polydata)[0]
                    else:
                        # NOTE: Simply taking the first boundary point might go wrong for double Acom!
                        opposite_boundary = find_boundary_points(11, 10, polydata)[0]
                    closest_node, _ = find_closest_node_to_point(loop_nodes, opposite_boundary, [aca_label, 10], polydata)
                    paths_to_remove = []
                    for path in small_loops:
                        if assert_node_on_path(closest_node, path):
                            paths_to_remove.append(path)
                    
                    assert len(paths_to_remove) == 2, 'Wrong number of paths to keep!'
                    path1, path2 = paths_to_remove[0], paths_to_remove[1]
                    path_to_remove = path1 if len(path1) > len(path2) else path2
                    logger.debug(f'\tPath to remove: {path_to_remove}')
                    cells_to_remove = []
                    for edge in path_to_remove:
                        cells_to_remove.append(get_cellId_for_edge(edge, polydata))
                else:
                    logger.warning(f'\tToo many small loops found: {small_loops}')
                    raise ValueError('\tToo many small loops found!')

                polydata = delete_cells(cells_to_remove, polydata)

    return polydata

def remove_ica_small_loops(polydata, loop_length_threshold=10):
    """
    This function finds and removes small loops in the ICA segments.
    The loop consists of three higher degree nodes and three loop paths between them.
    We find the loop nodes farthest away from the ICA start and remove the loop path between them.

    Args:
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with ACA/Acom small loops removed
    """
    
    for ica_label in [4, 6]:
        # find all small loops
        small_loops, loop_nodes = check_for_small_loops(polydata, [ica_label], max_length=loop_length_threshold)
        if len(small_loops) > 0:
            assert len(small_loops) > 1, 'Too few loop paths!'
            logger.warning(f'\tALERT: Small ICA loop detected! Deleting highest part...')
            if len(small_loops) == 3: 
                logger.debug(f'\tLoop consists of three paths: {small_loops}')
                ica_nodes_1 = get_nodes_of_degree_n(1, ica_label, polydata)
                ica_boundaries = []
                if ica_label == 4:
                    ica_boundaries += find_boundary_points(ica_label, 5, polydata)
                    ica_boundaries += find_boundary_points(ica_label, 8, polydata)
                    ica_boundaries += find_boundary_points(ica_label, 11, polydata)
                else:
                    ica_boundaries += find_boundary_points(ica_label, 7, polydata)
                    ica_boundaries += find_boundary_points(ica_label, 9, polydata)
                    ica_boundaries += find_boundary_points(ica_label, 12, polydata)
                
                for boundary in ica_boundaries:
                    if boundary in ica_nodes_1:
                        ica_nodes_1.remove(boundary)
                
                if len(ica_nodes_1) > 0:
                    ica_start = ica_nodes_1[0]
                    logger.debug(f'\tICA start: {ica_start}')
                    logger.debug(f'\tloop_nodes: {loop_nodes}')
                    closest_loop_node = None
                    min_dist = float('inf')
                    for node in loop_nodes:
                        dist = find_shortest_path(ica_start, node, polydata, ica_label)['length']
                        if dist < min_dist:
                            min_dist = dist
                            closest_loop_node = node
                    logger.debug(f'\tClosest loop node to ICA start: {closest_loop_node}')
                    loop_nodes.remove(closest_loop_node)

                else:
                    lowest_node = None
                    min_z = float('inf')
                    for node in loop_nodes:
                        coord = polydata.GetPoints().GetPoint(node)
                        if coord[2] < min_z:
                            min_z = coord[2]
                            lowest_node = node
                    logger.debug(f'\tLowest loop node: {lowest_node}')
                    loop_nodes.remove(lowest_node)
                
                assert len(loop_nodes) == 2, 'Wrong number of loop nodes found!'
                path_to_remove = find_shortest_path(loop_nodes[0], loop_nodes[1], polydata, ica_label)['path']
                if len(path_to_remove) < loop_length_threshold:
                    logger.debug(f'\tRemoving loop path: {path_to_remove}')
                    cells_to_remove = []
                    for edge in path_to_remove:
                        cells_to_remove.append(get_cellId_for_edge(edge, polydata))
                    polydata = delete_cells(cells_to_remove, polydata)
                else:
                    logger.warning(f'\tALERT: Loop path {path_to_remove} is longer than threshold {loop_length_threshold}. Not removing it.')
                    raise ValueError(f'ALERT: Loop path {path_to_remove} is longer than threshold {loop_length_threshold}. Not removing it.')

            else:
                raise NotImplementedError('Only loops with 3 paths are supported for ICA segments!')

    return polydata

def connect_broken_segments(polydata):
    """
    If a segment is broken, connect it with a single cell of the same label.
    The connection is done between the two closest points of degree 1. 

    Args:
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with connected segments
    """
    # the labels we have found to be disconnected in rare cases...
    labels = [11, 12, 15]
  
    for label in labels:
        num_components = get_number_of_connected_components(label, polydata)
        if num_components > 1:
            logger.warning(f'\tALERT: More than one connected component for label {label}. Connecting...')
            components = get_connected_components(label, polydata)
            closest_points, dist = find_closest_points_between_connected_components(components, polydata)
            logger.debug(f'\tClosest points: {closest_points}, distance: {dist}')
            pt1, pt2 = closest_points[0], closest_points[1]
            if pt1 != pt2:
                # connect by inserting cell
                polydata = connect_two_points(pt1, pt2, polydata, label)

    return polydata

def relabel_short_acom(nodes_dict, polydata):
    """
    For each connected Acom segment, if it is shorter than 3 edges, relabel neighboring ACA segments.

    Args:
    nodes_dict: dict, node dict
    polydata: vtkPolyData, polydata centerline graph

    Returns:
    polydata: vtkPolyData, polydata centerline graph with relabeled Acom segments
    """
    connected_components_acom = get_connected_components(10, polydata)
    for component in connected_components_acom:
        if len(component) < 3:
            logger.debug('\tShort Acom segment detected. Relabeling neighboring ACA segments...')
            r_acom_bif_ids = [n['id'] for n in nodes_dict['11']['Acom bifurcation']]
            r_aca_boundary_ids = [n['id'] for n in nodes_dict['11']['Acom boundary']]
            l_acom_bif_ids = [n['id'] for n in nodes_dict['12']['Acom bifurcation']]
            l_aca_boundary_ids = [n['id'] for n in nodes_dict['12']['Acom boundary']]
            _, r_dist = find_closest_node_to_point(r_acom_bif_ids, component[0], [11, 10], polydata)
            _, l_dist = find_closest_node_to_point(l_acom_bif_ids, component[0], [12, 10], polydata)
            if r_dist < l_dist and l_dist > 1:
                cell_id = get_cellId_for_edge((component[1], component[1]+1), polydata)
                polydata = set_label(10, cell_id, polydata)
                nodes_dict['12']['Acom boundary'][l_aca_boundary_ids.index(component[1])] = get_node_dict_entry(component[1]+1, 2, 12, polydata)[0]
                nodes_dict['10']['L-ACA boundary'][l_aca_boundary_ids.index(component[1])] = get_node_dict_entry(component[1]+1, 2, 10, polydata)[0]
            elif l_dist < r_dist and r_dist > 1:
                cell_id = get_cellId_for_edge((component[0], component[0]-1), polydata)
                polydata = set_label(10, cell_id, polydata)
                nodes_dict['11']['Acom boundary'][r_aca_boundary_ids.index(component[0])] = get_node_dict_entry(component[0]-1, 2, 11, polydata)[0]
                nodes_dict['10']['R-ACA boundary'][r_aca_boundary_ids.index(component[0])] = get_node_dict_entry(component[0]-1, 2, 10, polydata)[0]
    
    return polydata, nodes_dict
        



def run_postprocessing(input_graph: str, output_filename_without_extension: str, intermediate_dir: str, 
                       graph_dir: str, variant_dir: str, node_dir: str, window_size_smoothing: int = 5):
    """
    Run postprocessing on connected centerline graphs that were output by Voreen.
    This function performs multiple processing steps on the input graph:
    1. Changes background labels
    2. Identifies cow variant topology and writes to variant file
    3. Removes loops and short branches
    4. Identifies key nodes in various regions (BA-PCA, ICA-MCA, PCom, ACA-ACom)
    5. Reorders and smooths the centerline graph
    6. Performs sanity checks on the resulting topology

    Parameters:
    ---------
    input_graph (str): Path to input graph file (.vtp format)
    output_filename_without_extension (str): Base name for output files without extension
    intermediate_dir (str): Directory to store intermediate (non-smoothed) results
    graph_dir (str): Directory to store final smoothed graph files
    variant_dir (str): Directory to store variant information files
    node_dir (str): Directory to store final node files
    window_size_smoothing (int, optional): Window size used for smoothing. Defaults to 5.

    Returns:
    -------
    output_graph_smoothed (str): Path to the smoothed output graph file (.vtp)
    output_node_smoothed (str): Path to the smoothed output node file (.json)
    """
    
    # Check if input graph file exists
    if not os.path.exists(input_graph):
        logger.warning(f"Input graph file {input_graph} does not exist.")
        raise FileNotFoundError(f"Input graph file {input_graph} does not exist.")
    # Check if input graph file is in .vtp format
    if not input_graph.endswith('.vtp'):
        logger.warning(f"Input graph file {input_graph} is not in .vtp format.")
        raise ValueError(f"Input graph file {input_graph} is not in .vtp format.")

    # for saving intermediate (i.e. non-smoothed) graph
    output_graph_dir = os.path.join(intermediate_dir, 'cow_graphs')
    if not os.path.exists(output_graph_dir):
        os.makedirs(output_graph_dir)
    output_graph = os.path.join(output_graph_dir, f'{output_filename_without_extension}.vtp')
    
    # for saving final output (i.e. smoothed) graph
    output_graph_smoothed_dir = graph_dir
    if not os.path.exists(output_graph_smoothed_dir):
        os.makedirs(output_graph_smoothed_dir)
    output_graph_smoothed = os.path.join(output_graph_smoothed_dir, f'{output_filename_without_extension}.vtp')

    # for saving output variant file
    output_variant_dir = variant_dir
    if not os.path.exists(output_variant_dir):
        os.makedirs(output_variant_dir)
    output_variant = os.path.join(output_variant_dir, f'{output_filename_without_extension}.json')

    # for saving intermediate (i.e. non-smoothed) nodes file
    output_node_dir = os.path.join(intermediate_dir, 'cow_nodes')
    if not os.path.exists(output_node_dir):
        os.makedirs(output_node_dir)
    
    # for saving final output (i.e. smoothed) nodes file
    output_node_smoothed_dir = node_dir
    if not os.path.exists(output_node_smoothed_dir):
        os.makedirs(output_node_smoothed_dir)
    output_node = os.path.join(output_node_dir, f'{output_filename_without_extension}.json')
    output_node_smoothed = os.path.join(output_node_smoothed_dir, f'{output_filename_without_extension}.json')

    logger.info(f'Running post-processing with args:'
                f'\n\t- input_graph: {input_graph}'
                f'\n\t- output_filename_without_extension: {output_filename_without_extension}'
                f'\n\t- intermediate_dir: {intermediate_dir}'
                f'\n\t- graph_dir: {graph_dir}'
                f'\n\t- variant_dir: {variant_dir}'
                f'\n\t- node_dir: {node_dir}'
                f'\n\t- window_size_smoothing: {window_size_smoothing}')

    # get polydata
    logger.info(f'Getting polydata from {input_graph}...')
    polydata = get_vtk_polydata_from_file(input_graph)

    # 1) change background label
    polydata = change_background_label(polydata)

    # 2) identify cow variant and write to variant file
    # Identifying the variant is solely based on the conncectivity of the centerline graph
    # NOTE: Fenestration still missing. Get this information from graph later on while identifying key nodes
    logger.info('Identifying and extracting CoW variant...')
    variant_dict = get_cow_variant(polydata)

    # 3) Correcting graph
    logger.info('Correcting graph where necessary...')
    logger.info('Try: Relabeling ACA/PCA segments in case of Acom/Pcom label spilling')
    polydata = relabel_aca_pca_segments(variant_dict, polydata) # correct Acom/Pcom label spilling
    logger.info('Try: Removing self-loop edges')
    polydata = remove_self_loop_edges(polydata) # remove self-loops and duplicate edges
    logger.info('Try: Connecting broken ACA segments')
    polydata = connect_broken_segments(polydata) # connect segment (label 11, 12, 15 only) if it is broken
    logger.info('Try: Removing short branches')
    # remove short branches for ICAs, Pcoms, ACAs
    polydata = remove_short_branches(polydata, labellist=[4, 6, 8, 9, 11, 12], dist_threshold=10)
    polydata = remove_short_branches(polydata, labellist=[2, 3], dist_threshold=5) # remove short branches for PCAs
    polydata = remove_short_branches(polydata, labellist=[1], dist_threshold=2) # remove short branches for BA
    logger.info('Try: Moving 3rd-A2 origin')
    polydata = move_3rd_a2_origin(polydata) # moving 3rd-A2 origin to Acom bifurcation if necessary
    logger.info('Try: Removing self-loops')
    polydata = remove_self_loop(polydata, length_threshold=6) # remove self-loops in the graph
    logger.info('Try: Removing small loops in ACA/Acom segments')
    polydata = remove_aca_acom_small_loops(polydata, loop_length_threshold=10) # remove small loops in ACA and Acom segments
    logger.info('Try: Removing small loops in ICA segments')
    polydata = remove_ica_small_loops(polydata, loop_length_threshold=10) # remove small loops in ICA segments

    # 4) identify key nodes and write to node file + update variant file to include fenestrations
    logger.info('Identifying and extracting key nodes:')       
    ba_pca_nodes_dict, variant_dict, polydata = get_ba_pca_nodes(polydata, variant_dict)
    ica_mca_nodes_dict = get_ica_mca_nodes(polydata, variant_dict)
    pcom_nodes_dict = get_pcom_nodes(polydata, variant_dict)
    aca_acom_nodes_dict, variant_dict, polydata = get_aca_acom_nodes(polydata, variant_dict)

    # combine node dicts
    logger.info('Combining node dictionaries, updating node IDs and degrees')
    node_dict = {**ba_pca_nodes_dict, **ica_mca_nodes_dict, **pcom_nodes_dict, **aca_acom_nodes_dict}
    node_dict = update_ids_of_node_dict(node_dict, polydata)
    node_dict = update_degrees_of_node_dict(node_dict, polydata)

    # save to files
    write_vtk_polydata_to_file(polydata, output_graph)
    with open(output_variant, 'w') as f:
        json.dump(variant_dict, f, indent=3)
    with open(output_node, 'w') as f:
        json.dump(node_dict, f, indent=4)

    # 5) reorder and smooth
    logger.info('Reordering and smoothing centerline graph...')
    polydata_smooth, nodes_dict_smooth = reorder_and_smooth(node_dict, variant_dict, polydata, window_size=window_size_smoothing)
    logger.info('Try: Removing self-loop edges after smoothing')
    polydata_smooth = remove_self_loop_edges(polydata_smooth)
    logger.info('Updating node ids and degrees after smoothing')
    nodes_dict_smooth = update_ids_of_node_dict(nodes_dict_smooth, polydata_smooth)
    polydata_smooth, nodes_dict_smooth = relabel_short_acom(nodes_dict_smooth, polydata_smooth)
    nodes_dict_smooth = update_degrees_of_node_dict(nodes_dict_smooth, polydata_smooth)

    # 7) save files
    logger.info(f'Saving smoothed graph and nodes file to {output_graph_smoothed} and {output_node_smoothed}')
    write_vtk_polydata_to_file(polydata_smooth, output_graph_smoothed)

    with open(output_node_smoothed, 'w') as f:
        json.dump(nodes_dict_smooth, f, indent=4)

    # 8) Test nodes and topology
    logger.info('Running sanity checks on smoothed graph...')
    run_sanity_check(nodes_dict_smooth, variant_dict, polydata_smooth)
    
    logger.info('Postprocessing completed successfully.')

    return output_graph_smoothed, output_node_smoothed

