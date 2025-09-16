import os
import json
from utils.utils_feature_extraction import *
from utils.utils_graph_processing import *
from extract_features.extract_segments_for_feature_computation import extract_segments, get_consistent_segments

from logger import logger

def extract_segment_geometry(cnt_vtp_file: str, variant_dir: str, node_dir: str, feature_dir: str, modality: str = None, 
                             radius_attribute: str = 'ce_radius', median_p1: float = 7.2, median_a1: float = 15.6, median_c7: float = 7.1, 
                             margin_from_cow: int = 10, smooth_curve: bool = True, factor_num_points: int = 2, 
                             threshold_nan_radius: float = 0.5, threshold_broken_segment_removal: float = 0.66) -> str:
    """
    Run geometry extraction for all segments of the CoW. Splipy is used for bi-cubic interpolation.

    Parameters:
    cnt_vtp_file (str): Path to the centerline VTP file.
    variant_dir (str): Directory containing variant JSON files.
    node_dir (str): Directory containing node JSON files.
    feature_dir (str): Directory to save feature JSON files.
    modality (str): Imaging modality ('ct' or 'mr'). If None, it is inferred from the filename.
    radius_attribute (str): Attribute to use for radius computation.  ('ce_radius', 'avg_radius', 'min_radius', 'max_radius')
    median_p1 (float): Median length of P1 segment (for absent Pcom)
    median_a1 (float): Median length of A1 segment (for absent Acom)
    median_c7 (float): Median length of C7 segment (for absent Pcom)
    margin_from_cow (int): Margin from CoW to cap segments.
    smooth_curve (bool): If True, the (splipy) curve is smoothed before geometry computation.
    factor_num_points (int): Factor to increase the number of sample points along the curve.
    threshold_nan_radius (float): Threshold for the fraction of NaN radius values along a segment path to consider the radius computation failed.
    threshold_broken_segment_removal (float): Threshold for the fraction of the median length of a broken segment (A1 or P1) to remove it from the feature dict.

    Returns:
    savepath (str): Path to the saved geometry JSON file.
    """

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    filename = os.path.basename(cnt_vtp_file)

    if modality == None:
        if '_ct' in filename.lower():
            modality = 'ct'
        elif '_mr' in filename.lower():
            modality = 'mr'
        else:
            raise ValueError('Modality not specified!')
        
    logger.info(f'Running segment feature extraction for the {modality} modality with args:'
                f'\n\t- cnt_vtp_file={cnt_vtp_file}'
                f'\n\t- variant_dir={variant_dir}'
                f'\n\t- node_dir={node_dir}'
                f'\n\t- feature_dir={feature_dir}'
                f'\n\t- radius_attribute={radius_attribute}'
                f'\n\t- smooth_curve={smooth_curve}'
                f'\n\t- factor_num_points={factor_num_points}'
                f'\n\t- median_p1={median_p1}'
                f'\n\t- median_a1={median_a1}'
                f'\n\t- median_c7={median_c7}'
                f'\n\t- margin_from_cow={margin_from_cow}'
                f'\n\t- threshold_nan_radius={threshold_nan_radius}'
                f'\n\t- threshold_broken_segment_removal={threshold_broken_segment_removal}'
                )
        
    geometry_dict = {}
    savepath = os.path.join(feature_dir, filename.replace('.vtp', '.json'))
    
    nodes_file = os.path.join(node_dir, filename.replace('.vtp', '.json'))
    variant_file = os.path.join(variant_dir, filename.replace('.vtp', '.json'))
    polydata = get_vtk_polydata_from_file(cnt_vtp_file)
    with open(nodes_file, 'r') as f:
        nodes_dict = json.load(f)
    with open(variant_file, 'r') as f:
        variant_dict = json.load(f)
    
    logger.info(f'Extracting segments for feature computation...')
    segments = extract_segments(nodes_dict, variant_dict)
    segments = get_consistent_segments(segments, polydata, nodes_dict, modality, median_p1, median_a1, 
                                       median_c7, margin_from_cow)
    logger.debug(f'\tSegments: {segments}')
    logger.info(f'Extracting segment features (radius, length, tortuosity, volume, curvature) for...')
    for segment_name in segments:
        logger.info(f'...segment {segment_name}')
        segment = segments[segment_name]
        if len(segment) == 1:
            start_node, end_node, label = segment[0]
            # Find all paths
            paths = find_all_paths(start_node, end_node, polydata, label)
            if len(paths) == 1:
                path = paths[0]['path']
                # Radius along path
                path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                # Geometry along the path
                path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                geometry_dict[segment_name] = [get_dict_entry(segment[0], path_radius, path_geometry, nan_edges=nan_edges)]
            else:
                logger.warning(f'\tALERT: Multiple paths found for {segment_name}!') 
                # Fenestration of A1/P1 segment
                if segment_name in ['R-P1', 'L-P1', 'R-PCA', 'L-PCA', 'R-A1', 'L-A1', 'R-ACA', 'L-ACA']:
                    if segment_name == 'R-PCA' or segment_name == 'R-P1':
                        ba_boundary = nodes_dict['2']['BA boundary']
                        bif_id = nodes_dict['2']['Pcom bifurcation'][0]['id']
                        if len(ba_boundary) == 1:
                            paths_tmp = find_all_paths(nodes_dict['2']['BA boundary'][0]['id'], bif_id, polydata, label)
                            path_margin = find_shortest_path(bif_id, nodes_dict['2']['PCA end'][0]['id'], polydata, label)['path']
                        else:
                            # TODO: Implement fenestration with multiple BA boundaries
                            raise NotImplementedError 
                    elif segment_name == 'L-PCA' or segment_name == 'L-P1':
                        ba_boundary = nodes_dict['3']['BA boundary']
                        bif_id = nodes_dict['3']['Pcom bifurcation'][0]['id']
                        if len(ba_boundary) == 1:
                            paths_tmp = find_all_paths(nodes_dict['3']['BA boundary'][0]['id'], bif_id, polydata, label)
                            path_margin = find_shortest_path(bif_id, nodes_dict['3']['PCA end'][0]['id'], polydata, label)['path']
                        else:
                            # TODO: Implement fenestration with multiple BA boundaries
                            raise NotImplementedError
                    elif segment_name == 'R-ACA' or segment_name == 'R-A1':
                        bif_id = nodes_dict['11']['Acom bifurcation'][0]['id']
                        paths_tmp = find_all_paths(nodes_dict['11']['ICA boundary'][0]['id'], bif_id, polydata, label)
                        path_margin = find_shortest_path(bif_id, nodes_dict['11']['ACA end'][0]['id'], polydata, label)['path']
                    elif segment_name == 'L-ACA' or segment_name == 'L-A1':
                        bif_id = nodes_dict['12']['Acom bifurcation'][0]['id']
                        paths_tmp = find_all_paths(nodes_dict['12']['ICA boundary'][0]['id'], bif_id, polydata, label)
                        path_margin = find_shortest_path(bif_id, nodes_dict['12']['ACA end'][0]['id'], polydata, label)['path']

                    nodes_high = get_nodes_of_degree_n(3, label, polydata) + get_nodes_of_degree_n(4, label, polydata) + get_nodes_of_degree_n(5, label, polydata)
                    # find all high nodes on pahts
                    path_node_ids = [p[0] for p in paths_tmp[0]['path']] + [p[0] for p in paths_tmp[1]['path']]
                    margin_node_ids = [p[0] for p in path_margin]
                    nodes_on_path = list(set(nodes_high).intersection(set(path_node_ids)))
                    nodes_on_path_margin = list(set(nodes_on_path).intersection(set(margin_node_ids)))
                    # Compute fenestration paths that stop at different points
                    if len(nodes_on_path) > 0 and len(nodes_on_path_margin) > 0:
                        assert len(nodes_on_path) == 2, f'Wrong number of 3-nodes within fenestration loop!'
                        # find node closer to bifurcation
                        bif_close_id, distance_between_nodes = find_closest_node_to_point(nodes_on_path, bif_id, label, polydata)
                        # distance_between_nodes = find_shortest_path(bif_id, bif_close_id, polydata, label)['length']
                        
                        if segment_name.endswith('A1'):
                            if assert_node_on_path(bif_close_id, paths_tmp[0]['path']):
                                aca_index = 1
                            else:
                                aca_index = 0
                            path1 = paths[aca_index]['path']
                            path2 = paths[1 - aca_index]['path'][:-distance_between_nodes]
  
                        elif segment_name.endswith('ACA'):
                            assert aca_index is not None
                            path1 = paths[aca_index]['path']
                            path2 = paths[1 - aca_index]['path']
                        
                        if segment_name.endswith('P1'):
                            if assert_node_on_path(bif_close_id, paths_tmp[0]['path']):
                                pca_index = 1
                            else:
                                pca_index = 0
                            path1 = paths[pca_index]['path']
                            path2 = paths[1 - pca_index]['path'][:-distance_between_nodes]
                        elif segment_name.endswith('PCA'):
                            assert pca_index is not None
                            path1 = paths[pca_index]['path']
                            path2 = paths[1 - pca_index]['path']
                        
                        list_geom_dicts = []

                        for path in [path1, path2]:
                            # Radius along path
                            path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                            # Geometry along the path
                            path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                            list_geom_dicts.append(get_dict_entry((path[0][0], path[-1][1]), path_radius, path_geometry, nan_edges=nan_edges))
                        geometry_dict[segment_name] = list_geom_dicts
                    
                    else:
                        list_geom_dicts = []
                        for p in paths:
                            path = p['path']
                            # Radius along path
                            path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                            # Geometry along the path
                            path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                            list_geom_dicts.append(get_dict_entry(segment[0], path_radius, path_geometry, nan_edges=nan_edges))
                        geometry_dict[segment_name] = list_geom_dicts

                elif segment_name in ['R-P2', 'L-P2', 'R-A2', 'L-A2']:
                    path = find_shortest_path(start_node, end_node, polydata, label)['path']
                    # Radius along path
                    path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                    # Geometry along the path
                    path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                    geometry_dict[segment_name] = [get_dict_entry(segment[0], path_radius, path_geometry, nan_edges=nan_edges)]

                else:
                    list_geom_dicts = []
                    for p in paths:
                        path = p['path']
                        # Radius along path
                        path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                        # Geometry along the path
                        path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                        list_geom_dicts.append(get_dict_entry(segment[0], path_radius, path_geometry, nan_edges=nan_edges))
                    geometry_dict[segment_name] = list_geom_dicts

        if len(segment) > 1:
            logger.warning(f'\tALERT: Multiple segments found for {segment_name}!')
            list_geom_dicts = []
            for tpl in segment:
                start_node, end_node, label = tpl
                # Find all paths
                paths = find_all_paths(start_node, end_node, polydata, label)
                if len(paths) == 1:
                    path = paths[0]['path']
                    # Radius along path
                    path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                    # Geometry along the path
                    path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                    list_geom_dicts.append(get_dict_entry(tpl, path_radius, path_geometry, nan_edges=nan_edges))
                else:
                    logger.warning(f'\tALERT: Multiple paths found for {segment_name}!')
                    # NOTE: P1 fenestration for case topcow_mr_012
                    for p in paths:
                        path = p['path']
                        # Radius along path
                        path_radius, nan_edges = compute_radius_along_path(polydata, path, radius_attribute, nan_threshold=threshold_nan_radius)
                        # Geometry along the path
                        path_geometry = compute_geometry_along_path(polydata, path, radius_attribute=radius_attribute, factor_num_points=factor_num_points, comp_tor=False, smooth=smooth_curve, plot=False)
                        list_geom_dicts.append(get_dict_entry(tpl, path_radius, path_geometry, nan_edges=nan_edges))
            geometry_dict[segment_name] = list_geom_dicts
    
    # remove P1 entry if P1 broken (not touching BA) and segment shorter than 2/3 the median_p1 length
    # NOTE: This threshold might be changed
    threshold_p1 = threshold_broken_segment_removal * median_p1
    if not variant_dict['posterior']['R-P1']:
        if 'Pcom bifurcation' in nodes_dict['2'] and 'BA boundary' in nodes_dict['2']:
            if 'R-P1' in geometry_dict and geometry_dict['R-P1'][0]['length'] < threshold_p1:
                logger.warning(f'\tALERT: Removing R-P1 entry from feature dict!')
                geometry_dict.pop('R-P1')
    if not variant_dict['posterior']['L-P1']:
        if 'Pcom bifurcation' in nodes_dict['3'] and 'BA boundary' in nodes_dict['3']:
            if 'L-P1' in geometry_dict and geometry_dict['L-P1'][0]['length'] < threshold_p1:
                logger.warning(f'\tALERT: Removing L-P1 entry from feature dict!')
                geometry_dict.pop('L-P1')
    
    # remove A1 entry if A1 broken (not touching ICA) and segment shorter than 2/3 the median_a1 length
    # NOTE: This threshold might be changed
    threshold_a1 = threshold_broken_segment_removal * median_a1
    if not variant_dict['anterior']['R-A1']:
        if 'Acom bifurcation' in nodes_dict['11'] and 'ICA boundary' in nodes_dict['11']:
            if 'R-A1' in geometry_dict and geometry_dict['R-A1'][0]['length'] < threshold_a1:
                logger.warning(f'\tALERT: Removing R-A1 entry from feature dict!')
                geometry_dict.pop('R-A1')
    if not variant_dict['anterior']['L-A1']:
        if 'Acom bifurcation' in nodes_dict['12'] and 'ICA boundary' in nodes_dict['12']:
            if 'L-A1' in geometry_dict and geometry_dict['L-A1'][0]['length'] < threshold_a1:
                logger.warning(f'\tALERT: Removing L-A1 entry from feature dict!')
                geometry_dict.pop('L-A1')
    
    # save geometry dict
    geometry_dict = reorder_dict(geometry_dict)
    logger.debug(f'Segment feature dict: {geometry_dict}')
    logger.info(f'Saving segment feature dict to {savepath}')
    with open(savepath, 'w') as f:
        json.dump(geometry_dict, f, indent=2)

    return savepath

    
    