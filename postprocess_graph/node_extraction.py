import copy
from utils.utils_graph_processing import *

from logger import logger

def get_ba_pca_nodes(polydata, variant_dict):
    """
    Function to identify BA & PCA nodes and write to node dict. 
    It's a rule-based approach. Especially hard to cover all possible P1 fenestration cases.
    Might run into sum errors for messed-up cases --> TODO: Improve this in the future.

    Args:
    polydata: vtkPolyData, polydata object
    variant_dict: dict, dictionary with cow variant information

    Returns:
    nodes_dict: dict, dictionary with BA & PCA node information
    variant_dict: dict, dictionary with cow variant information (updated to contain fenestration info)
    """
    
    rpca_label, lpca_label = 2, 3
    rpcom_label, lpcom_label = 8, 9
    ba_label = 1

    nodes_dict = {}

    ba_nodes_1 = get_nodes_of_degree_n(1, ba_label, polydata)
    ba_nodes_3 = get_nodes_of_degree_n(3, ba_label, polydata)
    ba_nodes_4 = get_nodes_of_degree_n(4, ba_label, polydata)
    ba_nodes_high = ba_nodes_3 + ba_nodes_4
    ba_rpca_boundary = find_boundary_points(ba_label, rpca_label, polydata)
    ba_lpca_boundary = find_boundary_points(ba_label, lpca_label, polydata)

    nodes_dict['1'] = {}
    nodes_dict['2'] = {}
    nodes_dict['3'] = {}

    ba_boundary_rpca, ba_boundary_lpca = [], []
    rpca_boundary_ba, lpca_boundary_ba = [], []
    ba_start, ba_bif_node = [], []

    labels_array = get_label_array(polydata)

    # BA segment
    logger.info('1) Extracting BA nodes ...')
    # Case 1: Both P1 present
    if variant_dict['posterior']['R-P1'] and variant_dict['posterior']['L-P1']:
        logger.debug('\tBoth P1 present')
        assert len(ba_rpca_boundary) > 0 and len(ba_lpca_boundary) > 0, 'P1 present but BA not connected?!'
        if len(ba_nodes_high) == 1: # only 1 BA higher node -> must be BA bifurcation
            ba_bif_id = ba_nodes_high[0]
            ba_bif_node = get_node_dict_entry(ba_nodes_high[0], 3, ba_label, polydata)
            logger.debug(f'\tOnly 1 BA higher node present: {ba_bif_node}')
            
        else:
            # If both P1 present, there must be a BA higher node...
            assert len(ba_nodes_high) > 1, 'BA bifurcation not found!'
            # Identify BA bifurcation node as the one closest to both PCA boundaries
            ba_bif_id, _ = find_closest_node_to_point(ba_nodes_high, ba_rpca_boundary[0], ba_label, polydata)
            ba_bif_id_2, _ = find_closest_node_to_point(ba_nodes_high, ba_lpca_boundary[0], ba_label, polydata)
            assert ba_bif_id == ba_bif_id_2, 'BA bifurcation not uniquely found!'
            ba_bif_node = get_node_dict_entry(ba_bif_id, 3, ba_label, polydata)
            logger.debug(f'\tMultiple BA higher nodes present. BA bifurcation node is: {ba_bif_node}')

        if ba_bif_id in ba_nodes_4: # BA bifurcation of degree 4 -> P1 fenestration?
            if len(ba_rpca_boundary) == 2 and len(ba_lpca_boundary) == 1:
                variant_dict['fenestration']['R-P1'] = True
                logger.debug(f'\tBA bifurcation of degree 4 present. And two R-PCA boundaries found: {ba_rpca_boundary}. R-P1 fenestration!')
            elif len(ba_rpca_boundary) == 1 and len(ba_lpca_boundary) == 2:
                variant_dict['fenestration']['L-P1'] = True
                logger.debug(f'\tBA bifurcation of degree 4 present. And two L-PCA boundaries found: {ba_lpca_boundary}. L-P1 fenestration!')
            else:
                logger.warning(f'\tALERT: BA bifurcation of degree 4 present but wrong number of PCA boundaries! {ba_rpca_boundary}, {ba_lpca_boundary}')
                raise ValueError('\tBA bifurcation of degree 4 present but wrong number of PCA boundaries')
        
        # get boundary points between BA and PCA
        for pt in ba_rpca_boundary: 
            ba_boundary_rpca += get_node_dict_entry(pt, 2, ba_label, polydata)
            rpca_boundary_ba += get_node_dict_entry(pt, 2, rpca_label, polydata)
            ba_nodes_1.remove(pt)
        for pt in ba_lpca_boundary:
            ba_boundary_lpca += get_node_dict_entry(pt, 3, ba_label, polydata)
            lpca_boundary_ba += get_node_dict_entry(pt, 3, lpca_label, polydata)
            ba_nodes_1.remove(pt)

        logger.debug(f'\tR-PCA boundary IDs: {ba_rpca_boundary}')
        logger.debug(f'\tL-PCA boundary IDs: {ba_lpca_boundary}')

        # if only 1 BA 1-node present, it must be the start node
        if len(ba_nodes_1) == 1: 
            ba_start = get_node_dict_entry(ba_nodes_1[0], 1, ba_label, polydata)
            logger.debug(f'\tOnly 1 BA  1-node left; must be BA start: {ba_start}')
        else: # 0 or more than 1 BA 1-nodes present
            logger.warning(f'\tALERT: no BA 1-node found for BA start!  {len(ba_nodes_1)} BA 1-nodes left! Choosing lowst point instead...')
            # Choose lowest BA higher node (or lowest BA 1-node) as BA start instead
            ba_nodes_high_copy = copy.deepcopy(ba_nodes_high)
            ba_nodes_high_copy.remove(ba_bif_id)
            if len(ba_nodes_high) > 0:
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_high_copy]
                max_z_index = np.argmax(z_values)
                ba_start = get_node_dict_entry(ba_nodes_high_copy[max_z_index], 3, ba_label, polydata)
                logger.debug(f'\tLowest BA higher node chosen as BA start: {ba_start}')
            else:
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_1]
                min_z_index = np.argmin(z_values)
                ba_start = get_node_dict_entry(ba_nodes_1[min_z_index], 1, ba_label, polydata)
                logger.debug(f'\tLowest BA 1-node chosen as BA start: {ba_start}')

        # sanity check
        assert len(ba_bif_node) > 0, 'BA bifurcation not found!'

    # Case 2: Only R-P1 present
    elif variant_dict['posterior']['R-P1'] and not variant_dict['posterior']['L-P1']:
        logger.debug('\tOnly R-P1 present')
        assert len(ba_rpca_boundary) > 0 and len(ba_lpca_boundary) == 0, 'Wrong number of BA/PCA boundaries'
        if len(ba_rpca_boundary) == 1: # We expect only 1 BA/R-PCA boundary
            ba_boundary_rpca = get_node_dict_entry(ba_rpca_boundary[0], 2, ba_label, polydata)
            rpca_boundary_ba = get_node_dict_entry(ba_rpca_boundary[0], 2, rpca_label, polydata)
            ba_nodes_1.remove(ba_rpca_boundary[0])
            logger.debug(f'\tR-PCA boundary found: {ba_rpca_boundary}')
            if len(ba_nodes_high) == 0: 
                # if no BA higher node present, the only BA 1-node left must be the start node
                assert len(ba_nodes_1) == 1, 'No BA start found!'
                ba_start = get_node_dict_entry(ba_nodes_1[0], 1, ba_label, polydata)
                logger.debug(f'\tOnly 1 BA 1-node left; must be BA start: {ba_start}')
            elif len(ba_nodes_high) == 1: # Case with BA bifurcation but no L-P1
                assert len(ba_nodes_4) == 0, 'BA bifurcation of degree 4?!'
                ba_bif_node = get_node_dict_entry(ba_nodes_3[0], 3, ba_label, polydata)
                logger.debug(f'\tNo L-P1 but BA bifurcation of degree 3 present: {ba_bif_node}')
                assert len(ba_nodes_1) == 2, 'Wrong number of BA 1-nodes!'
                # Choose lower point as BA start
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_1]
                min_z_index = np.argmin(z_values)
                ba_start = get_node_dict_entry(ba_nodes_1[min_z_index], 1, ba_label, polydata)
                logger.debug(f'\tLowest BA 1-node chosen as BA start: {ba_start}')
                 
            else: # multiple BA higher nodes present
                # select closest BA higher node to the BA/R-PCA boundary as BA start...
                ba_bif_id, _ = find_closest_node_to_point(ba_nodes_high, ba_rpca_boundary[0], ba_label, polydata)
                ba_bif_node = get_node_dict_entry(ba_bif_id, 3, ba_label, polydata)
                logger.debug(f'\tBA bifurcation is closest node to R-PCA boundary: {ba_bif_node}')
                ba_nodes_high_copy = copy.deepcopy(ba_nodes_high)
                ba_nodes_high_copy.remove(ba_bif_id)
                # ... and choose the lowest BA higher node as BA start
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_high_copy]
                max_z_index = np.argmax(z_values)
                ba_start = get_node_dict_entry(ba_nodes_high_copy[max_z_index], 1, ba_label, polydata)
                logger.debug(f'\tLowest BA higher node chosen as BA start: {ba_start}')
        else:
            logger.warning(f'ALERT: Wrong number of BA/PCA boundaries! {ba_rpca_boundary}')
            raise ValueError('Wrong number of BA/PCA boundaries!')
    
    # Case 3: Only L-P1 present
    elif variant_dict['posterior']['L-P1'] and not variant_dict['posterior']['R-P1']:
        logger.debug('\tOnly L-P1 present')
        assert len(ba_lpca_boundary) > 0 and len(ba_rpca_boundary) == 0, 'Wrong number of BA/PCA boundaries'
        if len(ba_lpca_boundary) == 1: # We expect only 1 BA/L-PCA boundary
            ba_boundary_lpca = get_node_dict_entry(ba_lpca_boundary[0], 3, ba_label, polydata)
            lpca_boundary_ba = get_node_dict_entry(ba_lpca_boundary[0], 3, lpca_label, polydata)
            ba_nodes_1.remove(ba_lpca_boundary[0])
            logger.debug(f'\tL-PCA boundary found: {ba_lpca_boundary}')
            if len(ba_nodes_high) == 0:
                # if no BA higher node present, the only BA 1-node left must be the start node
                assert len(ba_nodes_1) == 1, 'No BA start found!'
                ba_start = get_node_dict_entry(ba_nodes_1[0], 1, ba_label, polydata)
                logger.debug(f'\tOnly 1 BA 1-node left; must be BA start: {ba_start}')
            elif len(ba_nodes_high) == 1: # Case with BA bifurcation but no R-P1
                assert len(ba_nodes_4) == 0, 'BA bifurcation of degree 4?!'
                ba_bif_node = get_node_dict_entry(ba_nodes_3[0], 3, ba_label, polydata)
                logger.debug(f'\tNo R-P1 but BA bifurcation of degree 3 present: {ba_bif_node}')
                assert len(ba_nodes_1) == 2, 'Wrong number of BA 1-nodes!'
                # Choose lowest BA 1-node as BA start
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_1]
                min_z_index = np.argmin(z_values)
                ba_start = get_node_dict_entry(ba_nodes_1[min_z_index], 1, ba_label, polydata)
                logger.debug(f'\tLowest BA 1-node chosen as BA start: {ba_start}')
                 
            else: # multiple BA higher nodes present
                # select closest BA higher node to the BA/L-PCA boundary as BA bifurcation...
                ba_bif_id, _ = find_closest_node_to_point(ba_nodes_high, ba_lpca_boundary[0], ba_label, polydata)
                ba_bif_node = get_node_dict_entry(ba_bif_id, 3, ba_label, polydata)
                logger.debug(f'\tBA bifurcation is closest node to L-PCA boundary: {ba_bif_node}')
                ba_nodes_high_copy = copy.deepcopy(ba_nodes_high)
                ba_nodes_high_copy.remove(ba_bif_id)
                z_values = [polydata.GetPoint(pt)[2] for pt in ba_nodes_high_copy]
                max_z_index = np.argmax(z_values)
                ba_start = get_node_dict_entry(ba_nodes_high_copy[max_z_index], 1, ba_label, polydata)
                logger.debug(f'\tLowest BA higher node chosen as BA start: {ba_start}')
        else:
            logger.warning(f'ALERT: Wrong number of BA/PCA boundaries! {ba_lpca_boundary}')
            raise ValueError('Wrong number of BA/PCA boundaries!')
    
    # Case 4: No P1 present
    else:
        logger.debug('\tNo P1 present')
        # no BA/PCA boundary points
        assert len(ba_rpca_boundary) == 0 and len(ba_lpca_boundary) == 0, 'Wrong number of BA/PCA boundaries'
        assert len(ba_nodes_1) == 2, 'Wrong number of BA 1-nodes'
        assert len(ba_nodes_3) == 0, 'BA bifurcation present?!'

        # Choose lower point as BA start
        z1, z2 = polydata.GetPoint(ba_nodes_1[0])[2], polydata.GetPoint(ba_nodes_1[1])[2]
        if z1 < z2:
            ba_start = get_node_dict_entry(ba_nodes_1[0], 1, ba_label, polydata)
            ba_bif_node = get_node_dict_entry(ba_nodes_1[1], 1, ba_label, polydata)
        else:
            ba_start = get_node_dict_entry(ba_nodes_1[1], 1, ba_label, polydata)
            ba_bif_node = get_node_dict_entry(ba_nodes_1[0], 1, ba_label, polydata)
        logger.debug(f'\tLowest BA 1-node chosen as BA start: {ba_start}')
        logger.debug(f'\tOther BA 1-node chosen as BA bifurcation: {ba_bif_node}')
    
    if len(ba_start) > 0:
        nodes_dict['1']['BA start'] = ba_start
    if len(ba_bif_node) > 0:
        nodes_dict['1']['BA bifurcation'] = ba_bif_node
    if len(ba_boundary_rpca) > 0:
        nodes_dict['1']['R-PCA boundary'] = ba_boundary_rpca
    if len(ba_boundary_lpca) > 0:
        nodes_dict['1']['L-PCA boundary'] = ba_boundary_lpca

    logger.debug(f'\tBA nodes extracted: {nodes_dict["1"]}')

    # PCA segments
    for pca_label, pcom_label, p1_name, pcom_name in zip([rpca_label, lpca_label],[rpcom_label, lpcom_label], ['R-P1', 'L-P1'], ['R-Pcom', 'L-Pcom']):
        if pca_label == 2:
            pca_boundary_ba = rpca_boundary_ba
            ba_pca_boundary = ba_rpca_boundary
        elif pca_label == 3:
            pca_boundary_ba = lpca_boundary_ba
            ba_pca_boundary = ba_lpca_boundary
        else:
            logger.warning(f'\tWrong PCA label: {pca_label}')
            raise ValueError('\tWrong PCA label!')
        
        logger.info(f'... and PCA nodes of label {pca_label}...')

        pca_nodes_1 = get_nodes_of_degree_n(1, pca_label, polydata)
        pca_nodes_3 = get_nodes_of_degree_n(3, [pca_label, pcom_label], polydata)
        pca_nodes_4 = get_nodes_of_degree_n(4, [pca_label, pcom_label], polydata)
        pca_nodes_5 = get_nodes_of_degree_n(5, [pca_label, pcom_label], polydata)
        pca_nodes_higher = pca_nodes_4 + pca_nodes_5
        pca_pcom_boundary = find_boundary_points(pca_label, pcom_label, polydata)

        pca_end = []
        pca_pcom_bif, pca_boundary_pcom = [], []
        
        # Case 1: P1 present
        if variant_dict['posterior'][p1_name]:
            assert len(pca_boundary_ba) > 0, 'BA/PCA boundary not found!'
            
            # No Pcom but P1 present
            if not variant_dict['posterior'][pcom_name]:
                logger.debug('\tP1 but no Pcom present.')
                logger.debug(f'\tBA boundary points: {pca_boundary_ba}')
                assert pcom_label not in labels_array
                assert len(pca_nodes_higher) == 0, 'PCA bifurcation present without Pcom?!'
                assert len(pca_boundary_ba) == 1, 'Wrong number of BA/PCA boundaries'
                if len(pca_nodes_3) == 0:
                    assert len(pca_nodes_1) == 2, 'Wrong number of PCA 1-nodes'      
                    pca_nodes_1.remove(pca_boundary_ba[0]['id'])
                    pca_end = get_node_dict_entry(pca_nodes_1[0], 1, pca_label, polydata)
                    logger.debug(f'\tOnly 1 PCA 1-node left; must be PCA end: {pca_end}')
                else:
                    # PCA end is branch point
                    assert len(pca_nodes_3) == 1, 'Wrong number of PCA 3-nodes!'
                    assert len(pca_nodes_1) == 3, 'Wrong number of PCA 1-nodes'
                    pca_end = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)  
                    logger.debug(f'\tPCA end is branch point: {pca_end}')
            
            # Pcom and P1 present
            else:
                logger.debug('\tP1 and Pcom present.')
                logger.debug(f'\tPCA boundary points: {pca_pcom_boundary}')
                assert pcom_label in labels_array
                assert len(pca_pcom_boundary) == 1, 'Wrong number of Pcom/PCA boundaries!'
                # getting the degree of the boundary point
                if pca_pcom_boundary[0] in pca_nodes_3: 
                    pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 3, pca_label, polydata)
                elif pca_pcom_boundary[0] in pca_nodes_higher:
                    pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 4, pca_label, polydata)
                else:
                    pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 2, pca_label, polydata)
                if len(pca_nodes_3) == 1: 
                    if len(pca_nodes_higher) == 0:
                        # Only 1 PCA 3-node present -> Pcom bifurcation
                        pca_pcom_bif = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)
                        logger.debug(f'\tOnly 1 PCA 3-node present; must be pcom bifurcation: {pca_pcom_bif}')
                        assert len(pca_boundary_ba) == 1, 'Wrong number of BA/PCA boundaries'
                    else: 
                        if len(pca_nodes_higher) == 1: # P1 fenestration with higher node (deg = 4) being the pcom bifurcation?
                            logger.warning('ALERT: P1 fenestration with higher node (deg = 4) being the pcom bifurcation?')
                            pca_nodes_high = pca_nodes_3 + pca_nodes_higher
                            pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_high, pca_pcom_boundary[0], pca_label, polydata)
                            logger.debug(f'\tPcom bifurcation ID: {pcom_bif_id}')
                            assert pcom_bif_id in pca_nodes_higher, 'PCA higher node is not Pcom bifurcation!'
                            pca_pcom_bif = get_node_dict_entry(pcom_bif_id, 4, pca_label, polydata)

                            loop = check_for_loop_between_nodes(pca_nodes_higher[0], pca_nodes_3[0], pca_label, polydata)
                            if not loop: # origin of fenestration must start at BA already
                                logger.debug('\tFenestration starting at BA already...?')
                                assert len(ba_pca_boundary) > 1, 'Wrong number of BA/PCA boundaries'
                                path = find_shortest_path(pca_nodes_3[0], ba_pca_boundary[0], polydata, pca_label)['path']
                                if assert_node_on_path(pca_nodes_higher[0], path):
                                    # PCA end is branch point
                                    pca_end = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)
                                    logger.debug(f'\tPCA end is branch point: {pca_end}')
                                else:
                                    logger.warning(f'\tALERT: PCA 3-node not PCA end?! {pca_nodes_3[0]}')
                                    raise ValueError('\tPCA 3-node not PCA end?!')
                            else: # origin of fenestration must be at the P1
                                assert len(ba_pca_boundary) == 1, 'Wrong number of BA/PCA boundaries'
                            
                            variant_dict['fenestration'][f'{p1_name}'] = True
                            
                        else:
                            logger.warning(f'\tALERT: More than 1 PCA higher node (deg >= 4) present?! {pca_nodes_higher}')
                            raise ValueError('More PCA higher nodes present?!')
                        
                elif len(pca_nodes_3) == 0: # fenestration with origin at BA?
                    logger.warning(f'\tALERT: P1 fenestration with origin at BA found...?')
                    if len(pca_nodes_higher) == 1:
                        pca_pcom_bif = get_node_dict_entry(pca_nodes_higher[0], 4, pca_label, polydata)
                        logger.debug(f'\tPcom bifurcation: {pca_pcom_bif}')
                        logger.debug(f'\tBA boundaries: {ba_pca_boundary}')
                        assert len(pca_boundary_ba) == 2, 'Wrong number of BA/PCA boundaries'
                        variant_dict['fenestration'][f'{p1_name}'] = True
                    else:
                        logger.warning(f'ALERT: 0 or more than 1 PCA higher node (deg >= 4) present?! {pca_nodes_higher}')
                        raise ValueError('No PCA 3-nodes found')
                elif len(pca_nodes_3) == 2: # PCA end is branch point and/or P1 fenestration...
                    logger.debug(f'\t2 PCA 3-nodes found: {pca_nodes_3}')
                    pca_nodes_3_copy = copy.deepcopy(pca_nodes_3)
                    if len(pca_nodes_higher) == 0:
                        pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_3, pca_pcom_boundary[0], pca_label, polydata)
                        pca_nodes_3_copy.remove(pcom_bif_id)
                        loop = check_for_loop_between_nodes(pca_nodes_3_copy[0], pcom_bif_id, pca_label, polydata)
                        assert not loop, 'Loop found between PCA 3-nodes. Should not be possible?!'
                        if len(ba_pca_boundary) == 1: # either PCA end is branching point or broken P1 fenestration
                            logger.debug('\tNo loop found between PCA 3-nodes, and only 1 BA boundary -> Either PCA end is branching point or broken P1 fenestration.')
                            path = find_shortest_path(pca_nodes_3_copy[0], ba_pca_boundary[0], polydata, pca_label)['path']
                            if assert_node_on_path(pcom_bif_id, path):
                                # PCA end is branching point
                                pca_end = get_node_dict_entry(pca_nodes_3_copy[0], 3, pca_label, polydata)
                                logger.debug(f'\tPCA end is branch point: {pca_end}')
                            else:
                                # Case broken fenestration! This might go wrong, check those cases manually...
                                logger.warning('ALERT: Broken fenestration?!')
                                variant_dict['fenestration'][f'{p1_name}'] = True
                                for pt in ba_pca_boundary:
                                    if pt in pca_nodes_1:
                                        pca_nodes_1.remove(pt)
                                for pt in pca_pcom_boundary:
                                    if pt in pca_nodes_1:
                                        pca_nodes_1.remove(pt)
                                assert len(pca_nodes_1) == 2, 'Wrong number of PCA 1-nodes!'
                                path1 = find_shortest_path(pca_nodes_1[0], ba_pca_boundary[0], polydata, pca_label)['path']
                                path2 = find_shortest_path(pca_nodes_1[1], ba_pca_boundary[0], polydata, pca_label)['path']
                                loose_end = None
                                if len(path1) > len(path2):
                                    pca_end = get_node_dict_entry(pca_nodes_1[0], 1, pca_label, polydata)
                                    loose_end = pca_nodes_1[1]
                                else:
                                    pca_end = get_node_dict_entry(pca_nodes_1[1], 1, pca_label, polydata)
                                    loose_end = pca_nodes_1[0]
                                logger.debug(f'\tPCA end: {pca_end}')
                                logger.debug(f'\tloose end for connecting: {loose_end}')
                                path_fen = find_shortest_path(loose_end, pca_nodes_3_copy[0], polydata, pca_label)['path']
                                nodes_fen = [p[0] for p in path_fen] + [loose_end]
                                path_p1 = find_shortest_path(ba_pca_boundary[0], pcom_bif_id, polydata, pca_label)['path']
                                nodes_p1 = [p[0] for p in path_p1]
                                points_pca = get_pointIds_for_label(pca_label, polydata)
                                # NOTE: Only connect to P2 segment so far. Might go wrong for some rare cases!
                                points_p2 = [pt for pt in points_pca if pt not in nodes_fen and pt not in nodes_p1]
                                logger.debug(f'\tConnecting loose end to P2 segment: {points_p2}')
                                polydata = connect_loose_end(loose_end, points_p2, pca_label, polydata)
                                pca_nodes_3 = get_nodes_of_degree_n(3, [pca_label, pcom_label], polydata)
                                pca_nodes_4 = get_nodes_of_degree_n(4, [pca_label, pcom_label], polydata)
                                pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_3+pca_nodes_4, pca_pcom_boundary[0], pca_label, polydata)
                                loop = check_for_loop_between_nodes(pca_nodes_3_copy[0], pcom_bif_id, pca_label, polydata)
                                # after connecting fenestration, there must be a loop between the nodes!
                                assert loop, 'No PCA loop present after connecting?!'
                        elif len(ba_pca_boundary) == 2:
                            logger.debug('\tNo loop found between PCA 3-nodes, and 2 BA boundaries.')
                            loop2 = check_for_loop_between_nodes(pca_nodes_3_copy[0], ba_bif_node['id'], [ba_label, pca_label], polydata)
                            if loop2: # P1 fenestration starting at BA
                                variant_dict['fenestration'][f'{p1_name}'] = True
                                for pt in ba_pca_boundary:
                                    if pt in pca_nodes_1:
                                        pca_nodes_1.remove(pt)
                                for pt in pca_pcom_boundary:
                                    if pt in pca_nodes_1:
                                        pca_nodes_1.remove(pt)
                                assert len(pca_nodes_1) == 1, 'Wrong number of PCA 1-nodes!'
                                pca_end = get_node_dict_entry(pca_nodes_1[0], 1, pca_label, polydata)
                                logger.debug(f'\tPCA end: {pca_end}')
                            else:
                                raise ValueError('PCA end is branch point and broken P1 fenestration starting at BA?!')

                    else: # P1 fenestration with higher node (deg = 4) being the pcom bifurcation and PCA end is branch point
                        logger.warning('\tALERT: P1 fenestration found with 2 PCA 3-nodes and PCA higher nodes!')
                        logger.debug(f'\tPCA higher nodes: {pca_nodes_higher}')
                        assert len(pca_nodes_higher) == 1, 'More than 1 PCA higher node present?!'
                        pca_nodes_high = pca_nodes_3 + pca_nodes_higher
                        pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_high, pca_pcom_boundary[0], pca_label, polydata)
                        logger.debug(f'\tPcom bifurcation ID: {pcom_bif_id}')
                        assert pcom_bif_id in pca_nodes_higher, 'PCA higher node is not Pcom bifurcation!'
                        assert len(ba_pca_boundary) == 1, 'Wrong number of BA/PCA boundaries'
                        path1 = find_shortest_path(pca_nodes_3[0], ba_pca_boundary[0], polydata, pca_label)['path']
                        if assert_node_on_path(pcom_bif_id, path1):
                            assert assert_node_on_path(pca_nodes_3[1], path1), 'PCA 3-node not on path!'
                            pca_end = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)
                            loop = check_for_loop_between_nodes(pca_nodes_3[1], pcom_bif_id, pca_label, polydata)
                            logger.debug(f'\tP1 loop node: {pca_nodes_3[1]}')
                            assert loop, 'No PCA loop present?!'
                        else:
                            path2 = find_shortest_path(pca_nodes_3[1], ba_pca_boundary[0], polydata, pca_label)['path']
                            if assert_node_on_path(pcom_bif_id, path2):
                                assert assert_node_on_path(pca_nodes_3[0], path2), 'PCA 3-node not on path!'
                                pca_end = get_node_dict_entry(pca_nodes_3[1], 3, pca_label, polydata)
                                loop = check_for_loop_between_nodes(pca_nodes_3[0], pcom_bif_id, pca_label, polydata)
                                logger.debug(f'\tP1 loop node: {pca_nodes_3[0]}')
                                assert loop, 'No PCA loop present?!'
                            else:
                                raise ValueError('PCA 3-node not PCA end?!')
                        logger.debug(f'\tPCA end is branch point: {pca_end}')
                        variant_dict['fenestration'][f'{p1_name}'] = True
                    
                    pca_pcom_bif = get_node_dict_entry(pcom_bif_id, 3, pca_label, polydata)
                    
                
                else:
                    # 3 or more PCA 3-nodes present
                    logger.debug(f'\t3 or more PCA 3-nodes found: {pca_nodes_3}')
                    logger.warning('\tALERT: P1 fenestration with 3 or more PCA 3-nodes found!')
                    assert len(pca_nodes_3) > 2, 'Too few PCA 3-nodes found!'
                    # Always fenestration?
                    variant_dict['fenestration'][f'{p1_name}'] = True
                    if len(pca_nodes_higher) == 0:
                        # Find PCA 3-node that is closest to the Pcom boundary
                        pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_3, pca_pcom_boundary[0], pca_label, polydata)
                        if len(pca_nodes_3) == 3: # "typical" P1 fenestration
                            logger.debug(f'\t3 PCA 3-nodes are loop nodes...?')
                            for k, nd1 in enumerate(pca_nodes_3[:-1]):
                                for nd2 in pca_nodes_3[k+1:]:
                                    assert check_for_loop_between_nodes(nd1, nd2, pca_label, polydata), 'No loop between PCA 3-nodes!'
                        else:
                            logger.debug(f'\tMore than 3 PCA 3-nodes found: {pca_nodes_3}')
                            assert len(pca_nodes_3) == 4, 'Wrong number of PCA 3-nodes!'
                            # Node farthest from BA boundary must be PCA end
                            longest_path = find_longest_shortest_path(pca_nodes_3, ba_pca_boundary[0], polydata, pca_label)['path']
                            pca_end = get_node_dict_entry(longest_path[0][0], 3, pca_label, polydata)
                            logger.debug(f'\tPCA end is branch point farthest from BA boundary: {pca_end}')
                            next_node = None
                            for edge in longest_path:
                                if edge[1] in pca_nodes_3:
                                    next_node = edge[1]
                                    break
                            loop = check_for_loop_between_nodes(next_node, pca_end[0]['id'], pca_label, polydata)
                            assert not loop, 'Loop found between pca end and fenestration!'
                    else:
                        logger.debug(f'\tPCA higher nodes: {pca_nodes_higher}')
                        assert len(pca_nodes_higher) == 1, 'Too many PCA higher nodes found!'
                        pca_nodes_high = pca_nodes_3 + pca_nodes_higher
                        pcom_bif_id, _ = find_closest_node_to_point(pca_nodes_high, pca_pcom_boundary[0], pca_label, polydata)
                        logger.warning(f'\tALERT: P1 fenestration with 3 or more PCA 3-nodes and 1 PCA higher node found?!')
                        variant_dict['fenestration'][f'{p1_name}'] = True
                        logger.warning(f'\tALERT: Case not implemented yet!')
                        raise ValueError('Too many PCA higher nodes found! Case not implemented yet!')

                    pca_pcom_bif = get_node_dict_entry(pcom_bif_id, 3, pca_label, polydata)
                
                # Get PCA end if not already done
                if len(pca_end) == 0:
                    logger.debug('\tPCA end not found yet...')
                    for pt in pca_pcom_boundary:
                        if pt in pca_nodes_1:
                            pca_nodes_1.remove(pt)
                    for pt in ba_pca_boundary:
                        if pt in pca_nodes_1:
                            pca_nodes_1.remove(pt)
                    if len(pca_nodes_1) == 1:
                        pca_end = get_node_dict_entry(pca_nodes_1[0], 1, pca_label, polydata)
                        logger.debug(f'\tOnly 1 PCA 1-node left; must be PCA end: {pca_end}')
                    else:
                        assert len(pca_nodes_1) == 2, 'Wrong number of R-PCA end points'
                        all_higher_nodes = pca_nodes_3 + pca_nodes_4 + pca_nodes_5
                        assert len(pca_nodes_3) > 0, 'Too few R-PCA higher 3-nodes found!'
                        assert len(all_higher_nodes) > 1, 'Too few R-PCA higher nodes found!'
                        if len(pca_nodes_3) == 1:
                            pca_end = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)

                        else:
                            pca_end_id, _ = find_closest_node_to_point(pca_nodes_3, pca_nodes_1[0], pca_label, polydata)
                            pca_end_id2, _ = find_closest_node_to_point(pca_nodes_3, pca_nodes_1[1], pca_label, polydata)
                            assert pca_end_id == pca_end_id2, 'PCA end not uniquely found!'
                            pca_end = get_node_dict_entry(pca_end_id, 3, pca_label, polydata)
                        
                        logger.debug(f'\tPCA is branch point: {pca_end}')

                # sanity check
                assert len(pca_pcom_bif) == 1, 'R-PCA bifurcation not found!'

        # Case 2: No P1, but Pcom present
        else:
            logger.debug('\tNo P1 but Pcom present.')
            assert pcom_label in labels_array
            assert variant_dict['posterior'][pcom_name], 'R-Pcom not present!'
            assert len(pca_pcom_boundary) == 1, 'Wrong number of Pcom/R-PCA boundaries!'
            # either branching PCA or broken P1! 
            if len(pca_nodes_3) > 0:
                if len(pca_nodes_3) == 1:
                    if len(pca_nodes_1) == 2:
                        # Case: Broken P1
                        if pca_pcom_boundary[0] == pca_nodes_3[0]:
                            pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 3, pca_label, polydata)
                            path = find_shortest_path(pca_nodes_1[0], pca_nodes_1[1], polydata, [pca_label])['path']
                            assert assert_node_on_path(pca_nodes_3[0], path)
                            logger.warning('\tALERT: P1 broken but present?!')
                            pcom_bif_id = pca_nodes_3[0]
                            pca_pcom_bif = get_node_dict_entry(pcom_bif_id, 3, pca_label, polydata)
                            logger.debug(f'\tPcom bifurcation: {pca_pcom_bif}')
                            path_ids = [p[0] for p in path] + [path[-1][1]]
                            # Split path_ids at pcom_bif_id
                            pcom_bif_index = path_ids.index(pcom_bif_id)
                            num_points = len(path_ids)
                            # Check if pcom bifurcation is in the first half of the path
                            if pcom_bif_index + 1 < num_points / 2:
                                # If bifurcation is in first half, first node is BA boundary
                                ba_boundary_id = pca_nodes_1[0]
                                pca_end_id = pca_nodes_1[1]
                            elif pcom_bif_index > num_points / 2:
                                # If bifurcation is in second half, second node is BA boundary
                                ba_boundary_id = pca_nodes_1[1]
                                pca_end_id = pca_nodes_1[0]
                            else:
                                logger.warning('\tALERT: P1 and P2 of same length?!')
                                raise ValueError('P1 and P2 of same length?!')

                            # Create the node dictionary entries
                            pca_boundary_ba = get_node_dict_entry(ba_boundary_id, 1, pca_label, polydata)
                            pca_end = get_node_dict_entry(pca_end_id, 1, pca_label, polydata)

                            logger.debug(f'\tBA boundary: {pca_boundary_ba}')
                            logger.debug(f'\tPCA end: {pca_end}')
                        else:
                            logger.warning('\tALERT: Pcom bifurcation entirely within Pcom?!')
                            raise ValueError('Pcom bifurcation entirely within Pcom?!')
                    
                    elif len(pca_nodes_1) == 3:
                        # pcom/pca boundary must be a 1-node
                        assert pca_pcom_boundary[0] in pca_nodes_1, 'Pcom/PCA boundary not a 1-node?!'
                        pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 2, pca_label, polydata)
                        path = find_shortest_path(pca_nodes_3[0], pca_pcom_boundary[0], polydata, [pca_label])['path']
                        if len(path) < 6: # NOTE: This is a hard-coded value! Might need to be adjusted! Set value small enough!
                            logger.warning('\tALERT: P1 broken but present?!')
                            pcom_bif_id = pca_nodes_3[0]
                            pca_pcom_bif = get_node_dict_entry(pcom_bif_id, 3, pca_label, polydata)
                            logger.debug(f'\tPcom bifurcation: {pca_pcom_bif}')
                            pca_nodes_1.remove(pca_pcom_boundary[0])
                            path = find_shortest_path(pca_nodes_1[0], pca_nodes_1[1], polydata, [pca_label])['path']
                            assert assert_node_on_path(pca_nodes_3[0], path)
                            path_ids = [p[0] for p in path] + [path[-1][1]]
                            # Split path_ids at pcom_bif_id
                            pcom_bif_index = path_ids.index(pcom_bif_id)
                            num_points = len(path_ids)
                            # Check if pcom bifurcation is in the first half of the path
                            if pcom_bif_index + 1 < num_points / 2:
                                # If bifurcation is in first half, first node is BA boundary
                                ba_boundary_id = pca_nodes_1[0]
                                pca_end_id = pca_nodes_1[1]
                            elif pcom_bif_index > num_points / 2:
                                # If bifurcation is in second half, second node is BA boundary
                                ba_boundary_id = pca_nodes_1[1]
                                pca_end_id = pca_nodes_1[0]
                            else:
                                logger.warning('\tALERT: P1 and P2 of same length?!')
                                raise ValueError('P1 and P2 of same length?!')
                            
                            # Create the node dictionary entries
                            pca_boundary_ba = get_node_dict_entry(ba_boundary_id, 1, pca_label, polydata)
                            pca_end = get_node_dict_entry(pca_end_id, 1, pca_label, polydata)

                            logger.debug(f'\tBA boundary: {pca_boundary_ba}')
                            logger.debug(f'\tPCA end: {pca_end}')
                            
                        elif len(path) > 15: # NOTE: This is a hard-coded value! Might need to be adjusted! Set value large enough!
                            logger.debug(f'\tPCA end is branch point: {pca_end}')
                            pca_end = get_node_dict_entry(pca_nodes_3[0], 3, pca_label, polydata)
                    
                    else:
                        logger.warning(f'\tALERT: More than 3 or less than 2 PCA 1-nodes present?! {pca_nodes_1}')
                        raise ValueError('More than 3 or less than 2 PCA 1-nodes present?!')
                            
                else:
                    logger.warning(f'\tALERT: More than 1 PCA 3-node present?! {pca_nodes_3}')
                    raise ValueError('More than 1 PCA 3-node present?!')

            else:
                assert len(pca_nodes_1) == 2, 'Wrong number of R-PCA 1-nodes'
                pca_boundary_pcom = get_node_dict_entry(pca_pcom_boundary[0], 2, pca_label, polydata)
                pca_nodes_1.remove(pca_pcom_boundary[0])
                pca_end = get_node_dict_entry(pca_nodes_1[0], 1, pca_label, polydata)
                logger.debug(f'\tOnly 1 PCA 1-node left; must be PCA end: {pca_end}')
        
        
        if len(pca_boundary_ba) > 0:
            nodes_dict[str(pca_label)]['BA boundary'] = pca_boundary_ba
        if len(pca_pcom_bif) > 0:
            nodes_dict[str(pca_label)]['Pcom bifurcation'] = pca_pcom_bif
        if len(pca_boundary_pcom) > 0:
            nodes_dict[str(pca_label)]['Pcom boundary'] = pca_boundary_pcom
        if len(pca_end) > 0:
            nodes_dict[str(pca_label)]['PCA end'] = pca_end 
        
        logger.debug(f'\tPCA nodes extracted: {nodes_dict[str(pca_label)]}')

    return nodes_dict, variant_dict, polydata

def get_ica_mca_nodes(polydata, variant_dict):
    """
    Function to identify ICA & MCA nodes and write to node dict. It's a rule-based approach. 
    Might run into sum errors for messed-up cases --> TODO: Improve this in the future.

    Args:
    polydata: vtkPolyData, polydata object
    variant_dict: dict, dictionary with cow variant information

    Returns:
    nodes_dict: dict, dictionary with ICA & MCA node information
    """
    rica_label, lica_label = 4, 6
    rmca_label, lmca_label = 5, 7
    rpcom_label, lpcom_label = 8, 9
    raca_label, laca_label = 11, 12

    nodes_dict = {}

    labels_array = get_label_array(polydata)

    logger.info('2) Extracting ICA & MCA nodes...')
    
    for ica_label, mca_label, pcom_label, aca_label, pcom_name, a1_name in zip([rica_label, lica_label], [rmca_label, lmca_label], [rpcom_label, lpcom_label], [raca_label, laca_label], ['R-Pcom', 'L-Pcom'], ['R-A1', 'L-A1']):
        logger.info(f'...of label {ica_label}, {mca_label}...')
        ica_start, ica_bif_node, pcom_bif_node = [], [], []
        ica_boundary_aca, ica_boundary_mca, ica_boundary_pcom = [], [], []
        mca_boundary_ica, mca_end = [], []


        if ica_label in labels_array:
            nodes_dict[str(ica_label)] = {}
            ica_nodes_1 = get_nodes_of_degree_n(1, ica_label, polydata)
            ica_nodes_3 = get_nodes_of_degree_n(3, [ica_label, pcom_label], polydata)
            ica_nodes_4 = get_nodes_of_degree_n(4, [ica_label, pcom_label], polydata)

            assert len(ica_nodes_4) == 0, "ICA node of degree 4 present!"

            ica_aca_boundary = find_boundary_points(ica_label, aca_label, polydata)

            # MCA present
            if mca_label in labels_array:
                nodes_dict[str(mca_label)] = {}
                mca_nodes_1 = get_nodes_of_degree_n(1, mca_label, polydata)
                mca_nodes_3 =  get_nodes_of_degree_n(3, mca_label, polydata)
                mca_nodes_4 = get_nodes_of_degree_n(4, mca_label, polydata)
                
                ica_mca_boundary = find_boundary_points(ica_label, mca_label, polydata)
                logger.debug(f'\tICA/MCA boundary points: {ica_mca_boundary}')
                assert len(ica_mca_boundary) == 1, 'Wrong number of ICA/MCA boundaries!'
                ica_boundary_mca = get_node_dict_entry(ica_mca_boundary[0], 2, ica_label, polydata)
                mca_boundary_ica = get_node_dict_entry(ica_mca_boundary[0], 2, mca_label, polydata)

                mca_nodes_high = mca_nodes_3 + mca_nodes_4
                
                # MCA end...
                if len(mca_nodes_high) == 0: # ... is a 1-node
                    assert len(mca_nodes_1) == 2, 'Wrong number of MCA 1-nodes!'
                    mca_nodes_1.remove(ica_mca_boundary[0])
                    mca_end = get_node_dict_entry(mca_nodes_1[0], 1, mca_label, polydata)
                    logger.debug(f'\tMCA end of degree 1: {mca_end}')
                else: # ... is a 3-node (closest to ICA boundary)
                    mca_end_id, _ = find_closest_node_to_point(mca_nodes_high, ica_mca_boundary[0], mca_label, polydata)
                    mca_end = get_node_dict_entry(mca_end_id, 3, mca_label, polydata)
                    logger.debug(f'\tMCA end of degree 3 is closest 3-node to ICA boundary: {mca_end}')
                
                # Pcom present
                if pcom_label in labels_array:
                    logger.debug('\tICA, MCA and Pcom present.')
                    assert variant_dict['posterior'][pcom_name]
                    ica_pcom_boundary = find_boundary_points(ica_label, pcom_label, polydata)
                    logger.debug(f'\tICA/Pcom boundary points: {ica_pcom_boundary}')
                    assert len(ica_pcom_boundary) == 1, 'Wrong number of ICA/Pcom boundaries!'
                    ica_boundary_pcom = get_node_dict_entry(ica_pcom_boundary[0], 2, ica_label, polydata)

                    # MCA, A1 and Pcom present
                    if len(ica_aca_boundary) > 0:
                        logger.debug(f'\tA1 present with boundary point {ica_aca_boundary}')
                        assert variant_dict['anterior'][a1_name]
                        assert len(ica_aca_boundary) == 1, 'Wrong number of ICA/ACA boundaries!'
                        ica_boundary_aca = get_node_dict_entry(ica_aca_boundary[0], 2, ica_label, polydata)
                        logger.debug(f'\tICA 3-nodes: {ica_nodes_3}')
                        if len(ica_nodes_3) == 2:
                            # Pcom bifurcation is closest 3-node to Pcom boundary
                            pcom_bif_id, _ = find_closest_node_to_point(ica_nodes_3, ica_pcom_boundary[0], ica_label, polydata)
                            pcom_bif_node = get_node_dict_entry(pcom_bif_id, 3, ica_label, polydata)
                            logger.debug(f'\tPcom bifurcation: {pcom_bif_node}')
                            ica_nodes_3.remove(pcom_bif_id)
                            ica_bif_node = get_node_dict_entry(ica_nodes_3[0], 3, ica_label, polydata)
                            logger.debug(f'\tICA bifurcation: {ica_bif_node}')
                            assert len(ica_nodes_1) == 4, 'Wrong number of ICA 1-nodes!'
                            ica_nodes_1.remove(ica_aca_boundary[0])
                            ica_nodes_1.remove(ica_mca_boundary[0])
                            ica_nodes_1.remove(ica_pcom_boundary[0])
                            ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                            logger.debug(f'\tICA start: {ica_start}')
                        elif len(ica_nodes_3) == 1:
                            logger.warning(f'\tALERT: Only 1 ICA 3-node present with A1, MCA and Pcom! Missing Pcom bif?')
                            path1 = find_shortest_path(ica_nodes_3[0], ica_aca_boundary[0], polydata, [ica_label])['path']
                            path2 = find_shortest_path(ica_nodes_3[0], ica_mca_boundary[0], polydata, [ica_label])['path']
                            assert len(path1) < 9 and len(path2) < 11, 'ICA bif too far away from boundaries?!' 
                            ica_bif_node = get_node_dict_entry(ica_nodes_3[0], 3, ica_label, polydata)
                            logger.debug(f'\tICA bifurcation: {ica_bif_node}')
                        else:
                            logger.warning(f'\tALERT: Too many ICA 3-nodes present with A1, MCA and Pcom!?! {ica_nodes_3}')
                            raise ValueError('Too many ICA 3-nodes present with A1, MCA and Pcom?!')
                    # No A1, MCA and Pcom present
                    else:
                        logger.debug('\tNo A1 present.')
                        logger.debug(f'\tICA 3-nodes: {ica_nodes_3}')
                        assert variant_dict['anterior'][a1_name] == False
                        assert len(ica_nodes_3) == 1, 'Wrong number of ICA 3-nodes!'
                        pcom_bif_node = get_node_dict_entry(ica_nodes_3[0], 3, ica_label, polydata)
                        logger.debug(f'\tPcom bifurcation: {pcom_bif_node}')
                        assert len(ica_nodes_1) == 3, 'Wrong number of ICA 1-nodes!'
                        ica_nodes_1.remove(ica_mca_boundary[0])
                        ica_nodes_1.remove(ica_pcom_boundary[0])
                        ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                        logger.debug(f'\tICA start: {ica_start}')
                
                else: # ICA and MCA present, but no Pcom
                    logger.debug('\tICA, MCA but no Pcom present.')
                    if len(ica_aca_boundary) > 0: # A1 present
                        logger.info(f'A1 present with boundary point {ica_aca_boundary}')
                        assert variant_dict['anterior'][a1_name]
                        assert len(ica_aca_boundary) == 1, 'Wrong number of ICA/ACA boundaries!'
                        ica_boundary_aca = get_node_dict_entry(ica_aca_boundary[0], 2, ica_label, polydata)
                        logger.debug(f'\tICA 3-nodes: {ica_nodes_3}')
                        assert len(ica_nodes_3) == 1, 'Wrong number of ICA 3-nodes!'
                        ica_bif_node = get_node_dict_entry(ica_nodes_3[0], 3, ica_label, polydata)
                        logger.debug(f'\tICA bifurcation: {ica_bif_node}')
                        assert len(ica_nodes_1) == 3, 'Wrong number of ICA 1-nodes!'
                        ica_nodes_1.remove(ica_aca_boundary[0])
                        ica_nodes_1.remove(ica_mca_boundary[0])
                        ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                        logger.debug(f'\tICA start: {ica_start}')
                    
                    else: # No A1, MCA and no Pcom
                        logger.debug('\tNo A1 present.')
                        assert variant_dict['anterior'][a1_name] == False
                        assert len(ica_nodes_3) == 0, 'Wrong number of ICA 3-nodes!'
                        assert len(ica_nodes_1) == 2, 'Wrong number of ICA 1-nodes!'
                        ica_nodes_1.remove(ica_mca_boundary[0])
                        ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                        logger.debug(f'\tICA start: {ica_start}')
       
            else: # No MCA
                logger.debug('\tICA but no MCA present.')
                # In that case we must have an A1!
                logger.debug(f'\tA1 present with boundary point {ica_aca_boundary}')
                assert len(ica_aca_boundary) == 1, 'Wrong number of ICA/ACA boundaries!'
                ica_boundary_aca = get_node_dict_entry(ica_aca_boundary[0], 2, ica_label, polydata)
                assert variant_dict['anterior'][a1_name]
                # No MCA, but A1 and Pcom present
                if pcom_label in labels_array:
                    logger.debug('\tPcom present')
                    assert variant_dict['posterior'][pcom_name]
                    ica_pcom_boundary = find_boundary_points(ica_label, pcom_label, polydata)
                    logger.debug(f'\tICA/Pcom boundary points: {ica_pcom_boundary}')
                    assert len(ica_pcom_boundary) == 1, 'Wrong number of ICA/Pcom boundaries!'
                    ica_boundary_pcom = get_node_dict_entry(ica_pcom_boundary[0], 2, ica_label, polydata)
                    logger.debug(f'\tICA 3-nodes: {ica_nodes_3}')
                    assert len(ica_nodes_3) == 1, 'Wrong number of ICA 3-nodes!'
                    pcom_bif_node = get_node_dict_entry(ica_nodes_3[0], 3, ica_label, polydata)
                    logger.debug(f'\tPcom bifurcation: {pcom_bif_node}')
                    assert len(ica_nodes_1) == 3, 'Wrong number of ICA 1-nodes!'
                    ica_nodes_1.remove(ica_pcom_boundary[0])
                    ica_nodes_1.remove(ica_aca_boundary[0])
                    ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                    logger.debug(f'\tICA start: {ica_start}')
                
                # No Pcom, No MCA, but A1 present
                else:
                    logger.debug('\tNo Pcom present')
                    assert variant_dict['posterior'][pcom_name] == False
                    assert len(ica_nodes_3) == 0, 'Wrong number of ICA 3-nodes!'
                    assert len(ica_nodes_1) == 2, 'Wrong number of ICA 1-nodes!'
                    ica_nodes_1.remove(ica_aca_boundary[0])
                    ica_start = get_node_dict_entry(ica_nodes_1[0], 1, ica_label, polydata)
                    logger.debug(f'\tICA start: {ica_start}')

        else: # No ICA
            logger.debug('\tNo ICA present.')
            assert variant_dict['posterior'][pcom_name] == False
            assert variant_dict['anterior'][a1_name] == False
            assert mca_label not in labels_array, 'MCA present without ICA?!'
            assert pcom_label not in labels_array, 'Pcom present without ICA?!'


        if len(ica_start) > 0:
            nodes_dict[str(ica_label)]['ICA start'] = ica_start
        if len(pcom_bif_node) > 0:
            nodes_dict[str(ica_label)]['Pcom bifurcation'] = pcom_bif_node
        if len(ica_boundary_pcom) > 0:
            nodes_dict[str(ica_label)]['Pcom boundary'] = ica_boundary_pcom
        if len(ica_bif_node) > 0:
            nodes_dict[str(ica_label)]['ICA bifurcation'] = ica_bif_node
        if len(ica_boundary_aca) > 0:
            nodes_dict[str(ica_label)]['ACA boundary'] = ica_boundary_aca
        if len(ica_boundary_mca) > 0:
            nodes_dict[str(ica_label)]['MCA boundary'] = ica_boundary_mca
        if len(mca_boundary_ica) > 0:
            nodes_dict[str(mca_label)]['ICA boundary'] = mca_boundary_ica
        if len(mca_end) > 0:
            nodes_dict[str(mca_label)]['MCA end'] = mca_end
        
        if ica_label in nodes_dict:
            logger.debug(f'\tICA nodes dict: {nodes_dict[str(ica_label)]}')
        if mca_label in nodes_dict:
            logger.debug(f'\tMCA nodes dict: {nodes_dict[str(mca_label)]}')

    return nodes_dict

def get_pcom_nodes(polydata, variant_dict):
    """
    Function to identify Pcom nodes and write to node dict. Rule-based approach.

    Args:
    polydata: vtkPolyData, polydata object
    variant_dict: dict, dictionary with cow variant information

    Returns:
    nodes_dict: dict, dictionary with Pcom node information
    """
    rpcom_label, lpcom_label = 8, 9
    rpca_label, lpca_label = 2, 3
    rica_label, lica_label = 4, 6

    nodes_dict = {}

    labels_array = get_label_array(polydata)

    logger.info('3) Extracting Pcom nodes...')

    for pcom_label, pca_label, ica_label, pcom_name in zip([rpcom_label, lpcom_label], [rpca_label, lpca_label], [rica_label, lica_label], ['R-Pcom', 'L-Pcom']):
        pcom_boundary_pca, pcom_boundary_ica = [], []

        if pcom_label in labels_array:
            logger.info(f'...of label {pcom_label}...')
            nodes_dict[str(pcom_label)] = {}

            assert variant_dict['posterior'][pcom_name]
            pcom_nodes_1 = get_nodes_of_degree_n(1, pcom_label, polydata)
            pcom_nodes_3 = get_nodes_of_degree_n(3, pcom_label, polydata)
            logger.debug(f'\tPcom 1-nodes: {pcom_nodes_1}')
            assert len(pcom_nodes_1) == 2, 'Wrong number of Pcom 1-nodes!'
            assert len(pcom_nodes_3) == 0, 'Wrong number of Pcom 3-nodes!'

            pcom_pca_boundary = find_boundary_points(pcom_label, pca_label, polydata)
            pcom_ica_boundary = find_boundary_points(pcom_label, ica_label, polydata)
            logger.debug(f'\tPcom/PCA boundary points: {pcom_pca_boundary}')
            logger.debug(f'\tPcom/ICA boundary points: {pcom_ica_boundary}')

            assert len(pcom_pca_boundary) == 1, 'Wrong number of Pcom/PCA boundaries!'
            assert len(pcom_ica_boundary) == 1, 'Wrong number of Pcom/ICA boundaries!'

            pcom_boundary_pca = get_node_dict_entry(pcom_pca_boundary[0], 2, pcom_label, polydata)
            pcom_boundary_ica = get_node_dict_entry(pcom_ica_boundary[0], 2, pcom_label, polydata)

        if len(pcom_boundary_ica) > 0:
            nodes_dict[str(pcom_label)]['ICA boundary'] = pcom_boundary_ica
        if len(pcom_boundary_pca) > 0:
            nodes_dict[str(pcom_label)]['PCA boundary'] = pcom_boundary_pca
            logger.debug(f'\tPcom nodes dict: {nodes_dict[str(pcom_label)]}')

    return nodes_dict

def get_aca_acom_nodes(polydata, variant_dict):
    """
    Function to identify ACA & Acom nodes (and 3rd-A2) and write to node dict. 
    Rule-based approach. Especially hard to cover all possible A1 fenestration cases.
    Might run into sum errors for messed-up cases --> TODO: Improve this in the future.

    Args:
    polydata: vtkPolyData, polydata object
    variant_dict: dict, dictionary with cow variant information

    Returns:
    nodes_dict: dict, dictionary with ACA & Acom node information
    variant_dict: dict, dictionary with cow variant information (updated to contain fenestration info)
    """
    raca_label, laca_label = 11, 12
    acom_label, a2_label = 10, 15
    rica_label, lica_label = 4, 6

    nodes_dict = {}
    

    labels_array = get_label_array(polydata)
    if acom_label in labels_array:
        nodes_dict[str(acom_label)] = {}

    logger.info('4) Extracting ACA nodes...')

    for aca_label, ica_label, a1_name, aca_name in zip([raca_label, laca_label], [rica_label, lica_label], ['R-A1', 'L-A1'], ['R-ACA', 'L-ACA']):
        logger.info(f'...of label {aca_label}...')
        aca_boundary_ica = []
        acom_bif_node = []
        aca_boundary_acom = []
        acom_boundary_aca = []
        aca_end = []
        acom_boundary_a2 = []
        a2_bifurcation = []
        a2_boundary_acom = []
        a2_end = []

        aca_nodes_1 = get_nodes_of_degree_n(1, aca_label, polydata)
        aca_nodes_3 = get_nodes_of_degree_n(3, aca_label, polydata)
        aca_nodes_4 = get_nodes_of_degree_n(4, aca_label, polydata)

        aca_acom_nodes_3 = get_nodes_of_degree_n(3, [aca_label, acom_label], polydata)
        aca_acom_nodes_4 = get_nodes_of_degree_n(4, [aca_label, acom_label], polydata)

        aca_acom_nodes_high = aca_acom_nodes_3 + aca_acom_nodes_4
        if 15 in labels_array:
            acom_a2_boundary = find_boundary_points(acom_label, a2_label, polydata)
            a2_start = find_acom_bif_for_3rd_a2(acom_a2_boundary[0], polydata)
            if a2_start in aca_acom_nodes_3:
                # Acom bifurcation and A2 origin do not coincide!
                deg = get_point_degree(a2_start, get_edge_list(polydata)[0])
                assert deg == 3
                aca_acom_nodes_high.remove(a2_start)
        
        # A1 present
        if variant_dict['anterior'][a1_name]:
            aca_ica_boundary = find_boundary_points(aca_label, ica_label, polydata)
            logger.debug(f'\tA1 present with boundary point {aca_ica_boundary}')
            assert len(aca_ica_boundary) == 1, 'Wrong number of ACA/ICA boundaries!'
            aca_boundary_ica = get_node_dict_entry(aca_ica_boundary[0], 2, aca_label, polydata)

            # Acom and A1 present
            if variant_dict['anterior']['Acom']:
                assert acom_label in labels_array
                aca_acom_boundary = find_boundary_points(aca_label, acom_label, polydata)
                logger.debug(f'\tAcom present with boundary point {aca_acom_boundary}')
                assert len(aca_acom_boundary) > 0, 'No ACA/Acom boundary found!'

                if len(aca_acom_boundary) == 1: # typical case
                    assert len(aca_acom_nodes_high) > 0, 'No ACA/Acom 3-nodes found!'
                    acom_bif_id, _ = find_closest_node_to_point(aca_acom_nodes_high, aca_acom_boundary[0], aca_label, polydata)
                    logger.debug(f'\tClosest ACA/Acom 3-node to ACA/Acom boundary is Acom bifurcation: {acom_bif_id}')
                    
                    # If path length too longth between acom bif and acom boundary -> broken fenestration!
                    path_acom_bif_boundary = find_shortest_path(acom_bif_id, aca_acom_boundary[0], polydata, aca_label)
                    if path_acom_bif_boundary['length'] > 9: # NOTE: This is a hard-coded value! Might need to be adjusted! Set value big enoug!
                        logger.warning(f'\tALERT: Path length between Acom bifurcation and Acom boundary too long: {path_acom_bif_boundary["length"]}. Broken A1 fenestration?!')
                        # connect boundary point to A2
                        path_a1 = find_shortest_path(aca_ica_boundary[0], acom_bif_id, polydata, [aca_label])['path']
                        nodes_a1 = [p[0] for p in path_a1] 
                        nodes_acom_bif_boundary = [p[0] for p in path_acom_bif_boundary['path']] + [path_acom_bif_boundary['path'][-1][1]]
                        points_aca = get_pointIds_for_label(aca_label, polydata)
                        # NOTE: Only connect to A2 segment -> might not work for some cases!
                        points_a2 = [pt for pt in points_aca if pt not in nodes_acom_bif_boundary and pt not in nodes_a1]
                        logger.debug(f'\tConnecting broken A1 fenestration to points A2: {points_a2}')
                        polydata = connect_loose_end(aca_acom_boundary[0], points_a2, aca_label, polydata)
                        loop = check_for_loop_between_nodes(aca_acom_boundary[0], acom_bif_id, aca_label, polydata)
                        # if connecting fenestration worked correctly, there should be a loop now!
                        assert loop, 'Still no loop after connecting A1 fenestration!'
                        variant_dict['fenestration'][f'{a1_name}'] = True
                        # Acom bif Id is now the boundary point
                        acom_bif_id = aca_acom_boundary[0]
                        logger.debug(f'\tNew Acom bifurcation: {acom_bif_id}')

                    if acom_bif_id in aca_acom_boundary:
                        aca_boundary_acom = get_node_dict_entry(aca_acom_boundary[0], 3, aca_label, polydata)
                        acom_bif_node = aca_boundary_acom
                        acom_boundary_aca = get_node_dict_entry(aca_acom_boundary[0], 3, acom_label, polydata)
                    else:
                        aca_boundary_acom = get_node_dict_entry(aca_acom_boundary[0], 2, aca_label, polydata)
                        acom_bif_node = get_node_dict_entry(acom_bif_id, 3, aca_label, polydata)
                        acom_boundary_aca = get_node_dict_entry(aca_acom_boundary[0], 2, acom_label, polydata)

                else: # more than one ACA/Acom boundary -> Acom fenestration?
                    logger.warning(f'\tALERT: More than 1 ACA/Acom boundary found: {aca_acom_boundary}. Acom fenestration?!')
                    variant_dict['fenestration']['Acom'] = True
                    # after relabeling!
                    if len(aca_acom_boundary) == 2:
                        # Acom fenestration
                        assert len(aca_acom_nodes_4) == 0, 'ACA/Acom 4-node present?!'
                        assert len(aca_acom_nodes_high) >= 2, 'Too few ACA/Acom 3-nodes!'
                        for pt in aca_acom_boundary:
                            if pt in aca_acom_nodes_high: # boundary is bifurcation
                                aca_boundary_acom += get_node_dict_entry(pt, 3, aca_label, polydata)
                                acom_boundary_aca += get_node_dict_entry(pt, 3, acom_label, polydata)
                                acom_bif_node += aca_boundary_acom
                                logger.debug(f'\tACA/Acom boundary point is Acom bifurcation: {acom_bif_node}')
                            elif pt in aca_nodes_1:
                                aca_boundary_acom += get_node_dict_entry(pt, 2, aca_label, polydata)
                                acom_boundary_aca += get_node_dict_entry(pt, 2, acom_label, polydata)
                                # closest 3-node is Acom bifurcation
                                acom_bif_id, _ = find_closest_node_to_point(aca_nodes_3, pt, aca_label, polydata)
                                acom_bif_node += get_node_dict_entry(acom_bif_id, 3, aca_label, polydata)
                                logger.debug(f'\tClosest 3-node is Acom bifurcation: {acom_bif_node}')
                            else:
                                raise ValueError('Acom boundary point not found!')
                    else:
                        raise ValueError('Wrong number of ACA/Acom boundaries! Not implemented yet!')

                # Check for fenestration
                loop = check_for_loop_between_nodes(aca_ica_boundary[0], acom_bif_node[0]['id'], aca_label, polydata)
                if loop: 
                    logger.warning(f'\tALERT: A1 fenestration found!')
                    variant_dict['fenestration'][f'{a1_name}'] = True
                
                # Get 3rd-A2
                if a2_label in labels_array and aca_label == 12:
                    assert variant_dict['anterior']['3rd-A2']
                    acom_a2_boundary = find_boundary_points(acom_label, a2_label, polydata)
                    logger.debug(f'\t3rd-A2 present with boundary point {acom_a2_boundary}')
                    a2_nodes_1 = get_nodes_of_degree_n(1, a2_label, polydata)
                    a2_nodes_3 = get_nodes_of_degree_n(3, a2_label, polydata)
                    assert len(a2_nodes_3) == 0, 'Wrong number of A2 3-nodes!'
                    assert len(a2_nodes_1) == 2, 'Wrong number of A2 1-nodes!'
                    if len(acom_a2_boundary) == 1:   
                        boundary_id = acom_a2_boundary[0]                 
                    else:
                        logger.warning(f'\tALERT: Wrong number of 3rd-A2/Acom boundary points: {acom_a2_boundary}.')
                        raise ValueError('3rd-A2 not touching anything?!')

                    acom_boundary_a2 = get_node_dict_entry(boundary_id, 2, acom_label, polydata)
                    a2_boundary_acom = get_node_dict_entry(boundary_id, 2, a2_label, polydata)
                    a2_nodes_1.remove(boundary_id)
                    assert len(a2_nodes_1) == 1, 'Wrong number of remaining A2 1-nodes!'
                    a2_end = get_node_dict_entry(a2_nodes_1[0], 1, a2_label, polydata)
                    logger.debug(f'\t3rd-A2 end: {a2_end}')
                    a2_bif_id = find_acom_bif_for_3rd_a2(acom_a2_boundary[0], polydata)
                    a2_bifurcation = get_node_dict_entry(a2_bif_id, 3, acom_label, polydata)
                    logger.debug(f'\t3rd-A2 bifurcation: {a2_bifurcation}')
            
            else: # No Acom but A1
                logger.debug('\tNo Acom present.')
                assert a2_label not in labels_array
                assert acom_label not in labels_array
                assert len(aca_nodes_4) == 0, 'ACA node of degree 4 present?!'    
        
        else: # No A1
            logger.debug('\tNo A1 present.')
            aca_acom_boundary = find_boundary_points(aca_label, acom_label, polydata)
            logger.debug(f'\tACA/Acom boundary points: {aca_acom_boundary}')
            assert a2_label not in labels_array # never seen 3rd-A2 without A1
            assert variant_dict['anterior']['Acom'] == True and acom_label in labels_array
            assert len(aca_acom_boundary) == 1, 'Wrong number of ACA/Acom boundaries!'
            aca_boundary_acom = get_node_dict_entry(aca_acom_boundary[0], 2, aca_label, polydata)
            acom_boundary_aca = get_node_dict_entry(aca_acom_boundary[0], 2, acom_label, polydata)
            try:
                aca_nodes_1.remove(aca_acom_boundary[0])
            except ValueError:
                logger.debug('\tACA/Acom boundary not in ACA 1-nodes. Might be a 3-node then.')
            # Either ACA end is branching point or broken but present A1
            if len(aca_nodes_1) == 2:
                if len(aca_acom_nodes_3) == 1:
                    acom_bif_id, min_dist = find_closest_node_to_point(aca_acom_nodes_3, aca_acom_boundary[0], aca_label, polydata)
                    if min_dist < 9: # NOTE: This is a hard-coded value! Might need to be adjusted! Set value small enough!
                        path = find_shortest_path(aca_nodes_1[0], aca_nodes_1[1], polydata, [aca_label])['path']
                        if assert_node_on_path(acom_bif_id, path):
                            logger.warning('\tALERT: A1 broken but present?!')
                            acom_bif_node = get_node_dict_entry(acom_bif_id, 3, aca_label, polydata)
                            logger.debug(f'\tClosest ACA/Acom 3-node to ACA/Acom boundary is Acom bifurcation: {acom_bif_node}')
                            # NOTE: If A1 is present but broken, we still label it as missing A1 (functionally, there is still no blood supply)
                            # variant_dict['anterior'][f'{a1_name}'] = True
                            z_values = [polydata.GetPoints().GetPoint(aca_nodes_1[0])[2], polydata.GetPoints().GetPoint(aca_nodes_1[1])[2]]
                            argmax_zvalue = np.argmax(z_values)
                            aca_end = get_node_dict_entry(aca_nodes_1[argmax_zvalue], 1, aca_label, polydata)
                            logger.debug(f'\tACA end is 1-node with max z-value: {aca_end}')
                            aca_boundary_ica = get_node_dict_entry(aca_nodes_1[(argmax_zvalue+1)%2], 1, aca_label, polydata)
                            logger.debug(f'\tICA boundary is remaining 1-node: {aca_boundary_ica}') 
            # ACA end is branching point and broken but present A1
            elif len(aca_nodes_1) == 3:
                logger.debug('\tACA end is branching point and broken but present A1.')
                assert len(aca_nodes_3) == 2, 'Wrong number of ACA 3-nodes!'
                acom_bif_id, min_dist = find_closest_node_to_point(aca_nodes_3, aca_acom_boundary[0], aca_label, polydata)
                aca_nodes_3.remove(acom_bif_id)
                if min_dist < 9: # NOTE: This is a hard-coded value! Might need to be adjusted! Set value small enough!
                    aca_end = get_node_dict_entry(aca_nodes_3[0], 3, aca_label, polydata)
                    acom_bif_node = get_node_dict_entry(acom_bif_id, 3, aca_label, polydata) 
                    path0 = find_shortest_path(aca_nodes_1[0], acom_bif_id, polydata, [aca_label])['path']
                    if not assert_node_on_path(aca_nodes_3[0], path0):
                        aca_boundary_ica = get_node_dict_entry(aca_nodes_1[0], 1, aca_label, polydata)
                    else:
                        path1 = find_shortest_path(aca_nodes_1[1], acom_bif_id, polydata, [aca_label])['path']
                        if not assert_node_on_path(aca_nodes_3[0], path1):
                            aca_boundary_ica = get_node_dict_entry(aca_nodes_1[1], 1, aca_label, polydata)
                        else:
                            aca_boundary_ica = get_node_dict_entry(aca_nodes_1[2], 1, aca_label, polydata)       

        # ACA end
        if len(aca_end) == 0:
            if len(aca_boundary_ica) > 0:
                aca_nodes_1.remove(aca_boundary_ica[0]['id'])
            if len(aca_boundary_acom) > 0:
                for pt in aca_acom_boundary:
                    if pt in aca_nodes_1:
                        aca_nodes_1.remove(pt)
            if len(aca_nodes_1) == 1: # typical case
                aca_end = get_node_dict_entry(aca_nodes_1[0], 1, aca_label, polydata)
                logger.debug(f'\tACA end of degree 1: {aca_end}')
            elif len(aca_nodes_1) >= 2: # ACA end is branch point and/or broken fenestration
                assert len(aca_nodes_3) > 0, 'Too few ACA 3-nodes found!'
                if len(acom_bif_node) == 0: # either no Acom or ACA end is branch point
                    assert len(aca_nodes_3) == 1, 'Wrong number of ACA 3-nodes!'
                    if acom_label in labels_array: # A1 missing and ACA end is branch point
                        assert variant_dict['anterior'][a1_name] == False, 'Acom bifurcation not found although acom is present and A1 not missing!'
                        logger.debug('\tA1 missing and ACA end is branch point.')
                        assert len(aca_nodes_1) == 2, 'Wrong number of ACA 1-nodes!'
                        aca_end = get_node_dict_entry(aca_nodes_3[0], 3, aca_label, polydata)
                        logger.debug(f'\tACA end is 3-node: {aca_end}')
                    else: # No Acom and ACA end is branch point
                        assert variant_dict['anterior']['Acom'] == False
                        logger.debug('\tNo Acom and ACA end is branch point.')
                        assert len(aca_nodes_1) == 2, 'Wrong number of ACA 1-nodes!'
                        aca_end = get_node_dict_entry(aca_nodes_3[0], 3, aca_label, polydata)
                        logger.debug(f'\tACA end is 3-node: {aca_end}')
        
                else: # acom bifurcation present
                    acom_bif_ids = [node['id'] for node in acom_bif_node]
                    aca_acom_nodes_high = aca_acom_nodes_3 + aca_acom_nodes_4
                    for pt in acom_bif_ids:
                        if pt in aca_acom_nodes_high:
                            aca_acom_nodes_high.remove(pt)
                        if pt in aca_nodes_3:
                            aca_nodes_3.remove(pt)
                    path_ica_acom = find_shortest_path(aca_ica_boundary[0], acom_bif_ids[0], polydata, [aca_label])['path']
                    fen_nodes = []
                    for node in aca_acom_nodes_high:
                        # 3-node befor Acom bifurcation -> Non-connected A1 fenestration?
                        if assert_node_on_path(node, path_ica_acom): 
                            if node in aca_nodes_3: 
                                aca_nodes_3.remove(node)
                            fen_nodes.append(node)
                    
                    loose_end = None
                    if len(fen_nodes) > 0:
                        logger.warning(f'\tALERT: Broken A1 fenestration found with loop node: {fen_nodes}')
                        assert len(fen_nodes) == 1, 'Wrong number of ACA fenestration nodes!'
                        assert len(acom_bif_node) == 1, 'A1 fenestration and Acom fenestration together?!'
                        loop = check_for_loop_between_nodes(fen_nodes[0], acom_bif_ids[0], aca_label, polydata)
                        assert not loop, 'Loop found between ACA fenestration and Acom bifurcation!'
                        logger.debug(f'\tConnecting possible fenestration for {aca_label}!')
                        variant_dict['fenestration'][f'{a1_name}'] = True
                        # get loose aca 1-node -> aca 1-node that is closest to the fenestration node
                        path_fen = find_shortest_shortest_path(aca_nodes_1, fen_nodes, polydata, aca_label)['path']
                        loose_end = path_fen[0][0]
                        nodes_fen = [p[0] for p in path_fen] + fen_nodes
                        path_a1 = find_shortest_path(aca_ica_boundary[0], acom_bif_node[0]['id'], polydata, [aca_label])['path']
                        nodes_a1 = [p[0] for p in path_a1[:-2]] # we allow the fenestration to be connected to A1 only very close to Acom bifurcation
                        points_aca = get_pointIds_for_label(aca_label, polydata)
                        # only connect to A2 segment?
                        # points_a2 = [pt for pt in points_aca if pt not in nodes_fen and pt not in nodes_a1]
                        # logger.debug(f'\tConnecting broken A1 fenestration to points A2: {points_a2}')
                        # NOTE: We allow connection also to A1 here (not just A2 as above)
                        points_a2 = [pt for pt in points_aca if pt not in nodes_fen]
                        logger.debug(f'\tConnecting broken A1 fenestration to points ACA: {points_a2}')
                        polydata = connect_loose_end(loose_end, points_a2, aca_label, polydata)
                        loop = check_for_loop_between_nodes(fen_nodes[0], acom_bif_ids[0], aca_label, polydata)
                        # if connecting fenestration worked correctly, there should be a loop now!
                        assert loop, 'Still no loop after connecting A1 fenestration!'

                    if len(aca_nodes_3) == 1: # we removed acom bifurcations and potential fenestration nodes already...
                        assert len(aca_nodes_1) >= 2, 'Wrong number of ACA 1-nodes!'
                        if loose_end is not None:
                            # broken A1 fenestration (fixed by now) and ACA end is branch point
                            assert len(aca_nodes_1) == 3, 'Wrong number of ACA 1-nodes!'
                        aca_end = get_node_dict_entry(aca_nodes_3[0], 3, aca_label, polydata)
                        logger.debug(f'\tACA end is 3-node: {aca_end}')
                    elif len(aca_nodes_3) == 0:
                        # ACA end is 1-node and broken A1 fenestration (fixed by now)
                        assert len(aca_nodes_1) == 2, 'Wrong number of ACA 1-nodes!'
                        assert loose_end is not None, 'No loose end found!'
                        aca_nodes_1.remove(loose_end)
                        aca_end = get_node_dict_entry(aca_nodes_1[0], 1, aca_label, polydata)
                        logger.debug(f'\tACA end is 1-node: {aca_end}')
                    else:
                        logger.warning(f'\tALERT: Couldnt determine ACA endè ACA 1-nodes: {aca_nodes_1}, ACA 3-nodes: {aca_nodes_3}')
                        raise ValueError('No ACA end found! Try to implement this case seprately!')
            else:
                logger.warning(f'\tALERT: Couldnt determine ACA end! Wrong number of ACA 1-nodes: {aca_nodes_1}')
                raise ValueError('Couldnt determine ACA end! Wrong number of ACA 1-nodes! Try to implement this case seprately!')

        if len(acom_boundary_aca) > 0:
            nodes_dict[str(acom_label)][f'{aca_name} boundary'] = acom_boundary_aca
        
        nodes_dict[str(aca_label)] = {}
        if len(aca_boundary_ica) > 0:
            nodes_dict[str(aca_label)]['ICA boundary'] = aca_boundary_ica
        if len(acom_bif_node) > 0:
            nodes_dict[str(aca_label)]['Acom bifurcation'] = acom_bif_node
        if len(aca_boundary_acom) > 0:
            nodes_dict[str(aca_label)]['Acom boundary'] = aca_boundary_acom
        if len(aca_end) > 0:
            nodes_dict[str(aca_label)]['ACA end'] = aca_end
        
        # Get A1 fenestration
        if len(aca_boundary_ica) > 0:
            loop = check_for_loop_between_nodes(aca_boundary_ica[0]['id'], aca_end[0]['id'], aca_label, polydata)
            if loop:
                assert variant_dict['fenestration'][f'{a1_name}'], 'Missing A1 fenestration?!'
                logger.debug(f'\tA1 fenestration: loop is correctly detected!')
        
        logger.debug(f'\tACA nodes dict: {nodes_dict[str(aca_label)]}')
    
    if variant_dict['anterior']['Acom']:
        logger.debug(f'\tAcom nodes dict: {nodes_dict[str(acom_label)]}')

    if len(acom_boundary_a2) > 0:
        nodes_dict[str(a2_label)] = {}
        nodes_dict[str(acom_label)]['3rd-A2 bifurcation'] = a2_bifurcation
        nodes_dict[str(acom_label)]['3rd-A2 boundary'] = acom_boundary_a2
    if len(a2_boundary_acom) > 0:
        nodes_dict[str(a2_label)]['Acom boundary'] = a2_boundary_acom
    if len(a2_end) > 0:
        nodes_dict[str(a2_label)]['3rd-A2 end'] = a2_end
    
    if variant_dict['anterior']['3rd-A2']:
        logger.debug(f'\t3rd-A2 nodes dict: {nodes_dict[str(a2_label)]}')

    return nodes_dict, variant_dict, polydata

def find_acom_bif_for_3rd_a2(acom_boundary, polydata, labels=[10, 15]):
    """
    Find the Acom bifurcation point for the 3rd-A2 segment.

    Args:
    acom_boundary: int, id of the Acom boundary point
    polydata: vtkPolyData, centerline polydata
    labels: list, list of labels to consider for path finding

    Returns:
    A2_start: int, id of the Acom bifurcation point
    """
    edge_list, _ = get_edge_list(polydata)
    acom_ids = get_pointIds_for_label(10, polydata)
    candidates = []
    for pt in acom_ids:
        degree = get_point_degree(pt, edge_list)
        if degree > 2:
            candidates.append(pt)

    if len(candidates) == 1:
        A2_start = candidates[0]
    elif acom_boundary in candidates:
        A2_start = acom_boundary
    elif len(candidates) > 1 and acom_boundary not in candidates:
        A2_start, _ = find_closest_node_to_point(candidates, acom_boundary, labels, polydata)
    elif len(candidates) == 0:
        logger.warning('No Acom point found for 3rd-A2 origin! Looking at ACAs now...')
        aca_nodes_higher = get_nodes_of_degree_n(3, [11, 12], polydata) + get_nodes_of_degree_n(4, [11, 12], polydata) + get_nodes_of_degree_n(5, [11, 12], polydata)
        A2_start, _ = find_closest_node_to_point(aca_nodes_higher, acom_boundary, [10, 11, 12], polydata)
    return A2_start








    
