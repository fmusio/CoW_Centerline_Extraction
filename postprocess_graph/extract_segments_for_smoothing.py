from utils.utils_graph_processing import *

def extract_segments(nodes_dict, variant_dict, polydata):
    """
    Extract all CoW segments, compare with CoW variant dict and return start and end node and labels for each segment
    for smoothing and reordering the centerline graph.
    The segments are defined as follows:
    - BA: BA start -> BA bifurcation
    - R/L-PCA: BA bifurcation -> PCA end (+ remaining segments after branching point)
    - R/L-ICA: ICA start -> ICA bifurcation
    - R/L-MCA: ICA bifurcation -> MCA end (+ remaining segments after branching point)
    - R/L-Pcom: ICA bifurcation -> Pcom bifurcation
    - R/L-ACA: ICA bifurcation -> ACA end
    - Acom: Acom bifurcation -> Acom bifurcation
    - 3rd-A2: Acom bifurcation -> 3rd-A2 end

    Args:
    nodes_dict: dict, dictionary containing the CoW nodes
    variant_dict: dict, dictionary containing the CoW variants
    polydata: vtkPolyData, centerline polydata

    Returns:
    segments: dict, dictionary containing the start and end nodes and labels for each segment
    """
    # get all nodes of degree 1 that are not end nodes (= remaining nodes of deg 1)
    edge_list, cell_ids = get_edge_list(polydata)
    all_nodeIds_deg_1 = get_all_nodes_of_deg_1(polydata)
    end_nodeIds_deg_1, remaining_nodeIds_deg_1 = match_end_nodes_and_nodes_of_deg_1(all_nodeIds_deg_1, nodes_dict)
    
    # get labels and node dict entries for these remaining deg 1 nodes
    remaining_labels_deg_1, remaining_nodes_deg_1 = [], []
    for nodeId in remaining_nodeIds_deg_1:
        cell_ids = get_cellIds_for_point(nodeId, polydata)
        assert len(cell_ids) == 1, f"Degree-1-Node {nodeId} has more than one cell id: {cell_ids}"
        cell_id = cell_ids[0]
        label = get_label_for_cell(cell_id, polydata)
        node = get_node_dict_entry(nodeId, 1, label, polydata)[0]
        remaining_labels_deg_1.append(label)
        remaining_nodes_deg_1.append(node)

    segments = {}
    posterior_top = variant_dict['posterior']
    anterior_top = variant_dict['anterior']

    # Helper to add remaining deg 1 nodes to segments
    def add_remaining_segments(label, key, ref_point, end_point, polydata):
        """
        Helper to add segments after a branching point that end in a remaining deg 1 node (for BA, MCA, ACA).
        This function checks if the end_point lies on the path from ref_point to the remaining deg 1 node(s).
        If not, the segment is added to the segments dictionary since it is a valid segment.

        Args:
        label: int, label of the remaining nodes
        key: str, key of the segment dictionary
        ref_point: int, reference point for the path finding
        end_point: int, end point of the segment as control
        polydata: vtkPolyData, centerline polydata
        """
        if label in remaining_labels_deg_1:
            idxs = [i for i, val in enumerate(remaining_labels_deg_1) if val == label]
            for i in idxs:
                node_id = remaining_nodes_deg_1[i]['id']
                try:
                    paths = find_all_paths(ref_point, node_id, polydata, label)
                    for p in paths:
                        path = p['path']
                        if not assert_node_on_path(end_point, path[1:]):
                            segments[key].append((ref_point, node_id, [label]))
                except AssertionError:
                    pass

    try:
        BA_dict = nodes_dict['1']
    except KeyError:
        logger.warning("\tkey 1 (BA) not found in nodes_dict")
        BA_dict = {}
    try:
        RPCA_dict = nodes_dict['2']
    except KeyError:
        logger.warning("\tkey 2 (R-PCA) not found in nodes_dict")
        RPCA_dict = {}
    try:
        LPCA_dict = nodes_dict['3']
    except KeyError:
        logger.warning("\tkey 3 (L-PCA) not found in nodes_dict")
        LPCA_dict = {}
    try:
        RICA_dict = nodes_dict['4']
    except KeyError:
        logger.warning("\tkey 4 (R-ICA) not found in nodes_dict")
        RICA_dict = {}
    try:
        LICA_dict = nodes_dict['6']
    except KeyError:
        logger.warning("\tkey 6 (L-ICA) not found in nodes_dict")
        LICA_dict = {}
    try:
        RACA_dict = nodes_dict['11']
    except KeyError:
        logger.warning("\tkey 11 (R-ACA) not found in nodes_dict")
        RACA_dict = {}
    try:
        LACA_dict = nodes_dict['12']
    except KeyError:
        logger.warning("\tkey 12 (L-ACA) not found in nodes_dict")
        LACA_dict = {}

    # Extract segments one by one
    # BA
    if '1' in nodes_dict:
        BA_dict = nodes_dict['1']
        if 'BA start' in BA_dict:
            ba_start = BA_dict['BA start'][0]['id']
            ba_bif = BA_dict['BA bifurcation'][0]['id'] if 'BA bifurcation' in BA_dict else None
            if ba_bif:
                if posterior_top['R-P1'] and posterior_top['L-P1']:
                    segments['BA'] = [(ba_start, ba_bif, [1])]
                elif posterior_top['R-P1']:
                    segments['BA'] = [(ba_start, RPCA_dict['PCA end'][0]['id'], [1, 2])]
                elif posterior_top['L-P1']:
                    segments['BA'] = [(ba_start, LPCA_dict['PCA end'][0]['id'], [1, 3])]
                else:
                    segments['BA'] = [(ba_start, ba_bif, [1])]
            else:
                if posterior_top['R-P1']:
                    segments['BA'] = [(ba_start, RPCA_dict['PCA end'][0]['id'], [1, 2])]
                elif posterior_top['L-P1']:
                    segments['BA'] = [(ba_start, LPCA_dict['PCA end'][0]['id'], [1, 3])]
            # Add remaining segments after branching point
            add_remaining_segments(1, 'BA', ba_bif, ba_start, polydata)

    # R-PCA
    if '2' in nodes_dict:
        RPCA_dict = nodes_dict['2']
        rp2_end = RPCA_dict['PCA end'][0]['id']
        if 'BA boundary' in RPCA_dict:
            if posterior_top['R-P1'] and posterior_top['L-P1']:
                segments['R-PCA'] = [(BA_dict['BA bifurcation'][0]['id'], rp2_end, [1, 2])]
            # Case: broken P1!
            elif not posterior_top['R-P1']:
                segments['R-PCA'] = [(RPCA_dict['BA boundary'][0]['id'], rp2_end, [1, 2])]
        else:
            rpca_start = RICA_dict['Pcom bifurcation'][0]['id']
            segments['R-Pcom'] = [(rpca_start, rp2_end, [2, 4, 8])]
        # Add remaining segments after branching point
        if 2 in remaining_labels_deg_1:
            idxs = [i for i, val in enumerate(remaining_labels_deg_1) if val == 2]
            for i in idxs:
                try:
                    segments['R-PCA'].append((rp2_end, remaining_nodes_deg_1[i]['id'], [2]))
                except KeyError:
                    if posterior_top['R-P1']:
                        segments['BA'].append((rp2_end, remaining_nodes_deg_1[i]['id'], [2]))
                    else:
                        segments['R-Pcom'].append((rp2_end, remaining_nodes_deg_1[i]['id'], [2]))

    # L-PCA
    if '3' in nodes_dict:
        LPCA_dict = nodes_dict['3']
        lp2_end = LPCA_dict['PCA end'][0]['id']
        if 'BA boundary' in LPCA_dict:
            if posterior_top['R-P1'] and posterior_top['L-P1']:
                segments['L-PCA'] = [(BA_dict['BA bifurcation'][0]['id'], lp2_end, [1, 3])]
            # Case: broken P1!
            elif not posterior_top['L-P1']:
                segments['L-PCA'] = [(LPCA_dict['BA boundary'][0]['id'], lp2_end, [1, 3])]
        else:
            LICA_dict = nodes_dict['6']
            lpca_start = LICA_dict['Pcom bifurcation'][0]['id']
            segments['L-Pcom'] = [(lpca_start, lp2_end, [3, 6, 9])]
        # Add remaining segments after branching point
        if 3 in remaining_labels_deg_1:
            idxs = [i for i, val in enumerate(remaining_labels_deg_1) if val == 3]
            for i in idxs:
                try:
                    segments['L-PCA'].append((lp2_end, remaining_nodes_deg_1[i]['id'], [3]))
                except KeyError:
                    if posterior_top['L-P1']:
                        segments['BA'].append((lp2_end, remaining_nodes_deg_1[i]['id'], [3]))
                    else:
                        segments['L-Pcom'].append((lp2_end, remaining_nodes_deg_1[i]['id'], [3]))

    # R-ICA
    if '4' in nodes_dict:
        RICA_dict = nodes_dict['4']
        if 'ICA start' in RICA_dict:
            rica_start = RICA_dict['ICA start'][0]['id']
            rica_labels = [4]
        else:
            rica_start = RPCA_dict['Pcom bounary'][0]['id']
            rica_labels = [2, 4, 8]
        if 'ICA bifurcation' in RICA_dict:
            segments['R-ICA'] = [(rica_start, RICA_dict['ICA bifurcation'][0]['id'], rica_labels)]
        else:
            if '5' in nodes_dict:
                segments['R-ICA'] = [(rica_start, nodes_dict['5']['MCA end'][0]['id'], [4, 5])]
            else:
                segments['R-ICA'] = [(rica_start, RACA_dict['ACA end'][0]['id'], [4, 11])]

    # R-MCA
    if '5' in nodes_dict:
        rmca_end = nodes_dict['5']['MCA end'][0]['id']
        if segments['R-ICA'][0][1] == rmca_end:
            # Add remaining segments after branching point
            add_remaining_segments(5, 'R-ICA', rmca_end, rmca_end, polydata)
        else:
            if 'ICA bifurcation' in RICA_dict:
                segments['R-MCA'] = [(RICA_dict['ICA bifurcation'][0]['id'], rmca_end, [4, 5])]
            else:
                segments['R-MCA'] = [(nodes_dict['5']['ICA boundary'][0]['id'], rmca_end, [5])]
            # Add remaining segments after branching point
            add_remaining_segments(5, 'R-MCA', rmca_end, rmca_end, polydata)

    # L-ICA
    if '6' in nodes_dict:
        LICA_dict = nodes_dict['6']
        if 'ICA start' in LICA_dict:
            lica_start = LICA_dict['ICA start'][0]['id']
            lica_labels = [6]
        else:
            lica_start = LPCA_dict['Pcom boundary'][0]['id']
            lica_labels = [3, 6, 9]
        if 'ICA bifurcation' in LICA_dict:
            segments['L-ICA'] = [(lica_start, LICA_dict['ICA bifurcation'][0]['id'], lica_labels)]
        else:
            if '7' in nodes_dict:
                segments['L-ICA'] = [(lica_start, nodes_dict['7']['MCA end'][0]['id'], [6, 7])]
            else:
                segments['L-ICA'] = [(lica_start, LACA_dict['ACA end'][0]['id'], [6, 12])]

    # L-MCA
    if '7' in nodes_dict:
        lmca_dict = nodes_dict['7']
        lmca_end = lmca_dict['MCA end'][0]['id']
        if segments['L-ICA'][0][1] == lmca_end:
            # Add remaining segments after branching point
            add_remaining_segments(7, 'L-ICA', lmca_end, lmca_end, polydata)
        else:
            if 'ICA bifurcation' in LICA_dict:
                segments['L-MCA'] = [(LICA_dict['ICA bifurcation'][0]['id'], lmca_end, [6, 7])]
            else:
                segments['L-MCA'] = [(lmca_dict['ICA boundary'][0]['id'], lmca_end, [7])]
            # Add remaining segments after branching point
            add_remaining_segments(7, 'L-MCA', lmca_end, lmca_end, polydata)

    # R-Pcom
    if ('8' in nodes_dict and posterior_top['R-P1'] and 'ICA start' in RICA_dict) \
        or ('8' in nodes_dict and 'BA boundary' in RPCA_dict and 'ICA start' in RICA_dict):
        rpcom_start = RICA_dict['Pcom bifurcation'][0]['id']
        rpcom_end = RPCA_dict['Pcom bifurcation'][0]['id']
        segments['R-Pcom'] = [(rpcom_start, rpcom_end, [2, 4, 8])]

    # L-Pcom
    if ('9' in nodes_dict and posterior_top['L-P1'] and 'ICA start' in LICA_dict) \
        or ('9' in nodes_dict and 'BA boundary' in LPCA_dict and 'ICA start' in LICA_dict):
        lpcom_start = LICA_dict['Pcom bifurcation'][0]['id']
        lpcom_end = LPCA_dict['Pcom bifurcation'][0]['id']
        segments['L-Pcom'] = [(lpcom_start, lpcom_end, [3, 6, 9])]

    # R-ACA
    if '11' in nodes_dict:
        RACA_dict = nodes_dict['11']
        ra2_end = RACA_dict['ACA end'][0]['id']
        # NOTE: if MCA is missing, we smooth ICA and ACA as single segment
        if '5' in nodes_dict:
            if 'ICA boundary' in RACA_dict and anterior_top['R-A1'] and 'ICA bifurcation' in RICA_dict:
                segments['R-ACA'] = [(RICA_dict['ICA bifurcation'][0]['id'], ra2_end, [4, 11])]
            elif 'ICA boundary' in RACA_dict and anterior_top['R-A1']:
                segments['L-ACA'] = [(RACA_dict['ICA boundary'][0]['id'], ra2_end, [11])]
            # case broken A1
            elif 'ICA boundary' in RACA_dict:
                assert 'ICA bifurcation' not in RICA_dict, 'R-A1 broken but ICA bifurcation present?!'
                segments['R-ACA'] = [(RACA_dict['ICA boundary'][0]['id'], ra2_end, [11])]
            else:
                segments['Acom'] = [(LACA_dict['Acom bifurcation'][0]['id'], ra2_end, [10, 11, 12])]
        # Add remaining segments after branching point
        add_remaining_segments(11, 'R-ACA', ra2_end, ra2_end, polydata) if 'R-ACA' in segments else None

    # L-ACA
    if '12' in nodes_dict:
        LACA_dict = nodes_dict['12']
        la2_end = LACA_dict['ACA end'][0]['id']
        # NOTE: if MCA is missing, we smooth ICA and ACA as single segment
        if '7' in nodes_dict:
            if 'ICA boundary' in LACA_dict and anterior_top['L-A1'] and 'ICA bifurcation' in LICA_dict:
                segments['L-ACA'] = [(LICA_dict['ICA bifurcation'][0]['id'], la2_end, [6, 12])]
            elif 'ICA boundary' in LACA_dict and anterior_top['L-A1']:
                segments['L-ACA'] = [(LACA_dict['ICA boundary'][0]['id'], la2_end, [12])]
            # case: broken A1
            elif 'ICA boundary' in LACA_dict:
                assert 'ICA bifurcation' not in LICA_dict, 'L-A1 broken but ICA bifurcation present?!'
                segments['L-ACA'] = [(LACA_dict['ICA boundary'][0]['id'], la2_end, [12])]
            else:
                segments['Acom'] = [(RACA_dict['Acom bifurcation'][0]['id'], la2_end, [10, 11, 12])]
        # Add remaining segments after branching point
        add_remaining_segments(12, 'L-ACA', la2_end, la2_end, polydata) if 'L-ACA' in segments else None

    # Acom
    if ('10' in nodes_dict and anterior_top['L-A1'] and anterior_top['R-A1'] and 'Acom bifurcation' in RACA_dict and 'Acom bifurcation' in LACA_dict) \
    or ('10' in nodes_dict and anterior_top['R-A1'] and 'Acom bifurcation' in RACA_dict and 'Acom bifurcation' in LACA_dict) \
    or ('10' in nodes_dict and anterior_top['L-A1'] and 'Acom bifurcation' in RACA_dict and 'Acom bifurcation' in LACA_dict):
        acom_starts = [d['id'] for d in RACA_dict['Acom bifurcation']]
        acom_ends = [d['id'] for d in LACA_dict['Acom bifurcation']]
        if len(acom_starts) == 1 and len(acom_ends) == 1:
            segments['Acom'] = [(acom_starts[0], acom_ends[0], [10, 11, 12])]
        elif len(acom_starts) == len(acom_ends) == 2:
            segments['Acom'] = [(acom_starts[i], acom_ends[i], [10, 11, 12]) for i in range(2)]
        elif len(acom_starts) == 2 and len(acom_ends) == 1:
            segments['Acom'] = [(acom_starts[0], acom_ends[0], [10, 11, 12]), (acom_starts[1], acom_ends[0], [10, 11, 12])]
        elif len(acom_starts) == 1 and len(acom_ends) == 2:
            segments['Acom'] = [(acom_starts[0], acom_ends[0], [10, 11, 12]), (acom_starts[0], acom_ends[1], [10, 11, 12])]
        else:
            raise ValueError('Acom fenestration not handled!')

    # 3rd-A2
    if '15' in nodes_dict and anterior_top['3rd-A2']:
        a2_dict = nodes_dict['15']
        a2_end = a2_dict['3rd-A2 end'][0]['id']
        # a2_start = find_acom_bif_for_3rd_a2(a2_dict['Acom boundary'][0]['id'], polydata)
        ACOM_dict = nodes_dict['10']
        a2_start = ACOM_dict['3rd-A2 bifurcation'][0]['id']
        segments['3rd-A2'] = [(a2_start, a2_end, [10, 15])]

    return segments
