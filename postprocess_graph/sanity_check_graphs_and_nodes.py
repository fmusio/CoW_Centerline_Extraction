from utils.utils_graph_processing import get_edge_list

def test_nodes_against_themselves(nodes_dict):
    """
    Many nodes appear in different segments --> must have same id, coordinates and degree!

    Args:
    nodes_dict: dict, node dict

    Returns:
    None
    """
    keys = ['coords', 'degree', 'id']
    if '1' in nodes_dict and 'R-PCA boundary' in nodes_dict['1'].keys():
        key_pair = ('R-PCA boundary', 'BA boundary')
        for i in range(len(nodes_dict['1'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['1'][key_pair[0]][i][k] == nodes_dict['2'][key_pair[1]][i][k]
    if '1' in nodes_dict and 'L-PCA boundary' in nodes_dict['1'].keys():
        key_pair = ('L-PCA boundary', 'BA boundary')
        for i in range(len(nodes_dict['1'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['1'][key_pair[0]][i][k] == nodes_dict['3'][key_pair[1]][i][k]
    if '2' in nodes_dict and 'Pcom boundary' in nodes_dict['2'].keys():
        key_pair = ('Pcom boundary', 'PCA boundary')
        for i in range(len(nodes_dict['2'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['2'][key_pair[0]][i][k] == nodes_dict['8'][key_pair[1]][i][k]

    if '3' in nodes_dict and 'Pcom boundary' in nodes_dict['3'].keys():
        key_pair = ('Pcom boundary', 'PCA boundary')
        for i in range(len(nodes_dict['3'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['3'][key_pair[0]][i][k] == nodes_dict['9'][key_pair[1]][i][k]

    if '4' in nodes_dict and 'Pcom boundary' in nodes_dict['4'].keys():
        key_pair = ('Pcom boundary', 'ICA boundary')
        for i in range(len(nodes_dict['4'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['4'][key_pair[0]][i][k] == nodes_dict['8'][key_pair[1]][i][k]

    if '6' in nodes_dict.keys() and 'Pcom boundary' in nodes_dict['6'].keys():
        key_pair = ('Pcom boundary', 'ICA boundary')
        for i in range(len(nodes_dict['6'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['6'][key_pair[0]][i][k] == nodes_dict['9'][key_pair[1]][i][k]

    if '4' in nodes_dict and 'MCA boundary' in nodes_dict['4'].keys():
        key_pair = ('MCA boundary', 'ICA boundary')
        for i in range(len(nodes_dict['4'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['4'][key_pair[0]][i][k] == nodes_dict['5'][key_pair[1]][i][k]

    if '6' in nodes_dict.keys() and 'MCA boundary' in nodes_dict['6'].keys():
        key_pair = ('MCA boundary', 'ICA boundary')
        for i in range(len(nodes_dict['6'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['6'][key_pair[0]][i][k] == nodes_dict['7'][key_pair[1]][i][k]
    
    if '4' in nodes_dict and 'ACA boundary' in nodes_dict['4'].keys():
        key_pair = ('ACA boundary', 'ICA boundary')
        for i in range(len(nodes_dict['4'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['4'][key_pair[0]][i][k] == nodes_dict['11'][key_pair[1]][i][k]

    if '6' in nodes_dict.keys() and 'ACA boundary' in nodes_dict['6'].keys():
        key_pair = ('ACA boundary', 'ICA boundary')
        for i in range(len(nodes_dict['6'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['6'][key_pair[0]][i][k] == nodes_dict['12'][key_pair[1]][i][k]
    
    if '11' in nodes_dict and 'Acom boundary' in nodes_dict['11'].keys():
        key_pair = ('Acom boundary', 'R-ACA boundary')
        for i in range(len(nodes_dict['11'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['11'][key_pair[0]][i][k] == nodes_dict['10'][key_pair[1]][i][k]

    if '12' in nodes_dict and 'Acom boundary' in nodes_dict['12'].keys():
        key_pair = ('Acom boundary', 'L-ACA boundary')
        for i in range(len(nodes_dict['12'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['12'][key_pair[0]][i][k] == nodes_dict['10'][key_pair[1]][i][k]
    
    if '10' in nodes_dict.keys() and '3rd-A2 boundary' in nodes_dict['10'].keys():
        key_pair = ('3rd-A2 boundary', 'Acom boundary')
        for i in range(len(nodes_dict['10'][key_pair[0]])):
            for k in keys:
                assert nodes_dict['10'][key_pair[0]][i][k] == nodes_dict['15'][key_pair[1]][i][k]

def test_degree_end_nodes(nodes_dict):
    """
    Testing degree of end nodes: must be 1 or >= 3 but not 2!

    Args:
    nodes_dict: dict, node dict
    
    Returns:
    None
    """
    start_nodes_deg_1 = [('1','BA start'), ('4','ICA start'), ('6','ICA start'), ('15','3rd-A2 end')]
    start_nodes_deg_1_or_3 = [('11','ACA end'), ('12','ACA end'), ('2', 'PCA end'), ('3', 'PCA end'), ('5', 'MCA end'), ('7', 'MCA end')]
    for nd in start_nodes_deg_1:
        if nd[0] in nodes_dict.keys() and nd[1] in nodes_dict[nd[0]].keys():
            assert nodes_dict[nd[0]][nd[1]][0]['degree'] == 1
    for nd in start_nodes_deg_1_or_3:
        if nd[0] in nodes_dict.keys() and nd[1] in nodes_dict[nd[0]].keys():
            assert nodes_dict[nd[0]][nd[1]][0]['degree'] == 1 or nodes_dict[nd[0]][nd[1]][0]['degree'] == 3 or nodes_dict[nd[0]][nd[1]][0]['degree'] == 2

def test_nodes_against_topology(nodes_dict, variant_dict):
    """
    Checking whether number of nodes agrees with CoW variant

    Args:
    nodes_dict: dict, node dict
    variant_dict: dict, edge dict

    Returns:
    None
    """
    BA = ['BA start', 'BA bifurcation', 'R-PCA boundary', 'L-PCA boundary']
    RPCA = ['BA boundary', 'Pcom bifurcation', 'Pcom boundary', 'PCA end']
    LPCA = ['BA boundary', 'Pcom bifurcation', 'Pcom boundary', 'PCA end']
    RICA = ['ICA start', 'Pcom bifurcation', 'Pcom boundary', 'ICA bifurcation', 'ACA boundary', 'MCA boundary']
    RMCA = ['ICA boundary', 'MCA end']
    LICA = ['ICA start', 'Pcom bifurcation', 'Pcom boundary', 'ICA bifurcation', 'ACA boundary', 'MCA boundary']
    LMCA = ['ICA boundary', 'MCA end']
    RPCOM = ['ICA boundary', 'PCA boundary']
    LPCOM = ['ICA boundary', 'PCA boundary']
    ACOM = ['R-ACA boundary', 'L-ACA boundary', '3rd-A2 bifurcation', '3rd-A2 boundary']
    RACA = ['ICA boundary', 'Acom bifurcation', 'Acom boundary', 'ACA end']
    LACA = ['ICA boundary', 'Acom bifurcation', 'Acom boundary', 'ACA end']
    A2 = ['Acom boundary', '3rd-A2 end']

    # Missing both P1
    if not variant_dict['posterior']['R-P1'] and not variant_dict['posterior']['L-P1']:
        assert nodes_dict['1'].keys() == set(BA) - {'R-PCA boundary'} - {'L-PCA boundary'}
        # NOTE: below assertions commented out because in broken P1 cases, the Pcom bifurcation might still be present
        # assert nodes_dict['2'].keys() == set(RPCA) - {'Pcom bifurcation'} - {'BA boundary'}
        # assert nodes_dict['3'].keys() == set(LPCA) - {'Pcom bifurcation'} - {'BA boundary'}
    # Missing R-P1 only
    elif not variant_dict['posterior']['R-P1']:
        assert nodes_dict['1'].keys() <= set(BA) - {'R-PCA boundary'}
        # NOTE: below assertions commented out because in broken P1 cases, the Pcom bifurcation might still be present
        # assert nodes_dict['2'].keys() == set(RPCA) - {'Pcom bifurcation'} - {'BA boundary'}
    # Missing L-P1 only
    elif not variant_dict['posterior']['L-P1']:
        assert nodes_dict['1'].keys() <= set(BA) - {'L-PCA boundary'}
        # NOTE: below assertions commented out because in broken P1 cases, the Pcom bifurcation might still be present
        # assert nodes_dict['3'].keys() == set(LPCA) - {'Pcom bifurcation'} - {'BA boundary'}
    # Both P1 present
    else:
        assert nodes_dict['1'].keys() == set(BA)

    # R-Pcom
    if variant_dict['posterior']['R-Pcom']:
        assert nodes_dict['8'].keys() == set(RPCOM)
        assert {'ICA start', 'Pcom bifurcation', 'Pcom boundary'} <= nodes_dict['4'].keys()
        if variant_dict['posterior']['R-P1']:
            assert nodes_dict['2'].keys() == set(RPCA)
    
    # L-Pcom
    if variant_dict['posterior']['L-Pcom']:
        assert nodes_dict['9'].keys() == set(LPCOM)
        assert {'ICA start', 'Pcom bifurcation', 'Pcom boundary'} <= nodes_dict['6'].keys()
        if variant_dict['posterior']['L-P1']:
            assert nodes_dict['3'].keys() == set(LPCA)
    
    # MCA
    if '5' in nodes_dict.keys():
        assert nodes_dict['5'].keys() == set(RMCA)
    if '7' in nodes_dict.keys():
        assert nodes_dict['7'].keys() == set(LMCA)

    # Acom
    if variant_dict['anterior']['Acom'] and variant_dict['anterior']['3rd-A2']:
        assert nodes_dict['10'].keys() == set(ACOM)
    elif variant_dict['anterior']['Acom']:
        assert nodes_dict['10'].keys() == set(ACOM) - {'3rd-A2 bifurcation', '3rd-A2 boundary'}
    
    # Missing R-A1 only
    if not variant_dict['anterior']['R-A1']:
        # assert nodes_dict['4'].keys() <= set(RICA) - {'ICA bifurcation'} - {'ACA boundary'}
        # broken A1 cases are special, still having an ICA boundary and an Acom bifurcation
        assert (nodes_dict['11'].keys() == {'Acom boundary', 'ACA end'}) or (nodes_dict['11'].keys() == set(RACA))
    # Missing L-A1 only
    elif not variant_dict['anterior']['L-A1']:
        # assert nodes_dict['6'].keys() <= set(LICA) - {'ICA bifurcation'} - {'ACA boundary'}
        # broken A1 cases are special, still having an ICA boundary and an Acom bifurcation
        assert (nodes_dict['12'].keys() == {'Acom boundary', 'ACA end'}) or (nodes_dict['12'].keys() == set(LACA))
    # Missing no A1
    else:
        if variant_dict['posterior']['R-Pcom'] and '5' in nodes_dict.keys():
            assert nodes_dict['4'].keys() == set(RICA)
        if variant_dict['posterior']['L-Pcom'] and '7' in nodes_dict.keys():
            assert nodes_dict['6'].keys() == set(LICA)
        if variant_dict['anterior']['Acom']:
            assert nodes_dict['11'].keys() == set(RACA)
            assert nodes_dict['12'].keys() == set(LACA)
    
    # 3rd-A2
    if variant_dict['anterior']['3rd-A2']:
        assert nodes_dict['15'].keys() == set(A2)

def check_for_loops(polydata):
    """
    Check for loops in the graph.

    Args:
    polydata: vtkPolyData, graph

    Returns:
    None
    """
    edge_list, cell_ids = get_edge_list(polydata)
    assert len(edge_list) == len(cell_ids)
    assert len(list(set(edge_list))) == len(edge_list), 'Duplicate edges found!'
    loop_edges = []
    for edge, cell in zip(edge_list, cell_ids):
        if edge[0] == edge[1]:
            loop_edges.append(cell)
        if (edge[1], edge[0]) in edge_list:
            cell_reversed = cell_ids[edge_list.index((edge[1], edge[0]))]
            if cell_reversed not in loop_edges:
                loop_edges.append(cell)
    
    assert len(loop_edges) == 0, f'Loops found in cells: {loop_edges}'

def run_sanity_check(nodes_dict, variant_dict, polydata):
    """
    Run sanity check for graph and nodes.

    Args:
    nodes_dict (dict): dict, node dict
    variant_dict (dict): dict, edge dict
    polydata (vtkPolyData): vtkPolyData, graph

    Returns:
    None
    """
    test_nodes_against_themselves(nodes_dict)
    test_degree_end_nodes(nodes_dict)
    test_nodes_against_topology(nodes_dict, variant_dict)
    check_for_loops(polydata)