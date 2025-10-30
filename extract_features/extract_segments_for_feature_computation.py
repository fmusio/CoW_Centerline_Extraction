from utils.utils_feature_extraction import find_endpoint_for_fixed_length, get_point_coords_along_path, compute_curve_length
from utils.utils_graph_processing import find_shortest_path

from logger import logger

def extract_segments(nodes_dict, variant_dict):
    """
    Extract all CoW segments, compare with variant dict and return start and end node for geometry computation.

    Parameters:
    nodes_dict (dict): Dictionary with all nodes
    variant_dict (dict): Dictionary with all variants

    Returns:
    segments (dict): Dictionary with all segments
    """
    segments = {}
    posterior_top = variant_dict['posterior']
    anterior_top = variant_dict['anterior']

    # BA: from start to bifurcation (unless P1 is missing)
    if '1' in nodes_dict:
        BA_dict = nodes_dict['1']
        if 'BA start' in BA_dict:
            BA_start = BA_dict['BA start'][0]['id']
            if 'BA bifurcation' in BA_dict:
                BA_end = BA_dict['BA bifurcation'][0]['id']
            else:
                assert posterior_top['R-P1'] == False or posterior_top['L-P1'] == False
                BA_end = BA_dict['R-PCA boundary'][0]['id'] if posterior_top['R-P1'] else BA_dict['L-PCA boundary'][0]['id']
            segments['BA'] = [(BA_start, BA_end, 1)]
    
    # R-PCA: Distinguish between P1, P2 and PCA
    if '2' in nodes_dict:
        RPCA_dict = nodes_dict['2']
        RP2_end = RPCA_dict['PCA end'][0]['id']
        if 'BA boundary' in RPCA_dict:
            # NOTE: For broken P1, we still have a BA boundary even if P1 is "missing"
            # assert posterior_top['R-P1']
            RPCA_start = [RPCA_dict['BA boundary'][i]['id'] for i in range(len(RPCA_dict['BA boundary']))]
            if 'Pcom bifurcation' in RPCA_dict:
                RP1_end = RPCA_dict['Pcom bifurcation'][0]['id']
                assert posterior_top['R-Pcom']
                segments['R-P1'] = [(RPCA_start[i], RP1_end, 2) for i in range(len(RPCA_start))]
                segments['R-P2'] = [(RP1_end, RP2_end, 2)]
            segments['R-PCA'] = [(RPCA_start[i], RP2_end, 2) for i in range(len(RPCA_start))]
        else:
            assert posterior_top['R-P1'] == False
            assert 'Pcom boundary' in RPCA_dict
            RPCA_start = RPCA_dict['Pcom boundary'][0]['id']
            segments['R-P2'] = [(RPCA_start, RP2_end, 2)]

    # L-PCA: Distinguish between P1, P2 and PCA
    if '3' in nodes_dict:
        LPCA_dict = nodes_dict['3']
        LP2_end = LPCA_dict['PCA end'][0]['id']
        if 'BA boundary' in LPCA_dict:
            # NOTE: For broken P1, we still have a BA boundary even if P1 is "missing"
            # assert posterior_top['L-P1']
            LPCA_start = [LPCA_dict['BA boundary'][i]['id'] for i in range(len(LPCA_dict['BA boundary']))]
            if 'Pcom bifurcation' in LPCA_dict:
                LP1_end = LPCA_dict['Pcom bifurcation'][0]['id']
                assert posterior_top['L-Pcom']
                segments['L-P1'] = [(LPCA_start[i], LP1_end, 3) for i in range(len(LPCA_start))]
                segments['L-P2'] = [(LP1_end, LP2_end, 3)]
            segments['L-PCA'] = [(LPCA_start[i], LP2_end, 3) for i in range(len(LPCA_start))]
                
        else:
            assert posterior_top['L-P1'] == False
            assert 'Pcom boundary' in LPCA_dict
            LPCA_start = LPCA_dict['Pcom boundary'][0]['id']
            segments['L-P2'] = [(LPCA_start, LP2_end, 3)]

    # R-ICA: 
    if '4' in nodes_dict:
        RICA_dict = nodes_dict['4']
        if 'ICA start' in RICA_dict:
            RICA_start = RICA_dict['ICA start'][0]['id']
        else:
            assert posterior_top['R-Pcom']
            RICA_start = RICA_dict['Pcom boundary'][0]['id']
        
        if 'ICA bifurcation' in RICA_dict:
            RICA_end = RICA_dict['ICA bifurcation'][0]['id']
            segments['R-ICA'] = [(RICA_start, RICA_end, 4)]
        else:
            assert anterior_top['R-A1'] == False or '5' not in nodes_dict
            if '5' in nodes_dict:
                RICA_end = RICA_dict['MCA boundary'][0]['id']
            else:
                RICA_end = RICA_dict['ACA boundary'][0]['id']
            segments['R-ICA'] = [(RICA_start, RICA_end, 4)]
    
    # R-MCA
    if '5' in nodes_dict:
        RMCA_dict = nodes_dict['5']
        RMCA_start = RMCA_dict['ICA boundary'][0]['id']
        RMCA_end = RMCA_dict['MCA end'][0]['id']
        segments['R-MCA'] = [(RMCA_start, RMCA_end, 5)]

    # L-ICA:
    if '6' in nodes_dict:
        LICA_dict = nodes_dict['6']
        if 'ICA start' in LICA_dict:
            LICA_start = LICA_dict['ICA start'][0]['id']
        else:
            assert posterior_top['L-Pcom']
            LICA_start = LICA_dict['Pcom boundary'][0]['id']
        if 'ICA bifurcation' in LICA_dict:
            LICA_end = LICA_dict['ICA bifurcation'][0]['id']
            segments['L-ICA'] = [(LICA_start, LICA_end, 6)]
        else:
            # assert anterior_top['L-A1'] == False or '7' not in nodes_dict
            if '7' in nodes_dict:
                LICA_end = LICA_dict['MCA boundary'][0]['id']
            else:
                LICA_end = LICA_dict['ACA boundary'][0]['id']
            segments['L-ICA'] = [(LICA_start, LICA_end, 6)]
    
    # L-MCA
    if '7' in nodes_dict:
        LMCA_dict = nodes_dict['7']
        LMCA_start = LMCA_dict['ICA boundary'][0]['id']
        LMCA_end = LMCA_dict['MCA end'][0]['id']
        segments['L-MCA'] = [(LMCA_start, LMCA_end, 7)]
    
    # R-Pcom
    if '8' in nodes_dict:
        assert posterior_top['R-Pcom']
        RPcom_dict = nodes_dict['8']
        RPcom_start = RPcom_dict['ICA boundary'][0]['id']
        RPcom_end = RPcom_dict['PCA boundary'][0]['id']
        segments['R-Pcom'] = [(RPcom_start, RPcom_end, 8)]
    
    # L-Pcom
    if '9' in nodes_dict:
        assert posterior_top['L-Pcom']
        LPcom_dict = nodes_dict['9']
        LPcom_start = LPcom_dict['ICA boundary'][0]['id']
        LPcom_end = LPcom_dict['PCA boundary'][0]['id']
        segments['L-Pcom'] = [(LPcom_start, LPcom_end, 9)]

    # Acom
    if '10' in nodes_dict:
        assert anterior_top['Acom']
        Acom_dict = nodes_dict['10']
        Acom_start = [Acom_dict['R-ACA boundary'][i]['id'] for i in range(len(Acom_dict['R-ACA boundary']))]
        if len(Acom_start) > 1:
            logger.warning('\tALERT: Acom fenestration on the right!')
        Acom_end = [Acom_dict['L-ACA boundary'][i]['id'] for i in range(len(Acom_dict['L-ACA boundary']))]
        if len(Acom_end) > 1:
            logger.warning('\tALERT: Acom fenestration on the left!')

        if len(Acom_start) == 1 and len(Acom_end) == 1:
            segments['Acom'] = [(Acom_start[0], Acom_end[0], 10)]
        elif len(Acom_start) == 2 and len(Acom_end) == 2:
            segments['Acom'] = [(Acom_start[i], Acom_end[i], 10) for i in range(len(Acom_start))]
        elif len(Acom_start) == 1 and len(Acom_end) == 2:
            segments['Acom'] = [(Acom_start[0], Acom_end[i], 10) for i in range(len(Acom_end))]
        else:
            segments['Acom'] = [(Acom_start[i], Acom_end[0], 10) for i in range(len(Acom_start))]
    
    # R-ACA: Distinguish between A1, A2 and ACA
    if '11' in nodes_dict:
        RACA_dict = nodes_dict['11']
        RA2_end = RACA_dict['ACA end'][0]['id']
        if 'ICA boundary' in RACA_dict:
            if anterior_top['R-A1']:
                RACA_start = RACA_dict['ICA boundary'][0]['id']
                if 'Acom bifurcation' in RACA_dict:
                    RA1_end = RACA_dict['Acom bifurcation'][0]['id']
                    assert anterior_top['Acom']
                    segments['R-A1'] = [(RACA_start, RA1_end, 11)]
                    segments['R-A2'] = [(RA1_end, RA2_end, 11)]
                segments['R-ACA'] = [(RACA_start, RA2_end, 11)]
            else: # broken A1!
                assert anterior_top['Acom'], 'A1 broken but no Acom?!'
                assert 'Acom bifurcation' in RACA_dict, 'A1 broken but no Acom?!'
                RACA_start = RACA_dict['ICA boundary'][0]['id']
                RA1_end = RACA_dict['Acom bifurcation'][0]['id']
                segments['R-A1'] = [(RACA_start, RA1_end, 11)]
                segments['R-A2'] = [(RA1_end, RA2_end, 11)]
                segments['R-ACA'] = [(RACA_start, RA2_end, 11)]
        else:
            assert anterior_top['R-A1'] == False
            assert 'Acom boundary' in RACA_dict
            RACA_start = RACA_dict['Acom boundary'][0]['id']
            segments['R-A2'] = [(RACA_start, RA2_end, 11)]

    # L-ACA: Distinguish between A1, A2 and ACA
    if '12' in nodes_dict:
        LACA_dict = nodes_dict['12']
        LA2_end = LACA_dict['ACA end'][0]['id']
        if 'ICA boundary' in LACA_dict:
            if anterior_top['L-A1']:
                LACA_start = LACA_dict['ICA boundary'][0]['id']
                if 'Acom bifurcation' in LACA_dict:
                    # Take node closer to ICA boundary
                    LA1_end = LACA_dict['Acom bifurcation'][0]['id']
                    assert anterior_top['Acom']
                    segments['L-A1'] = [(LACA_start, LA1_end, 12)]
                    segments['L-A2'] = [(LA1_end, LA2_end, 12)]
                segments['L-ACA'] = [(LACA_start, LA2_end, 12)]
            else: # broken A1!
                assert anterior_top['Acom'], 'A1 broken but no Acom?!'
                assert 'Acom bifurcation' in LACA_dict, 'A1 broken but no Acom?!'
                LACA_start = LACA_dict['ICA boundary'][0]['id']
                LA1_end = LACA_dict['Acom bifurcation'][0]['id']
                segments['L-A1'] = [(LACA_start, LA1_end, 12)]
                segments['L-A2'] = [(LA1_end, LA2_end, 12)]
                segments['L-ACA'] = [(LACA_start, LA2_end, 12)]
        else:
            assert anterior_top['L-A1'] == False
            assert 'Acom boundary' in LACA_dict
            LACA_start = LACA_dict['Acom boundary'][0]['id']
            segments['L-A2'] = [(LACA_start, LA2_end, 12)]
    
    # 3rd-A2
    if '15' in nodes_dict:
        assert anterior_top['3rd-A2']
        A2_dict = nodes_dict['15']
        A2_start = A2_dict['Acom boundary'][0]['id']
        A2_end = A2_dict['3rd-A2 end'][0]['id']
        segments['3rd-A2'] = [(A2_start, A2_end, 15)]

    # sanity check:
    if 'R-A1' in segments and segments:
        assert segments['R-A1'][0][0] == segments['R-ACA'][0][0]
        assert segments['R-A2'][0][0] == segments['R-A1'][0][1]
        assert segments['R-ACA'][0][1] == segments['R-A2'][0][1]
    if 'L-A1' in segments:	
        assert segments['L-A1'][0][0] == segments['L-ACA'][0][0]
        assert segments['L-A2'][0][0] == segments['L-A1'][0][1]
        assert segments['L-ACA'][0][1] == segments['L-A2'][0][1]
    if 'R-P1' in segments:
        assert segments['R-P1'][0][0] == segments['R-PCA'][0][0]
        assert segments['R-P2'][0][0] == segments['R-P1'][0][1]
        assert segments['R-PCA'][0][1] == segments['R-P2'][0][1]
    if 'L-P1' in segments and type(segments['L-P1']) == int:
        assert segments['L-P1'][0][0] == segments['L-PCA'][0][0]
        assert segments['L-P2'][0][0] == segments['L-P1'][0][1]
        assert segments['L-PCA'][0][1] == segments['L-P2'][0][1]
    

    return segments

def cap_segment_at_length(segment, polydata, length_cap=10):
    """
    Cap segment at length of 10mm. For vessels adjoining the CoW: BA, P2, C6, MCA, A2, 3rd-A2

    Parameters:
    segment (list): List of tuples with start and end node and label
    polydata (vtkPolyData): Centerline VTP file
    length_cap (int): Length to cap the segment

    Returns:
    segment (list): List of tuples with capped segment
    """
    start, end, label = segment[0]
    path = find_shortest_path(start, end, polydata, label)['path']
    if len(path) > 0:
        coords_points = get_point_coords_along_path(path, polydata)
        length = compute_curve_length(coords_points)
        if length > length_cap:
            # find new end point that is closest to 10mm
            new_end = find_endpoint_for_fixed_length(path, polydata, length_threshold=length_cap, stop_at_bif_point=False)
            segment = [(start, new_end, label)]
        else:
            # do nothing
            pass
        return segment
    
    else:
        return None


def get_consistent_segments(segments, polydata, nodes_dict, modality='ct', median_p1=7.2, median_a1=15.6, 
                            median_c7=7.1, margin_from_cow=10):
    """
    Work with consistent definitions of segments for better comparison.
    BA/MCA/A2/P2/3rd-A2 segments: Set length=10mm if possible
    P1 segment: If no pcom, set length to median P1 length = 7.2mm (better criterion?)
    A1 segment: If no Acom, set length to median A1 length = 15.6mm (better criterion?)
    ICA: 
        -> with pcom: use c7 betweeen pcom and ica bif, c6 10mm if possible (5mm for CT)
        -> without pcom: use median c7 length = 7.1
    
    Parameters:
    segments (dict): Dictionary with all segments
    polydata (vtkPolyData): Centerline VTP file
    nodes_dict (dict): Dictionary with all nodes
    modality (str): Imaging modality ('ct' or 'mr')
    median_p1 (float): Median length of P1 segment
    median_a1 (float): Median length of A1 segment
    median_c7 (float): Median length of C7 segment
    margin_from_cow (int): Margin from CoW to cap segments

    Returns:
    segments (dict): Dictionary with consistent segment definitions
    """
    # BA: Set length=10mm if possible
    if 'BA' in segments.keys():
        segment_key = 'BA'
        assert len(segments[segment_key]) == 1
        segment = [(segments[segment_key][0][1], segments[segment_key][0][0], segments[segment_key][0][2])]
        segments[segment_key] = cap_segment_at_length(segment, polydata, length_cap=margin_from_cow)
        segments[segment_key] = [(segments[segment_key][0][1], segments[segment_key][0][0], segments[segment_key][0][2])]
    
    # MCA/3rd-A2: Set length=10mm if possible
    for segment_key in ['R-MCA', 'L-MCA', '3rd-A2']:
        if segment_key in segments:
            assert len(segments[segment_key]) == 1
            segments[segment_key] = cap_segment_at_length(segments[segment_key], polydata, length_cap=margin_from_cow)
    
    # PCA
    sides = ['R-', 'L-']
    for side in sides:
        if f'{side}P1' in segments:
            # Case Pcom
            assert f'{side}P2' in segments, f'{side}P2 missing!'
            assert f'{side}PCA' in segments, f'{side}PCA missing!'
            # Cap P2 at 10mm
            segments[f'{side}P2'] = cap_segment_at_length(segments[f'{side}P2'], polydata, length_cap=margin_from_cow)
            if len(segments[f'{side}PCA']) == 1:
                segments[f'{side}PCA'] = [(segments[f'{side}PCA'][0][0], segments[f'{side}P2'][0][1], segments[f'{side}PCA'][0][2])]
            elif len(segments[f'{side}PCA']) == 2:
                segments[f'{side}PCA'] = [(segments[f'{side}PCA'][i][0], segments[f'{side}P2'][0][1], segments[f'{side}PCA'][i][2]) for i in range(len(segments[f'{side}PCA']))]
            else:
                raise NotImplementedError
        elif f'{side}P2' in segments:
            # Case missing P1
            assert f'{side}PCA' not in segments, f'{side}PCA should not be present!'
            assert f'{side}P1' not in segments, f'{side}P1 should not be present!'
            # Cap P2 at 10mm
            segments[f'{side}P2'] = cap_segment_at_length(segments[f'{side}P2'], polydata, length_cap=margin_from_cow)
        elif f'{side}PCA' in segments:
            assert f'{side}P1' not in segments, f'{side}P1 should not be present!'
            assert f'{side}P2' not in segments, f'{side}P2 should not be present!'
            # Cap P1 at mean P1 length
            segments[f'{side}P1'] = cap_segment_at_length(segments[f'{side}PCA'], polydata, length_cap=median_p1)
            # Cap P2 at 10mm
            segment_p2 = [(segments[f'{side}P1'][0][1], segments[f'{side}PCA'][0][1], segments[f'{side}PCA'][0][2])]
            segment_p2 = cap_segment_at_length(segment_p2, polydata, length_cap=margin_from_cow)
            if segment_p2 is not None:
                segments[f'{side}P2'] = segment_p2
                segments[f'{side}PCA'] = [(segments[f'{side}PCA'][0][0], segments[f'{side}P2'][0][1], segments[f'{side}PCA'][0][2])]
            else:
                segments[f'{side}PCA'] = [(segments[f'{side}PCA'][0][0], segments[f'{side}PCA'][0][1], segments[f'{side}PCA'][0][2])]
        else:
            logger.warning(f'{side}PCA segment missing altogether?!')
            pass
    
    # ACA
    sides = ['R-', 'L-']
    for side in sides:
        if f'{side}A1' in segments:
            # Case Acom
            assert f'{side}A2' in segments, f'{side}A2 missing!'
            assert f'{side}ACA' in segments, f'{side}ACA missing!'
            # Cap A2 at 10mm
            segments[f'{side}A2'] = cap_segment_at_length(segments[f'{side}A2'], polydata, length_cap=margin_from_cow)
            assert len(segments[f'{side}ACA']) == 1
            segments[f'{side}ACA'] = [(segments[f'{side}ACA'][0][0], segments[f'{side}A2'][0][1], segments[f'{side}ACA'][0][2])]
        elif f'{side}A2' in segments:
            # Case missing A1
            assert f'{side}ACA' not in segments, f'{side}ACA should not be present!'
            assert f'{side}A1' not in segments, f'{side}A1 should not be present!'
            # Cap A2 at 10mm
            segments[f'{side}A2'] = cap_segment_at_length(segments[f'{side}A2'], polydata, length_cap=margin_from_cow)
        elif f'{side}ACA' in segments:
            assert f'{side}A1' not in segments, f'{side}A1 should not be present!'
            assert f'{side}A2' not in segments, f'{side}A2 should not be present!'
            # Cap A1 at mean A1 length
            segments[f'{side}A1'] = cap_segment_at_length(segments[f'{side}ACA'], polydata, length_cap=median_a1)
            # Cap A2 at 10mm
            segments[f'{side}A2'] = [(segments[f'{side}A1'][0][1], segments[f'{side}ACA'][0][1], segments[f'{side}ACA'][0][2])]
            segments[f'{side}A2'] = cap_segment_at_length(segments[f'{side}A2'], polydata, length_cap=margin_from_cow)
            if segments[f'{side}A2'] is not None:
                segments[f'{side}ACA'] = [(segments[f'{side}ACA'][0][0], segments[f'{side}A2'][0][1], segments[f'{side}ACA'][0][2])]
            else: 
                segments[f'{side}ACA'] = [(segments[f'{side}ACA'][0][0], segments[f'{side}ACA'][0][1], segments[f'{side}ACA'][0][2])]
        else:
            raise NotImplementedError
        
    # ICA
    c6_stop = margin_from_cow
    if modality == 'ct':
        # Use smaller distance for CT because of ICA entering bones
        c6_stop = 5
    sides = ['R-', 'L-']
    for side in sides:
        if f'{side}ICA' in segments:
            ica_label = segments[f'{side}ICA'][0][2]
            if f'{side}Pcom' in segments:
                # Case with pcom
                if 'ICA start' in nodes_dict[f'{ica_label}']:
                    assert 'Pcom bifurcation' in nodes_dict[f'{ica_label}'], f'Pcom bifurcation missing for {side}ICA!'
                    pcom_bif_id = nodes_dict[str(ica_label)]['Pcom bifurcation'][0]['id']
                    segments[f'{side}C7'] = [(pcom_bif_id, segments[f'{side}ICA'][0][1], ica_label)]
                    c6_segment_reversed = [(pcom_bif_id, segments[f'{side}ICA'][0][0], ica_label)]
                    c6_segment_reversed = cap_segment_at_length(c6_segment_reversed, polydata, length_cap=c6_stop)
                    if c6_segment_reversed is not None:
                        segments[f'{side}C6'] = [(c6_segment_reversed[0][1], pcom_bif_id, ica_label)]
                        segments[f'{side}ICA'] = [(segments[f'{side}C6'][0][0], segments[f'{side}ICA'][0][1], ica_label)]
                    else:
                        segments[f'{side}ICA'] = [(segments[f'{side}C7'][0][0], segments[f'{side}C7'][0][1], ica_label)]
                else:
                    pcom_bif_id = nodes_dict[str(ica_label)]['Pcom boundary'][0]['id']
                    segments[f'{side}C7'] = [(pcom_bif_id, segments[f'{side}ICA'][0][1], ica_label)]
                    del segments[f'{side}ICA']
            else:
                # case without pcom
                ica_segment_reversed = [(segments[f'{side}ICA'][0][1], segments[f'{side}ICA'][0][0], ica_label)]
                c7_segment_reversed = cap_segment_at_length(ica_segment_reversed, polydata, length_cap=median_c7)
                c6_segment_reversed = [(c7_segment_reversed[0][1], segments[f'{side}ICA'][0][0], ica_label)]
                c6_segment_reversed = cap_segment_at_length(c6_segment_reversed, polydata, length_cap=c6_stop)
                segments[f'{side}C7'] = [(c7_segment_reversed[0][1], segments[f'{side}ICA'][0][1], ica_label)]
                if c6_segment_reversed is not None:
                    segments[f'{side}C6'] = [(c6_segment_reversed[0][1], c6_segment_reversed[0][0], ica_label)]
                    segments[f'{side}ICA'] = [(segments[f'{side}C6'][0][0], segments[f'{side}C7'][0][1], ica_label)]
                else:
                    segments[f'{side}ICA'] = [(segments[f'{side}C7'][0][0], segments[f'{side}C7'][0][1], ica_label)]
                
                

    return segments