from utils.utils_graph_processing import *

from logger import logger

def get_cow_variant(polydata):
    """""
    Identifies CoW variant from a centerline polydata object.
    This function analyzes the connectivity between various segments of the centerline 
    to determine the presence or absence of the anterior segments [Acom, 3rd-A2, R-A1, L-A1] and 
    the posterior segments [R-Pcom, L-Pcom, R-P1, L-P1]. 

    Parameters:
    ----------
    polydata (vtkPolyData): labeled centerline polydata object

    Returns:
    -------
    variant_dict: dictionary containing the CoW variantinformation (presence/absence of segments)

    Raises:
    --------
    ValueError: If detected connections are anatomically implausible (e.g., Acom not
                connected to both ACAs, 3rd-A2 not connected to Acom, Pcom not connected
                to both ICA and PCA)
    AssertionError: If a key arterial segment is missing or if a required compensatory
                    pathway is absent (e.g., if A1 is absent, Acom must be present)
    
    Note:
    ------
    Fenestration detection is mentioned in the returned dictionary but is not implemented 
    in this function. Neither are fetal PCA.
    """
    labels = get_label_array(polydata)

    # initialize variant dict with default values
    # AV and PV same as TopCoW
    variant_dict = {
        'anterior': {
            "L-A1": True,
            "Acom": False,
            "3rd-A2": False,
            "R-A1": True
        },
        'posterior': {
            "L-Pcom": False,
            "L-P1": True,
            "R-P1": True,
            "R-Pcom": False
        },
        'fetal': {
            "L-PCA": False,
            "R-PCA": False
        },
        'fenestration': {
            "L-A1": False,
            "Acom": False,
            "R-A1": False,
            "L-P1": False,
            "R-P1": False
        }
    }

    raca_label, laca_label = 11, 12
    rpca_label, lpca_label = 2, 3
    rica_label, lica_label = 4, 6
    ba_label = 1

    # Acom present if label present and connected to both ACAs
    acom_label = 10
    if acom_label in labels:
        logger.info('\tAcom present')
        raca_boundary = find_boundary_points(acom_label, raca_label, polydata)
        laca_boundary = find_boundary_points(acom_label, laca_label, polydata)
        if len(raca_boundary) > 0 and len(laca_boundary) > 0:
            variant_dict['anterior']['Acom'] = True
        else:
            logger.warning('\tALERT: Acom not connected to both ACAs?!')
            raise ValueError('Acom not connected to both ACAs')
    else:
        logger.info('\tAcom not present')
    
    # 3rd-A2 present if label present and connected to Acom
    a2_label = 15
    if a2_label in labels:
        assert variant_dict['anterior']['Acom'], 'Acom not present!'
        logger.info('\t3rd-A2 present')
        a2_boundary = find_boundary_points(a2_label, acom_label, polydata)
        a2_boundary_aca = find_boundary_points(a2_label, raca_label, polydata) + find_boundary_points(a2_label, laca_label, polydata)
        if len(a2_boundary) > 0:
            variant_dict['anterior']['3rd-A2'] = True
        elif len(a2_boundary_aca) > 0:
            logger.warning('\tALERT: 3rd-A2 connected to ACA instead of Acom!')
            variant_dict['anterior']['3rd-A2'] = True
        else:
            logger.warning('\tALERT: 3rd-A2 not connected to Acom!')
            raise ValueError('3rd-A2 not connected to Acom')
    else:
        logger.info('\t3rd-A2 not present')
    
    # R-A1 absent if label not connected to R-ICA
    assert raca_label in labels, 'R-ACA not present!'
    rica_boundary = find_boundary_points(raca_label, rica_label, polydata)
    if len(rica_boundary) == 0:
        logger.info('\tR-A1 not present')
        variant_dict['anterior']['R-A1'] = False
        assert variant_dict['anterior']['Acom'], 'Acom not present!'
    else:
        logger.info('\tR-A1 present')

    # L-A1 absent if label not connected to L-ICA
    assert laca_label in labels, 'L-ACA not present!'
    lica_boundary = find_boundary_points(laca_label, lica_label, polydata)
    if len(lica_boundary) == 0:
        logger.info('\tL-A1 not present')
        variant_dict['anterior']['L-A1'] = False
        assert variant_dict['anterior']['Acom'], 'Acom not present!'
    else:
        logger.info('\tL-A1 present')
    
    # R-Pcom present if label present and connected to R-PCA and R-ICA
    rpcom_label = 8
    if rpcom_label in labels:
        logger.info('\tR-Pcom present')
        rpca_boundary = find_boundary_points(rpcom_label, rpca_label, polydata)
        rica_boundary = find_boundary_points(rpcom_label, rica_label, polydata)
        if len(rpca_boundary) > 0 and len(rica_boundary) > 0:
            variant_dict['posterior']['R-Pcom'] = True
        else:
            logger.warning('\tALERT: R-Pcom not connected to both ICA and PCA!')
            raise ValueError('R-Pcom not connected to both ICA and PCA')
    else:
        logger.info('\tR-Pcom not present')
    
    # L-Pcom present if label present and connected to L-PCA and L-ICA
    lpcom_label = 9
    if lpcom_label in labels:
        logger.info('\tL-Pcom present')
        lpca_boundary = find_boundary_points(lpcom_label, lpca_label, polydata)
        lica_boundary = find_boundary_points(lpcom_label, lica_label, polydata)
        if len(lpca_boundary) > 0 and len(lica_boundary) > 0:
            variant_dict['posterior']['L-Pcom'] = True
        else:
            logger.warning('\tALERT: L-Pcom not connected to both ICA and PCA!')
            raise ValueError('L-Pcom not connected to both ICA and PCA')
    else:
        logger.info('\tL-Pcom not present')
    
    # R-P1 absent if label not connected to BA
    if rpca_label in labels:
        rpca_boundary = find_boundary_points(rpca_label, ba_label, polydata)
        if len(rpca_boundary) == 0:
            logger.info('\tR-P1 not present')
            variant_dict['posterior']['R-P1'] = False
            assert variant_dict['posterior']['R-Pcom'], 'R-Pcom not present!'
        else:
            logger.info('\tR-P1 present')
    else:
        logger.warning('\tR-PCA not present, thus R-P1 not present!')
        variant_dict['posterior']['R-P1'] = False
    
    # L-P1 absent if label not connected to BA
    if lpca_label in labels:
        lpca_boundary = find_boundary_points(lpca_label, ba_label, polydata)
        if len(lpca_boundary) == 0:
            logger.info('\tL-P1 not present')
            variant_dict['posterior']['L-P1'] = False
            assert variant_dict['posterior']['L-Pcom'], 'L-Pcom not present!'
        else:
            logger.info('\tL-P1 present')
    else:
        logger.warning('\tL-PCA not present, thus L-P1 not present!')
        variant_dict['posterior']['L-P1'] = False

    return variant_dict