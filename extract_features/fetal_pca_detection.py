import os
import sys
# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from utils.utils_graph_processing import get_vtk_polydata_from_file, get_edge_list, get_cellId_for_edge, find_shortest_path

from logger import logger

def collect_radius_along_path(polydata, path, radius_attribute='ce_radius'):
    """
    Given a path of edges, compute radius along path.

    Parameters:
    polydata (vtkPolyData): VTK PolyData object
    path (list): List of edges
    radius_attribute (str): Attribute name for radius, ('ce_radius', 'mis_radius', 'voreen_radius')

    Returns:
    list: List of radii along the path
    """
    radius = polydata.GetCellData().GetArray(radius_attribute)

    _, cell_ids_cow = get_edge_list(polydata)

    cellIds_path = []
    for edge in path:
        cellIds_path.append(get_cellId_for_edge(edge, polydata))
    
    radii = []
    for id in cellIds_path:
        assert cell_ids_cow[id] == id, "Cell ID does not match!"
        radii.append(radius.GetValue(id))

    return radii

def check_for_fetal_pca(feature_dict, polydata, side, percentile=25, factor=1.05):
    """
    Detect the presence of a fetal PCA variant on the specified side.
    This function determines if a fetal PCA variant exists by either:
    1. Checking if the P1 segment is absent while Pcom is present, or
    2. Comparing the radii of P1 and Pcom segments - if Pcom is larger than P1 
        (based on the specified percentile and factor), it indicates a fetal PCA variant.
    
    Parameters
    ----------
        feature_dict (dict): cow feature dictionary
        polydata (vtkPolyData): CNT polydata object
        side (str): The side to check, either 'right' or 'left'
        percentile (int, optional): The percentile to use for radius comparison (default 25)
        factor (float, optional): The multiplier for P1 radius when comparing to Pcom radius (default 1.05)
    
    Returns
    -------
        bool: True if a fetal PCA variant is detected, False otherwise
    """

    if side == 'right':
        pcom_label = '8'
        pca_label = '2'
    elif side == 'left':
        pcom_label = '9'
        pca_label = '3'
    else:
        raise ValueError("Side must be either 'right' or 'left'.")
    
    rad_feature = None
    if percentile == 0:
        rad_feature = 'min'
    elif percentile == 25:
        rad_feature = 'q1'
    elif percentile == 50:
        rad_feature = 'median'
    elif percentile == 75:
        rad_feature = 'q3'
    elif percentile == 100:
        rad_feature = 'max'
    
    fetal_pca = False
    
    # right side
    if pcom_label in feature_dict:
        if 'P1' not in feature_dict[pca_label]:
            fetal_pca = True
        else:
            if rad_feature is not None:
                rad_p1 = feature_dict[pca_label]['P1'][0]['radius'][rad_feature]
                rad_pcom = feature_dict[pcom_label]['Pcom'][0]['radius'][rad_feature]
                if rad_pcom > factor * rad_p1:
                    fetal_pca = True
            else:
                p1_start = feature_dict[pca_label]['P1'][0]['segment']['start']
                p1_end = feature_dict[pca_label]['P1'][0]['segment']['end']
                p1_path = find_shortest_path(p1_start, p1_end, polydata, int(pca_label))['path']
                pcom_start = feature_dict[pcom_label]['Pcom'][0]['segment']['start']
                pcom_end = feature_dict[pcom_label]['Pcom'][0]['segment']['end']
                pcom_path = find_shortest_path(pcom_start, pcom_end, polydata, int(pcom_label))['path']
                pcom_radii = collect_radius_along_path(polydata, pcom_path, radius_attribute='ce_radius')
                p1_radii = collect_radius_along_path(polydata, p1_path, radius_attribute='ce_radius')

                # if n-percentile of pcom is slightly greater than n-percentile of p1, then it is fetal
                if np.percentile(pcom_radii, percentile) > factor * np.percentile(p1_radii, percentile):
                    fetal_pca = True
    
    return fetal_pca

def run_fetal_detection(cnt_vtp_file: str, variant_dir: str, feature_dir: str, percentile: int = 25, factor: float = 1.05):
    """
    Detects fetal PCA by analyzing and comparing vessel diameters of the P1 and Pcom segments.
    It loads previously extracted features, performs the fetal PCA detection, and updates the variant file
    with the detection results.
    
    Parameters:
    ----------
        cnt_vtp_file (str): Path to the CNT polydata object.
        variant_dir (str): Directory containing cow variant files (cow variant files).
        feature_dir (str): Directory containing cow feature files.
        percentile (int, optional): Percentile of segment radii used for comparing P1 and Pcom diameters
        factor (float, optional): Scaling factor used in fetal PCA detection algorithm (Pcom > factor * P1).
    
    Returns:
    -------
        tuple: A pair of boolean values (r_fetal, l_fetal) indicating whether
               fetal PCA configurations were detected on the right and left sides.
    """
    
    filename = os.path.basename(cnt_vtp_file)
    
    variant_file = os.path.join(variant_dir, filename.replace('.vtp', '.json'))
    feature_file = os.path.join(feature_dir, filename.replace('.vtp', '.json'))
    polydata = get_vtk_polydata_from_file(cnt_vtp_file)

    logger.info(f"Running fetal PCA detection with args:"
                f"\n\t- cnt_vtp_file: {cnt_vtp_file}"
                f"\n\t- variant_dir: {variant_dir}"
                f"\n\t- feature_dir: {feature_dir}"
                f"\n\t- percentile: {percentile}"
                f"\n\t- factor: {factor}")

    with open(variant_file, 'r') as f:
        variant_dict = json.load(f)
    with open(feature_file, 'r') as f:
        feature_dict = json.load(f)

    logger.info(f"Checking fetal PCA on the right side")
    r_fetal = check_for_fetal_pca(feature_dict, polydata, 'right', percentile=percentile, factor=factor)
    logger.info(f"\tfetal R-PCA={r_fetal}")
    logger.info(f"Checking fetal PCA on the left side")
    l_fetal = check_for_fetal_pca(feature_dict, polydata, 'left', percentile=percentile, factor=factor)
    logger.info(f"\tfetal L-PCA={l_fetal}")

    variant_dict['fetal']['R-PCA'] = r_fetal
    variant_dict['fetal']['L-PCA'] = l_fetal

    # Save the updated variant dictionary
    logger.info(f"Saving updated variant dictionary to {variant_file}")
    with open(variant_file, 'w') as f:
        json.dump(variant_dict, f, indent=4)
    
    return r_fetal, l_fetal