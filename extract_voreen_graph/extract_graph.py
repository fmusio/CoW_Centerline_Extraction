import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.ndimage import binary_dilation
import uuid
import concurrent.futures
import nibabel as nib
import functools
import copy
from extract_voreen_graph.voreen_vesselgraphextraction import extract_vessel_graph

from logger import logger

def extract_graph_features(img_nii: nib.Nifti1Image,
                           image_name: str,
                           output_dir: str,
                           bulge_size: float,
                           voreen_tool_path: str,
                           voreen_temp_dir: str,
                           voreen_cache_dir: str,
                           workspace_file: str,
                           generate_graph_file=True,
                           verbose=False,
                           ):
    """
    Extract graph features from the given image using Voreen.
    This function saves the image in a temporary directory and then calls the 
    Voreen tool to extract the graph features.

    Parameters:
    ----------
    img_nii (nib.Nifti1Image): The input image in NIfTI format.
    image_name (str): The name of the image.
    output_dir (str): The directory where the output files will be saved.
    bulge_size (float): The bulge size for the graph extraction.
    voreen_tool_path (str): The path to the Voreen tool.
    voreen_temp_dir (str): The temporary directory for Voreen.
    voreen_cache_dir (str): The cache directory for Voreen.
    workspace_file (str): The workspace file for Voreen.
    generate_graph_file (bool): Whether to generate the graph file.
    verbose (bool): Whether to run in verbose mode.
    
    Returns:
    -------
    None: The function saves the output files in the specified output directory.
    """


    while True:
        tempdir = f"{voreen_temp_dir}/{str(uuid.uuid4())}/"
        if not os.path.isdir(tempdir):
            break
    os.makedirs(tempdir)
    nii_path = os.path.join(tempdir, f'{image_name}.nii')
    nib.save(img_nii, nii_path)

    logger.info(f'Calling voreen_vesselgraphextraction.extract_vessel_graph...')
    _ = extract_vessel_graph(nii_path, 
        output_dir+"/",
        tempdir,
        voreen_cache_dir,
        bulge_size,
        workspace_file,
        voreen_tool_path,
        name=image_name,
        generate_graph_file=generate_graph_file,
        verbose=verbose
    )

def compute_neighbors(img_data, val):
    """
    Compute the 26-neighbors of a given value in a 3D image using binary dilation.
    This function identifies the neighbors of the specified value in the image data
    and returns a list of unique neighboring values.

    Parameters:
    ----------
    img_data (numpy.ndarray): The input image data.
    val (int): The value for which to find neighbors.

    Returns:
    -------
    list: A list of unique neighboring values.
    """

    mask = img_data == val
    mask = binary_dilation(mask, structure=np.ones((3,3,3))).astype(int)-mask  # 26 connected neighborhood
    mask = img_data*mask
    unique_neighbour = list(np.unique(mask[mask>0.0]))
    return unique_neighbour


def run_graph_extraction(input_path: str, output_dir: str, voreen_tool_path: str, voreen_temp_dir: str, 
                         voreen_cache_dir: str, workspace_file: str, bulge_size: float = 0.5, 
                         remove_connections: bool = True):
    """
    Run Voreen graph extraction for a given skeleton mask file.
    This function loads the mask, optionally removes invalid connections
    and calls the voreen graph extraction routine to generate the centerline graph from the skeleton mask.
    
    Parameters:
    -----------
    input_path (str): Path to the input mask file. 
    output_dir (str): Directory where the output files will be saved.
    voreen_tool_path (str): Path to the Voreen tool used for graph extraction.
    voreen_temp_dir (str): Temporary directory for storing intermediate Voreen files.
    voreen_cache_dir (str): Cache directory to be used by the Voreen tool.
    workspace_file (str): Path to the Voreen workspace file required for graph extraction.
    bulge_size (int, optional): Parameter to specify the bulge size in graph extraction; defaults to 1.
    remove_connections (bool, optional): If True, process the image to remove invalid connections based on 
                                         a predefined adjacency criterion.
    Returns:
    --------
    None: The function saves the output files in the specified output directory.
    """
    
    extension = ".nii.gz" if input_path.endswith(".nii.gz") else "." + input_path.split(".")[-1]
    image_name = os.path.basename(input_path)[:-len(extension)]

    logger.info(f'Extracting Voreen graph with args:'
                f'\n\t- input_path={input_path}'
                f'\n\t- output_dir={output_dir}'
                f'\n\t- voreen_tool_path={voreen_tool_path}'
                f'\n\t- voreen_temp_dir={voreen_temp_dir}'
                f'\n\t- voreen_cache_dir={voreen_cache_dir}'
                f'\n\t- workspace_file={workspace_file}'
                f'\n\t- bulge_size={bulge_size}'
                f'\n\t- remove_connections={remove_connections}')
    
    # Load the skeleton mask
    try:
        logger.info(f"Loading file {input_path}")
        img = nib.load(input_path)
    except Exception as e:
        logger.error(f"Error loading file {input_path}: {e}")
        return None

    # remove invalid connections
    if remove_connections:
        logger.debug("Try: Removing invalid connections")
        
        affine = img.affine
        img_data = img.get_fdata().astype(np.uint8)
        
        unique_val = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0]
        valid_adjacency = np.array([[0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
        [1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
        [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	0],
        [0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
        [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
        [0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1],
        [0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0],
        [0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0]])
        
        ### Check for Connections
        # find all values of neighbors of all the unique values in the 26 connected neighborhood
        # pass partial argument img_data to compute_neighbors function 
        compute_neighbors_ = functools.partial(compute_neighbors, copy.deepcopy(img_data))      
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            neighbors = list(executor.map(compute_neighbors_, unique_val))

        # find the adjacency matrix
        adjacency = np.zeros((len(unique_val), len(unique_val)))
        for i, val in enumerate(unique_val):
            for neighbor in neighbors[i]:
                if neighbor in unique_val:
                    adjacency[i, np.where(unique_val==neighbor)[0][0]] = 1

        invalid_connection = ((adjacency-valid_adjacency)>0).astype(int)
        invalid_edges = np.where(invalid_connection)
        invalid_edges = list(np.array(invalid_edges).T)
        if np.sum(invalid_connection) > 0:
            logger.debug(f"\tRemoving {len(invalid_edges)} invalid connections: {invalid_edges}")
            for tuple_ in invalid_edges:
                mask1 = img_data == unique_val[tuple_[0]]
                mask2 = img_data == unique_val[tuple_[1]]
                mask = (binary_dilation(mask1, structure=np.ones((3,3,3))).astype(int)-mask1).astype(int)  # 26 connected neighborhood
                mask = mask2*mask
                img_data[mask==1] = 0

        # make nifti file
        img = nib.Nifti1Image(img_data.astype(np.uint8), affine)

    # save multi-class nifti file
    seg_path = os.path.join(output_dir, image_name+'_multi' + extension)
    logger.info(f"Saving multi-class mask to {seg_path} for graph extraction")
    nib.save(img, seg_path)

    extract_graph_features(img_nii=img,
                        image_name=image_name,
                        output_dir=output_dir,
                        bulge_size=bulge_size,
                        voreen_tool_path=voreen_tool_path,
                        voreen_temp_dir=voreen_temp_dir,
                        voreen_cache_dir=voreen_cache_dir,
                        workspace_file=workspace_file,
                        generate_graph_file=True,
                        verbose=False,)

    logger.info(f"Voreen graph extraction completed for {image_name}")
    
