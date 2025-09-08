import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nibabel as nib
import pyvista
import numpy as np
import re
import gzip
import json
import pandas as pd
import copy
from utils.utils_meshing import get_vtk_mesh

from logger import logger

def consecutive_pairwise_average_with_ends(input_list):
    """
    Computes the pairwise average of a list of numeric values while preserving the boundary elements.

    Parameters:
    ----------
    input_list (list): A list of numeric values.

    Returns:
    -------
    list: A new list where each element is the average of the corresponding elements in the input list.
          The first and last elements of the input list are preserved.
    """
    if len(input_list) == 1:
        return [input_list[0], input_list[0]]

    result = [(input_list[i-1] + input_list[i]) / 2.0 for i in range(1, len(input_list))]
    result = [input_list[0]] + result + [input_list[-1]]
    return result


def vvg_to_df(vvg_path):
    """
    Generates pandas DataFrames for edges and nodes from a given VVG JSON file.
    The JSON data is expected to have a "graph" key containing "nodes" and "edges".
    
    Parameters:
    -----------
    vvg_path (str): The file path to the VVG JSON file. 

    Returns:
    --------
    tuple: A tuple containing two pandas DataFrames:
        - df_edge: DataFrame with edge information. Columns §lude:
            "pos", "node1", "node2", "minDistToSurface", "maxDistToSurface", 
            "avgDistToSurface", "numSurfaceVoxels", "volume", "nearOtherEdge".
        - df_nodes: DataFrame with node information. Columns include:
            "voxel_pos", "radius".  
    """
    if vvg_path[-3:] == ".gz":
        with gzip.open(vvg_path, "rt") as gzipped_file:
            # Read the decompressed JSON data
            json_data = gzipped_file.read()
            data = json.loads(json_data)

    else:
        f = open(vvg_path)
        data = json.load(f)
        f.close()

    id_col = []
    pos_col = []
    node1_col = []
    node2_col = []
    minDistToSurface_col = []
    maxDistToSurface_col = []
    avgDistToSurface_col = []
    numSurfaceVoxels_col = []
    volume_col = []
    nearOtherEdge_col = []

    node_id_col = []
    node_voxel_col = []
    node_radius_col = []

    for i in data["graph"]["nodes"]:
        node_id_col.append(i["id"])
        node_voxel_col.append(i["voxels_"])
        node_radius_col.append(i["radius"])

    d_nodes = {'id': node_id_col,'voxel_pos' : node_voxel_col, "radius": node_radius_col}
    df_nodes = pd.DataFrame(d_nodes)
    df_nodes.set_index('id')

    for i in data["graph"]["edges"]:
        positions = []
        minDistToSurface = []
        maxDistToSurface = []
        avgDistToSurface = []
        numSurfaceVoxels = []
        volume = []
        nearOtherEdge = []

        try:
            i["skeletonVoxels"]
        except KeyError:
            continue

        id_col.append(i["id"])
        node1_col.append(i["node1"])
        node2_col.append(i["node2"])

        for j in i["skeletonVoxels"]:
            positions.append(np.asarray(j["pos"]))
            minDistToSurface.append(j["minDistToSurface"])
            maxDistToSurface.append(j["maxDistToSurface"])
            avgDistToSurface.append(j["avgDistToSurface"])
            numSurfaceVoxels.append(j["numSurfaceVoxels"])
            volume.append(j["volume"])
            nearOtherEdge.append(j["nearOtherEdge"] )
        
        pos_col.append(positions)
        minDistToSurface_col.append(minDistToSurface)
        maxDistToSurface_col.append(maxDistToSurface)
        avgDistToSurface_col.append(avgDistToSurface)
        numSurfaceVoxels_col.append(numSurfaceVoxels)
        volume_col.append(volume)
        nearOtherEdge_col.append(nearOtherEdge)

    d = {'id': id_col,'pos' : pos_col, "node1" : node1_col, "node2" : node2_col, "minDistToSurface": minDistToSurface_col,"maxDistToSurface":maxDistToSurface_col, "avgDistToSurface":avgDistToSurface_col, "numSurfaceVoxels":numSurfaceVoxels_col, "volume":volume_col,"nearOtherEdge":nearOtherEdge_col }
    df_edge = pd.DataFrame(d)
    df_edge.set_index('id')
    return df_edge, df_nodes

def generate_vtp(voreen_output: str, mlt_mask_file: str, do_mesh_smoothing: bool = True,
                 pass_band: float = 0.25, add_mesh_labels: bool = True):
    """
    Generate VTP files for the segmentation mesh and centerline graph from a Voreen output directory.

    Parameters:
    -----------
    voreen_output (str): Directory containing Voreen output files.
    mlt_mask_file (str): Segmentation mask file.
    do_mesh_smoothing (bool, optional): Whether to smooth the mesh. Default is True.
    pass_band (float, optional): Pass band for mesh smoothing. Default is 0.25.
    add_mesh_labels (bool, optional): Whether to add labels to the mesh. Default is True.

    Returns:
    --------
    tuple: (cnt_path, mesh_path) paths to the centerline graph and segmentation mesh VTP files.
    """
    seg_regex = re.compile(r'.*\_multi.nii.*')
    centerline_regex = re.compile(r'.*\.vvg')

    seg_file = None
    centerline_file = None

    for root, dirs, files in os.walk(voreen_output):
        for file in files:
            if seg_regex.match(file):
                seg_file = os.path.join(voreen_output, file)
            elif centerline_regex.match(file):
                centerline_file = os.path.join(voreen_output, file)

    seg_file_orig = mlt_mask_file

    if seg_file is None:
        raise ValueError("Segmentation file not found in the specified directory.")
    if centerline_file is None:
        raise ValueError("Centerline file not found in the specified directory.")
    
    logger.info(f'Generating vtp file with args:'
                f'\n\t- voreen_output={voreen_output}'
                f'\n\t- mlt_mask_file={mlt_mask_file}'
                f'\n\t- do_mesh_smoothing={do_mesh_smoothing}'
                f'\n\t- pass_band={pass_band}'
                f'\n\t- add_mesh_labels={add_mesh_labels}'
               )

    # segmentation mesh
    nii = nib.load(seg_file_orig)
    affine = nii.affine
    # Compute the inverse
    affine_inv = np.linalg.inv(affine)
    seg_data = nii.get_fdata()
    
    mesh_path = seg_file.replace(".nii.gz", "_mesh.vtp")
    logger.info(f"Generating .vtp mesh from resampled mlt mask {seg_file} with smoothing={do_mesh_smoothing} and adding labels={add_mesh_labels}")
    mesh_structured = get_vtk_mesh(nii, do_smoothing=do_mesh_smoothing, pass_band=pass_band, 
                                   add_labels=add_mesh_labels, save_mesh=True, savepath=mesh_path, verbose=True)
    
    # centerline graph
    logger.info(f"Generating .vtp centerline graph from {centerline_file}")
    cnt_edges_df, cnt_nodes_df = vvg_to_df(centerline_file)
    cnt_coord = np.array([np.array(item_).mean(0) for item_ in cnt_nodes_df['voxel_pos'].values])
    cnt_edges = np.stack([cnt_edges_df['node1'].values,cnt_edges_df['node2'].values], axis=1)
    
    # # cnt_radius = cnt_edges_df['radius'].values
    cnt_lines = cnt_edges_df['pos'].values
    cnt_radii = cnt_edges_df['avgDistToSurface'].values
    node_id = np.arange(len(cnt_coord))
    
    # centerline graph
    cent_edges = []
    cent_radii = []
    cent_labels = []

    # metric graph
    met_coord = copy.deepcopy(cnt_coord)
    met_edges = []
    met_radii = []
    met_labels = []
    for cnt_, edge_, radii_ in zip(cnt_lines, cnt_edges, cnt_radii):
        new_node_id = np.arange(len(node_id), len(node_id)+len(cnt_))
        cnt_coord_ = np.concatenate([cnt_coord, cnt_], axis=0)
        node_id_ = np.concatenate((node_id, new_node_id))
        node_0, node_1 = node_id_[edge_[0]], node_id_[edge_[1]]
        node_pairs = [node_0]+list(new_node_id)+[node_1]
        edge__ = list(map(list, zip(*[node_pairs[:-1], node_pairs[1:]])))

        edge___ = np.array(edge__)
        p1_, p2_ = edge___[:,0], edge___[:,1]
        p1_coord = cnt_coord_[p1_]
        p2_coord = cnt_coord_[p2_]
        p1, p2 = np.zeros(p1_coord.shape), np.zeros(p2_coord.shape)
        # Compute voxel coordinates (i,j,k) from scanner coordinates (x, y, z) using inverse of affine
        # https://nipy.org/nibabel/coordinate_systems.html
        for n, (pt1, pt2) in enumerate(zip(p1_coord, p2_coord)):
            p1[n] = np.dot(affine_inv, np.insert(pt1, 3, 1, axis=0))[:3] 
            p2[n] = np.dot(affine_inv, np.insert(pt2, 3, 1, axis=0))[:3]

        p1 = p1.astype(np.int16)
        p2 = p2.astype(np.int16)

        label_1 = seg_data[p1[:,0],p1[:,1],p1[:,2]]
        label_2 = seg_data[p2[:,0],p2[:,1],p2[:,2]]

        # get point wise max between the two label arrays
        label_ = np.maximum(label_1, label_2)

        radii_ = consecutive_pairwise_average_with_ends(radii_)

        # count the number of consecutive same labels, restart count when it changes
        length_ = []
        intermed_node_idx = []
        met_label_ = []
        met_radii_ = []
        count_ = 0
        met_label__ = []
        met_radii__ = []
        for idx_ in range(1, len(label_)):
            if label_[idx_] == label_[idx_-1]:
                count_ += 1
                met_label__.append(label_[idx_])
                met_radii__.append(radii_[idx_])
            else:
                length_.append(count_)
                intermed_node_idx.append(idx_-1)
                count_ = 0
                met_label_.append(met_label__)
                met_radii_.append(met_radii__)
                met_label__ = []
                met_radii__ = []
            if idx_ == len(label_)-1:
                length_.append(count_)
                met_label_.append(met_label__)
                met_radii_.append(met_radii__)

        if len(length_) > 1:
            length_ = np.array(length_)
            keep_branch = length_ > 20
            keep_node = np.array([True if keep_branch[idx_]==keep_branch[idx_+1]==True else False for idx_ in range(len(keep_branch)-1)])
        else:
            keep_node = np.array([False])
        
        if keep_node.any():
            logger.debug(f'keeping_node {np.array(cnt_)[np.array(intermed_node_idx)][keep_node]}')

        if label_.min() > -1:  # always true
            cnt_coord = cnt_coord_
            node_id = node_id_
            cent_labels.extend(list(label_))
            cent_edges.extend(edge__)
            cent_radii.extend(radii_)

            if keep_node.any():
                keep_node_coord = np.array(cnt_)[np.array(intermed_node_idx)][keep_node]
                new_met_node_id = np.arange(len(met_coord), len(met_coord)+len(keep_node_coord))
                met_coord = np.concatenate([met_coord, keep_node_coord], axis=0)
                met_node_pairs = [node_0]+list(new_met_node_id)+[node_1]
                met_edge_ = list(map(list, zip(*[met_node_pairs[:-1], met_node_pairs[1:]])))
                met_edges.extend(met_edge_)

                met_label__ = met_label_[0]
                met_radii__ = met_radii_[0]
                for idx__, keep_node_ in enumerate(keep_node):
                    if keep_node_:
                        met_labels.extend([np.median(met_label__)])
                        met_radii.extend([np.mean(met_radii__)])
                        met_label__ = []
                        met_radii__ = []
                    else:
                        met_label__.extend(met_label_[idx__+1])
                        met_radii__.extend(met_radii_[idx__+1])
                    if idx__ == len(keep_node)-1:
                        met_label__.extend(met_label_[idx__+1])
                        met_radii__.extend(met_radii_[idx__+1])
                        met_labels.extend([np.median(met_label__)])
                        met_radii.extend([np.mean(met_radii__)])

            else:
                met_edges.append([node_0, node_1])
                met_labels.extend([np.median(label_)])
                met_radii.extend([np.mean(radii_)])

        elif len(label_) > 20: # to include long edge segments
            cnt_coord = cnt_coord_
            node_id = node_id_
            cent_labels.extend(list(label_))
            cent_edges.extend(edge__)
            cent_radii.extend(radii_)

            if keep_node.any():
                keep_node_coord = np.array(cnt_)[np.array(intermed_node_idx)][keep_node]
                new_met_node_id = np.arange(len(met_coord), len(met_coord)+len(keep_node_coord))
                met_coord = np.concatenate([met_coord, keep_node_coord], axis=0)
                met_node_pairs = [node_0]+list(new_met_node_id)+[node_1]
                met_edge_ = list(map(list, zip(*[met_node_pairs[:-1], met_node_pairs[1:]])))
                met_edges.extend(met_edge_)

                met_label__ = met_label_[0]
                met_radii__ = met_radii_[0]
                for idx__, keep_node_ in enumerate(keep_node):
                    if keep_node_:
                        met_labels.extend([np.median(met_label__)])
                        met_radii.extend([np.mean(met_radii__)])
                        met_label__ = []
                        met_radii__ = []
                    else:
                        met_label__.extend(met_label_[idx__+1])
                        met_radii__.extend(met_radii_[idx__+1])
                    if idx__ == len(keep_node)-1:
                        met_label__.extend(met_label_[idx__+1])
                        met_radii__.extend(met_radii_[idx__+1])
                        met_labels.extend([np.median(met_label__)])
                        met_radii.extend([np.mean(met_radii__)])

            else:
                met_edges.append([node_0, node_1])
                met_labels.extend([np.median(label_)])
                met_radii.extend([np.mean(radii_)])

        else:
            logger.debug(f"skipping edge (background), {len(label_)}, {np.unique(label_).shape[0]}")
    
    cent_edges = np.array(cent_edges)
    cent_radii = np.array(cent_radii)
    cent_labels = np.array(cent_labels)
    cent_edges = np.concatenate((np.int32(2 * np.ones((cent_edges.shape[0], 1))), cent_edges), 1)
    cnt_mesh = pyvista.UnstructuredGrid(cent_edges.flatten(), np.array([4] * len(cent_edges)), cnt_coord)
    cnt_mesh.cell_data['labels'] = cent_labels
    cnt_mesh_structured = cnt_mesh.extract_surface().clean()
    cnt_path = seg_file.replace(".nii.gz", "_cnt_graph.vtp")
    logger.info(f'Graph generation complete, saving to {cnt_path}')
    cnt_mesh_structured.save(cnt_path)

    return cnt_path, mesh_path


    