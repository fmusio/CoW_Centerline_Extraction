import os
import datetime
import pathlib
from shutil import copyfile
import h5py
import numpy as np


def extract_vessel_graph(volume_path: str,
                         outdir: str,
                         tempdir: str,
                         cachedir:str,
                         bulge_size: float,
                         workspace_file: str,
                         voreen_tool_path: str,
                         name='',
                         generate_graph_file=False,
                         verbose=False):
    """
    Extracts a vessel graph from a volumetric image using a customized Voreen workspace.
    This function updates a Voreen workspace file with the provided file paths and parameters,
    creates a temporary directory for processing, and executes the Voreen tool to extract the vessel
    graph. After the command-line processing, it reads the resulting HDF5 output, applies post-processing
    (rotating and flipping the data), and returns the processed vessel graph as a NumPy array.
    
    Parameters:
    ----------
    volume_path (str): Path to the input volume file (e.g., a NIfTI image).
    outdir (str): Directory where the output files (nodes, edges, graph) will be saved.
    tempdir (str): Temporary directory used for intermediate file creation and processing.
    cachedir (str): Directory to cache intermediate results during processing.
    bulge_size (float): Parameter specifying the minimum bulge size, used to customize the workspace file.
    workspace_file (str): Path to the Voreen workspace file that contains the extraction pipeline configuration.
    voreen_tool_path (str): Path to the directory containing the Voreen command-line tool executable.
    name (str, optional): Optional prefix for naming output files; defaults to an empty string.
    generate_graph_file (bool, optional): Flag indicating whether to generate and retain a separate graph file; defaults to False.
    verbose (bool, optional): If False, minimizes logging output during processing; defaults to False.
    
    Returns:
    -------
    numpy.ndarray: A 2D NumPy array representing the extracted vessel graph after post-processing.
    
    Note:
    -------
    - The function uses system calls to run the Voreen tool, and proper permissions are required.
    - Temporary files and directories are created and removed during execution.
    """
    bulge_size_identifier = f'{bulge_size}'
    bulge_size_identifier = bulge_size_identifier.replace('.','_')

    bulge_path = f'<Property mapKey="minBulgeSize" name="minBulgeSize" value="{bulge_size}"/>'

    bulge_size_identifier = f'{bulge_size}'
    bulge_size_identifier = bulge_size_identifier.replace('.','_')
    edge_path = f'{outdir}{name}_edges.csv'
    node_path = f'{outdir}{name}_nodes.csv'
    graph_path = f'{outdir}{name}_graph.vvg'

    # create temp directory
    temp_directory = os.path.join(tempdir,datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    pathlib.Path(temp_directory).mkdir(parents=True, exist_ok=True)

    voreen_workspace = 'feature-vesselgraphextraction_customized_command_line.vws'
    copyfile(workspace_file,os.path.join(temp_directory,voreen_workspace))

    # Read in the file
    with open(os.path.join(temp_directory,voreen_workspace), 'r') as file :
        filedata = file.read()

    out_path = f'{tempdir}sample.h5'

    # Replace the target string
    filedata = filedata.replace("volume.nii", volume_path)
    filedata = filedata.replace("nodes.csv", node_path)
    filedata = filedata.replace("edges.csv", edge_path)
    filedata = filedata.replace("graph.vvg", graph_path)
    filedata = filedata.replace('<Property mapKey="continousSave" name="continousSave" value="false" /> <Property mapKey="graphFilePath"',
                                f'<Property mapKey="continousSave" name="continousSave" value="{str(generate_graph_file).lower()}" /> <Property mapKey="graphFilePath"')
    filedata = filedata.replace('<Property mapKey="minBulgeSize" name="minBulgeSize" value="3" />', bulge_path)
    filedata = filedata.replace("input.nii", volume_path)
    filedata = filedata.replace("output.h5", out_path)

    # Write the file out again
    with open(os.path.join(temp_directory,voreen_workspace), 'w') as file:
        file.write(filedata)

    workspace_file = os.path.join(os.path.join(os. getcwd(),temp_directory),voreen_workspace)

    absolute_temp_path = os.path.join(tempdir)

    # extract graph and delete temp directory
    os.system(f'cd {voreen_tool_path} ; ./voreentool \
        --workspace {workspace_file} \
        -platform minimal --trigger-volumesaves --trigger-geometrysaves  --trigger-imagesaves \
        --workdir {outdir} --tempdir {tempdir} --cachedir {cachedir}' + ("" if verbose else "--logLevel error >/dev/null 2>&1")
    )
    if generate_graph_file:
        os.rename(graph_path, graph_path.replace(".vvg", ".vvg"))
    with h5py.File(out_path, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        os.system(f"rm -rf '{absolute_temp_path}' 2> /dev/null")
    ret = ds_arr[1]
    ret = np.flip(np.rot90(ret),0)
    return ret