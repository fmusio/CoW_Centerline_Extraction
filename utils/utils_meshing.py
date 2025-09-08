import vtk
import numpy as np

from logger import logger


def get_vtk_mesh(nii_mask, do_smoothing=True, pass_band=0.25, add_labels=False, 
                 save_mesh=False, savepath=None, verbose=False):
    """
    Use vtk for meshing a segmentation mask using the marching cube algorithm. 
    Triangulation filter is applied for better triangulation.
    Optionally, apply Taubin smoothing to the mesh.

    Args:
    nii_mask: nibabel.nifti1.Nifti1Image, segmentation mask
    do_smoothing: bool, whether to apply Taubin smoothing to the mesh
    pass_band: float, frequency filtering for Taubin smoothing. Higher values better preserve volume
    add_labels: bool, whether to add cell labels to the mesh
    save_mesh: bool, whether to save the transformed mesh
    savepath: str, path to save the transformed mesh
    verbose: bool, whether to print verbose messages

    Returns:
    mesh: vtk.vtkPolyData, mesh of the segmentation mask
    """

    # Suppress VTK logging messages
    try:
        # For newer VTK versions
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)  # Show only errors
    except AttributeError:
        # Fallback for older VTK versions
        output_window = vtk.vtkOutputWindow.GetInstance()
        output_window.SetGlobalWarningDisplay(0)

    nii_data = nii_mask.get_fdata().astype(np.uint8)
    # Set the dimensions of the VTK image
    dimensions = nii_data.shape
    affine = nii_mask.affine

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(dimensions)

    # Allocate scalars for the VTK image
    vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Loop through the numpy array and set the values in the VTK image
    for z in range(dimensions[2]):
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                vtk_image.SetScalarComponentFromFloat(x, y, z, 0, nii_data[x, y, z])


    # Create a contour filter to generate the mesh
    if verbose:
        logger.debug("Creating contour filter to generate the mesh")
    contour_filter = vtk.vtkContourFilter()
    contour_filter.SetInputData(vtk_image)
    contour_filter.SetValue(0, 0.5)  # Adjust the value based on your segmentation mask

    # Update the contour filter to generate the mesh
    contour_filter.Update()

    # Get the output of the contour filter (the mesh)
    mesh = contour_filter.GetOutput()
    
    # remove the ImageScalars array if it exists
    if mesh.GetPointData().HasArray('ImageScalars'):
        mesh.GetPointData().RemoveArray('ImageScalars')

    # Create a transform to apply the direction cosines to the VTK image
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()

    # Set the affine transformation matrix
    for i in range(4):
        for j in range(4):
            transform_matrix.SetElement(i, j, affine[i, j])

    transform.SetMatrix(transform_matrix)

    # Apply transform to mesh
    if verbose:
        logger.debug(f"Transforming mesh to world coordinates")
        logger.debug(f"Transform matrix: {affine}")
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    mesh_transformed = transform_filter.GetOutput()

    # Add triangle filter to ensure clean triangulation
    if verbose:
        logger.debug("Applying triangle filter to ensure clean triangulation of the mesh")
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh_transformed)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    mesh_triangulated = triangle_filter.GetOutput()

    if do_smoothing:
        # Alternative: use Taubin smoothing which preserves volume better
        if verbose:
            logger.debug(f"Applying Taubin smoothing to the mesh with pass band {pass_band}")
        taubin = vtk.vtkWindowedSincPolyDataFilter()
        taubin.SetInputData(mesh_triangulated)
        taubin.SetNumberOfIterations(20)
        taubin.SetPassBand(pass_band)  # Controls frequency filtering
        taubin.SetBoundarySmoothing(False)
        taubin.SetFeatureEdgeSmoothing(True)
        taubin.SetFeatureAngle(120)  # Only smooth edges with angle < 120°
        taubin.SetNormalizeCoordinates(True)  # Helps preserve volume
        taubin.NonManifoldSmoothingOn()
        taubin.Update()
        mesh = taubin.GetOutput()
    
    else:
        mesh = mesh_triangulated

    if add_labels:
        if verbose:
            logger.debug("Adding labels of underlying mask to the mesh cells")
        affine_inv = np.linalg.inv(affine)
        # Add cell labels from the segmentation file
        # Compute cell centers (in world coordinates)
        cell_centers_filter = vtk.vtkCellCenters()
        cell_centers_filter.SetInputData(mesh)
        cell_centers_filter.Update()
        cell_centers_polydata = cell_centers_filter.GetOutput()
        cell_centers = np.array([cell_centers_polydata.GetPoint(i) for i in range(cell_centers_polydata.GetNumberOfPoints())])
        # Convert cell centers from world coordinates to voxel coordinates using affine inverse
        homogeneous_centers = np.c_[cell_centers, np.ones(cell_centers.shape[0])]
        voxel_coords = (affine_inv @ homogeneous_centers.T).T[:, :3]
        # Round to nearest integer voxel coordinates and clip within volume bounds
        voxel_coords_round = np.round(voxel_coords).astype(np.uint32)
        for dim in range(3):
            voxel_coords_round[:, dim] = np.clip(voxel_coords_round[:, dim], 0, nii_data.shape[dim] - 1)
        # Sample seg_data at voxel coordinates to obtain labels
        cell_labels = nii_data[voxel_coords_round[:, 0],
                            voxel_coords_round[:, 1],
                            voxel_coords_round[:, 2]].copy()


        for i, label in enumerate(cell_labels):
            if label == 0:
                x, y, z = voxel_coords_round[i]
                new_label = get_nonzero_label(x, y, z, nii_data, radius=1)
                cell_labels[i] = new_label

        # Create a new cell data array
        cell_data = vtk.vtkDoubleArray()
        cell_data.SetName('labels')
        cell_data.SetNumberOfComponents(1)
        cell_data.SetNumberOfValues(len(cell_labels))
        
        # Add the label values to the array
        for i, label in enumerate(cell_labels):
            cell_data.SetValue(i, label)
            
        # Add the array to the mesh's cell data
        mesh.GetCellData().AddArray(cell_data)

    if save_mesh:
        if verbose:
            logger.info(f"Saving processed mesh to {savepath}")
        # save transformed mesh
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(savepath)
        writer.SetInputData(mesh)
        writer.Write()

    return mesh

def get_nonzero_label(x, y, z, seg, radius=2):
    """
    For cells with background label, search for the most common non-zero label in a neighborhood 
    of the cell.

    Args:
    x: int, x-coordinate of the cell
    y: int, y-coordinate of the cell
    z: int, z-coordinate of the cell
    seg: np.ndarray, segmentation mask
    radius: int, radius of the neighborhood

    Returns:
    label: int, most common non-zero label in the neighborhood
    """
    x_min = max(x - radius, 0)
    x_max = min(x + radius + 1, seg.shape[0])
    y_min = max(y - radius, 0)
    y_max = min(y + radius + 1, seg.shape[1])
    z_min = max(z - radius, 0)
    z_max = min(z + radius + 1, seg.shape[2])
    neighborhood = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    nonzero = neighborhood[neighborhood > 0]
    if nonzero.size > 0:
        # Return the most common non-zero label in the neighborhood
        return np.bincount(nonzero.astype(int)).argmax()
    else:
        # If no non-zero label is found, increase radius and try again
        if radius < 5:  # Set a maximum radius to prevent excessive searching
            logger.debug(f'increasing radius to {radius + 1} for voxel ({x}, {y}, {z})')
            return get_nonzero_label(x, y, z, seg, radius + 1)
        logger.warning(f'ALERT: no non-zero label found for voxel ({x}, {y}, {z})')
        return 0