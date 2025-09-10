import numpy as np
from math import hypot
from splipy import curve_factory
import splipy.utils.smooth as sm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from utils.utils_graph_processing import *

from logger import logger

def extract_node_entries(nodes_dict, label, node_names):
    """
    Extract bifurcation, boundary and vessel end points as specified in node_names

    Args:
    nodes_dict: dict, dictionary containing node entries
    label: int, label of the vessel
    node_names: list, list of node names to extract

    Returns:
    points: list, list of extracted node entries
    """
    label = str(label)
    points = []
    for n in node_names: 
        try:
            points.append(nodes_dict[label][n])
        except KeyError:
            points.append(None)
    
    return points

def get_dict_entry(segment, radius, geometry, nan_edges=0):
    """
    Create dictionary entry for segment geometry.
    NOTE: We don't add torsion for now

    Parameters:
    segment (tuple): tuple with segment start and end node and label
    radius (list): List with radius values
    geometry (list): List with geometric property values

    Returns:
    dict: Dictionary with segment geometry
    """
    length, tortuosity, curvature, volume, torsion = geometry

    return {
        'segment': {
            'start': segment[0], 
            'end': segment[1],
            'nan_edges': nan_edges,
        },
        'radius': {
            'mean': np.round(radius[0], 3).astype(float),
            'sd': np.round(radius[1], 3),
            'median': np.round(radius[2], 3),
            'min': np.round(radius[3], 3),
            'q1': np.round(radius[4], 3),
            'q3': np.round(radius[5], 3),
            'max': np.round(radius[6], 3)
        },
        'length': np.round(length, 3),
        'tortuosity': np.round(tortuosity, 3),
        'volume': np.round(volume, 3),
        'curvature': {
            'mean': np.round(curvature[0], 3),
            'sd': np.round(curvature[1], 3),
            'median': np.round(curvature[2], 3),
            'min': np.round(curvature[3], 3),
            'q1': np.round(curvature[4], 3),
            'q3': np.round(curvature[5], 3),
            'max': np.round(curvature[6], 3)
        }
    }

def reorder_dict(geometry_dict):
    """
    reorder dictionary with specified labelmap and segment order.

    Parameters:
    geometry_dict (dict): Dictionary with segment geometry

    Returns:
    new_dict (dict): Dictionary with restructured segments
    """
    labelmap = {'1': 'BA', '2': ['R-PCA', 'R-P1', 'R-P2'], '3': ['L-PCA', 'L-P1', 'L-P2'], '4': ['R-ICA', 'R-C6','R-C7'], '5': 'R-MCA', 
                '6': ['L-ICA', 'L-C6', 'L-C7'], '7': 'L-MCA', '8': 'R-Pcom', '9': 'L-Pcom', '10': 'Acom', '11': ['R-ACA', 'R-A1', 'R-A2'], 
                '12': ['L-ACA', 'L-A1', 'L-A2'], '15': '3rd-A2'}
    new_dict = {}
    for label in labelmap:
        if isinstance(labelmap[label], str):
            if labelmap[label] in geometry_dict:
                name = labelmap[label]
                if name.startswith('R-') or name.startswith('L-'):
                    name = name[2:]
                new_dict[label] = {name: geometry_dict[labelmap[label]]}

        elif isinstance(labelmap[label], list):
            new_dict[label] = {}
            for l in labelmap[label]:
                if l in geometry_dict:
                    name = l
                    if name.startswith('R-') or name.startswith('L-'):
                        name = name[2:]
                    
                    new_dict[label][name] = geometry_dict[l]
            # delete if new_dict[label] is empty
            if not new_dict[label]:
                del new_dict[label]

    return new_dict

def get_point_coords_along_path(path, polydata):
    """
    Given a path of edges, return the coordinates of the points along the path

    Parameters:
    path (list): List of edges
    polydata (vtkPolyData): VTK PolyData object

    Returns:
    coords (list): List of coordinates of the points along the path
    """
    points = polydata.GetPoints()
    pointIds_path = [p[0] for p in path] + [path[-1][1]]
    coords = []
    for id in pointIds_path:
        coords.append(points.GetPoint(id))

    return coords

def find_endpoint_for_fixed_length(path, polydata, length_threshold=10, stop_at_bif_point=True):
    """
    Find endpoint of a segment that corresponds to fixed length along the path. There is the option 
    to stop early at a bifurcation point along the path (if there is any, e.g. A1 fenestration). 

    Parameters:
    path (list): List of edges
    polydata (vtkPolyData): VTK PolyData object
    length_threshold (float): Fixed length along the path
    stop_at_bif_point (bool): Stop at bifurcation point

    Returns:
    endpoint (int): Endpoint of the segment
    """
    coords_points = get_point_coords_along_path(path, polydata)
    length = 0
    stopping_index = None
    for i in range(len(coords_points)-1):
        length_before = length
        length += hypot(coords_points[i+1][0] - coords_points[i][0], coords_points[i+1][1] - coords_points[i][1], coords_points[i+1][2] - coords_points[i][2])
        if length >= length_threshold:
            if np.abs(length - length_threshold) < np.abs(length_before - length_threshold):
                stopping_index = i
            else:
                stopping_index = i - 1
            break
        
    if stop_at_bif_point:
        edge_list, _ = get_edge_list(polydata)
        pointIds_path = [p[1] for p in path[:stopping_index+1]]
        for id in pointIds_path:
            deg = get_point_degree(id, edge_list)
            if deg > 2:
                stopping_index = pointIds_path.index(id)
                break
    
    return path[stopping_index][1]

def compute_radius_along_path(polydata, path, radius_attribute='ce_radius', nan_threshold=0.5):
    """
    Given a path of edges, compute radius statistics along path.

    Parameters:
    polydata (vtkPolyData): VTK PolyData object
    path (list): List of edges
    radius_attribute (str): Attribute name for radius, ('ce_radius', 'avg_radius', 'min_radius', 'max_radius')
    nan_threshold (float): Threshold for NaN values in radius statistics

    Returns:
    mean, std, median, min, max (float): Statistics of the radius along the path
    """
    avg_radius = polydata.GetCellData().GetArray(radius_attribute)

    _, cell_ids_cow = get_edge_list(polydata)

    cellIds_path = []
    for edge in path:
        cellIds_path.append(get_cellId_for_edge(edge, polydata))
    
    nan_edges = 0
    avg_radii = []
    for id in cellIds_path:
        assert cell_ids_cow[id] == id, "Cell ID does not match!"
        avg_radii.append(avg_radius.GetValue(id))
        if np.isnan(avg_radius.GetValue(id)):
            nan_edges += 1

    if nan_edges/len(avg_radii) < nan_threshold:
        mean, std, median, min, q1, q3, max = compute_statistics(avg_radii, treat_nan='ignore')
    else:
        logger.warning(f'\tALERT: more than {nan_threshold*100}% NaN values in radius along path!')
        mean, std, median, min, q1, q3, max = compute_statistics(avg_radii, treat_nan='none')
    
    return (mean, std, median, min, q1, q3, max), nan_edges

def compute_geometry_along_path(polydata, path, radius_attribute='avg_radius', factor_num_points=2, 
                                comp_tor=True, num_points_torsion=1000, smooth=True, plot=False):
    """
    Given a path of edges, compute segment geometry: radius, length, tortuosity, curvature, volume, torsion
    
    Parameters:
    polydata (vtkPolyData): VTK PolyData object
    path (list): List of edges
    radius_attribute (str): Attribute name for radius, ('ce_radius', 'avg_radius', 'min_radius', 'max_radius')
    factor_num_points (int): Factor by which to multiply centerline points for sampling from curve
    comp_tor (bool): Compute torsion
    num_points_torsion (int): Number of points for torsion computation
    smooth (bool): Smooth the curve
    plot (bool): Plot the curve

    Returns:
    length (float): Length of the segment
    tortuosity (float): Tortuosity of the segment
    curvature (tuple): Mean, std, median, min, max of curvature
    volume (float): Volume of the segment
    torsion (tuple): Mean, std, median, min, max of torsion
    """
    radius = polydata.GetCellData().GetArray(radius_attribute)

    cellIds_path = []
    for edge in path:
        cellIds_path.append(get_cellId_for_edge(edge, polydata))
    
    point_coords = get_point_coords_along_path(path, polydata)
        
    radii = []
    for id in cellIds_path:
        radii.append(radius.GetValue(id))

    # TODO: treat NaN values
    volume = compute_volume(point_coords, radii)

    # Work with parametrized curve
    curve = return_curve(point_coords, factor_num_points, smooth, plot)

    # NOTE: torsion converges to stable value only for large number of sample points
    num_points = factor_num_points * len(point_coords)
    curvature, torsion, length, tortuosity = compute_curve_features(curve, num_points, comp_tor, num_points_torsion)

    return length, tortuosity, curvature, volume, torsion

def compute_statistics(list_of_values, treat_nan='none'):
    """
    Compute statistics of an input list from start to end.

    Parameters:
    list_of_values (list): List of values
    treat_nan (str): How to treat NaN values, options are 'none' (default), 'ignore', or 'zero'.

    Returns:
    mean (float): Mean of the values
    std (float): Standard deviation of the values
    median (float): Median of the values
    min (float): Minimum value
    q1 (float): First quartile (25th percentile)
    q3 (float): Third quartile (75th percentile)
    max (float): Maximum value
    """
    values = np.array(list_of_values)
    
    if treat_nan == 'ignore':
        mean = np.nanmean(values)
        median = np.nanmedian(values)
        min = np.nanmin(values)
        max = np.nanmax(values)
        std = np.nanstd(values)
        q1 = np.nanquantile(values, 0.25)
        q3 = np.nanquantile(values, 0.75)
    elif treat_nan == 'zero':
        values = np.nan_to_num(values, nan=0.0)
        mean = np.mean(values)
        median = np.median(values)
        min = np.min(values)
        max = np.max(values)
        std = np.std(values)
        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
    else:  # 'none' - default behavior
        mean = np.mean(values)
        median = np.median(values)
        min = np.min(values)
        max = np.max(values)
        std = np.std(values)
        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
        
    return mean, std, median, min, q1, q3, max

def compute_curve_length(list_of_points):
    """
    Compute curve length from a list of points (3D coords) by summing all the distances 
    between consecutive points

    Parameters:
    list_of_points (list): List of 3D coordinates

    Returns:
    curve_length (float): Length of the
    """
    centerline_points = np.array(list_of_points)
    return np.sum(np.linalg.norm(np.diff(centerline_points, axis=0), axis=1))

def compute_tortuosity(curve_length, start_pt, end_pt):
    """
    Compute tortuosity of a curve.

    Parameters:
    curve_length (float): Length of the curve
    start_pt (tuple): Start point of the curve
    end_pt (tuple): End point of the curve

    Returns:
    tortuosity (float): Tortuosity of the curve
    """
    euclidean_distance = hypot(start_pt[0] - end_pt[0], start_pt[1] - end_pt[1], start_pt[2] - end_pt[2])
    # Definition used by VMTK!
    # https://discourse.slicer.org/t/how-the-definition-of-tortuosity-in-vmtk/12503
    tortuosity = curve_length / euclidean_distance - 1
    return tortuosity

def return_tangent_vectors(list_of_points):
    """
    Compute tangent vectors at each point using the finite differences method.

    Parameters:
    list_of_points (list): List of 3D coordinates

    Returns:
    normalized_tangent_vectors (np.array): Array of normalized tangent vectors
    """
    control_points = np.array(list_of_points)
    curve = curve_factory.polygon(control_points)
    knots = curve.knots()[0]
    x_points = control_points[:, 0]
    y_points = control_points[:, 1]
    z_points = control_points[:, 2]

    # Compute finite differences for each coordinate
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    dz = np.diff(z_points)
    dt = np.diff(knots)

    # Approximate derivatives (slopes)
    x_derivative = dx / dt
    y_derivative = dy / dt
    z_derivative = dz / dt

    tangent_x, tangent_y, tangent_z = np.zeros(len(control_points)), np.zeros(len(control_points)), np.zeros(len(control_points))

    tangent_x[0], tangent_y[0], tangent_z[0] = x_derivative[0], y_derivative[0], z_derivative[0]
    tangent_x[-1], tangent_y[-1], tangent_z[-1] = x_derivative[-1], y_derivative[-1], z_derivative[-1]

    for i in range(2, len(control_points)-1):
        tangent_x[i] = (x_derivative[i-1] + x_derivative[i]) / 2
        tangent_y[i] = (y_derivative[i-1] + y_derivative[i]) / 2
        tangent_z[i] = (z_derivative[i-1] + z_derivative[i]) / 2

    # Compute tangent vectors
    tangent_vectors = np.column_stack((tangent_x, tangent_y, tangent_z))
    tangent_magnitudes = np.linalg.norm(tangent_vectors, axis=1)
    normalized_tangent_vectors = tangent_vectors / (tangent_magnitudes[:, np.newaxis]+10e-3)

    return normalized_tangent_vectors

def return_curve(list_of_points, factor_num_pts=2, smooth=True, plot=False):
    """
    Return a curve object using splipy and the cubic_curve method.

    Parameters:
    list_of_points (list): List of 3D coordinates
    factor_num_pts (int): Factor by which to multiply centerline points for sampling from curve
    smooth (bool): Smooth the curve
    plot (bool): Plot the curve

    Returns:
    curve (splipy.curve.Curve): Curve object
    """
    control_points = np.array(list_of_points)
    
    # if number of points is less than 4, add midpoints until we have at least 4 points
    while len(control_points) < 4:
        new_points = []
        for i in range(len(control_points) - 1):
            new_points.append(control_points[i])
            midpoint = (control_points[i] + control_points[i+1]) / 2.0
            new_points.append(midpoint)
        new_points.append(control_points[-1])
        control_points = np.array(new_points)
    curve = curve_factory.cubic_curve(control_points)

    if smooth:
        # Smooth the control points
        # NOTE: Curvature and torsion are not preserved after smoothing
        sm.smooth(curve)

    if plot:
        # Evaluate the curve at multiple points
        num_pts = factor_num_pts * len(control_points)
        t_values = np.linspace(curve.start()[0], curve.end()[0], num=num_pts)
        curve_points = curve(t_values)

        # Plot the 3D curve
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], label="3D Curve")
        ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], color="red", label="Control Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D B-spline Curve with Control Points")
        plt.legend()
        plt.show()
    return curve

def compute_curve_features(curve, num_points=100, comp_tor=True, num_points_torsion=10000):
    """
    Compute curve features of the parametrized curve of a vessel segment: curvature, torsion, tortuosity, length
    NOTE: Torsion needs a lot of points to converge to a stable value

    Parameters:
    curve (splipy.curve.Curve): Curve object
    num_points (int): Number of points for curvature computation
    comp_tor (bool): Compute torsion
    num_points_torsion (int): Number of points for torsion computation

    Returns:
    curvature (tuple): Mean, std, median, min, max of curvature
    torsion (tuple): Mean, std, median, min, max of torsion
    length (float): Length of the curve
    tortuosity (float): Tortuosity of the curve
    """
    # Compute curvature and torsion along the curve
    t_values = np.linspace(curve.start()[0], curve.end()[0], num=num_points)
    curve_points = curve(t_values)
    length = curve.length(curve.start()[0], curve.end()[0])
    curvature = curve.curvature(t_values)

    mean_curvature, std_curvature, median_curvature, min_curvature, q1_curvature, q3_curvature, max_curvature = compute_statistics(curvature)
    if comp_tor:
        
        t_values_torsion = np.linspace(curve.start()[0], curve.end()[0], num=num_points_torsion)
        torsion = curve.torsion(t_values_torsion)
        mean_torsion, std_torsion, median_torsion, min_torsion, q1_torsion, q3_torsion, max_torsion = compute_statistics(torsion)
    else:
        mean_torsion, std_torsion, median_torsion, min_torsion, q1_torsion, q3_torsion, max_torsion = None, None, None, None, None, None, None

    # Calculate straight-line distance (distance between start and end points)
    start_point = curve_points[0]
    end_point = curve_points[-1]
    straight_line_distance = np.linalg.norm(end_point - start_point)

    # Compute tortuosity ratio
    # Same definition as VMTK Slicer: https://discourse.slicer.org/t/how-the-definition-of-tortuosity-in-vmtk/12503
    tortuosity = length / straight_line_distance - 1
    
    return (mean_curvature, std_curvature, median_curvature, min_curvature, q1_curvature, q3_curvature, max_curvature), (mean_torsion, std_torsion, median_torsion, min_torsion, q1_torsion, q3_torsion, max_torsion), length, tortuosity


def compute_volume(list_of_coordinates, list_of_radii, nan_threshold=0.1):
    """
    Compute volume of a segment. 
    The volume is computed as the sum of the volumes of cylinders defined by the points and radii.

    Args:
    list_of_coordinates: list, list of 3D coordinates of the points along the segment
    list_of_radii: list, list of radii corresponding to the points
    nan_threshold: float, threshold for the ratio of NaN values in the radii list to decide how to handle NaNs

    Returns:
    volume: float, volume of the segment
    """
    assert len(list_of_coordinates)-1 == len(list_of_radii), "Number of radii does not match number of points!"

    # Calculate ratio of NaN values
    nan_count = np.sum(np.isnan(list_of_radii))
    nan_ratio = nan_count / len(list_of_radii)

    volume = 0
    if nan_ratio < nan_threshold:
        # If few NaNs, ignore them in the calculation
        for n, (pt1, pt2) in enumerate(zip(list_of_coordinates, list_of_coordinates[1:])):
            if not np.isnan(list_of_radii[n]):
                length = hypot(pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2])
                volume += np.pi * list_of_radii[n]**2 * length
    else:
        # If many NaNs, compute as in the original code
        for n, (pt1, pt2) in enumerate(zip(list_of_coordinates, list_of_coordinates[1:])):
            length = hypot(pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2])
            volume += np.pi * list_of_radii[n]**2 * length
    
    return volume

def compute_normal_vector(pt1, pt2):
    """
    Compute normal vector between two points

    Args:
    pt1: tuple, point 1
    pt2: tuple, point 2

    Returns:
    nvec: np.array, normal vector
    """
    vec = np.array(pt2) - np.array(pt1)
    nvec = vec / np.linalg.norm(vec)
    return nvec

def compute_angle(v1, v2):
    """
    Compute angle between normal vectors

    Args:
    v1: np.array, normal vector 1
    v2: np.array, normal vector 2

    Returns:
    angle: float, angle between normal
    """
    # angle v1, v2 in degree
    angle = np.arccos(np.dot(v1, v2)) * 180 / np.pi
    if np.isnan(angle):
        # If angle is NaN, return 0
        return 0.0
    else:
        return angle

def get_distances_to_boundaries(bifurcation, boundary1, boundary2):
    """
    Compute distances from bifurcation point to boundary points.

    Args:
    bifurcation: list, list of bifurcation points (node ditionary entry)
    boundary1: list, list of boundary points (node dictionary entry)
    boundary2: list, list of boundary points (node dictionary entry)

    Returns:
    distances: list, list of euclidean distances
    """
    distances = []
    for i in range(len(boundary1)):
        point1 = bifurcation[0]['coords']
        point2 = boundary1[i]['coords']
        distances.append(hypot(point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]))
    for i in range(len(boundary2)):
        point1 = bifurcation[0]['coords']
        point2 = boundary2[i]['coords']
        distances.append(hypot(point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]))
    
    return distances

def compute_ratios(r_p, r_c1, r_c2):
    """
    ratios of parent and child vessels
    See https://www.researchgate.net/figure/Scaling-principle-governing-vessel-diameters-in-a-bifurcation-For-a-large-side-branch_fig1_51097526
        Finet: ratio=0.678
    
    Args:
    r_p: float, radius of parent vessel
    r_c1: float, radius of first child vessel
    r_c2: float, radius of second child vessel

    Returns:
    ratio_finet: float, ratio of parent and child vessel
    ratio_pc1: float, ratio of parent and first child vessel
    ratio_pc2: float, ratio of parent and second child vessel
    """

    return r_p / (r_c1 + r_c2), r_c1 / r_p, r_c2 / r_p, r_c1 / r_c2, r_p**2 / (r_c1**2 + r_c2**2)

def compute_bifurcation_exponent(r_p, r_c1, r_c2, x_min=0, x_max=7, x_step=0.05):
    """
    find bifurcation exponent x s.t. r_p**x = r_1**x + r_2**x
    see https://www.researchgate.net/figure/Scaling-principle-governing-vessel-diameters-in-a-bifurcation-For-a-large-side-branch_fig1_51097526
        Murray: x=3
        HK: x=7/3
        Flow conservation: x=2
    
    Args:
    r_p: float, radius of parent vessel
    r_c1: float, radius of first child vessel
    r_c2: float, radius of second child vessel
    x_min: float, minimum value for exponent
    x_max: float, maximum value for exponent
    x_step: float, step size for exponent

    Returns:
    opt: list, list of optimal exponents
    """

    dist = np.inf
    x_opt = 0
    for x in np.arange(x_min, x_max+x_step, x_step):
        dist_new = np.abs(r_p**x - (r_c1**x + r_c2**x))
        if dist_new < dist:
            dist = dist_new
            x_opt = x
    
    return x_opt

def get_normal_vectors(bif_id, pointId_parent, pointId_child1, pointId_child2, polydata, 
                       nr_of_points_for_avg=1, start_parent=None, end_child1=None, end_child2=None):
    """
    Get normal vectors at bifurcation point.

    Args:
    bif_id: int, id of the bifurcation point
    pointId_parent: int, id of the parent vessel point
    pointId_child1: int, id of the first child vessel point
    pointId_child2: int, id of the second child vessel point
    polydata: vtkPolyData, polydata of the vessel
    nr_of_points_for_avg: int, number of points to average for normal vector computation
                            (starting at pointIds given above)
    start_parent: int, start point for parent vessel
    end_child1: int, end point for first child vessel
    end_child2: int, end point for second child vessel

    Returns:
    parent_nv: list, list of normal vectors for parent vessel
    """
    points_cow = polydata.GetPoints()
    bif_point = points_cow.GetPoint(bif_id)
    if nr_of_points_for_avg == 1:
        point_parent = points_cow.GetPoint(pointId_parent)
        point_child1 = points_cow.GetPoint(pointId_child1)
        point_child2 = points_cow.GetPoint(pointId_child2)
        parent_nv = compute_normal_vector(bif_point, point_parent)
        child1_nv = compute_normal_vector(bif_point, point_child1)
        child2_nv = compute_normal_vector(bif_point, point_child2)
        
    else:
        parent_nv = np.zeros(3)
        child1_nv = np.zeros(3)
        child2_nv = np.zeros(3)
        max_dist = [nr_of_points_for_avg]
        if start_parent is not None:
            max_dist.append(pointId_parent - start_parent + 1)
        if end_child1 is not None:
            max_dist.append(end_child1 - pointId_child1 + 1)
        if end_child2 is not None:
            max_dist.append(end_child2 - pointId_child2 + 1)
        nr_of_points_for_avg = min(max_dist)
        for i in range(nr_of_points_for_avg):
            point_parent = points_cow.GetPoint(pointId_parent-i)
            point_child1 = points_cow.GetPoint(pointId_child1+i)
            point_child2 = points_cow.GetPoint(pointId_child2+i)
            parent_nv += compute_normal_vector(bif_point, point_parent)
            child1_nv += compute_normal_vector(bif_point, point_child1)
            child2_nv += compute_normal_vector(bif_point, point_child2)
        parent_nv /= np.linalg.norm(parent_nv)
        child1_nv /= np.linalg.norm(child1_nv)
        child2_nv /= np.linalg.norm(child2_nv)
        
    return parent_nv, child1_nv, child2_nv

def compute_branching_plane_deviation(parent_nv, child1_nv, child2_nv):
    """
    Compute the deviation of parent vessel from the branching plane defined by the child vessels.
    The branching plane is defined by the normal vector of the child vessels.

    Args:
    parent_nv: np.array, normal vector of the parent vessel
    child1_nv: np.array, normal vector of the first child vessel
    child2_nv: np.array, normal vector of the second child vessel

    Returns:
    branching_plane_deviation: float, deviation of the branching plane
    """
    # Compute the cross product of the parent and first child normal vectors
    cross_product = np.cross(child1_nv, child2_nv)
    # Normalize the cross product to get the normal vector of the branching plane
    cross_product = cross_product / np.linalg.norm(cross_product)
    # assert np.round(np.linalg.norm(cross_product), 3) == 1 and np.round(np.linalg.norm(parent_nv), 3) == 1, "Normal vectors are not normalized!"
    
    # Compute the angle between the cross product and the plane (cosine gives angle between normal and vector, i.e. 90 - angle)
    angle = np.arcsin(np.abs(np.dot(cross_product, parent_nv))) * 180 / np.pi
    
    return angle


def compute_angles(bif_id, pointId_parent, pointId_child1, pointId_child2, polydata, 
                    nr_of_points_for_avg=1, start_parent=None, end_child1=None, end_child2=None):
    """
    compute bifurcation angles between parent and child vessels

    Args:
    bif_id: int, id of the bifurcation point
    pointId_parent: int, id of the parent vessel point
    pointId_child1: int, id of the first child vessel point
    pointId_child2: int, id of the second child vessel point
    polydata: vtkPolyData, polydata of the vessel
    nr_of_points_for_avg: int, number of points to average for normal vector computation
                            (starting at pointIds given above)  
    start_parent: int, start point for parent vessel
    end_child1: int, end point for first child vessel
    end_child2: int, end point for second child vessel

    Returns:
    angle_parent_child1: float, angle between parent and first child vessel
    angle_parent_child2: float, angle between parent and second child vessel
    angle_child1_child2: float, angle between first and second child vessel
    """
    parent_nv, child1_nv, child2_nv = get_normal_vectors(bif_id, pointId_parent, pointId_child1, 
                                                         pointId_child2, polydata, nr_of_points_for_avg,
                                                         start_parent, end_child1, end_child2)
    
    angle_parent_child1 = compute_angle(parent_nv, child1_nv)
    angle_parent_child2 = compute_angle(parent_nv, child2_nv)
    angle_child1_child2 = compute_angle(child1_nv, child2_nv)

    # branching_plane_deviation = compute_branching_plane_deviation(parent_nv, child1_nv, child2_nv)

    return angle_parent_child1, angle_parent_child2, angle_child1_child2

def compute_radii(pointId_parent, pointId_child1, pointId_child2, polydata, radius_attribute='avg_radius',
                  nr_of_edges_for_avg=5, start_parent=None, end_child1=None, end_child2=None):
    """
    Compute radii for ratio compututation

    Args:
    pointId_parent: int, id of the parent vessel point
    pointId_child1: int, id of the first child vessel point
    pointId_child2: int, id of the second child vessel point
    polydata: vtkPolyData, polydata of the vessel
    radius_attribute (str): Attribute name for radius, ('ce_radius', 'avg_radius', 'min_radius', 'max_radius')
    nr_of_edges_for_avg: int, number of edges to average for radius computation
    start_parent: int, start point for parent vessel
    end_child1: int, end point for first child vessel
    end_child2: int, end point for second child vessel

    Returns:
    radius_parent: float, radius of parent vessel
    radius_child1: float, radius of first child vessel
    radius_child2: float, radius of second child vessel 
    """
    radii_parent = []
    radii_child1 = []
    radii_child2 = []

    avg_radii = polydata.GetCellData().GetArray(radius_attribute)

    if nr_of_edges_for_avg > 1:
        max_dist = [nr_of_edges_for_avg]
        if start_parent is not None:
            max_dist.append(pointId_parent - start_parent)
        if end_child1 is not None:
            max_dist.append(end_child1 - pointId_child1)
        if end_child2 is not None:
            max_dist.append(end_child2 - pointId_child2)
        # remove negative values
        max_dist = [d for d in max_dist if d > 0]
        nr_of_edges_for_avg = min(max_dist)
    
    stop_parent = False
    stop_child1 = False
    stop_child2 = False
    for i in range(nr_of_edges_for_avg):
        if stop_parent:
            pass
        elif start_parent == pointId_parent - i:
            stop_parent = True
            if len(radii_parent) == 0:
                cellId_parent = get_cellId_for_edge((pointId_parent-i, pointId_parent-i+1), polydata)
                radii_parent.append(avg_radii.GetValue(cellId_parent))
        else:
            cellId_parent = get_cellId_for_edge((pointId_parent-1-i, pointId_parent-i), polydata)
            radii_parent.append(avg_radii.GetValue(cellId_parent))
        if stop_child1:
            pass
        elif start_parent == pointId_child1 + i:
            stop_child1 = True
            if len(radii_child1) == 0:
                cellId_child1 = get_cellId_for_edge((pointId_child1+i, pointId_child1+1+i), polydata)
                radii_child1.append(avg_radii.GetValue(cellId_child1))
        else:
            cellId_child1 = get_cellId_for_edge((pointId_child1+i, pointId_child1+i+1), polydata)
            radii_child1.append(avg_radii.GetValue(cellId_child1))
        if stop_child2:
            pass
        elif start_parent == pointId_child2 + i:
            stop_child2 = True
            if len(radii_child2) == 0:    
                cellId_child2 = get_cellId_for_edge((pointId_child2+i, pointId_child2+1+i), polydata)
                radii_child2.append(avg_radii.GetValue(cellId_child2))
        else:
            cellId_child2 = get_cellId_for_edge((pointId_child2+i, pointId_child2+i+1), polydata)
            radii_child2.append(avg_radii.GetValue(cellId_child2))

    radius_parent = np.mean(radii_parent)
    radius_child1 = np.mean(radii_child1)
    radius_child2 = np.mean(radii_child2)

    return radius_parent, radius_child1, radius_child2

def check_for_nan(path, end_id, polydata, buffer=5, radius_attribute='ce_radius'):
    """
    Check if the path contains NaN values from start to end_id with a buffer.
    If so, return True, otherwise return False.

    Args:
    path (list): List of edges
    end_id (int): Endpoint of the path
    buffer (int): Buffer to check for NaN values

    Returns:
    bool: True if NaN values are found, False otherwise
    """
    radii = polydata.GetCellData().GetArray(radius_attribute)
    ids = [edge[1] for edge in path]
    end_idx = ids.index(end_id) if end_id in ids else -1
    if end_idx == -1:
        return False  # end_id not found in path
    else:
        # add buffer to end_ids
        end_idx = min(end_idx + buffer, len(path) - 1)
    for edge in path[:end_idx+1]:
        cellId = get_cellId_for_edge(edge, polydata)
        rad = radii.GetValue(cellId)
        if np.isnan(rad):
            return True
        

