import vtk
import numpy as np
import networkx as nx
from collections import Counter
import copy

from logger import logger

def get_array_type(array):
    """
    Get array type of vtk array

    Args:
    array: vtkArray, vtk array
    
    Returns:
    array_type: str, array type
    """
    if array.IsA("vtkStringArray"):
        return "vtkStringArray"
    elif array.IsA("vtkFloatArray"):
        return "vtkFloatArray"
    elif array.IsA("vtkDoubleArray"):
        return "vtkDoubleArray"
    elif array.IsA("vtkIntArray"):
        return "vtkIntArray"
    elif array.IsA("vtkIdTypeArray"):
        return "vtkIdTypeArray"
    # Add more checks as needed
    else:
        return "Unknown Array Type"
    
def get_vtk_polydata_from_file(file_path):
    """
    Read a vtk file and return the polydata

    Args:
    file_path: str, path to file

    Returns:
    polydata: vtkPolyData, polydata object
    """
    # Load your VTK PolyData 
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def write_vtk_polydata_to_file(polydata, file_path):
    """
    Write a vtk polydata to a file

    Args:
    polydata: vtkPolyData, polydata object
    file_path: str, path to file

    Returns:
    None
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(file_path)
    writer.Write()

def delete_cells(cells_to_remove, polydata):
    """
    Delete a list of cells from a polydata object.

    Args:
    cells_to_remove: list of int, list of cell IDs to remove
    polydata: vtkPolyData, polydata object

    Returns:
    polydata: vtkPolyData, polydata object with cells removed
    """
    
    for cell in cells_to_remove:
        polydata.DeleteCell(cell)
    polydata.RemoveDeletedCells()
    polydata.Modified()

    return clean_polydata(polydata)

def clean_polydata(polydata):
    """
    Clean a vtk polydata

    Args:
    polydata: vtkPolyData, polydata object

    Returns:
    clean: vtkPolyData, cleaned polydata
    """
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(polydata)
    clean_filter.SetTolerance(0.001)  
    clean_filter.Update()
    return clean_filter.GetOutput()

def get_label_array(polydata):
    """
    Get the label array from a polydata object.

    Args:
    polydata: vtkPolyData, polydata object

    Returns:
    labels: np.array, array of labels
    """
    labels = polydata.GetCellData().GetArray('labels') 
    return np.array(labels)

def get_pointIds_for_cell(cell_id, polydata):
    """
    Given a cell ID, get the point IDs that belong to that cell. 
    Each cell is defined by two points.

    Args:
    cell_id: int, cell id
    polydata: vtkPolyData, polydata object

    Returns:
    points: tuple, tuple of point ids
    """
    cell_points = vtk.vtkIdList()
    polydata.GetCellPoints(cell_id, cell_points)
    num_points = cell_points.GetNumberOfIds()
    # assert num_points == 2, f'Cell {cell_id} has {num_points} points, expected 2'
    if num_points == 2:
        points = (cell_points.GetId(0), cell_points.GetId(1))
    elif num_points == 1:
        points = (cell_points.GetId(0), cell_points.GetId(0))
    else:
        raise ValueError(f'Cell {cell_id} has {num_points} points, expected 2')
    return points

def get_edge_list(polydata, label=None):
    """
    Get the edge list from a polydata object.
    This function returns the edges - defined as tuples (pt1, pt2) - and the corresponding cell IDs 
    for certain labels. If label is None, all edges of the polydata object are returned.

    Args:
    polydata: vtkPolyData, polydata object
    label: int or list, label to filter edges for

    Returns:
    edge_list: list of tuples, list of edges
    """
    labels_arr = get_label_array(polydata)
    if label is None:
        cell_ids = np.where(labels_arr > -1)[0]
    elif isinstance(label, list):
        cell_ids = [i for l in label for i in np.where(labels_arr == l)[0]]
    else:
        cell_ids = np.where(labels_arr == label)[0]
    
    edge_list = []
    for l in cell_ids:
        ids = get_pointIds_for_cell(l, polydata)
        edge_list.append(ids)
       
    return edge_list, cell_ids

def get_cellId_for_edge(edge, polydata):
    """
    Get the cell ID for a certain edge. The edge is given by a tuple of point IDs.

    Args:
    edge: tuple, edge defined by two point IDs
    polydata: vtkPolyData, polydata object

    Returns:
    cell_id: int, cell ID
    """
    edge_list, cell_ids = get_edge_list(polydata)
    for e, id in zip(edge_list, cell_ids):
        if set(e) == set(edge):
            return id
        
def get_cellIds_for_point(point_id, polydata):
    """
    Get the cell IDs for a certain point. I.e. all cells that adjoin the given point.

    Args:
    point_id: int, point ID
    polydata: vtkPolyData, polydata object

    Returns:
    cell_ids: list of int, list of cell IDs
    """
    edge_list, cell_ids = get_edge_list(polydata)
    cell_ids_for_point = []
    for edge, id in zip(edge_list, cell_ids):
        if point_id in edge:
            cell_ids_for_point.append(id)
    return cell_ids_for_point

def get_midpoint(edge, graph):
    """
    Get the midpoint of an edge in the graph.

    Args:
    edge: tuple, edge in the graph
    graph: vtkPolyData, graph

    Returns:
    midpoint: list, 3D coordinates of the midpoint
    """
    point1 = graph.GetPoint(edge[0])
    point2 = graph.GetPoint(edge[1])
    midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2]
    return midpoint

def get_neighbors_for_point(point_id, polydata):
    """
    Get the neighbors of a point in a polydata object. 
    This function looks at all edges and finds the points that share an edge with the given point.

    Args:
    point_id: int, point id for which to find neighbors
    polydata: vtkPolyData, polydata object

    Returns:
    neighbors: list of int, point ids of neighbors
    """
    edge_list, _ = get_edge_list(polydata)
    neighbors = []
    for edge in edge_list:
        if point_id in edge:
            for id in edge:
                if id != point_id:
                    neighbors.append(id)
    
    return list(set(neighbors))

def get_neighbors(cell_id, polydata):
    """
    Get the neighbors of a cell in a polydata object. 
    This function looks at all edges and finds the cells that share a point with the given cell.

    Args:
    cell_id: int, cell id for which to find neighbors
    polydata: vtkPolyData, polydata object

    Returns:
    neighbors: list of int, cell ids of neighbors
    """
    edge_list, _ = get_edge_list(polydata)
    neighbors = []
    cell_point_ids = get_pointIds_for_cell(cell_id, polydata)
    for id in cell_point_ids:
        for edge in edge_list:
            if id in edge and edge != cell_point_ids:
                neighbors.append(edge_list.index(edge))
    
    return list(set(neighbors))

def get_label_for_cell(cell_id, polydata):
    """
    Get the label of a cell in a polydata

    Args:
    cell_id: int, cell id
    polydata: vtkPolyData, polydata object

    Returns:
    label: int, label of cell
    """
    labels = polydata.GetCellData().GetArray('labels')
    return labels.GetValue(cell_id)

def set_label(label, cell_id, polydata):
    """
    Changing label of a cell in a polydata object. 

    Args:
    label: int, new label
    cell_id: int, cell id to change label for
    polydata: vtkPolyData, polydata object

    Returns:
    polydata: vtkPolyData, polydata object with changed label
    """
    labels = polydata.GetCellData().GetArray('labels') 
    labels.SetValue(cell_id, label)
    return polydata

def get_pointIds_for_label(label, polydata):
    """
    Get all the point IDs from the cells with a given label.

    Args:
    label: int, label to filter for
    polydata: vtkPolyData, polydata object

    Returns:
    pointIDs: list of int, list of point IDs
    """
    labels_arr = get_label_array(polydata)
    cell_ids_label = np.where(labels_arr == label)[0]
    pointIDs = []
    for l in cell_ids_label:
        p0, p1 = get_pointIds_for_cell(l, polydata)
        pointIDs.append(p0)
        pointIDs.append(p1)

    return list(set(pointIDs))

def find_boundary_points(label1, label2, polydata):
    """
    Find boundary points between neighboring segments with labels label1 and label2.
    
    Args:
    label1: int, label of first segment
    label2: int, label of second segment
    polydata: vtkPolyData, polydata object

    Returns:
    boundary_points: list of int, list of boundary points
    """
    points_segment_1 = get_pointIds_for_label(label1, polydata)
    points_segment_2 = get_pointIds_for_label(label2, polydata)
    boundary_points = []
    for id in points_segment_1:
        if id in points_segment_2:
            boundary_points.append(id)

    return boundary_points

def get_degree_lookup_table(polydata):
    """
    Get a lookup table for the degree of each point in the polydata object.
    The degree is determined based on the edge_list of the polydata object.

    Args:
    polydata: vtkPolyData, polydata object

    Returns:
    degree_lookup_table: dict, dictionary with point IDs as keys and degrees as values
    """
    edge_list, _ = get_edge_list(polydata)
    all_point_ids = list(set([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]))
    
    degree_lookup_table = {}
    for point in all_point_ids:
        degree_lookup_table[point] = all_point_ids.count(point)

    return degree_lookup_table

def get_point_degree(pointId, edge_list):
    """
    Get the degree of a point in a polydata object. The degree is determined based on the edge_list
    of the polydata object by counting the number of occurences of the point in the edge_list.

    Args:
    pointId: int, point ID
    edge_list: list of tuples, list of edges

    Returns:
    degree: int, degree of point
    """
    all_point_ids = [edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]
    return all_point_ids.count(pointId)

def get_nodes_of_degree_n(degree, label, polydata):
    """
    For the polydata object, get all nodes of a given degree and label.

    Args:
    degree: int, degree of node
    label: int, label of node
    polydata: vtkPolyData, polydata object

    Returns:
    nodes_with_degree: list of int, list of nodes with degree n
    """

    edge_list, _ = get_edge_list(polydata, label)
    point_ids = list(set([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list]))
    nodes_with_degree = [point for point in point_ids if get_point_degree(point, edge_list) == degree]
    return nodes_with_degree

def get_node_dict_entry(node_id, degree, label, polydata):
    """
    Get the node dict entry for a node with a given node ID, degree and label.
    The coordinates are rounded to 5 decimal places.

    Args:
    node_id: int, node ID
    degree: int, degree of node
    label: int, label of node
    polydata: vtkPolyData, polydata object

    Returns:
    node_dict_entry: dict, node dict entry
    """
    node_dict_entry = {
        "id": node_id,
        "degree": degree,
        "label": label,
        "coords": list(np.round(polydata.GetPoint(node_id), 5))
    }
    return [node_dict_entry]

def find_all_paths(start_point, end_point, polydata, label=None):
    """
    Find all paths between two points in the centerline polydata object.
    This function uses the networkx library to find all simple edge paths between two points.

    Args:
    start_point: int, start point
    end_point: int, end point
    polydata: vtkPolyData, polydata object
    label: int or list, label of segment

    Returns:
    path_data: list of dict, list of paths
    """
    if start_point == end_point:
        return [{'path': [], 'length': 0}]
    else:
        G = nx.Graph()

        if label is None:
            edge_list, cell_ids = get_edge_list(polydata)
            for edge, id in zip(edge_list, cell_ids):
                G.add_edge(edge[0], edge[1])
                G[edge[0]][edge[1]]['label'] = None
                G[edge[0]][edge[1]]['cell_id'] = id

        else:
            if isinstance(label, int):
                label = [label]
            for l in label:
                edge_list_segment, cell_ids_segment = get_edge_list(polydata, label)
                for edge, id in zip(edge_list_segment, cell_ids_segment):
                    G.add_edge(edge[0], edge[1])
                    G[edge[0]][edge[1]]['label'] = label
                    G[edge[0]][edge[1]]['cell_id'] = id

        paths = nx.all_simple_edge_paths(G, start_point, end_point)
        path_data = []
        for path in paths:
            path_dict = {
                'path': path,
                'length': len(path)
            }
            
            path_data.append(path_dict)

        assert len(path_data) > 0, 'No path found!'

        return path_data

def find_shortest_path(start_point, end_point, polydata, label=None):
    """ 
    Find the shortest path between two points in the cow graph.
    This function uses the output of find_all_paths to find all paths between two points
    in the centerline polydata and select the shortest one in terms of number of edges.

    Args:
    start_point: int, start point
    end_point: int, end point
    polydata: vtkPolyData, polydata object
    label: int or list, label of segment

    Returns:
    best_path: dict, best path
    """
    path_data = find_all_paths(start_point, end_point, polydata, label)
    
    best_path = None
    min_len = 10000
    for path in path_data:
        if path['length'] < min_len:
            best_path = path
            min_len = path['length']

    return best_path

def find_longest_path(start_point, end_point, polydata, label=None):
    """ 
    Find the longest path between two points in the cow graph.
    This function uses the output of find_all_paths to find all paths between two points
    in the centerline polydata and select the longest one in terms of number of edges.

    Args:
    start_point: int, start point
    end_point: int, end point
    polydata: vtkPolyData, polydata object
    label: int or list, label of segment

    Returns:
    best_path: dict, best path
    """
    path_data = find_all_paths(start_point, end_point, polydata, label)
    
    best_path = None
    max_len = 0
    for path in path_data:
        if path['length'] > max_len:
            best_path = path
            max_len = path['length']

    return best_path

def find_closest_node_to_point_euclidean(nodes, ref_point, polydata, deg=None):
    """
    Find the closest node to a certain reference point in terms of euclidean distance.
    This function computes the euclidean distance between the reference point and all nodes
    and returns the closest one. Optionally, the degree of the node can be specified as a constraint.

    Args:
    nodes: list of int, list of node IDs
    ref_point: int, reference point ID
    label: int, label of segment
    polydata: vtkPolyData, polydata object
    deg: int, degree of node (optional)

    Returns:
    closest_node: int, closest node
    """
    if deg is not None:
        edge_list = get_edge_list(polydata)[0]
        nodes = [node for node in nodes if get_point_degree(node, edge_list) == deg]

    min_dist = np.inf
    points = polydata.GetPoints()
    ref_point_coords = np.array(points.GetPoint(ref_point))
    closest_node = None
    for node in nodes:
        point = points.GetPoint(node)
        dist = np.linalg.norm(ref_point_coords - np.array(point))
        if dist < min_dist:
            min_dist = dist
            closest_node = node

    return closest_node, min_dist

def find_closest_node_to_point(nodes, ref_point, label, polydata):
    """
    Find the closest node to a certain reference point in terms of path length (i.e. number of edges
    between the points).

    Args:
    nodes: list of int, list of node IDs
    ref_point: int, reference point ID
    label: int, label of segment
    polydata: vtkPolyData, polydata object

    Returns:
    closest_node: int, closest node
    """
    min_dist = np.inf
    closest_node = None
    for node in nodes:
        if node == ref_point:
            return node, 0
        else:
            try:
                path_length = find_shortest_path(node, ref_point, polydata, label)['length']
                if path_length < min_dist:
                    min_dist = path_length
                    closest_node = node
            except:
                continue

    return closest_node, min_dist

def get_all_nodes_of_deg_1(polydata):
    """
    Find all nodes of degree 1 in a polydata object

    Args:
    polydata: vtkPolyData, polydata object

    Returns:
    nodes_deg_1: list of int, list of node IDs of degree 1
    """
    edge_list, _ = get_edge_list(polydata)
    cnt = Counter(node for edge in edge_list for node in edge)
    return [n for n, c in cnt.items() if c == 1]

def match_end_nodes_and_nodes_of_deg_1(all_nodes_deg_1, nodes_dict):
    """
    Match end nodes of degree 1 (as in nodes_dict json) with all nodes of degree 1 (all_nodes_deg_1).
    All the nodes of degree 1 that are not in the nodes_dict are considered as remaining nodes.
    The function returns the end nodes and the remaining nodes.

    Args:
    all_nodes_deg_1: list, list of ints of all nodes (node IDs) of degree 1
    nodes_dict: dict, dictionary containing the CoW nodes

    Returns:
    end_nodes: list, list of end nodes of degree 1 (as in nodes_dict json)
    remaining_nodes: list, list of remaining nodes of degree 1 
    """
    remaining_nodes = set(all_nodes_deg_1)
    end_nodes = []
    for node_id in all_nodes_deg_1:
        for segment_data in nodes_dict.values():
            for sub_data in segment_data.values():
                if isinstance(sub_data, dict) and sub_data.get('id') == node_id:
                    end_nodes.append(sub_data)
                    remaining_nodes.discard(node_id)
                elif isinstance(sub_data, list):
                    for item in sub_data:
                        if item.get('id') == node_id:
                            end_nodes.append(item)
                            remaining_nodes.discard(node_id)
    remaining_nodes = list(remaining_nodes)
    
    return end_nodes, remaining_nodes

def get_point_coord_lookup(polydata):
    """
    Get a dictionary of point coordinates from a polydata object with the keys being the point IDs.

    Args:
    polydata: vtkPolyData, polydata object

    Returns:
    point_coord_dict: dict, dictionary of point coordinates
    """
    points = polydata.GetPoints()
    point_coord_dict = {}
    for i in range(points.GetNumberOfPoints()):
        point_coord_dict[i] = list(points.GetPoint(i))
    
    return point_coord_dict

def find_id_for_coord(coord, point_coords_dict, tolerance=0.01):
    """
    Get point ID for a certain coordinate. 
    This function compares the coordinate with the coordinates in the point_coords_dict of the polydata.
    It returns the ID of the point that is closest to the given coordinate with a tolerance of 0.01.

    Args:
    coord: tuple, coordinate
    point_coords_dict: dict, dictionary of point coordinates of the polydata object
    tolerance: float, tolerance for matching coordinates

    Returns:
    new_id: int, point ID
    """
    for i in range(len(point_coords_dict)):
        x, y, z = point_coords_dict[i]
        # coord_rounded = (np.round(coord[0], 2), np.round(coord[1], 2), np.round(coord[2], 2))
        if (coord[0]-x)**2+(coord[1]-y)**2+(coord[2]-z)**2 < tolerance:
            new_id = i
            break
    assert new_id is not None
    return new_id

def connect_two_points(point1, point2, polydata, label):
    """
    Connect two points in a polydata object with a line segment of the given label.

    Args:
    point1: int, point 1
    point2: int, point 2
    polydata: vtkPolyData, polydata object
    label: int, label of segment

    Returns:
    polydata: vtkPolyData, polydata object with points connected
    """
    # Create a vtkCellArray to store the new line
    lines = polydata.GetLines()
    if not lines:
        lines = vtk.vtkCellArray()
        polydata.SetLines(lines)

    # Create a new line between the points
    new_line = vtk.vtkLine()
    new_line.GetPointIds().SetId(0, point1)
    new_line.GetPointIds().SetId(1, point2)
    lines.InsertNextCell(new_line)

    # Update the polydata object with the new line
    polydata.SetLines(lines)
    polydata.Modified()  # Ensure the polydata is updated
    polydata.BuildCells()  # Ensure cells are built
    polydata.BuildLinks()  # Ensure links are built
    original_cell_ids = polydata.GetCellData().GetArray("vtkOriginalCellIds")
    original_cell_ids.InsertNextValue(polydata.GetNumberOfCells())
    polydata.GetCellData().GetArray('labels').InsertNextValue(label)

    return polydata

def connect_loose_end(loose_1_node, segment_points, label, polydata, avg_dist=0.4): 
    """
    For a loose end, connect it to the closest point of the segment given by segment_points.
    The average distance between points is used to determine how many points to add between the two points.
    The function adds the new points to the polydata object and updates the lines and labels.

    Args:
    loose_1_node: int, loose end node
    segment_points: list of int, list of segment points
    label: int, label of segment
    polydata: vtkPolyData, polydata
    avg_dist: float, average distance between points

    Returns:
    polydata: vtkPolyData, polydata object with loose end connected
    """
    cow_points = polydata.GetPoints()
    segment_point_coords = [np.array(cow_points.GetPoint(pt)) for pt in segment_points]
    loose_point = np.array(cow_points.GetPoint(loose_1_node))
    dists = [np.linalg.norm(pt - loose_point) for pt in segment_point_coords]
    closest_point = segment_points[np.argmin(dists)]

    points_to_connect = [loose_1_node, closest_point]
    dist = np.min(dists)
    logger.debug(f'Connecting loose end {loose_1_node} to closest point {closest_point} with dist {dist}')
    # Add points between the two points until the average distance is reached
    while dist > avg_dist:
        dist_temp = 0
        num_dist = 0
        points_to_connect_freeze = copy.deepcopy(points_to_connect)
        # insert new points between any two neighboring points in the list
        for n, i in enumerate(range(len(points_to_connect_freeze)-1)):
            p1 = np.array(cow_points.GetPoint(points_to_connect_freeze[i]))
            p2 = np.array(cow_points.GetPoint(points_to_connect_freeze[i+1]))
            new_point = (p1 + p2) / 2
            # insert into list at correct position
            points_to_connect.insert(n+i+1, polydata.GetNumberOfPoints())
            polydata.GetPoints().InsertNextPoint(new_point)
            original_ids = polydata.GetPointData().GetArray("vtkOriginalPointIds")
            original_ids.InsertNextValue(polydata.GetNumberOfPoints())
            dist1 = np.linalg.norm(p1 - new_point)
            dist2 = np.linalg.norm(p2 - new_point)
            dist_temp += dist1 + dist2
            num_dist += 2
        
        dist = dist_temp / num_dist

    # Add Lines to connect the points
    for i in range(len(points_to_connect)-1):
        # Create a vtkCellArray to store the new line
        lines = polydata.GetLines()
        if not lines:
            lines = vtk.vtkCellArray()
            polydata.SetLines(lines)

        # Create a new line between the points
        new_line = vtk.vtkLine()
        new_line.GetPointIds().SetId(0, points_to_connect[i])
        new_line.GetPointIds().SetId(1, points_to_connect[i+1])
        lines.InsertNextCell(new_line)

        # Update the polydata object with the new line
        polydata.SetLines(lines)
        polydata.Modified()  # Ensure the polydata is updated
        polydata.BuildCells()  # Ensure cells are built
        polydata.BuildLinks()  # Ensure links are built
        original_cell_ids = polydata.GetCellData().GetArray("vtkOriginalCellIds")
        original_cell_ids.InsertNextValue(polydata.GetNumberOfCells())
        polydata.GetCellData().GetArray('labels').InsertNextValue(label)
         
    return polydata
        
def get_number_of_connected_components(label, polydata):
    """
    Get the number of connected components in a segment with a given label.
    This function uses the networkx library to find the number of connected components in the graph.

    Args:
    label: int, label of segment
    polydata: vtkPolyData, polydata object

    Returns:
    n_components: int, number of connected components
    """
    edge_list, _ = get_edge_list(polydata, label)
    G = nx.Graph()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    return nx.number_connected_components(G)

def get_connected_components(label, polydata):
    """
    Get the connected components in a segment with a given label.
    This function uses the networkx library to find and return the connected components in the graph.

    Args:
    label: int, label of segment
    polydata: vtkPolyData, polydata object

    Returns:
    components: list of list, list of connected components
    """
    edge_list, _ = get_edge_list(polydata, label)
    G = nx.Graph()
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    components = [list(c) for c in nx.connected_components(G)]
    return components

def find_shortest_shortest_path(pointIds1, pointIds2, polydata, labels=None):
    """
    Find the shortest path among all shortest paths between two sets of points.
    This function computes the shortest paths between all pairs of points in pointIds1 and pointIds2
    and returns the shortest one in terms of number of edges.

    Args:
    pointIds1: list of int, list of point IDs
    pointIds2: list of int, list of point IDs
    polydata: vtkPolyData, polydata object
    labels: np.array, array of labels

    Returns:
    shortest_path: dict, shortest path
    """
    shortest_path = None
    shortest_length = 10000
    for pt1 in pointIds1:
        for pt2 in pointIds2:
            if pt1 == pt2:
                shortest_length = 0
                shortest_path = {'path': [], 'length': 0}
            else:
                try:
                    path = find_shortest_path(pt1, pt2, polydata, labels)
                    if path['length'] < shortest_length:
                        shortest_length = path['length']
                        shortest_path = path
                except:
                    continue
        
    return shortest_path

def find_longest_shortest_path(pointIds1, pointIds2, polydata, labels=None):
    """
    Find the longest shortest path between two sets of points.
    This function computes the shortest paths between all pairs of points in pointIds1 and pointIds2
    and returns the longest one in terms of number of edges.


    Args:
    pointIds1: list of int, list of point IDs
    pointIds2: list of int, list of point IDs
    polydata: vtkPolyData, polydata object
    labels: np.array, array of labels

    Returns:
    longest_path: dict, longest path
    """
    longest_path = None
    longest_length = 0
    for pt1 in pointIds1:
        for pt2 in pointIds2:
            if pt1 == pt2:
                pass
            else:
                try:
                    path = find_shortest_path(pt1, pt2, polydata, labels)
                    if path['length'] > longest_length:
                        longest_length = path['length']
                        longest_path = path
                except:
                    continue
        
    return longest_path

def check_for_small_loops(polydata, labels, max_length=10):
    """
    Check for small loops in the graph with a given maximum length.

    Args:
    polydata: vtkPolyData, polydata object
    labels: list, list of labels
    max_length: int, maximum length of halft the loop

    Returns:
    loops: list, list of loop paths (paths between loop nodes)
    loop_nodes: list, list of higher degree nodes in the loop
    """

    # check for small loops
    nodes_3 = get_nodes_of_degree_n(3, labels, polydata)
    nodes_4 = get_nodes_of_degree_n(4, labels, polydata)
    nodes_5 = get_nodes_of_degree_n(5, labels, polydata)
    higher_nodes = nodes_3 + nodes_4 + nodes_5
    
    loops, loop_nodes = [], []
    if len(higher_nodes) > 1:
        for k, nd1 in enumerate(higher_nodes[:-1]):
            for nd2 in higher_nodes[k+1:]:
                paths = find_all_paths(nd1, nd2, polydata, labels)
                if len(paths) > 1:
                    all_paths_combined = []
                    for p in paths:
                        all_paths_combined += p['path']
                    if len(list(set(all_paths_combined))) == len(all_paths_combined):
                        other_nodes = copy.deepcopy(higher_nodes)
                        other_nodes.remove(nd1)
                        other_nodes.remove(nd2)
                        loop_paths = []
                        for p in paths:
                            path = p['path']
                            end_ids =  [e[1] for e in path]
                            for node in other_nodes:
                                if node in end_ids:
                                    higher_nodes.remove(node)
                                    loop_paths.append(path[:end_ids.index(node)+1])
                                    loop_paths.append(path[end_ids.index(node)+1:])
                                else:
                                    loop_paths.append(path)
                        
                        if all([len(p) < max_length for p in loop_paths]):
                            loops = loop_paths
                            loop_nodes = list(set([p[0][0] for p in loop_paths] + [p[-1][1] for p in loop_paths]))
                            return loops, loop_nodes
                            
    return loops, loop_nodes

def check_for_loop_between_nodes(node1, node2, labels, polydata):
    """
    Check if there is a loop between two nodes node1 and node2.
    A loop is present if there are more than 1 paths between the nodes.
    This function uses the find_all_paths function to find all paths between the two nodes
    and checks if there are more than 1 paths.

    Args:
    node1: int, node 1
    node2: int, node 2
    labels: list, list of labels
    polydata: vtkPolyData, polydata object

    Returns:
    loop: list, list of nodes in loop
    """
    paths = find_all_paths(node1, node2, polydata, labels)
    if len(paths) > 1:
        return True
    else:
        return False
    
def assert_node_on_path(node, path):
    """
    Assert that a node is on a path.

    Args:
    node: int, node
    path: list of tuples, path

    Returns:
    is_on_path: bool, whether node is on path
    """
    is_on_path = False
    node_ids = [e[0] for e in path] + [path[-1][1]]
    if node in node_ids:
        is_on_path = True
    return is_on_path

def find_closest_points_between_connected_components(components, polydata):
    """
    The function finds the closest two points between multiple connected components and 
    returns the closest points and the euclidean distance between them. 
    The function only considers points of degree 1. 
    The number of connected components must be greater than 1.

    Args:
    components: list of list, list of connected components
    polydata: vtkPolyData, polydata object

    Returns:
    closest_points: list of int, list of closest points
    """
    edge_list = get_edge_list(polydata)[0]
    closest_points = None
    min_dist = np.inf
    assert len(components) > 1, 'Two few components!'
    for k, cmp1 in enumerate(components[:-1]):
        for cmp2 in components[k+1:]:
            deg1_nodes_cmp1 = [node for node in cmp1 if get_point_degree(node, edge_list) == 1]
            for node in deg1_nodes_cmp1:
                closest_node, dist = find_closest_node_to_point_euclidean(cmp2, node, polydata, deg=1)
                if dist < min_dist:
                    assert get_point_degree(closest_node, edge_list) == 1, f'Node {closest_node} is not of degree 1!'
                    assert get_point_degree(node, edge_list) == 1, f'Node {node} is not of degree 1!'
                    min_dist = dist
                    closest_points = (node, closest_node)
    return closest_points, min_dist






    