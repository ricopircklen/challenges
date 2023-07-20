from shapely.geometry import MultiPoint
from geopy.distance import great_circle, geodesic
from typing import Tuple, List


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculates the distance (in kilometers) between two starting points.
    great_circle is faster and good enough for short distances (within or around a city), but in this case we rely
    on geodesic in case of distant outliers.
    """
    return geodesic(point1, point2).kilometers


def get_centermost_point(cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Gets the center-most point from a cluster by taking a set of points (i.e., a cluster) 
    and returning the point within it that is nearest to some reference point (in this case, the cluster’s centroid).

    Args:
        cluster: A list of tuples, where each tuple represents the coordinates of a point in the cluster.
        centroid: A tuple representing the coordinates of the centroid.

    Returns:
        A tuple representing the coordinates of the center-most point in the cluster.
    """
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
