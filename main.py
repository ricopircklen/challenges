import argparse
import sys
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from shapely.geometry import MultiPoint
from geopy.distance import great_circle, geodesic
import uuid
import math
from typing import Tuple, List

def open_csv_file(file_path: str) -> pd.DataFrame:
    """
    Open a CSV file, handling regular and compressed files (.csv.gz).
    """
    try:
        if file_path.endswith('.csv.gz'):
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)
        return df
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening file: {e}")
        sys.exit(1)


def calculate_distance(start_point1: Tuple[float, float], start_point2: Tuple[float, float]) -> float:
    """
    Calculate the distance (in kilometers) between two starting points.
    """
    return geodesic(start_point1, start_point2).kilometers


def get_centermost_point(cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
    '''
    This function returns the center-most point from a cluster by taking a set of points (i.e., a cluster) 
    and returning the point within it that is nearest to some reference point (in this case, the cluster’s centroid).
    '''
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def group_users(df: pd.DataFrame) -> pd.DataFrame:
    max_group_size = 40
    group_data = []
    unique_clusters = df['cluster_label'].unique().tolist()

    for cluster_label in unique_clusters:
        cluster_points = df[df['cluster_label'] == cluster_label]
        cluster_members = cluster_points['user_id'].tolist()
        unique_users = cluster_points['user_id'].nunique()
        num_groups = max(1, math.ceil(unique_users / max_group_size))
        group_size = math.ceil(unique_users / num_groups)
        for _ in range(num_groups):
            group_id = str(uuid.uuid4())
            members = []
            if unique_users >= group_size:
                members.extend(cluster_members[:group_size])
                cluster_members = cluster_members[group_size:] 
                unique_users -= group_size
            else:
                members.extend(cluster_members)
                cluster_members = []
                unique_users = 0
            group_data.append({'cluster_label': cluster_label, 'group_id': group_id, 'members': members, 'group_size': len(members)})
    
    # Assign group_id and potential_group_members to df
    df['group_id'] = None
    df['potential_group_members'] = None
    df['group_size'] = None
    for group in group_data:
        members = group['members']
        group_id = group['group_id']
        group_size = group['group_size']
        df.loc[df['user_id'].isin(members), 'group_id'] = group_id
        df.loc[df['user_id'].isin(members), 'potential_group_members'] = ','.join(members)
        df.loc[df['user_id'].isin(members), 'group_size'] = group_size
    
    return df

def cluster_data_points(df: pd.DataFrame) -> pd.DataFrame:
    kms_per_radian = 6371.0088
    epsilon = 1 / kms_per_radian
    min_samples = 5

    df['latitude_scaled'] = np.radians(df['latitude'])
    df['longitude_scaled'] = np.radians(df['longitude'])
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric="haversine")
    df['cluster_label'] = dbscan.fit_predict(df[['latitude_scaled', 'longitude_scaled']])

    cluster_labels = dbscan.labels_
    num_clusters = len(set(cluster_labels))
    coords = df[['latitude', 'longitude']].values

    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters) if len(coords[cluster_labels == n]) > 0]) 
    centermost_points = clusters.map(get_centermost_point)
    
    # Include the outlier data points into a group, if the individual is not already in a cluster
    users_without_cluster = []
    for user_id in df[df['cluster_label'] == -1]['user_id'].unique():
        if df[df['user_id'] == user_id]['cluster_label'].nunique() == 1:
            users_without_cluster.append(user_id)

    # Assign the closest cluster for users without a cluster by finding the closest one from all the users points.
    for user_id in users_without_cluster:
        user_coords = df.loc[df['user_id'] == user_id, ['latitude', 'longitude']].values
        closest_cluster = min(centermost_points, key=lambda point: min(great_circle(point, coord).m for coord in user_coords))
        closest_cluster_idx = centermost_points[centermost_points == closest_cluster].index[0]
        df.loc[df['user_id'] == user_id, 'cluster_label'] = closest_cluster_idx

    # Drop the rows that have cluster_label -1. 
    # At this point, users that have -1 label, already belong to another cluster, so these outliers can be neglected.
    df = df[df['cluster_label'] != -1]

    # Calculate distances for each row
    df['distance'] = df.apply(lambda row: great_circle(
        (row['latitude'], row['longitude']),
        centermost_points[row['cluster_label']]
    ).kilometers, axis=1)

    # Get the indices of the rows with the smallest distance for each user_id
    min_distance_indices = df.groupby('user_id')['distance'].idxmin()

    # Filter the DataFrame to keep only the rows with the smallest distance for each user_id
    df = df.loc[min_distance_indices]

    # Create a dictionary mapping cluster labels to starting point IDs and add starting points
    start_point_ids = {index: uuid.uuid4() for index in centermost_points.index}

    df['start_point_id'] = df['cluster_label'].map(start_point_ids)
    df['start_point_latitude'] = df['cluster_label'].map(lambda x: centermost_points[x][0])
    df['start_point_longitude'] = df['cluster_label'].map(lambda x: centermost_points[x][1])
    return df

def generate_output(df):
    output_data = df[['user_id', 'start_point_id', 'start_point_latitude', 'start_point_longitude', 'potential_group_members', 'group_id', 'group_size', 'distance']]

    # Clean user itself from potential members
    for i, row in output_data.iterrows():
        user_id = row['user_id']
        members = row['potential_group_members'].split(',')
        members = [member for member in members if member != user_id]
        joined_members = ','.join(members)
        output_data.loc[i, 'potential_group_members'] = joined_members

    return output_data

def generate_newsletter_csv(input_file, output_file):
    """
    Group individuals and generate a CSV file for the newsletter with the required columns.
    """
    input_df = open_csv_file(input_file)   
    clustered_df = cluster_data_points(input_df)
    grouped_df = group_users(clustered_df)
    output_df = generate_output(grouped_df)
    output_df.to_csv(output_file, index=False)
    print(f"Generated newsletter CSV: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a newsletter CSV file for group cycling events.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    generate_newsletter_csv(input_file, output_file)

if __name__ == '__main__':
    main()
