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


class NewsletterGenerator:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    def open_csv_file(self):
        """
        Open a CSV file, handling regular and compressed files (.csv.gz).
        Exit in case of issues opening the file.
        """
        try:
            if self.input_file.endswith('.csv.gz'):
                self.df = pd.read_csv(self.input_file, compression='gzip')
            else:
                self.df = pd.read_csv(self.input_file)
        except (FileNotFoundError, IOError) as e:
            print(f"Error opening file: {e}")
            sys.exit(1)

    def preprocessing(self) -> pd.DataFrame:
        """
        Data preprocessing includes removal of null value rows and incorrectly formatted geolocation values.
        """
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        self.df.dropna(inplace=True)
        return self.df

    def check_data_quality(self) -> pd.DataFrame:
        """
        Check the data quality of the input DataFrame.
        The handled edge cases are missing or incorrectly labeled columns, and datasets that have less than 5 data points
        (minimum for dbscan). Exit in case of not fulfilling these criteria.

        Returns:
            pd.DataFrame: The preprocessed DataFrame without null values.
        """
        required_columns = {'latitude', 'longitude', 'user_id'}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            missing_columns_str = ', '.join(missing_columns)
            error_msg = f"The input DataFrame is missing the following required columns: {missing_columns_str}."
            sys.exit(error_msg)

        preprocessed_df = self.preprocessing()

        if preprocessed_df.shape[0] < 5:
            sys.exit("The input dataset has fewer than 5 rows of values in correct format.")

        return self.df

    def calculate_distance(self, start_point1: Tuple[float, float], start_point2: Tuple[float, float]) -> float:
        """
        Calculate the distance (in kilometers) between two starting points.
        great_circle is faster and good enough for short distances (within or around a city), but in this case we rely
        on geodesic in case of distant outliers.
        """
        return geodesic(start_point1, start_point2).kilometers

    def get_centermost_point(self, cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
        '''
        This function returns the center-most point from a cluster by taking a set of points (i.e., a cluster) 
        and returning the point within it that is nearest to some reference point (in this case, the cluster’s centroid).
        '''
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    def group_users(self) -> pd.DataFrame:
        """
        Group users to groups of maximum size of 40.
        """
        max_group_size = 40
        group_data        = []
        unique_clusters = self.df['cluster_label'].unique().tolist()

        for cluster_label in unique_clusters:
            cluster_points = self.df[self.df['cluster_label'] == cluster_label]
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
                group_data.append(
                    {'cluster_label': cluster_label, 'group_id': group_id, 'members': members, 'group_size': len(members)})

        # Assign group_id and potential_group_members to df
        self.df['group_id'] = None
        self.df['potential_group_members'] = None
        self.df['group_size'] = None
        for group in group_data:
            members = group['members']
            group_id = group['group_id']
            group_size = group['group_size']
            self.df.loc[self.df['user_id'].isin(members), 'group_id'] = group_id
            self.df.loc[self.df['user_id'].isin(members), 'potential_group_members'] = ','.join(members)
            self.df.loc[self.df['user_id'].isin(members), 'group_size'] = group_size

        return self.df

    def run_dbscan(self, epsilon: float, min_samples: int) -> Tuple[pd.Series, List]:
        """
        Run DBSCAN clustering on the given dataframe using the specified epsilon and min_samples.

        Args:
            epsilon (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The minimum number of samples required for a group to be considered as a cluster.

        Returns:
            pd.Series: The cluster labels assigned by DBSCAN.
            list: dbscan labels.
        """
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric="haversine")
        self.df['cluster_label'] = dbscan.fit_predict(self.df[['latitude_rad', 'longitude_rad']])
        return self.df['cluster_label'], dbscan.labels_

    def cluster_data_points(self) -> pd.DataFrame:
        """
        Assign each data point into a cluster.
        """
        kms_per_radian = 6371.0088
        epsilon = 1 / kms_per_radian
        min_samples = 5

        self.df['latitude_rad'] = np.radians(self.df['latitude'])
        self.df['longitude_rad'] = np.radians(self.df['longitude'])
        self.df['cluster_label'], cluster_labels = self.run_dbscan(epsilon, min_samples)
        num_clusters = len(set(cluster_labels))

        coords = self.df[['latitude', 'longitude']].values
        clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters) if len(coords[cluster_labels == n]) > 0])
        centermost_points = clusters.map(self.get_centermost_point)

        # Include the outlier data points into a group, if the individual is not already in a cluster
        users_without_cluster = []
        for user_id in self.df[self.df['cluster_label'] == -1]['user_id'].unique():
            if self.df[self.df['user_id'] == user_id]['cluster_label'].nunique() == 1:
                users_without_cluster.append(user_id)

        # Assign the closest cluster for users without a cluster by finding the closest one from all the user's points.
        for user_id in users_without_cluster:
            user_coords = self.df.loc[self.df['user_id'] == user_id, ['latitude', 'longitude']].values
            closest_cluster = min(centermost_points,
                                  key=lambda point: min(great_circle(point, coord).m for coord in user_coords))
            closest_cluster_idx = centermost_points[centermost_points == closest_cluster].index[0]
            self.df.loc[self.df['user_id'] == user_id, 'cluster_label'] = closest_cluster_idx

        # At this point every user belongs to a cluster, so neglect remaining outliers and
        # drop the rows that still might have cluster_label -1.
        self.df = self.df[self.df['cluster_label'] != -1]

        # Calculate distances for each row
        self.df['distance'] = self.df.apply(lambda row: self.calculate_distance(
            (row['latitude'], row['longitude']),
            centermost_points[row['cluster_label']]
        ), axis=1)

        # Get the indices of the rows with the smallest distance for each user_id
        min_distance_indices = self.df.groupby('user_id')['distance'].idxmin()

        # Filter the DataFrame to keep only the rows with the smallest distance for each user_id
        self.df = self.df.loc[min_distance_indices]

        # Create a dictionary mapping cluster labels to starting point IDs and add starting points
        start_point_ids = {index: uuid.uuid4() for index in centermost_points.index}

        self.df['start_point_id'] = self.df['cluster_label'].map(start_point_ids)
        self.df['start_point_latitude'] = self.df['cluster_label'].map(lambda x: centermost_points[x][0])
        self.df['start_point_longitude'] = self.df['cluster_label'].map(lambda x: centermost_points[x][1])
        return self.df

    def generate_output(self):
        """
        Generate the output data and save it to a CSV file.
        The generated output contains the following columns: user_id, start_point_id, start_point_latitude,
        start_point_longitude,potential_group_members, group_id, group_size, and distance.
        Distance is in kilometers and measured to the user's closest starting point.
        """
        output_data = self.df[
            ['user_id', 'start_point_id', 'start_point_latitude', 'start_point_longitude', 'potential_group_members',
             'group_id', 'group_size', 'distance']]

        # Clean user itself from potential members
        for i, row in output_data.iterrows():
            user_id = row['user_id']
            members = row['potential_group_members'].split(',')
            members = [member for member in members if member != user_id]
            joined_members = ','.join(members)
            output_data.loc[i, 'potential_group_members'] = joined_members
        try:
            output_data.to_csv(self.output_file, index=False)
            print(f"Generated newsletter CSV: {self.output_file}")
        except Exception as e:
            print(f"Error occurred during CSV output generation: {str(e)}")

        return

    def generate_newsletter_csv(self):
        """
        Group individuals and generate a CSV file for the newsletter with the required columns.
        """
        self.open_csv_file()
        self.check_data_quality()
        self.cluster_data_points()
        self.group_users()
        self.generate_output()

def main():
    parser = argparse.ArgumentParser(description='Generate a newsletter CSV file for group cycling events.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    newsletter_generator = NewsletterGenerator(input_file, output_file)
    newsletter_generator.generate_newsletter_csv()

if __name__ == '__main__':
    main()