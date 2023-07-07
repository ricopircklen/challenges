import argparse
import sys
import pandas as pd
from sklearn.cluster import OPTICS
import numpy as np
from shapely.geometry import MultiPoint
from geopy.distance import great_circle, geodesic
import uuid
import math
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)


class NewsletterGenerator:
    """
    The NewsletterGenerator class provides functionality to generate a CSV newsletter for group cycling events.
    It reads input from a CSV file, processes it, and outputs a CSV file containing organized information.

    Attributes:
    -----------
    input_file : str
        The name or the path of the input CSV file.
    output_file : str
        The name or the path of the output CSV file.
    df : pandas.DataFrame
        The dataframe used to store and manipulate data.
    centermost_points: pandas.Series
        The centermost points of clusters used to reassign users.
    """

    MAX_GROUP_SIZE = 40
    MAX_CHUNK_SIZE = 10000
    MIN_CLUSTER_SIZE = 5

    def __init__(self, input_file: str, output_file: str) -> None:
        if not input_file.endswith(('.csv', '.csv.gz')):
            sys.exit("Error: Input file is not a CSV file")
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.centermost_points = None

    def open_csv_file(self) -> None:
        """
        Reads a CSV file and stores the data into a dataframe. It handles both regular and gzip-compressed CSV files.
        If an error occurs while opening the file, an error message is logged and the exception is re-raised.
        Reads large files in chunks and concatenates them for reducing memory usage. Chunk size specified in MAX_CHUNK_SIZE.

        Raises:
        -------
        FileNotFoundError
            If the file cannot be found.
        IOError
            If an I/O error occurs while opening the file.
        """
        try:
            if self.input_file.endswith('.csv.gz'):
                chunks = pd.read_csv(
                    self.input_file, compression='gzip', chunksize=self.MAX_CHUNK_SIZE)
            else:
                chunks = pd.read_csv(
                    self.input_file, chunksize=self.MAX_CHUNK_SIZE)
            for chunk in chunks:
                self.df = pd.concat([self.df, chunk])
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Error opening file: {e}")
            raise

    def preprocessing(self) -> pd.DataFrame:
        """
        Preprocesses the data in the dataframe by removing null value rows and incorrectly formatted geolocation values.

        Returns:
        --------
        pd.DataFrame
            The preprocessed dataframe without null values.
        """
        self.df['latitude'] = pd.to_numeric(
            self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(
            self.df['longitude'], errors='coerce')
        self.df.dropna(inplace=True)
        return self.df

    def check_data_quality(self) -> pd.DataFrame:
        """
        Checks for missing or incorrectly labeled columns, and verifies that the dataframe has at least 5 data points.
        If these criteria are not met, an appropriate error message is logged and the system exits.

        Returns:
        --------
        pd.DataFrame
            The dataframe that has passed the quality check.
        """
        required_columns = {'latitude', 'longitude', 'user_id'}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            missing_columns_str = ', '.join(missing_columns)
            error_msg = f"The input DataFrame is missing the following required columns: {missing_columns_str}."
            sys.exit(error_msg)

        preprocessed_df = self.preprocessing()

        if preprocessed_df.shape[0] < 5:
            sys.exit(
                "The input dataset has fewer than 5 rows of values in correct format.")

        return self.df

    def calculate_distance(self, start_point1: Tuple[float, float], start_point2: Tuple[float, float]) -> float:
        """
        Calculates the distance (in kilometers) between two starting points.
        great_circle is faster and good enough for short distances (within or around a city), but in this case we rely
        on geodesic in case of distant outliers.
        """
        return geodesic(start_point1, start_point2).kilometers

    def get_centermost_point(self, cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
        '''
        Gets the center-most point from a cluster by taking a set of points (i.e., a cluster) 
        and returning the point within it that is nearest to some reference point (in this case, the cluster’s centroid).
        '''
        centroid = (MultiPoint(cluster).centroid.x,
                    MultiPoint(cluster).centroid.y)
        centermost_point = min(
            cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    def group_users(self) -> pd.DataFrame:
        """
        Groups users to groups of maximum size specified in self.MAX_GROUP_SIZE.
        """
        group_data = []
        unique_clusters = self.df['cluster_label'].unique().tolist()

        for cluster_label in unique_clusters:
            cluster_points = self.df[self.df['cluster_label'] == cluster_label]
            cluster_members = cluster_points['user_id'].tolist()
            unique_users = cluster_points['user_id'].nunique()
            num_groups = max(1, math.ceil(unique_users / self.MAX_GROUP_SIZE))
            group_size = math.ceil(unique_users / num_groups)

            self._handle_grouping(
                group_data, cluster_members, unique_users, group_size, num_groups, cluster_label)

        self._assign_group_data(group_data)

        return self.df

    def _handle_grouping(self, group_data: List[dict], cluster_members: List[str], unique_users: int, group_size: int, num_groups: int, cluster_label: int) -> None:
        """
        Handles the process of grouping the users within a specific cluster.
        The method creates a unique group ID, fetches the members of the group, and 
        then appends this information to the group_data list.
        """
        for _ in range(num_groups):
            group_id = str(uuid.uuid4())
            members = self._get_group_members(
                cluster_members, unique_users, group_size)
            unique_users -= len(members)
            group_data.append({'cluster_label': cluster_label, 'group_id': group_id,
                              'members': members, 'group_size': len(members)})

    def _get_group_members(self, cluster_members: List[str], unique_users: int, group_size: int) -> List[str]:
        """
        Gets the members of a group from the list of cluster members.
        The method either gets a list of members equal to the group size or all remaining members
        if the number of unique users is less than the group size.
        """
        members = []
        try:
            if unique_users >= group_size:
                members.extend(cluster_members[:group_size])
                del cluster_members[:group_size]
            else:
                members.extend(cluster_members)
                del cluster_members[:]
        except IndexError:
            logging.error(
                "Attempting to access an empty list: cluster_members. Please check your data.")
            raise
        return members

    def _assign_group_data(self, group_data: List[dict]) -> None:
        """
        Assigns the group data (group ID, potential group members, group size) to the DataFrame.
        """
        self.df['group_id'] = None
        self.df['potential_group_members'] = None
        self.df['group_size'] = None
        for group in group_data:
            members = group['members']
            group_id = group['group_id']
            group_size = group['group_size']
            self.df.loc[self.df['user_id'].isin(
                members), ['group_id', 'potential_group_members', 'group_size']] = [group_id, ','.join(members), group_size]

    def run_optics(self, min_samples: int, xi: float) -> Tuple[pd.Series, List]:
        """
        Runs OPTICS clustering on the given dataframe using the specified min_samples and xi.
        Haversine is approriate for geospatial data.

        Args:
            min_samples (int): The minimum number of samples required for a group to be considered as a cluster.
            xi (float): The parameter for the OPTICS algorithm, specifying a minimum steepness on the reachability plot.

        Returns:
            pd.Series: The cluster labels assigned by OPTICS.
            list: optics labels.
        """
        optics = OPTICS(min_samples=min_samples, xi=xi, metric="haversine")
        self.df['cluster_label'] = optics.fit_predict(
            self.df[['latitude_rad', 'longitude_rad']])
        return self.df['cluster_label'], optics.labels_

    def _assign_cluster_for_outliers(self):
        '''Assign the closest cluster for users without a cluster by finding the closest one from all the users points'''
        users_without_cluster = []
        for user_id in self.df[self.df['cluster_label'] == -1]['user_id'].unique():
            if self.df[self.df['user_id'] == user_id]['cluster_label'].nunique() == 1:
                users_without_cluster.append(user_id)

        for user_id in users_without_cluster:
            user_coords = self.df.loc[self.df['user_id']
                                      == user_id, ['latitude', 'longitude']].values
            closest_cluster = min(self.centermost_points, key=lambda point: min(
                great_circle(point, coord).m for coord in user_coords))
            closest_cluster_idx = self.centermost_points[self.centermost_points ==
                                                         closest_cluster].index[0]
            self.df.loc[self.df['user_id'] == user_id,
                        'cluster_label'] = closest_cluster_idx

    def _filter_outliers_and_keep_closest_points(self):
        '''Drop outliers and keep only the rows with the smallest distance for each user_id'''
        # Drop the rows that have cluster_label -1.
        self.df = self.df[self.df['cluster_label'] != -1]
        # Calculate distances for each row
        self.df['distance'] = self.df.apply(lambda row: self.calculate_distance(
            (row['latitude'], row['longitude']),
            self.centermost_points[row['cluster_label']]
        ), axis=1)
        # Get the indices of the rows with the smallest distance for each user_id
        min_distance_indices = self.df.groupby('user_id')['distance'].idxmin()

        # Filter the DataFrame to keep only the rows with the smallest distance for each user_id
        self.df = self.df.loc[min_distance_indices]

    def _reassign_small_clusters(self):
        '''Assign the closest cluster for users in small clusters by finding the closest one from all the users points'''

        # Identify small clusters and valid (non-small) clusters
        cluster_sizes = self.df.groupby('cluster_label')['user_id'].nunique()
        small_clusters = cluster_sizes[cluster_sizes <
                                       self.MIN_CLUSTER_SIZE].index
        valid_clusters = cluster_sizes[cluster_sizes >=
                                       self.MIN_CLUSTER_SIZE].index

        # Filter centermost_points to include only valid clusters
        self.centermost_points = self.centermost_points[self.centermost_points.index.isin(
            valid_clusters)]
        users_in_small_clusters = []
        for cluster in small_clusters:
            users_in_small_clusters.extend(
                self.df[self.df['cluster_label'] == cluster]['user_id'].unique())

        for user_id in users_in_small_clusters:
            user_coords = self.df.loc[self.df['user_id']
                                      == user_id, ['latitude', 'longitude']].values
            closest_cluster = min(self.centermost_points, key=lambda point: min(
                great_circle(point, coord).m for coord in user_coords))
            closest_cluster_idx = self.centermost_points[self.centermost_points ==
                                                         closest_cluster].index[0]
            self.df.loc[self.df['user_id'] == user_id,
                        'cluster_label'] = closest_cluster_idx

    def _assign_starting_points(self) -> None:
        """
        Create a dictionary mapping cluster labels to starting point IDs and add starting points.

        Parameters:
        -----------
        centermost_points : pd.Series
            The series of centermost points for each cluster.
        """
        start_point_ids = {index: uuid.uuid4()
                           for index in self.centermost_points.index}

        self.df['start_point_id'] = self.df['cluster_label'].map(
            start_point_ids)
        self.df['start_point_latitude'] = self.df['cluster_label'].map(
            lambda x: self.centermost_points[x][0])
        self.df['start_point_longitude'] = self.df['cluster_label'].map(
            lambda x: self.centermost_points[x][1])

    def cluster_data_points(self) -> pd.DataFrame:
        """
        Assign each data point into a cluster.
        """
        xi = 0.01

        self.df['latitude_rad'] = np.radians(self.df['latitude'])
        self.df['longitude_rad'] = np.radians(self.df['longitude'])
        self.df['cluster_label'], cluster_labels = self.run_optics(
            self.MIN_CLUSTER_SIZE, xi)
        num_clusters = len(set(cluster_labels))

        coords = self.df[['latitude', 'longitude']].values
        clusters = pd.Series([coords[cluster_labels == n] for n in range(
            num_clusters) if len(coords[cluster_labels == n]) > 0])
        self.centermost_points = clusters.map(self.get_centermost_point)

        self._assign_cluster_for_outliers()
        self._filter_outliers_and_keep_closest_points()
        self._reassign_small_clusters()
        self._assign_starting_points()

    def generate_output(self) -> None:
        """
        Generate the output data and save it to a CSV file.
        The generated output contains the following columns: user_id, start_point_id, start_point_latitude,
        start_point_longitude,potential_group_members, group_id, group_size, and distance.
        Distance is in kilometers and measured to the user's closest starting point.
        """
        output_data = self.df[
            ['user_id', 'start_point_id', 'start_point_latitude', 'start_point_longitude', 'potential_group_members',
             'group_id', 'group_size', 'distance']]

        self._clean_self_from_members(output_data)

        with open(self.output_file, 'w') as output_file:
            try:
                output_data.to_csv(output_file, index=False)
                logging.info(f"Generated newsletter CSV: {self.output_file}")
            except Exception:
                logging.exception(
                    "An error occurred during CSV output generation.")

    def _clean_self_from_members(self, output_data: pd.DataFrame) -> None:
        """
        Cleans the 'potential_group_members' column by removing the user's own ID from the list of group members.
        """
        for i, row in output_data.iterrows():
            user_id = row['user_id']
            members = row['potential_group_members'].split(',')
            members = [member for member in members if member != user_id]
            joined_members = ','.join(members)
            output_data.loc[i, 'potential_group_members'] = joined_members

    def generate_newsletter_csv(self) -> None:
        """
        Group individuals and generate a CSV file for the newsletter with the required columns.
        """
        try:
            self.open_csv_file()
        except (FileNotFoundError, IOError):
            sys.exit(1)

        try:
            self.check_data_quality()
            self.cluster_data_points()
            self.group_users()
            self.generate_output()
        except Exception:
            logging.exception(
                "An error occurred during newsletter CSV generation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a newsletter CSV file for group cycling events.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    newsletter_generator = NewsletterGenerator(input_file, output_file)
    newsletter_generator.generate_newsletter_csv()


if __name__ == '__main__':
    main()
