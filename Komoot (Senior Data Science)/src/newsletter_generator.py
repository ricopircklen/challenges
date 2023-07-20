from .utils import calculate_distance, get_centermost_point
import pandas as pd
from sklearn.cluster import OPTICS
import numpy as np
from geopy.distance import great_circle
import uuid
import math
from typing import Tuple, List
import logging
from .exceptions import FileFormatError, MissingColumnsError, InsufficientRowsError


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

    def __init__(self, input_file: str, output_file: str, max_group_size: int = 40, max_chunk_size: int = 10000,
                 min_cluster_size: int = 5, xi: float = 0.01) -> None:
        if not input_file.endswith(('.csv', '.csv.gz')):
            raise FileFormatError("Input file is not a CSV file")
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.centermost_points = None

    def _read_csv_file(self) -> None:
        """
        Reads a CSV file and stores the data into a dataframe. It handles both regular and gzip-compressed CSV files.
        If an error occurs while opening the file, an error message is logged and the exception is re-raised.
        Reads large files in chunks and concatenates them for reducing memory usage. Chunk size specified in max_chunk_size.

        Raises:
        -------
        FileNotFoundError
            If the file cannot be found.
        IOError
            If an I/O error occurs while opening the file.
        """
        try:
            if self.input_file.endswith('.csv.gz'):
                chunks = pd.read_csv(self.input_file, compression='gzip', chunksize=self.max_chunk_size)
            else:
                chunks = pd.read_csv(self.input_file, chunksize=self.max_chunk_size)
            for chunk in chunks:
                self.df = pd.concat([self.df, chunk])
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Error opening file: {e}")
            raise
        logging.info(f"Finished reading file {self.input_file}")

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses the data in the dataframe by removing null value rows and incorrectly formatted geolocation values.

        Returns:
        --------
        pd.DataFrame
            The preprocessed dataframe without null values.
        """
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(
            self.df['longitude'], errors='coerce')
        self.df.dropna(inplace=True)
        return self.df

    def _validate_data(self):
        """
        Checks for missing or incorrectly labeled columns, and verifies that the dataframe has at least 5 data points.
        If these criteria are not met, an appropriate error message is raised.
        """
        required_columns = {'latitude', 'longitude', 'user_id'}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            missing_columns_str = ', '.join(missing_columns)
            raise MissingColumnsError(
                f"The input DataFrame is missing the following required columns: {missing_columns_str}.")

        preprocessed_df = self._preprocess_data()
        if preprocessed_df.shape[0] < self.min_cluster_size:
            raise InsufficientRowsError("The input dataset has fewer than 5 rows of values in correct format.")

        logging.info("Data validation passed")
        return self.df

    def _cluster_data_points(self) -> pd.DataFrame:
        """
        Assign each data point into a cluster.
        """
        logging.info("Clustering data points")
        self.df['latitude_rad'] = np.radians(self.df['latitude'])
        self.df['longitude_rad'] = np.radians(self.df['longitude'])
        self.df['cluster_label'], cluster_labels = self.run_optics(self.min_cluster_size, xi=self.xi)
        num_clusters = len(set(cluster_labels))

        coords = self.df[['latitude', 'longitude']].values
        clusters = pd.Series([coords[cluster_labels == n] for n in range(
            num_clusters) if len(coords[cluster_labels == n]) > 0])
        self.centermost_points = clusters.map(get_centermost_point)

        self._assign_cluster_for_outliers()
        self._filter_unclustered_and_keep_closest_points()
        self._reassign_small_clusters()
        self._assign_starting_points_for_each_cluster()
        logging.info(f"Finished clustering data points into {len(self.centermost_points)} clusters")

    def _find_closest_cluster(self, user_id):
        user_coords = self.df.loc[self.df['user_id'] == user_id, ['latitude', 'longitude']].values
        closest_cluster = min(self.centermost_points, key=lambda point: min(
            great_circle(point, coord).m for coord in user_coords))
        closest_cluster_idx = self.centermost_points[self.centermost_points == closest_cluster].index[0]
        return closest_cluster_idx

    def run_optics(self, min_samples: int = 5, xi: float = 0.01) -> Tuple[pd.Series, List]:
        """
        Runs OPTICS clustering on the given dataframe using the specified min_samples and xi.
        Utilises Haversine metric that is approriate for geospatial data.

        Args:
            min_samples (int): The minimum number of samples required for a group to be considered as a cluster.
            xi (float): The parameter for the OPTICS algorithm, specifying a minimum steepness on the reachability plot.

        Returns:
            Tuple[pd.Series, List]:
                - pd.Series: The cluster labels assigned by OPTICS.
                - List: The labels from the OPTICS object.
        """
        optics = OPTICS(min_samples=min_samples, xi=xi, metric="haversine")
        self.df['cluster_label'] = optics.fit_predict(
            self.df[['latitude_rad', 'longitude_rad']])
        return self.df['cluster_label'], optics.labels_

    def _assign_users_to_groups(self) -> pd.DataFrame:
        """
        Groups users to groups of maximum size specified in self.max_group_size.
        """
        group_data = []
        unique_clusters = self.df['cluster_label'].unique().tolist()

        for cluster_label in unique_clusters:
            cluster_points = self.df[self.df['cluster_label'] == cluster_label]
            cluster_members = cluster_points['user_id'].tolist()
            unique_users = cluster_points['user_id'].nunique()
            num_groups = max(1, math.ceil(unique_users / self.max_group_size))
            group_size = math.ceil(unique_users / num_groups)
            self._handle_grouping(group_data, cluster_members, unique_users, group_size, num_groups, cluster_label)
        self._assign_group_data(group_data)

    def _handle_grouping(self, group_data: List[dict], cluster_members: List[str], unique_users: int, group_size: int, num_groups: int, cluster_label: int) -> None:
        """
        Handles the process of grouping the users within a specific cluster into groups with a unique group ID. 
        """
        for _ in range(num_groups):
            group_id = str(uuid.uuid4())
            members = self._get_group_members(cluster_members, unique_users, group_size)
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
            logging.error("Attempting to access an empty list: cluster_members. Please check your data.")
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
            self.df.loc[self.df['user_id'].isin(members), ['group_id', 'potential_group_members', 'group_size']] = [
                group_id, ','.join(members), group_size]

    def _assign_cluster_for_outliers(self):
        '''Assigns the closest cluster for users without a cluster by finding the closest one from all the users points'''
        users_without_cluster = []
        for user_id in self.df[self.df['cluster_label'] == -1]['user_id'].unique():
            if self.df[self.df['user_id'] == user_id]['cluster_label'].nunique() == 1:
                users_without_cluster.append(user_id)

        for user_id in users_without_cluster:
            closest_cluster_idx = self._find_closest_cluster(user_id)
            self.df.loc[self.df['user_id'] == user_id, 'cluster_label'] = closest_cluster_idx

    def _filter_unclustered_and_keep_closest_points(self):
        '''
        Drops unclustered points and keep only the rows with the smallest distance for each user_id
        '''
        self.df = self.df[self.df['cluster_label'] != -1]
        self.df['distance'] = self.df.apply(lambda row: calculate_distance(
            (row['latitude'], row['longitude']), self.centermost_points[row['cluster_label']]), axis=1)
        min_distance_indices = self.df.groupby('user_id')['distance'].idxmin()
        self.df = self.df.loc[min_distance_indices]

    def _reassign_small_clusters(self):
        '''
        Assigns the closest cluster for users in small clusters by finding the closest one from all the users points
        '''

        # Identify small clusters and valid (non-small) clusters
        cluster_sizes = self.df.groupby('cluster_label')['user_id'].nunique()
        small_clusters = cluster_sizes[cluster_sizes < self.min_cluster_size].index
        valid_clusters = cluster_sizes[cluster_sizes >= self.min_cluster_size].index

        # Filter centermost_points to include only valid clusters
        self.centermost_points = self.centermost_points[self.centermost_points.index.isin(valid_clusters)]
        users_in_small_clusters = []
        for cluster in small_clusters:
            users_in_small_clusters.extend(self.df[self.df['cluster_label'] == cluster]['user_id'].unique())

        for user_id in users_in_small_clusters:
            closest_cluster_idx = self._find_closest_cluster(user_id)
            self.df.loc[self.df['user_id'] == user_id, 'cluster_label'] = closest_cluster_idx

    def _assign_starting_points_for_each_cluster(self) -> None:
        """
        Creates a dictionary mapping cluster labels to starting point IDs and add starting points.
        """
        start_point_ids = {index: uuid.uuid4() for index in self.centermost_points.index}
        self.df['start_point_id'] = self.df['cluster_label'].map(start_point_ids)
        self.df['start_point_latitude'] = self.df['cluster_label'].map(lambda x: self.centermost_points[x][0])
        self.df['start_point_longitude'] = self.df['cluster_label'].map(lambda x: self.centermost_points[x][1])

    def _clean_self_from_members(self, output_data: pd.DataFrame) -> None:
        """
        Cleans the 'potential_group_members' column by removing the user's own ID from the list of group members.
        """
        for i, row in output_data.iterrows():
            user_id = row['user_id']
            members = (member for member in row['potential_group_members'].split(
                ',') if member != user_id)
            output_data.loc[i, 'potential_group_members'] = ','.join(members)

    def _generate_output_csv(self) -> None:
        """
        Generates the output data and save it to a CSV file.
        The output distance is in kilometers and measured to the user's closest starting point.
        """
        logging.info("Start generating output CSV")
        output_data = self.df[['user_id', 'start_point_id', 'start_point_latitude', 'start_point_longitude', 'potential_group_members',
                               'group_id', 'group_size', 'distance']]

        self._clean_self_from_members(output_data)

        with open(self.output_file, 'w') as output_file:
            try:
                output_data.to_csv(output_file, index=False)
                logging.info(f"Generated newsletter CSV: {self.output_file}")
            except Exception:
                logging.exception("An error occurred during CSV output generation.")

    def generate_newsletter(self) -> None:
        logging.info("Start generating newsletter")
        try:
            self._read_csv_file()
            self._validate_data()
            self._cluster_data_points()
            self._assign_users_to_groups()
            self._generate_output_csv()
        except Exception:
            logging.exception(
                "An error occurred during newsletter CSV generation.")
            raise
