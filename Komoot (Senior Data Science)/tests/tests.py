import pandas as pd
import numpy as np
import unittest
from main import NewsletterGenerator, main
from src.exceptions import FileFormatError, MissingColumnsError, InsufficientRowsError
import sys
from src.utils import calculate_distance, get_centermost_point


class TestNewsletterGenerator(unittest.TestCase):
    def setUp(self):
        self.newsletter_generator = NewsletterGenerator("input.csv", "output.csv")
        self.sample_data = pd.DataFrame({
            'user_id': ['101', '102', '103', None, '105'],
            'latitude': [50.7749, None, 51.8781, 51.7604, 50.7392],
            'longitude': [13.4194, 13.2437, 13.6298, 13.3698, 13.9903],
        })

        self.sample_data['latitude_rad'] = np.radians(self.sample_data['latitude'])
        self.sample_data['longitude_rad'] = np.radians(self.sample_data['longitude'])

    def test_validate_data(self):
        self.newsletter_generator.df = self.sample_data.drop('user_id', axis=1)
        self.assertRaises(MissingColumnsError, self.newsletter_generator._validate_data)

        # Test data with insufficient rows assuming MIN_CLUSTER_SIZE = 5
        self.newsletter_generator.df = self.sample_data.dropna()
        if len(self.newsletter_generator.df) < 5:
            self.assertRaises(InsufficientRowsError, self.newsletter_generator._validate_data)

    def test_init_with_invalid_file_format(self):
        with self.assertRaises(FileFormatError):
            NewsletterGenerator("input.txt", "output.csv")

    def test_main_with_invalid_args(self):
        sys.argv = ['main.py', 'input.txt', 'output.csv']
        with self.assertRaises(SystemExit):
            main()

    def test_preprocessing(self):
        self.newsletter_generator.df = self.sample_data
        # Test the preprocessing function
        preprocessed_sample_data = self.newsletter_generator._preprocess_data()
        # Check that null value rows are removed
        self.assertEqual(len(preprocessed_sample_data), 3)
        # Check that latitude and longitude columns are in correct format
        self.assertTrue(pd.api.types.is_numeric_dtype(
            preprocessed_sample_data['latitude']))
        self.assertTrue(pd.api.types.is_numeric_dtype(
            preprocessed_sample_data['longitude']))

    def test_calculate_distance(self):
        start_point1 = (50.7749, 13.4194)
        start_point2 = (50.0522, 13.2437)
        distance = calculate_distance(start_point1, start_point2)
        self.assertAlmostEqual(distance, 81.3551782656741, places=1)

    def test_get_centermost_point(self):
        cluster = [(50.7749, 13.4194), (50.0522, 14.2437), (50.8781, 13.6298), (51.7604, 12.3698), (29.7604, 13.3698)]
        centermost_point = get_centermost_point(cluster)
        self.assertEqual(centermost_point, (50.0522, 14.2437))


if __name__ == '__main__':
    unittest.main()
