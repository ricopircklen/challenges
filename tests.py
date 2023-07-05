import pandas as pd
import numpy as np
import unittest
from main import preprocessing, check_data_quality, run_dbscan

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'latitude': [1.0, 2.0, 3.0, np.nan, 5.0],
            'longitude': [10.0, 20.0, np.nan, 40.0, 50.0],
            'user_id': [101, 102, 103, 104, 105],
            'latitude_rad': np.radians([1.0, 2.0, 3.0, np.nan, 5.0]),
            'longitude_rad': np.radians([10.0, 20.0, np.nan, 40.0, 50.0])
        })

    def test_preprocessing(self):
        # Test the preprocessing function
        preprocessed_df = preprocessing(self.df)

        # Check that null value rows are removed
        self.assertEqual(len(preprocessed_df), 3)

        # Check that latitude and longitude columns are in correct format
        self.assertTrue(pd.api.types.is_numeric_dtype(preprocessed_df['latitude']))
        self.assertTrue(pd.api.types.is_numeric_dtype(preprocessed_df['longitude']))

    def test_check_data_quality(self):
        # Test the check_data_quality function
        # Case: Missing required columns
        missing_columns_df = pd.DataFrame({'latitude': [1.0, 2.0, 3.0, 5.0], 'longitude': [10.0, 20.0, 40.0, 50.0]})
        with self.assertRaises(SystemExit):
            check_data_quality(missing_columns_df)

        # Case: Less than 5 rows
        less_rows_df = pd.DataFrame({
            'latitude': [1.0, 2.0, 3.0],
            'longitude': [10.0, 20.0, 30.0],
            'user_id': [101, 102, 103]
        })
        with self.assertRaises(SystemExit):
            check_data_quality(less_rows_df)

    def test_run_dbscan(self):
        # Test the run_dbscan function
        epsilon = 0.5
        min_samples = 3

        self.df = preprocessing(self.df)
        # Run DBSCAN
        labels, _ = run_dbscan(self.df, epsilon, min_samples)

        # Check that cluster labels are assigned correctly
        self.assertEqual(labels.tolist(), [-1, -1, -1])

if __name__ == '__main__':
    unittest.main()
