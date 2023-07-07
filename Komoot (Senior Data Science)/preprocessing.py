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