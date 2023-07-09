# Komoot Challenge

## Installation

1. Recommended to run a virtual env (e.g. `python3 -m venv myenv`; Win: `myenv\Scripts\activate` / MacOS & Linux: `source myenv/bin/activate`)
2. Install required packages by running `pip install -r requirements.txt`
3. Run the script by `python main.py tours.csv.gz output.csv` where the first argument is the name of the input file and the second argument the name of the output file.
4. Tests are in tests.py and can be run with `python -m unittest tests/tests.py`

## Logic

- The selected strategy is to use OPTICS to cluster users into points so that each group would have an arbitraty minimum of 5 individuals.
- After clustering, outlier data points that weren't clustered, are assigned to their closest existing cluster, respectively.
- Next, for every unique user, we save only their closest cluster. This is defined as the smallest distance to any single starting point the user has. We propose only one group for each user to avoid the situation where a user will get spammed by dozens of groups.
- Lastly, clusters that end up having less than 5 individuals, will be reassigned to their closest clusters, respectively.
- Next, each cluster is grouped into sets of maximum 40 members in order to have a reasonable group size for a collaborative activity. Groups are assigned randomly but fairly, e.g. a cluster of 41 users is divided into groups of 20 and 21 users.
- The starting point is chosen as the point that is nearest to the cluster’s centroid. This is based on the assumption that the existing input starting points are already proven to be "good".

## Todos and improvements

- First thing would be to define the requirements, especially regarding the expected input dataset size, as well as whether the input is considered to be a single area or a bigger region.
- Error handling and testing: The code has not been rigorously tested on edge cases e.g. wrong input and output file formats, empty or massive datasets, and cases e.g. wrong format of latitudes and longitudes.
- Memory management and performance: The code doesn’t consider that the input datasets might be massive, slowing down the system or running out of memory. Also tools like Dask or Vaex may perform better than pandas for bigger datasets.
- Data validation: Additional checks can be added for the incoming data to ensure its validity. For example, checking that the latitude and longitude are within valid ranges.
- Result validation: Processing potential outliers, e.g. removing users from newsletters that are unreasonably far away from the starting point might make sense.
- Code structure: If the project size would increase, the code shall be split into multiple files and modules for better maintainability and readability, starting from separating the clustering algorithm.
- Improved feature engineering and clustering algorithm for better results. E.g. prioritizing areas where the user has higher weight of points, potentially indicating preference towards such starting point. Such user specific preferences could be analyzed further and used to improve the algorithm.
- Improved logging
- Pinning dependencies to specific version in requirements.txt
- Refine the grouping process
- Improved UX so that a non-coder user could utilize the script could be beneficial, e.g. through streamlit.
- Simple tricks e.g. shuffling the input data tends to be suggested for ML algorithms and might affect the behaviour depending on how the input dataset was formed.
- Single user might have multiple points in a cluster, and therefore have a more significant effect on calculating the centroid. This can be considered positive since an active user might deserve a higher weight on deciding the starting point. Alternative, the centroid could be calculated equally, considering only one point per user.
- The starting point is likely to be in an inconvenient location, and we could consider using google maps or open map data to automatically detect and propose the nearest landmark for a more practical and easier starting point.
- Other considerations: Due to GDPR, it sounds slightly sketchy to share information of users that live or like to cycle close to your location, and might be more suitable for the US than the European market in the proposed newletter format.

## Process to getting to the solution

The process was rather straightforward: I started with a small EDA to have a basic understanding of the data (eda.ipynb). After this, I coded a quick and simple brute force script to find the nearest users in order to have a benchmark. This was clearly a very slow and simple approach, so I improved it using the tools and logic I was familiar and knew to perform well for geolocations (DBSCAN). For comparison, HDBSCAN and OPTICS were tested. The rest, e.g. grouping user came on the fly by noticing that a cluster of thousands of users does not sound practical for the suggested purpose.

## Strengths and limitations

- DBScan (especially with the selected ball tree algo) is very simple and straightforward algorithm that is much more performant (especially computationally) than brute forcing data points, and is known to be a robust clustering algorithm for geolocations. Therefore, a strong candidate as a first choice for exploratory analysis.
- HDBSCAN doesn't not require DBSCAN's epsilon, and can find clusters of varying densities - a clear improvement for the selected use case. HDBSCAN is especially a good choice when you don't know how many clusters to expect and you believe there might be noise and varying density. However, the algorithm is computationally much more intensive.
- Compared to HDBSCAN, OPTICS provides additional insight into the structure of the data, which may be useful for more complex analyses.
- In general, the limitations of the method can be seen to include the hard coded minimum number of samples in a cluster.
- Other limitations of the solution in general are discussed in todos and improvements -section, e.g. lack of personal preferences in cases where a user has multiple data points. Also, the selected method would become computationally very intensive in cases of massive amount of datapoints. Improved preprocessing could help, e.g. selecting for each user only a few most significant data points.

## Learnings

I haven't dealt with unsupervised learning and clustering in a while so it was refreshing to play around with such models for a change. My main learnings however came from the domain side and data by starting to consider the behaviour and preferences of a user, and all the different ways the results could be improved and to make them better suit for the use case.

I'm expecting the main learnings to come from the feedback and expect feedback especially regarding the code quality, and the selected algorithm.
