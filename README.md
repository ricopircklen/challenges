# Komoot Challenge

## Installation

1. Recommended to run a virtual env (e.g. `python3 -m venv myenv`; Win: `myenv\Scripts\activate` / MacOS & Linux: `source myenv/bin/activate`)
2. Install required packages by running `pip install -r requirements.txt`
3. Run the script by `python main.py tours.csv.gz output.csv` where the first argument is the name of the input file and the second argument the name of the output file.

## Logic

- The selected strategy is to use DBSCAN to cluster users into points so that each group would have an arbitraty minimum 5 individuals and maximum 40 to make the potential group to have a reasonable size for a collaborative activity.
- The starting point is chosen as the point that is nearest to the cluster’s centroid. This is based on the assumption that the starting points are already proven to be "good".
- Users that do not fall into any cluster, are appointed to a cluster that is closest to him/her. We propose only one group for each user_id to avoid the situation where a user will get spammed by dyzens of different groups.
- Each user might have multiple points and belong to multiple clusters, yet only the cluster where the distance to the starting point is the smallest, is selected.
- Each cluster that has over 40 members is split into smaller groups to make the activity easier to organize.

## Todos and improvements

This code was generated in the proposed 2 hours, and lots were left out.

- There's a high risk for typos and bugs, and a slightly more comprehensive analysis (e.g. manually checking if the generated results make sense) would be recommended before deploying such script into production.
- From coding perspective, there is a lot of room for improvements, including error handling, adding tests, logging, typed inputs, reformatting, and cleaning the code in general. If utilized for bigger datasets, a further optimization would be recommended. Also a better UX so that a non-coder user could utilize the script could be beneficial, e.g. through streamlit, including also comments on wrong behaviour (e.g. wrong input type or column names).
- We have set an arbitraty value of 1km for DBSCAN epsilon. Selecting a metric e.g. mean distance to the starting point, and using grid search or other means to optimize the hyperparameters.
- Additionally, we could consider running multiple DBSCANs, e.g. first clustering tightly those within 100m from each other, then extending the search to 1km, and so on until each member is in a cluster. We could also consider other algorithms e.g. HDBSCAN.
- Single user might have multiple points in a cluster, and therefore have a more significant effect on calculating the centroid. This can be considered positive since an active user might deserve a higher weight on deciding the starting point. Alternative, the centroid could be calculated equally, considering only one point per user.
- The starting point is likely to be in an inconvenient location, and we could consider using google maps or open map data to automatically detect and propose the nearest landmark for a more practical and easier starting point.
- There is a chance that a group is formed around multiple data points from a single user, and after cleaning out, the group does not have the arbitrarily selected minimum of 5 members. Such groups could be considered to be appointed to other groups.
- The maximum distance of the most distant user is currently almost 40km. Such cases could be considered to be filtered out or handled with an improved clustering algorithm, if the point is to ride to the starting point for a 50km ride.
- Other considerations: Due to GDPR, it sounds slightly sketchy to share information of users that live or like to cycle close to your location, and might be more suitable for the US than the European market in the proposed newletter format.
- Potential improvements also include considering if users have some preferences regarding a starting point, e.g. one starting point might be close to them but the user might tend to prefer some more distant starting point, which could be indicated by the user having multiple data points in a specific region. Such user specific preferences could be analyzed further to detect specific preferences.

## Process to getting to the solution

The process was rather straightforward: I started with a small EDA to have a basic understanding of the data. After this, I coded a quick and simple brute force script to find the nearest users in order to have a benchmark. This was clearly a very slow and simple approach, so I improved it using the tools and logic I was familiar and knew to perform well for geolocations (DBSCAN). The rest, e.g. grouping users came on the fly by noticing that a cluster of thousands of users does not sound practical for the suggested purpose. Most of such weaknesses were acknowed above but left in place for the sake of keepign the task compact.

## Strengths and limitations

DBScan (especially with the selected ball tree algo) is very simple and straightforward algorithm that is much more performant (especially computationally) than brute forcing data points, and is known to be a robust clustering algorithm for geolocations. The limitations of the method come e.g. from selecting a single epsilon and minimum number of samples. In order to improve this, we would need either run multiple DBSCANs or to utilize a different algo, to get better clustering for varying environments, e.g. in the city center users are within hundreds of meters from each other, and the clustering could be much more tighter while further away, the clustering would benefit from being looser.

Other limitations come from the practical point of view, e.g. potentially ending up with too big or small clusters, since a single user with multiple data points might generate a cluster of his/her own. Such cases would be easiest to take into account through post processing the results.

From code and real life usability perspective, the limitations are mostly discussed in todos and improvements -section.

## Learnings

I haven't dealt with unsupervised learning and clustering in a while so it was refreshing to deal with such models and data for a change. My most learnings however came from the application side by starting to consider the behaviour and preferences of a user and all the ways such data could be user and the results could be improved by taking all that into account.
