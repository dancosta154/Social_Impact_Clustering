# Social Impact
### Evaluating state level economies by employment sector

### Executive Summary

Are US state economies distinct enough by employment sector to be clustered? Can these clusters provide insight into which US states or state economies are most resilient?

---

### File Structure
Notbooks should be run in order from `00 - 03`.

```
project
│     Clustering - Data Dictionary.md
│     EDA - Data Dictionary.md
│     README.md
│     00 - Data Collection.ipynb
│     01 - Clustering.ipynb
│     02 - Cluster Merge.ipynb
│     03 - EDA.ipynb
│         
│   
└── source_data
│     2018_with_clusters.csv
│     economies_2018.csv
│     economies_2021.csv
│
│
└── state_data
│     HTML files by state and industry
│
│
└── state_employment
│     Employment files by state 
│       
│          
└── state_metrics
      quarterly_personal_income_raw.csv
      Quarterly_Personal_Income.csv
```

---

### Background

How well does the US economy bounce back from economic downfalls? How would you measure economic resiliency? During the coronavirus, the global economy was shaken. Overall, the United States faired pretty well in comparison to many, but do particular sectors of the US economy or even particular states do better when various crisis cause economic hits? If so, why? Using mainly the unemployment rate, and employment-population ratio overtime as metrics to measure the state of the economy; these are all questions I set out to answer. 

---

### Data Formatting and Modeling

In order to collect the data that needed for this project, I scraped multiple pages from the Bureau of Labor Statistics. Initially, I was interested in extracting subsegmented industry data by state, which I was able to extract leveraging the BeautifulSoup library. I then stored these outputs in both dictionaries and dataframes, depending on their intended use cases. Once compiled, I was able to then transform these outputs to become more usable for the clustering models, by calculating aggregated statistical measures by state. 

Additionally, I found the need to extract employment/unemployment data by state, which was also stored within the Bureau of Labor Statistics website. In this case, the data was already in the appropriate format, so there was not a need to transform it. For the scraping process, I diverted from the previous method of using BeautifulSoup, and instead wrote HTML files to a directory to then be extracted. Finally, I appended the state information to the output to be used in clustering.

Now that the necessary data was in the preferred format, I progressed to clustering the data. In exploring the available data I thought it would be interesting to see what kind of relationships a model would form between the different US states and if these relationships could then provide insight of significance.

The clusters were formed based on economic industry. The different states were then set as the dataframe index with 22 features, ten distinct industries, and one as `Total Non-farm` which was essentially the sum of the others. From here, a subset of the data was scaled with sklearn's `StandardScaler`, then joined back with all of the data using a `FeatureUnion`. Once this data was compiled, it was then fed into `for loops` designed to determine the best clustering method and number of clusters. The KMeans loop ran through possible clusters from 2 - 25 and kept running track of the silhouette scores. A loop for Kmeans also ran to calculate and plot inertia scores in order to find the ideal elbow point of diminishing returns. The DBScan `for loop` ran through various epsilons and minimum samples per cluster to find the best possible silhouette score for cluster of 2 or more. Kmeans had the better and more cohesive scores. The inertia elbow provided a bit of wiggle room, as to which cluster size (3, 4, or 5) to choose as the scores were strong and relatively close. I chose 5 clusters as I believed this would lead to more interesting analysis and EDA with more granular groupings of states.

For EDA, I was able to input all of the clustering data into pivot tables for visualization. Once pivoted, I started with looking at the unemployment rate as a way to analyze how well the economy was doing. There were a total of five clusters. We observed a very large spike in unemployment rates during covid, looking at about a 10-15% unemployment rate increase across all clusters. The second cluster of states by far has the largest overall unemployment rate across ass industries. 

Overall the model seems to have clustered according to the rate of unemployment. California seems to have been an outlier because there was the least amount of data available.

Because of this EDA, we could argue that if the model clustered according the unemployment rate, that the first cluster (0 in our jupyter notebooks) seems to be the most economically stable states. But causality is hard to conclude in this region because there are a few potential factors:
1. Population. The population in the more economically stable regions seems to be less than those grouped in the second cluster. We could argue that a large reason their unemployment rate is lower is because they have fewer people.
2. Dominant industry present. The most dominant industry in the cluster 0’s group seems to be that of farming type industries. Because of this, we can conclude the areas are probably more rural. This means that maybe the population is living father apart from one another, extracting covid as a lesser rate, and therefore the state's economy being less effected by the pandemic. 
3. Age. Age could potentially play a part in the unemployment rate. If the states in cluster 0’s have a higher aged population, their unemployment rate may be deemed as more steady because less of the population was employed to begin with.

---

### Observations

Yes, US state economies exhibit distinctions by employment sector, allowing them to be clustered differently. In general, states with the greatest economic workforce are impacted the most.

---

### Next Steps

This analysis led to many possible avenues to continue exploring in our next steps. Incorporating additional factors, such as: age, homeless population, or population could prove very useful in trying to better understand a state's economic resiliency.

From a modeling perspective, a next step for this project would be to build a predictive model to ideally predict unemployment drops based on the clusters.
