# Cluster_analysis_in_BioColour_project
## Introduction
We apply common unsupervised learning methods to discover hidden clusters emerging among bio-dyed textile samples, and show the potential of clustering techniques in this application domain. In detail, we use agglomerative clustering, Fuzzy C-means, OPTICS as well as a well-known artificial neural network (ANN), namely self-organizing maps (SOM), resulting in an investigation that combines data visualization and cluster analysis.

## Requirements:
| Software  | Version |
| ------------- | ------------- |
| Python  | 3.7.0  |
| Numpy  | 1.17.2  |
| pandas  | 1.0.4  |
| matplotlib  | 3.2.1  |
| scikit-learn  | 0.21.3  |
| colormath  | 3.0.0  |
## Citation:
In particular, we use its implementation of the evaluation measures:

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html 
* as well as for the **agglomerative hierarchical clustering algorithm**:

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html 
* and the **OPTICS algorithm**:

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html 

For the other algorithms, we use separate existing implementations, respectively
* for the **Fuzzy C-means algorithms**: Madson Luiz Dantas Dias. (2019). fuzzy-c-means: An implementation of Fuzzy C-means clustering algorithm., Zenodo, doi = 10.5281/zenodo.3066222 

https://github.com/omadson/fuzzy-c-means 
* for the **two-dimensional Self-Organizing Maps**: Giuseppe Vettigli. (2018). Mini-Som: minimalistic and NumPy-based implementation of the Self Organizing Map.

https://github.com/JustGlowing/minisom 
* for the **Growing Hierarchical Self-Organizing Map**: Civitelli E., Teotini F. (2018). An implementation of Growing Hierarchical SOM algorithm. 

https://github.com/enry12/growing_hierarchical_som


## Some results
 The distribution of bio-dyed samples in 2D space after PCA and corresponding clustering results by different unsupervised learning methods.
 ![image](https://user-images.githubusercontent.com/34623632/134513246-adad5653-700c-4491-8617-5b3f8742d9ff.png)


 One example in cluster analysis by 2D SOM in size of 140 neurons：
![140](https://user-images.githubusercontent.com/34623632/129462337-79c85620-7694-41a1-9bf9-5051b90e55c4.png)
 
  One example in cluster analysis by GHSOM in size of 20 neurons：
  ![image](https://user-images.githubusercontent.com/34623632/134515098-8ef2097c-5732-4621-a9bc-d0f48d90f274.png)

