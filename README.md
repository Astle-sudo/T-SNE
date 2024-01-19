#  T-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data into a space of two or three dimensions for the purpose of visualization. It is capable of capturing complex data structures while retaining local structure, making it ideal for visualizing dataset geometry.

The main idea behind t-SNE is to model the similarity between data points in the input high dimensional space and also between mapped points in the reduced space, while optimizing the embeddings to reflect structure at many scales.

### Mathematical Details 

Given a dataset X={x1, x2, ..., xN} with N points that contains features in a high-dimensional space Rd, t-SNE first converts the high-dimensional Euclidean distances between datapoints into conditional probabilities that represent similarities. The similarity between xi and xj is calculated as:

$$p_{j|i}=\frac{\exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i-x_k||^2/2\sigma_i^2)}$$

Where σi is the variance of the Gaussian distribution centered on datapoint xi. 

In the reduced space R^d (d=2 or 3), t-SNE defines a Student t-distribution to measure similarity between two points yi and yj:

$$q_{j|i}=\frac{(1+||y_i-y_j||^2)^{-1}}{\sum_{k \neq i} (1+||y_i-y_k||^2)^{-1}}$$

t-SNE optimizes the locations of the points in the reduced space by minimizing the Kullback–Leibler divergence between the conditional distributions P and Q using gradient descent. The cost function with respect to y1, ..., yN is:  

$$C = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

The gradients of this cost function with respect to the mapped points yt are used to iteratively update the locations of points in the low-dimensional map to reflect meaningful patterns in the original high-dimensional data.

The end result is a projection of the data into two or three dimensions which models both the local and global structure in such a way that visually similar datapoints will be modeled as nearby points.
