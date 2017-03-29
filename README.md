# A generalization of Convolutional Neural Networks to Graph-Structured Data

This is supplementary code to "A generalization of Convolutional Neural Networks to Graph-Structured Data", by Yotam Hechtlinger, Purvasha Chakravarti and Jining Qin.

------------------

### Basic example
The graph convolution is implemented as a Keras layer, receiving the matrix Q of the expected number of visits as an argument. The implementation details are discussed in the paper.
```python
from graph_convolution import GraphConv

g_model = Sequential()
g_model.add(GraphConv(nb_filter=nb_filter, Q_matrix=Q_matrix, 
                      nb_neighbors=nb_neighbors, input_shape=(1000,1)))
```

------------------

### Merck Dataset
The DPP4 dataset is part of the Merck Molecular Activity Challenge, a previous [Kaggle](https://www.kaggle.com/c/MerckActivity) competition. The data used here was downloaded from the [supplementary material](http://pubs.acs.org/doi/suppl/10.1021/ci500747n) of the paper "Deep Neural Nets as a Method for Quantitative Structure–Activity Relationships" by Ma et al.

------------------

### Dependencies
Requires Keras using the Theano backend. 


