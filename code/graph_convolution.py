from keras.layers.core import *

from keras import backend as K
from keras.engine.topology import Layer


class GraphConv(Layer):
    '''Convolution operator for graphs.

    REQUIRES THEANO BACKEND (see line 133).
	
    Implementation reduce the convolution to tensor product, 
    as described in "A generalization of Convolutional Neural 
    Networks to Graph-Structured Data".  

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers e.g. `(1000, 1)` for a graph 
    with 1000 features (or nodes) and a single filter.

    # Arguments
        nb_filter: Number of convolution kernels to use
            (dimensionality of the output).
		nb_neighbors: the number of neighbors the convolution
		would be applied on (analogue to filter length)
        neighbors_ix_mat: A matrix with dimensions
	    (variables, nb_neighbors) where the entry [Q]_ij
	    denotes for the i's variable the j's closest neighbor.
        weights: list of numpy arrays to set as initial weights.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: Number of filters/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        3D tensor with shape:
        `(batch_size, features, filters)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, features, nb_filter)`.
    '''
    def __init__(self, nb_filter, nb_neighbors,
			neighbors_ix_mat, weights=None,
                 init='uniform', activation='linear',
                 W_regularizer=None, b_regularizer=None, 
                 activity_regularizer=None, W_constraint=None, 
			b_constraint=None, bias=True,
			input_dim=None, input_length=None, **kwargs):

        if K.backend() != 'theano':
            raise Exception("GraphConv Requires Theano Backend.")

        self.nb_filter = nb_filter     
        self.nb_neighbors = nb_neighbors
        self.neighbors_ix_mat = neighbors_ix_mat
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        self.initial_weights = weights

        self.init = initializations.get(init, dim_ordering='th')
        self.activation = activations.get(activation)
        
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)	
        super(GraphConv, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.W_shape = (self.nb_neighbors, input_dim, self.nb_filter)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        
        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)
        
        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
                
    def call(self, x, mask=None):
        x_expanded = x[:,self.neighbors_ix_mat, :]
	  
        #Tensor dot implementation requires theano backend
        output = K.T.tensordot(x_expanded, self.W, [[2,3],[0,1]])
        if self.bias:
            output += K.reshape(self.b, (1, 1, self.nb_filter))
        output = self.activation(output)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.nb_filter)

