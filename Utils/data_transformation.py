import tensorflow as tf
import numpy as np
import logging
import pdb

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


def se_func(a, b):
    """compute squared error"""
    assert(len(np.shape(a))==len(np.shape(b)))
    return np.square(a-b)

def ae_func(a, b):
    """compute absolute error"""
    assert (len(np.shape(a)) == len(np.shape(b)))
    return np.abs(a-b)

def mse_func(a, b):
    """compute mean squared error"""
    return np.mean(se_func(a,b), axis=0)


def mae_func(a, b):
    """compute mean absolute error"""
    return np.mean(ae_func(a,b), axis=0)

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

class AverageTracker:
    """
    The object tracks the mean value of the accumulated values.
    The object can be initialized to numpy array or to a floating point scalar.
    Can repeatedly receive values (using update function) to accumulate.
    Upon demand, the mean() function returns the average values (normalized by
    the number of the accumulations.
    """

    def __init__(self, initial_value):
        self.value = initial_value
        self.num_accumulations = 0

    def mean(self):
        return self.value / float(self.num_accumulations)

    def sum(self):
        return self.value

    def update(self, value_to_accumulate):
    #type(value_to_accumulate) != type(self.value)
        if ((np.isscalar(value_to_accumulate) != np.isscalar(self.value))):
            ld("Warning: Attempted to update AverageTracker which was "
               "configured to receive type {}, by a value "
               "from type {}. No update will take place.".format(type(self.value), type(value_to_accumulate)))
            return
        elif (type(self.value) == np.ndarray and self.value.shape != value_to_accumulate.shape):
            ld("Warning: Attempted to update AverageTracker which was "
               "configured to receive numpy ndarray of the shape {}, by a numpy ndarray "
               "of the shape {}. No update will take place.".format(self.value.shape, value_to_accumulate.shape))
            return
        self.value += value_to_accumulate
        self.num_accumulations += 1

    def reset(self):
        self.value *= 0.0
        self.num_accumulations = 0

class IdentityScaler:
    def __init__(self):
        pass
    def transform(self,data):
        return data
    def inverse_transform(self,data):
        return data
    def fit(self,data):
        pass
    def fit_transform(self,data):
        return data

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        self.scale_ = self._reshape(self._scaler.scale_)
        self.mean_ = self._reshape(self._scaler.mean_)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


class NDMinMaxScaler(TransformerMixin):
    def __init__(self, feature_range=(0,1), copy=True):
        self._scaler = MinMaxScaler(feature_range, copy)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        self.scale_ = self._reshape(self._scaler.scale_)
        self.min_ = self._reshape(self._scaler.min_)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


def tf_inverse_transform(scikitlearnScaler, tensor, tf_dtype):
    """
    Receives a tensor and a Scaler object which can be:
        scikitlearn's MinMaxScaler
        scikitlearn's StandardScaler
        this project's IdentityScaler

    :return: inverse_transformed tensorflow op applied on the tensor,
             with respect to  the scaler provided
    """
    orig_shape = tf.shape(tensor)
    flat_tensor = tf.reshape(tensor, [orig_shape[0], -1])
    if isinstance(scikitlearnScaler, MinMaxScaler) or isinstance(scikitlearnScaler, NDMinMaxScaler):
        scale_t = tf.constant(scikitlearnScaler.scale_, dtype=tf_dtype)
        min_t = tf.constant(scikitlearnScaler.min_, dtype=tf_dtype)
        flat_tensor_scaled = tf.divide(tf.subtract(tensor, min_t),scale_t)
        return tf.reshape(flat_tensor_scaled, orig_shape)
    elif isinstance(scikitlearnScaler, StandardScaler) or isinstance(scikitlearnScaler, NDStandardScaler):
        scale_t = tf.constant(scikitlearnScaler.scale_, dtype=tf_dtype)
        mean_t = tf.constant(scikitlearnScaler.mean_, dtype=tf_dtype)
        flat_tensor_scaled = tf.add(tf.multiply(flat_tensor, scale_t), mean_t)
        return tf.reshape(flat_tensor_scaled, orig_shape)
    elif isinstance(scikitlearnScaler, IdentityScaler):
        return tensor
    else:
        ld("The scaler provided to tf_inverse_transform() has type {} which is not supported".format(type(scikitlearnScaler)))


def tf_transform(scikitlearnScaler, tensor, tf_dtype):
    """
    Receives a tensor and a Scaler object which can be:
        scikitlearn's MinMaxScaler
        scikitlearn's StandardScaler
        this project's IdentityScaler

    :return: transformed tensorflow op applied on the tensor,
             with respect to the scaler provided
    """
    if isinstance(scikitlearnScaler, MinMaxScaler):
        scale_t = tf.constant(scikitlearnScaler.scale_, dtype=tf_dtype)
        min_t = tf.constant(scikitlearnScaler.min_, dtype=tf_dtype)
        return tf.add(tf.multiply(tensor,scale_t), min_t)
    elif isinstance(scikitlearnScaler, StandardScaler):
        scale_t = tf.constant(scikitlearnScaler.scale_, dtype=tf_dtype)
        mean_t = tf.constant(scikitlearnScaler.mean_, dtype=tf_dtype)
        return tf.divide(tf.subtract(tensor, mean_t), scale_t)
    elif isinstance(scikitlearnScaler, IdentityScaler):
        return tensor
    else:
        ld("The scaler provided to tf_inverse_transform() has type {} which is not supported".format(type(scikitlearnScaler)))
