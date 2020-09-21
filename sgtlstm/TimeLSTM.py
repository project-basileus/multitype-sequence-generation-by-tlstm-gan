"""
Implementation of time-lstm models in TF.keras

Models proposed by paper:
[What to Do Next: Modeling User Behaviors by Time-LSTM]
https://www.researchgate.net/publication/318830264_What_to_Do_Next_Modeling_User_Behaviors_by_Time-LSTM
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, RNN
from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask, _standardize_args, _is_multiple_state,\
 _generate_zero_filled_state_for_cell, _generate_zero_filled_state, _caching_device, RECURRENT_DROPOUT_WARNING_MSG


""" All Kinds of Time-LSTM Cells"""

class TimeLSTMCell0(DropoutRNNCellMixin, Layer):
  """Time-LSTM 0: no peepholes, classical LSTM with time gate
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    time_kernel_initializer: Initializer for the `time_kernel` weights matrix,
      used for the linear transformation of the time difference.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    time_kernel_regularizer: Regularizer function applied to
      the `time_kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    time_kernel_constraint: Constraint function applied to
      the `time_kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
  Call arguments:
    combined_inputs: A tuple of 2D tensors [inputs, delta_t].
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               time_kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               time_kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               time_kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(TimeLSTMCell0, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.time_kernel_initializer = initializers.get(time_kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.time_kernel_regularizer = regularizers.get(time_kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.time_kernel_constraint = constraints.get(time_kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
    # and fixed after 2.7.16. Converting the state_size to wrapper around
    # NoDependency(), so that the base_layer.__setattr__ will not convert it to
    # ListWrapper. Down the stream, self.states will be a list since it is
    # generated from nest.map_structure with list, and tuple(list) will work
    # properly.
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    # changed kernel shape to 5, added W_xt; input_dim=D_emb+1, the last dimension is for delta_t
    self.kernel = self.add_weight(
        shape=(input_dim-1, self.units * 5), 
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    # changed: added time kernel, [V_tt, V_to]
    self.time_kernel = self.add_weight(
        shape=(1, self.units * 2),
        name='time_kernel',
        initializer=self.time_kernel_initializer,
        regularizer=self.time_kernel_regularizer,
        constraint=self.time_kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight( 
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 3,), *args, **kwargs),  # changed bias init
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 5,),  # changed bias shape
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  # changed input arguments, added delta_t_v
  def _compute_carry_and_output(self, x, delta_t_v, h_tm1, c_tm1): 
    """Computes carry and output using split kernels."""    
    # changed x unpacking, inputs-weight product term x_T in time gate T
    x_i, x_f, x_c, x_o, x_T = x 
    # changed, added time-weight product terms as delta_t_v
    delta_t_v_T, delta_t_v_o = delta_t_v
    
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1    
    # changed gate T, added time gate
    T = self.recurrent_activation(
        x_T + self.recurrent_activation(delta_t_v_T))
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    # changed gate c, added time gate output T in pointwise product 
    c = f * c_tm1 + i * T * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    # changed gate o, added delta_t_v_o in output gate
    o = self.recurrent_activation(
        x_o + delta_t_v_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))

    return c, o, T

    # changed, ignore implementation 2 for now
#   def _compute_carry_and_output_fused(self, z, c_tm1):
#     """Computes carry and output using fused kernels."""
#     z0, z1, z2, z3 = z
#     i = self.recurrent_activation(z0)
#     f = self.recurrent_activation(z1)
#     c = f * c_tm1 + i * self.activation(z2)
#     o = self.recurrent_activation(z3)
#     return c, o
    
  # changed, replace feature-only inputs with a combined feature tensor of shape (N, T, D_emb+1)
  def call(self, combined_inputs, states, training=None):
    # changed, split inputs and delta_t
    inputs, delta_t = tf.split(combined_inputs, [combined_inputs.shape[-1]-1,1], axis=-1)
        
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    # changed, add 1 more dp mask count, for x_T, notebly, delta_t must not be dropped out
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=5)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
        inputs_T = inputs * dp_mask[4] # changed, added dropout-masked inputs for gate T 
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        inputs_T = inputs # changed, added non-masked inputs for gate T
      # changed, split kernel to 5 camps instead of 4, added k_T
      k_i, k_f, k_c, k_o, k_T = array_ops.split(
          self.kernel, num_or_size_splits=5, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      x_T = K.dot(inputs_T, k_T) # changed, added inputs-weight product term in time gate T

      # changed, split time kernel to 2 camps v_tt, v_to
      v_tt, v_to = array_ops.split(
          self.time_kernel, num_or_size_splits=2, axis=1)
      delta_t_v_T = K.dot(delta_t, v_tt) # changed, added time-weight product term in time gate T 
      delta_t_v_o = K.dot(delta_t, v_to) # changed, added time-weight product term in time gate o
      if self.use_bias:
        # changed, split bias to 5 camps instead of 4, added b_T
        b_i, b_f, b_c, b_o, b_T = array_ops.split(
            self.bias, num_or_size_splits=5, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)
        x_T = K.bias_add(x_T, b_T) # changed, added bias for x_T
      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o, x_T) # changed, added x_T into packed x
      delta_t_v = (delta_t_v_T, delta_t_v_o) # changed, added pack delta_t_v
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o, T = self._compute_carry_and_output(x, delta_t_v, h_tm1, c_tm1) # changed, added delta_t_v in input args 
    else: 
        # changed, ignore implementation 2 for now
        raise NotImplementedError
#       if 0. < self.dropout < 1.:
#         inputs = inputs * dp_mask[0]
#       z = K.dot(inputs, self.kernel)
#       z += K.dot(h_tm1, self.recurrent_kernel)
#       if self.use_bias:
#         z = K.bias_add(z, self.bias)

#       z = array_ops.split(z, num_or_size_splits=4, axis=1)
#       c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return [h, T], [h, c]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'time_kernel_initializer':
            initializers.serialize(self.time_kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'time_kernel_regularizer':
            regularizers.serialize(self.time_kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'time_kernel_constraint':
            constraints.serialize(self.time_kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(TimeLSTMCell0, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


class TimeLSTMCell1(TimeLSTMCell0):
  """Time-LSTM 1: equivalent to TimeLSTMCell0 class but adds peephole connections.
  """

  def build(self, input_shape):
    super(TimeLSTMCell1, self).build(input_shape)
    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.
    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)
      
  def _compute_carry_and_output(self, x, delta_t_v, h_tm1, c_tm1): 
    """Computes carry and output using split kernels with peephole weights."""            
    x_i, x_f, x_c, x_o, x_T = x     
    delta_t_v_T, delta_t_v_o = delta_t_v
    
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1    
    # no peephole weights for time gate
    T = self.recurrent_activation(
        x_T + self.recurrent_activation(delta_t_v_T))
    # add peephole weights for input gate
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
         self.input_gate_peephole_weights * c_tm1)
    # add peephole weights for forget gate
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + 
        self.forget_gate_peephole_weights * c_tm1)    
    c = f * c_tm1 + i * T * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    # add peephole weights for output gate
    o = self.recurrent_activation(
        x_o + delta_t_v_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + 
        self.output_gate_peephole_weights * c)

    return c, o, T


class TimeLSTMCell2(TimeLSTMCell0):
  """Time-LSTM 2: has 2 time gates T1 and T2.
  """

  def build(self, input_shape):
    super(TimeLSTMCell2, self).build(input_shape)

    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    # changed kernel shape to 6, added W_xt1 and W_xt2; input_dim=D_emb+1, the last dimension is for delta_t
    self.kernel = self.add_weight(
        shape=(input_dim-1, self.units * 6), 
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    # changed: added time kernel, [V_tt1, V_tt2, V_to]
    self.time_kernel = self.add_weight(
        shape=(1, self.units * 3),
        name='time_kernel',
        initializer=self.time_kernel_initializer,
        regularizer=self.time_kernel_regularizer,
        constraint=self.time_kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight( 
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 4,), *args, **kwargs),  # changed bias init
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 6,),  # changed bias shape
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.

    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)
      
  def _compute_carry_and_output(self, x, delta_t_v, h_tm1, c_tm1): 
    """Computes carry and output using split kernels with peephole weights."""            
    x_i, x_f, x_c, x_o, x_T1, x_T2 = x     
    delta_t_v_T1, delta_t_v_T2, delta_t_v_o = delta_t_v
    
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1    
    # no peephole weights for time gate 1
    T1 = self.recurrent_activation(
        x_T1 + self.recurrent_activation(delta_t_v_T1))
    # no peephole weights for time gate 2
    T2 = self.recurrent_activation(
        x_T2 + self.recurrent_activation(delta_t_v_T2))
    # add peephole weights for input gate
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
         self.input_gate_peephole_weights * c_tm1)
    # add peephole weights for forget gate
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + 
        self.forget_gate_peephole_weights * c_tm1)    
    # changed, c1 and c2 differs only on wich time game to use
    candidate = self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    # use c1 to calculate the current hidden state, but pass c2 to the next cell at c
    c1 = f * c_tm1 + i * T1 * candidate
    c2 = f * c_tm1 + i * T2 * candidate
    # add peephole weights for output gate
    o = self.recurrent_activation(
        x_o + delta_t_v_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + 
        self.output_gate_peephole_weights * c1)

    return c1, c2, o, T1, T2

  # changed, replace feature-only inputs with a combined feature tensor of shape (N, T, D_emb+1)
  def call(self, combined_inputs, states, training=None):
    # changed, split inputs and delta_t
    inputs, delta_t = tf.split(combined_inputs, [combined_inputs.shape[-1]-1,1], axis=-1)
        
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    # changed, add 2 more dp mask count, for x_T1, x_T2 notebly, delta_t must not be dropped out
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=6)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
        inputs_T1 = inputs * dp_mask[4] # changed, added dropout-masked inputs for gate T1
        inputs_T2 = inputs * dp_mask[5] # changed, added dropout-masked inputs for gate T2 
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        inputs_T1 = inputs # changed, added non-masked inputs for gate T1
        inputs_T2 = inputs # changed, added non-masked inputs for gate T2
      # changed, split kernel to 5 camps instead of 4, added k_T
      k_i, k_f, k_c, k_o, k_T1, k_T2 = array_ops.split(
          self.kernel, num_or_size_splits=6, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      x_T1 = K.dot(inputs_T1, k_T1) # changed, added inputs-weight product term in time gate T
      x_T2 = K.dot(inputs_T2, k_T2) # changed, added inputs-weight product term in time gate T

      # changed, split time kernel to 3 camps v_tt1, v_tt2, v_to
      v_tt1, v_tt2, v_to = array_ops.split(
          self.time_kernel, num_or_size_splits=3, axis=1)
      # changed, add a constraint of TimeLSTM2, v_tt1 <= 0
      v_tt1 = tf.clip_by_value(v_tt1, clip_value_min=-float('inf'), clip_value_max=0)

      delta_t_v_T1 = K.dot(delta_t, v_tt1) # changed, added time-weight product term in time gate T1
      delta_t_v_T2 = K.dot(delta_t, v_tt2) # changed, added time-weight product term in time gate T1
      delta_t_v_o = K.dot(delta_t, v_to) # changed, added time-weight product term in time gate o
      if self.use_bias:
        # changed, split bias to 5 camps instead of 4, added b_T
        b_i, b_f, b_c, b_o, b_T1, b_T2 = array_ops.split(
            self.bias, num_or_size_splits=6, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)
        x_T1 = K.bias_add(x_T1, b_T1) # changed, added bias for x_T1
        x_T2 = K.bias_add(x_T2, b_T2) # changed, added bias for x_T2
      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o, x_T1, x_T2) # changed, added x_T1 and x_T2 into packed x
      delta_t_v = (delta_t_v_T1, delta_t_v_T2, delta_t_v_o) # changed, added pack delta_t_v
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c1, c2, o, T1, T2 = self._compute_carry_and_output(x, delta_t_v, h_tm1, c_tm1) # changed, added delta_t_v in input args 
    else: 
        # changed, ignore implementation 2 for now
        raise NotImplementedError
    # use c1 to calculate the current hidden state, but pass c2 to the next cell at c
    h = o * self.activation(c1)
    return [h, T1, T2], [h, c2]


class TimeLSTMCell3(TimeLSTMCell0):
  """Time-LSTM 3: Based on TimeLSTM2, removed forget gate, and instead using Coupled input and forget gates. 
  """

  def build(self, input_shape):
    super(TimeLSTMCell3, self).build(input_shape)

    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    # changed kernel shape to 5, added W_xt1 and W_xt2, removed W_f; input_dim=D_emb+1, the last dimension is for delta_t
    self.kernel = self.add_weight(
        shape=(input_dim-1, self.units * 5), 
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    # changed: added time kernel, [V_tt1, V_tt2, V_to]
    self.time_kernel = self.add_weight(
        shape=(1, self.units * 3),
        name='time_kernel',
        initializer=self.time_kernel_initializer,
        regularizer=self.time_kernel_regularizer,
        constraint=self.time_kernel_constraint,
        caching_device=default_caching_device)
    # changed: removed recurrent kernel for forget gate 
    self.recurrent_kernel = self.add_weight( 
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if self.unit_forget_bias: # changed, add warning but no actions required.
        logging.warning('Invalid argument unit_forget_bias: No forget gate in TimeLSTMCell3.')

      bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 5,),  # changed bias shape
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.

    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)    
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)
      
  def _compute_carry_and_output(self, x, delta_t_v, h_tm1, c_tm1): 
    """Computes carry and output using split kernels with peephole weights."""            
    x_i, x_c, x_o, x_T1, x_T2 = x     
    delta_t_v_T1, delta_t_v_T2, delta_t_v_o = delta_t_v
    
    h_tm1_i, h_tm1_c, h_tm1_o = h_tm1    
    # no peephole weights for time gate 1
    T1 = self.recurrent_activation(
        x_T1 + self.recurrent_activation(delta_t_v_T1))
    # no peephole weights for time gate 2
    T2 = self.recurrent_activation(
        x_T2 + self.recurrent_activation(delta_t_v_T2))
    # add peephole weights for input gate
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
         self.input_gate_peephole_weights * c_tm1)    
    # changed, c1 and c2 differs only on wich time game to use
    candidate = self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units:self.units * 2]))
    # use c1 to calculate the current hidden state, but pass c2 to the next cell at c
    c1 = (1 - i * T1) * c_tm1 + i * T1 * candidate
    c2 = (1 - i) * c_tm1 + i * T2 * candidate  # notebly c1 and c2 are not symmetric in TimeLSTMCell3
    # add peephole weights for output gate
    o = self.recurrent_activation(
        x_o + delta_t_v_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 2:]) + 
        self.output_gate_peephole_weights * c1)

    return c1, c2, o, T1, T2

  # changed, replace feature-only inputs with a combined feature tensor of shape (N, T, D_emb+1)
  def call(self, combined_inputs, states, training=None):
    # changed, split inputs and delta_t
    inputs, delta_t = tf.split(combined_inputs, [combined_inputs.shape[-1]-1,1], axis=-1)
        
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    # changed, add 1 more dp mask count, for x_T1, x_T2 but removed x_f notebly, delta_t must not be dropped out
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=5)
    # changed, removed recurrent mask for forget gate
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]        
        inputs_c = inputs * dp_mask[1]
        inputs_o = inputs * dp_mask[2]
        inputs_T1 = inputs * dp_mask[3] # changed, added dropout-masked inputs for gate T1
        inputs_T2 = inputs * dp_mask[4] # changed, added dropout-masked inputs for gate T2 
      else:
        inputs_i = inputs        
        inputs_c = inputs
        inputs_o = inputs
        inputs_T1 = inputs # changed, added non-masked inputs for gate T1
        inputs_T2 = inputs # changed, added non-masked inputs for gate T2
      # changed, split kernel to 5 camps instead of 4, added k_T
      k_i, k_c, k_o, k_T1, k_T2 = array_ops.split(
          self.kernel, num_or_size_splits=5, axis=1)
      x_i = K.dot(inputs_i, k_i)      
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      x_T1 = K.dot(inputs_T1, k_T1) # changed, added inputs-weight product term in time gate T
      x_T2 = K.dot(inputs_T2, k_T2) # changed, added inputs-weight product term in time gate T

      # changed, split time kernel to 3 camps v_tt1, v_tt2, v_to
      v_tt1, v_tt2, v_to = array_ops.split(
          self.time_kernel, num_or_size_splits=3, axis=1)
      # changed, add a constraint of TimeLSTM2, v_tt1 <= 0
      v_tt1 = tf.clip_by_value(v_tt1, clip_value_min=-float('inf'), clip_value_max=0)

      delta_t_v_T1 = K.dot(delta_t, v_tt1) # changed, added time-weight product term in time gate T1
      delta_t_v_T2 = K.dot(delta_t, v_tt2) # changed, added time-weight product term in time gate T1
      delta_t_v_o = K.dot(delta_t, v_to) # changed, added time-weight product term in time gate o
      if self.use_bias:
        # changed, split bias to 5 camps instead of 4, added b_T
        b_i, b_c, b_o, b_T1, b_T2 = array_ops.split(
            self.bias, num_or_size_splits=5, axis=0)
        x_i = K.bias_add(x_i, b_i)        
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)
        x_T1 = K.bias_add(x_T1, b_T1) # changed, added bias for x_T1
        x_T2 = K.bias_add(x_T2, b_T2) # changed, added bias for x_T2
      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]      
        h_tm1_c = h_tm1 * rec_dp_mask[1]
        h_tm1_o = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_i = h_tm1        
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_c, x_o, x_T1, x_T2) # changed, added x_T1 and x_T2 into packed x, removed x_f
      delta_t_v = (delta_t_v_T1, delta_t_v_T2, delta_t_v_o) # changed, added pack delta_t_v
      h_tm1 = (h_tm1_i, h_tm1_c, h_tm1_o)
      c1, c2, o, T1, T2 = self._compute_carry_and_output(x, delta_t_v, h_tm1, c_tm1) # changed, added delta_t_v in input args 
    else: 
        # changed, ignore implementation 2 for now
        raise NotImplementedError
    # use c1 to calculate the current hidden state, but pass c2 to the next cell at c
    h = o * self.activation(c1)
    return [h, T1, T2], [h, c2]


"""Wrapper Layers of Time-LSTM Cells"""

@keras_export(v1=['keras.experimental.TimeLSTM0'])
class TimeLSTM0(RNN):
  """Time Long Short-Term Memory layer - What to Do Next: Modeling User Behaviors by Time-LSTM.
   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    time_kernel_initializer: Initializer for the `time_kernel` weights matrix,
      used for the linear transformation of the time difference.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    time_kernel_regularizer: Regularizer function applied to
      the `time_kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    time_kernel_constraint: Constraint function applied to
      the `time_kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               time_kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               time_kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               time_kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = TimeLSTMCell0(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        time_kernel_initializer=time_kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        time_kernel_regularizer=time_kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        time_kernel_constraint=time_kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(TimeLSTM0, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(TimeLSTM0, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def time_kernel_initializer(self):
    return self.cell.time_kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def time_kernel_regularizer(self):
    return self.cell.time_kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def time_kernel_constraint(self):
    return self.cell.time_kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'time_kernel_initializer':
            initializers.serialize(self.time_kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'time_kernel_regularizer':
            regularizers.serialize(self.time_kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'time_kernel_constraint':
            constraints.serialize(self.time_kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(TimeLSTM0, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


@keras_export(v1=['keras.experimental.TimeLSTM1'])
class TimeLSTM1(TimeLSTM0):
  """Equivalent to TimeLSTM0, but uses TimeLSTMCell1
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               time_kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               time_kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               time_kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = TimeLSTMCell1(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        time_kernel_initializer=time_kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        time_kernel_regularizer=time_kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        time_kernel_constraint=time_kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(TimeLSTM0, self).__init__( # invoke the init method of grandfather (RNN)
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]


@keras_export(v1=['keras.experimental.TimeLSTM2'])
class TimeLSTM2(TimeLSTM0):
  """TimeLSTM with two time gates
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               time_kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               time_kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               time_kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = TimeLSTMCell2(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        time_kernel_initializer=time_kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        time_kernel_regularizer=time_kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        time_kernel_constraint=time_kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(TimeLSTM0, self).__init__( # invoke the init method of grandfather (RNN)
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]


@keras_export(v1=['keras.experimental.TimeLSTM3'])
class TimeLSTM3(TimeLSTM0):
  """Almost Itentical with TimeLSTM2 Wrapper, notebly, all forget gate related arguments can be neglected \
      since there's no forget gate in TimeLSTM3Cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               time_kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               time_kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               time_kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = TimeLSTMCell3(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        time_kernel_initializer=time_kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        time_kernel_regularizer=time_kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        time_kernel_constraint=time_kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(TimeLSTM0, self).__init__( # invoke the init method of grandfather (RNN)
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]