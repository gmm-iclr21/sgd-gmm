'''
Created on 08.04.2020

@author: Benedikt
'''
import abc
import tensorflow     as tf


class Layer(abc.ABC):
  initializer_zeros = tf.compat.v1.initializers.zeros()
  initializer_ones  = tf.compat.v1.constant_initializer(1.)

  @abc.abstractmethod
  def _build_tf_graph(self): pass

  def train_step(self, feed_dict): pass
  def test_step(self, feed_dict) : pass

  def __str__(self):
    s     = f'INFO: {self._name}'
    max_  = max([ len(key) for key in vars(self).keys() ])
    for k, v in vars(self).items():
      if not k.startswith('_'): s += f'  {k:>{max_}}: {v}'
    return s

  def info(self):
    pass

  @classmethod
  def _tensor_shape(cls, tensor):
    ''' return a tensors shape as list (dynamic types are not converted) '''
    shape = list()
    for dimension in tensor.get_shape():
      try   : shape += [dimension.value] # TODO: do not use exceptions
      except: shape += [dimension      ]
    return shape

  def get_variable(self, *names):
    ''' return a list or a single variable or use dot (.) notation for the access of variables '''
    variables = [ getattr(self, name) for name in names ]
    return variables[0] if len(variables) == 1 else variables

  def _init_visualization(self): pass
  def visualize(self): pass
