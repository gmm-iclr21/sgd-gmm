''' a simple linear classifier '''

import math
import numpy      as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from . import Layer
from utils.Visualization        import Visualization as Vis

class Linear_Classifier_Layer(Layer, Tensor):

  def __init__(self, xs, ys, **kwargs):
    self.xs = xs
    self.ys = ys

    self.num_classes    = kwargs.get('num_classes' , 10  )
    self.regEps         = kwargs.get('regEps'      , 0.05)
    self.lr_reduction   = kwargs.get('lr_reduction', 0.01) # learning rate reduction factor lr * (1 - lr_redution)

    self.vis_samples_   = kwargs.get('vis_samples', False)

    self.w              = kwargs.get('w'          , 28   ) # only for visualization (remove or derive/calculate)
    self.h              = kwargs.get('h'          , 28   )
    self.c              = kwargs.get('c'          , 1    )

    input_shape         = self._tensor_shape(xs)
    self.channels_in    = np.prod(input_shape[1:])
    self.channels_out   = self.num_classes

    self._init_tf_variables()
    self._build_tf_graph(xs)
    self._construct_update_ops()
    self._init_visualization()

    output_tensor = getattr(self, kwargs.get('output_tensor', 'mean_loss'))
    Tensor.__init__(self, output_tensor._op, output_tensor._value_index, output_tensor._dtype)
    self._name    = kwargs.get('name', 'linear_layer')


  def _init_tf_variables(self):
    initializer_bias       = self.initializer_zeros
    initializer_bias_tmp   = self.initializer_zeros
    initializer_weight     = tf.compat.v1.initializers.truncated_normal(stddev=1. / math.sqrt(self.channels_in))
    initializer_weight_tmp = tf.compat.v1.initializers.random_uniform(-0.01, 0.01)

    weight_shape           = (self.channels_in, self.channels_out)
    bias_shape             = (self.channels_out)

    self.W                 = tf.compat.v1.get_variable(name='weight'              , shape=weight_shape, dtype=tf.float64, initializer=initializer_weight    )
    self.W_tmp             = tf.compat.v1.get_variable(name='weightTMP'           , shape=weight_shape, dtype=tf.float64, initializer=initializer_weight_tmp)
    self.b                 = tf.compat.v1.get_variable(name='bias'                , shape=bias_shape  , dtype=tf.float64, initializer=initializer_bias      )
    self.b_tmp             = tf.compat.v1.get_variable(name='biasTMP'             , shape=bias_shape  , dtype=tf.float64, initializer=initializer_bias_tmp  )

    self.lambda_W_factor   = tf.compat.v1.placeholder_with_default(name='lambda_W', shape=[]          , input=tf.compat.v1.constant(1.0, dtype=tf.float64)  ) # constant to change the adaption rate by SGD step (Ws)
    self.lambda_b_factor   = tf.compat.v1.placeholder_with_default(name='lambda_b', shape=[]          , input=tf.compat.v1.constant(1.0, dtype=tf.float64)  ) # constant to change the adaption rate by SGD step (bs)


  def _build_tf_graph(self, tensor):
    ''' classifier operator (supervised)
      train a linear classifier (output: predicted class label)
     '''
    self.logits         = tf.nn.bias_add(tf.matmul(tensor, self.W), self.b                             , name='logits'       ) # readout layer
    self.loss           = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.logits, name='loss'         )
    self.mean_loss      = tf.reduce_mean(self.loss                                                     , name='mean_loss'    )
    self.true_ys        = tf.argmax(self.ys     , axis=1                                               , name='true_y'       )
    self.pred_ys        = tf.argmax(self.logits , axis=1                                               , name='pred_y'       ) # class value e.g. [0, 1, 3, ...]
    self.pred_ys_onehot = tf.one_hot(tf.cast(self.pred_ys, dtype=tf.int32), self.num_classes           , name='pred_y_onehot') # class value in one-hot format


  def test_step(self, feed_dict):
    sess = tf.compat.v1.get_default_session()
    return sess.run([self.true_ys, self.pred_ys], feed_dict=feed_dict)


  def train_step(self, feed_dict, loss=None):
    sess = tf.compat.v1.get_default_session()

    # feed_dict.update({ # TODO: add this stuff only if a command line parameter is given?
    #   self.lambda_W_factor: self.lambda_W, # 1.0
    #   self.lambda_b_factor: self.lambda_b, # 1.0
    #   })
    sess.run(self.update1 ,feed_dict=feed_dict)
    sess.run(self.update)


  def _classify_batch(self, feed_dict):
    ''' predict class labels '''
    sess   = tf.compat.v1.get_default_session()
    labels = sess.run(self.pred_ys_onehot, feed_dict=feed_dict)
    self._visualize(feed_dict, labels)
    return labels


  def _construct_update_ops(self):
    ''' build operations for update GMM '''
    (self.grad_Ws,
     self.grad_bs,
     )           = tf.gradients(-self.loss        , [self.W, self.b])

    updateWs1    = tf.compat.v1.assign(self.W_tmp, self.W + self.lambda_W_factor * self.regEps * self.grad_Ws, name='tmp_Ws'   )
    updatebs1    = tf.compat.v1.assign(self.b_tmp, self.b + self.lambda_b_factor * self.regEps * self.grad_bs, name='tmp_bs'   )

    updateWs     = tf.compat.v1.assign(self.W    , self.W_tmp                                         , name='assign_Ws')
    updatebs     = tf.compat.v1.assign(self.b    , self.b_tmp                                         , name='assign_bs')

    self.update1 = [updateWs1, updatebs1]
    self.update  = [updateWs , updatebs ]


  def _init_visualization(self):
    self.sample_visualizer = Vis(
      enabled        = self.vis_samples_       ,
      name           = 'sampled'               ,
      x_plots        = 10                      ,
      y_plots        = 10                      ,
      width          = self.w                  ,
      height         = self.h                  ,
      channels       = self.c                  ,
      dimensionality = self.w * self.h * self.c,
      h_out          = 1                       ,
      w_out          = 1                       ,
      )


  def _visualize(self, feed_dict, labels):
    if not self.vis_samples_: return
    X      = tf.get_default_graph().get_tensor_by_name('X:0')
    data   = feed_dict[X].reshape(-1, 28, 28, 1) # FIXME: visualize generated samples only images with 28 x 28 x 1 resolution
    labels = np.argmax(labels, axis=1)
    self.sample_visualizer.visualize(data, labels=labels) # visualize one batch of generated data
