''' a (un)folding layer '''

from . import Layer
import numpy      as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

class Folding_Layer(Layer, Tensor):

  def __init__(self, tensor, patch_height=-1, patch_width=-1, stride_y=-1, stride_x=-1, name='folding_layer', **kwargs):
    ''' (un)fold image 2D (tensor)

    @param tensor      : input tensor in NWHC format
    @param patch_height: filter patch size y
    @param patch_width : filter patch size x
    @param stride_y    : patches shift y
    @param stride_x    : patches shift X
    @param object_     : if True, self is returned (default=False) else output tensor

    @return: conversation operator (tf.tensor)
    '''

    self.tensor            = tensor

    (self.batch_size, # not used
    self.w,
    self.h,
    self.c)                = self._tensor_shape(tensor)

    self.batch_size        = tf.shape(tensor)[0]

    self.patch_height      = patch_height if patch_height > 0 else self.h
    self.patch_width       = patch_width  if patch_width  > 0 else self.w
    self.stride_y          = stride_y     if stride_y     > 0 else self.h
    self.stride_x          = stride_x     if stride_x     > 0 else self.w

    self.h_out              = int((self.h - self.patch_height) / (self.stride_y) + 1)
    self.w_out              = int((self.w - self.patch_width) / (self.stride_x) + 1)
    self.c_out              = self.patch_height * self.patch_width * self.c
    self.output_size        = self.h_out * self.w_out * self.c_out

    self.output_tensor = self._build_tf_graph(
      tensor       = self.tensor      , # input tensor
      width        = self.w           , # image width
      height       = self.h           , # image height
      channels     = self.c           , # number of channels
      filterSizeY  = self.patch_height, # filter patch size y
      filterSizeX  = self.patch_width , # filter patch size x
      strideY      = self.stride_y    , # patch shift in y
      strideX      = self.stride_x    , # patch shift in x
      )

    Tensor.__init__(self, self.output_tensor._op, self.output_tensor._value_index, self.output_tensor._dtype)
    self._name = name

  def _build_tf_graph(self, tensor, width, height, channels, filterSizeY, filterSizeX, strideY, strideX):
    tensor_flattened        = tf.reshape(tensor, [-1])
    indicesOneSample        = np.zeros([int(self.h_out * self.w_out * self.c_out)], dtype=np.int32)

    shape_tf                = tf.shape(tensor)
    batch_size_tf           = shape_tf[0]

    currentFilterY          = 0
    currentFilterX          = 0
    dstIndex                = 0
    offsetsMap              = tf.expand_dims(tf.range(0, batch_size_tf, delta=1), 1) * width * height * channels

    while currentFilterY < self.h_out and currentFilterX < self.w_out:
      for yInFilter in range(filterSizeY):
        for xInFilter in range(filterSizeX):
          for currentC in range(channels):
            currentY                   = currentFilterY * strideY + yInFilter
            currentX                   = currentFilterX * strideX + xInFilter
            srcIndex                   = (currentY * width + currentX) * channels + currentC
            indicesOneSample[dstIndex] = srcIndex
            dstIndex += 1
      currentFilterX += 1
      if currentFilterX >= self.w_out:
        currentFilterX  = 0
        currentFilterY += 1
    indicesMat  = tf.reshape(tf.tile(indicesOneSample, shape_tf[0:1]), (batch_size_tf, self.output_size))
    indicesMat += offsetsMap
    indicesMat  = tf.reshape(indicesMat, [-1])
    gatherFlat  = tf.gather(tensor_flattened, indicesMat)

    convert_op  = tf.reshape(gatherFlat, (batch_size_tf, self.w_out, self.h_out, self.c_out))
    return convert_op



  def _init_visualization(self): pass
  def _visualize(self): pass