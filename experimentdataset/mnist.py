
from . import Dataset as DS
import numpy as np


class MNIST(DS):
  ''' MNIST database of handwritten digits

  source (TesorFlow implementation): https://storage.googleapis.com/cvdf-datasets/mnist/

  website: http://yann.lecun.com/exdb/mnist/
  training 60000 (28 x 28 gray scale) images in 10 classes
  test 10000 (28 x 28 gray scale) images in 10 classes

  @ARTICLE{MNIST,
      author   = {Y. Lecun and L. Bottou and Y. Bengio and P. Haffner},
      journal  = {Proceedings of the IEEE},
      title    = {Gradient-based learning applied to document recognition},
      year     = {1998},
      volume   = {86},
      number   = {11},
      pages    = {2278-2324},
      keywords = {backpropagation;convolution;multilayer perceptrons;optical character recognition;2D shape variability;GTN;back-propagation;cheque reading;complex decision surface synthesis;convolutional neural network character recognizers;document recognition;document recognition systems;field extraction;gradient based learning technique;gradient-based learning;graph transformer networks;handwritten character recognition;handwritten digit recognition task;high-dimensional patterns;language modeling;multilayer neural networks;multimodule systems;performance measure minimization;segmentation recognition;Character recognition;Feature extraction;Hidden Markov models;Machine learning;Multi-layer neural network;Neural networks;Optical character recognition software;Optical computing;Pattern recognition;Principal component analysis},
      doi      = {10.1109/5.726791},
      ISSN     = {0018-9219},
      month    = {Nov},
  }
  '''

  def __init__(self):
    super(MNIST, self).__init__()

  @property
  def name(self): return 'MNIST'

  @property
  def shape(self): return (28, 28, 1)

  @property
  def pickle_file(self): return 'MNIST.pkl.gz'

  @property
  def num_train(self): return 55000

  @property
  def num_test(self): return 10000

  @property
  def num_classes(self): return 10

  def download_data(self, force=False):
    print('download MNIST')
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

    self.data = read_data_sets(DS.DOWNLOAD_DIR + '/MNIST', one_hot=True)

  def extract_data(self, force=False):
    pass

  def prepare_data(self):
    print('prepare MNIST')
    self.data_train   = self.data.train.images
    self.labels_train = self.data.train.labels.astype(np.float32)

    self.data_test    = self.data.test.images
    self.labels_test  = self.data.test.labels.astype(np.float32)

  def clean_up(self):
    print('cleanup MNIST')
    DS.file_delete('MNIST/')


if __name__ == '__main__':
  MNIST().get_data(clean_up=False)
