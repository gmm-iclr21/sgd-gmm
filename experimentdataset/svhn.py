import scipy.io

from . import Downloader as DWL
from . import Dataset as DS

import numpy as np


class SVHN(DS):
  ''' Street View house Numbers (SVHN) Dataset

  source: http://ufldl.stanford.edu/housenumbers/

  website: http://ufldl.stanford.edu/housenumbers/
  training 73257 digits (32 x 32 color) images in 10 classes
  testing 26032 digits (32 x 32 color) images in 10 classes

  @article{article,
      author    = {Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Y Ng, Andrew},
      year      = {2011},
      month     = {01},
      title     = {Reading Digits in Natural Images with Unsupervised Feature Learning},
      booktitle = {NIPS}
  }
  '''

  def __init__(self):
      super(SVHN, self).__init__()

  @property
  def name(self): return 'Street View house Numbers (SVHN) Dataset'

  @property
  def shape(self): return (32, 32, 3)

  @property
  def pickle_file(self): return 'SVHN.pkl.gz'

  @property
  def num_train(self): return 73257

  @property
  def num_test(self): return 26032

  @property
  def num_classes(self): return 10

  def download_data(self, force=False):
    self.train_mat = DWL.download_file('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', FORCE=force)
    self.test_mat  = DWL.download_file('http://ufldl.stanford.edu/housenumbers/test_32x32.mat' , FORCE=force)

    # TODO: merge extra data into the normal dataset?
    if False: self.extra_mat = DWL.download_file('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat')

  def extract_data(self, force=False): pass

  def prepare_data(self):

    def load_matrix(data_mat):
      data_mat = scipy.io.loadmat(data_mat)
      data     = np.rollaxis(data_mat['X'], 3, 0).astype('float16') # from [32, 32 3, ?] to [?, 32, 32, 3]
      labels   = data_mat['y']

      return data, labels

    self.data_train, self.labels_train = load_matrix(self.train_mat)
    self.data_test , self.labels_test  = load_matrix(self.test_mat)

    # TODO: merge extra data into the normal dataset?
    if False: self.data_extra, self.label_extra = load_matrix(self.extra_mat)

  def clean_up(self):
    DS.file_delete('train_32x32.mat')
    DS.file_delete('test_32x32.mat')


if __name__ == '__main__':
  SVHN().get_data(clean_up=False)
