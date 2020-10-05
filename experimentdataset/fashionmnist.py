
from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

import numpy as np


class FashionMNIST(DS):
  ''' FashionMNIST

  source: https://www.kaggle.com/zalando-research/fashionmnist

  website: https://github.com/zalandoresearch/fashion-mnist
  training 60,000 (28x28 grayscale) images in 10 classes.
  test 10,000 (28x28 grayscale) images in 10 classes.

  @online{xiao2017/online,
    author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
    title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
    date         = {2017-08-28},
    year         = {2017},
    eprintclass  = {cs.LG},
    eprinttype   = {arXiv},
    eprint       = {cs.LG/1708.07747},
  }
  '''

  def __init__(self):
    super(FashionMNIST, self).__init__()

  @property
  def name(self): return 'FashionMNIST'

  @property
  def shape(self): return (28, 28, 1)

  @property
  def pickle_file(self): return 'FashionMNIST.pkl.gz'

  @property
  def num_train(self): return 60000

  @property
  def num_test(self): return 10000

  @property
  def num_classes(self): return 10

  def download_data(self, force=False):
    self.compressed_file_data_train  = DWL.download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', FORCE=force)
    self.compressed_file_label_train = DWL.download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz', FORCE=force)
    self.compressed_file_data_test   = DWL.download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', FORCE=force)
    self.compressed_file_label_test  = DWL.download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', FORCE=force)

  def extract_data(self, force=False):
    self.extracted_files_data_train  = DL.unzip_ubyte_gz(self.compressed_file_data_train)
    self.extracted_files_label_train = DL.unzip_ubyte_gz(self.compressed_file_label_train)
    self.extracted_files_data_test   = DL.unzip_ubyte_gz(self.compressed_file_data_test)
    self.extracted_files_label_test  = DL.unzip_ubyte_gz(self.compressed_file_label_test)

  def prepare_data(self):
    self.data_train   = self.strip_1_dims(self.extracted_files_data_train)
    self.labels_train = self.extracted_files_label_train.astype(np.float32)

    self.data_test    = self.strip_1_dims(self.extracted_files_data_test)
    self.labels_test  = self.extracted_files_label_test.astype(np.float32)

  def clean_up(self):
    DS.file_delete('t10k-images-idx3-ubyte.gz')
    DS.file_delete('t10k-labels-idx1-ubyte.gz')
    DS.file_delete('train-images-idx3-ubyte.gz')
    DS.file_delete('train-labels-idx1-ubyte.gz')


if __name__ == '__main__':
  FashionMNIST().get_data(clean_up=False)

