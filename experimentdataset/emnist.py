import os

from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS


class EMNIST(DS):
  ''' EMNIST (Extended MNIST) eMNIST

  derived from NIST Special Database 19

  source: https://www.kaggle.com/crawford/emnist

  website: https://www.nist.gov/itl/iad/image-group/emnist-dataset
  train: 697,932 images (28 x 28 gray scale) in 62 unbalanced classes.
  test: 116,323 images (28 x 28 gray scale) in 62 unbalanced classes.

  TODO: fix uint overflow in zlib (split?)
  TODO: add function for: select only digits, only letters or both?
    current: 10 classes with most elements (digits)

  @article{DBLP:journals/corr/CohenATS17,
    author        = {Gregory Cohen and Saeed Afshar and Jonathan Tapson and Andr{\'{e}} van Schaik},
    title         = {{EMNIST:} an extension of {MNIST} to handwritten letters},
    journal       = {CoRR},
    volume        = {abs/1702.05373},
    year          = {2017},
    url           = {http://arxiv.org/abs/1702.05373},
    archivePrefix = {arXiv},
    eprint        = {1702.05373},
    timestamp     = {Wed, 07 Jun 2017 14:41:07 +0200},
    biburl        = {https://dblp.org/rec/bib/journals/corr/CohenATS17},
    bibsource     = {dblp computer science bibliography, https://dblp.org}
  }
  '''

  def __init__(self):
    super(EMNIST, self).__init__()

  @property
  def name(self): return 'Extended MNIST Dataset'

  @property
  def shape(self): return (28, 28, 1)

  @property
  def pickle_file(self): return 'EMNIST.pkl.gz'

  @property
  def num_train(self): return 697932

  @property
  def num_test(self): return 116323

  @property
  def num_classes(self): return 10 # all=62

  def download_data(self, force=False):
    self.compressed_file = DWL.download_from_kaggle('crawford/emnist', filename='emnist.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)
    self.extracted_files = os.path.join(self.extracted_files, 'emnist_source_files')

  def prepare_data(self):
    data_training_ubyte   = os.path.join(self.extracted_files, 'emnist-byclass-train-images-idx3-ubyte')
    labels_training_ubyte = os.path.join(self.extracted_files, 'emnist-byclass-train-labels-idx1-ubyte')
    data_test_ubyte       = os.path.join(self.extracted_files, 'emnist-byclass-test-images-idx3-ubyte')
    lables_test_ubyte     = os.path.join(self.extracted_files, 'emnist-byclass-test-labels-idx1-ubyte')

    self.data_train       = DL.read_idx(data_training_ubyte)
    self.labels_train     = DL.read_idx(labels_training_ubyte)
    self.data_test        = DL.read_idx(data_test_ubyte)
    self.labels_test      = DL.read_idx(lables_test_ubyte)

    # find num_classes with most elements
    DL.reduce(self)

    # True, if all classes were created/used
    self.validate_shape = False

  def clean_up(self):
    DS.file_delete('emnist')
    DS.file_delete('emnist.zip')


if __name__ == '__main__':
  EMNIST().get_data(clean_up=False)
