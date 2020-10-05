
from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class CIFAR10(DS):
  ''' CIFAR-10 dataset

  source: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

  website: https://www.cs.toronto.edu/~kriz/cifar.html
  50000 training 32x32 color images in 10 classes, with 6000 images per class
  10000 test 32x32 color images in 10 classes, with 6000 images per class


  @inproceedings{Krizhevsky2009LearningML,
    title  = {Learning Multiple Layers of Features from Tiny Images},
    author = {Alex Krizhevsky},
    year   = {2009}
  }
  '''

  def __init__(self):
    super(CIFAR10, self).__init__()

  @property
  def name(self): return 'CIFAR10'

  @property
  def shape(self): return (32, 32, 3)

  @property
  def pickle_file(self): return 'CIFAR10.pkl.gz'

  @property
  def num_train(self): return 50000

  @property
  def num_test(self): return 10000

  @property
  def num_classes(self): return 10

  def download_data(self, force=False):
    self.compressed_file = DWL.download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', FORCE=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

  def prepare_data(self):
    pickle_batch_size = 10000
    sub_dir            = '/cifar-10-batches-py/'

    # training data and labels files
    train_pickle_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    train_pickle_files = [ self.extracted_files + sub_dir + x for x in train_pickle_files ]

    # test data and labels file
    test_pickle_files  = ['test_batch']
    test_pickle_files  = [ self.extracted_files + sub_dir + x for x in test_pickle_files ]

    # load training data and labels
    self.data_train, self.labels_train = DL.load_from_pickles(train_pickle_files, pickle_batch_size, self.shape[0], self.shape[2])
    self.data_test, self.labels_test   = DL.load_from_pickles(test_pickle_files , pickle_batch_size, self.shape[0], self.shape[2])

    # self.validate_shape = False

  def clean_up(self):
    DS.file_delete('cifar-10-python.tar.gz')
    DS.file_delete('cifar-10-python')


if __name__ == '__main__':
  CIFAR10().get_data(True)

