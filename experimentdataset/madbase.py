from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class MADBase(DS):
  ''' Arabic Handwritten Digits Dataset (Modified version of the ADBase (MADBase))

  source: https://www.kaggle.com/mloey1/ahdd1

  website: http://datacenter.aucegypt.edu/shazeem/
  training set 60,000 digits 6000 (28x28 grayscale) images per class
  test set 10,000 digits 1000 (28x28 grayscale) images per class
  '''

  def __init__(self):
    super(MADBase, self).__init__()

  @property
  def name(self): return 'MADBase Handwritten Digits Dataset'

  @property
  def shape(self): return (28, 28, 1)

  @property
  def pickle_file(self): return 'MADBase.pkl.gz'

  @property
  def num_train(self): return 60000

  @property
  def num_test(self): return 10000

  @property
  def num_classes(self): return 10

  def download_data(self, force=False):
    self.compressed_file = DWL.download_from_kaggle('mloey1/ahdd1', filename='ahdd1.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

  def prepare_data(self):
    data_train        = DL.csv_loader(self.extracted_files + '/csvTrainImages 60k x 784.csv')
    labels_train      = DL.csv_loader(self.extracted_files + '/csvTrainLabel 60k x 1.csv')
    data_test         = DL.csv_loader(self.extracted_files + '/csvTestImages 10k x 784.csv')
    labels_test       = DL.csv_loader(self.extracted_files + '/csvTestLabel 10k x 1.csv')

    # data train
    self.data_train   = self.strip_1_dims(data_train)
    self.labels_train = self.strip_1_dims(labels_train)

    # data test
    self.data_test    = self.strip_1_dims(data_test)
    self.labels_test  = self.strip_1_dims(labels_test)

  def clean_up(self):
    DS.file_delete('ahdd1')
    DS.file_delete('ahdd1.zip')


if __name__ == '__main__':
  MADBase().get_data(clean_up=True)
