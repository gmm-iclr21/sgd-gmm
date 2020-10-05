import os

from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class Devanagari(DS):
  ''' The Devanagari Character Dataset (DHCD) handwritten digits

    source: https://www.kaggle.com/rishianand/devanagari-character-set

    website: https://web.archive.org/web/20160105230017/http://cvresearchnepal.com/wordpress/dhcd/
    training set of 78200 images (32 x 32 grey scale) for 46 characters classes (1700 images per class)
    test set of 13800 images (32 x 32 grey scale) for 46 characters classes (300 images per class)

    @INPROCEEDINGS{7400041,
        author    = {S. Acharya and A. K. Pant and P. K. Gyawali},
        booktitle = {2015 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA)},
        title     = {Deep learning based large scale handwritten Devanagari character recognition},
        year      = {2015},
        pages     = {1-6},
        doi       = {10.1109/SKIMA.2015.7400041},
        month     = {Dec},
        }
  '''

  def __init__(self):
    super(Devanagari, self).__init__()

  @property
  def name(self): return 'Devanagari Character Dataset'

  @property
  def shape(self): return (32, 32, 1)

  @property
  def pickle_file(self): return 'Devanagari.pkl.gz'

  @property
  def num_train(self): return 82800

  @property
  def num_test(self): return 9200

  @property
  def num_classes(self): return 10 # 46

  def download_data(self, force=False):
    self.compressed_file = DWL.download_from_kaggle('rishianand/devanagari-character-set', 'devanagari-character-set.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

  def prepare_data(self):
    self.extracted_files               = os.path.join(self.extracted_files, 'Images', 'Images') + os.sep
    self.data_train, self.labels_train = DL.load_directory_as_dataset(self.extracted_files, self.shape, max_classes=self.num_classes)

    # Only True, if 10% (default) of data split in to training and testing
    self.validate_shape = False

  def clean_up(self):
    DS.file_delete('devanagari-character-set.zip')
    DS.file_delete('devanagari-character-set/')


if __name__ == '__main__':
  Devanagari().get_data(clean_up=False)
