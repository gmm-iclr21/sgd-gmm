import os

from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class CUB200(DS):
  ''' CUB200
  source: http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

  website: http://www.vision.caltech.edu/visipedia
  Number of categories/classes: 200
  Number of images: 11,788

  @techreport{WahCUB_200_2011,
      Title       = {{The Caltech-UCSD Birds-200-2011 Dataset}},
      Author      = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
      Year        = {2011}
      Institution = {California Institute of Technology},
      Number      = {CNS-TR-2011-001}
  }
  '''

  def __init__(self):
    super(CUB200, self).__init__()

  @property
  def name(self): return 'CUB_200_2011'

  @property
  def shape(self): return (100, 100, 3)

  @property
  def pickle_file(self): return 'CUB200.pkl.gz'

  @property
  def num_train(self): return 70000

  @property
  def num_test(self): return 10000

  @property
  def num_classes(self): return 10 # all = 200 TODO: fix zlib bug (file size limit 4GB) or split pickle files?

  def download_data(self, force=False):
    self.compressed_file = DWL.download_file('http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz', FORCE=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

  def prepare_data(self):
    self.extracted_files               = os.path.join(self.extracted_files, 'CUB_200_2011', 'images')
    self.data_train, self.labels_train = DL.load_directory_as_dataset(self.extracted_files, (100, 100, 3), reshape=True, max_classes=self.num_classes)

    # True, if all present classes were created/used
    self.validate_shape = False

  def clean_up(self):
    DS.file_delete('CUB_200_2011.tgz')
    DS.file_delete('CUB_200_2011')


if __name__ == '__main__':
  CUB200().get_data(clean_up=True)
