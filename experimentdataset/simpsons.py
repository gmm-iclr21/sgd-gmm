import os

from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class Simpsons(DS):
  ''' The Simpsons Characters Data
  souce: https://www.kaggle.com/alexattia/the-simpsons-characters-dataset/data

  website: https://github.com/alexattia/SimpsonRecognition
           https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras-d8e1796eae36

  20 folders (one for each character) with 0-2000 pictures in each folder (images have different sizes (color))
  '''

  def __init__(self):
    super(Simpsons, self).__init__()

  @property
  def name(self): return 'The Simpsons Characters Data'

  @property
  def shape(self): return (100, 100, 3)

  @property
  def pickle_file(self): return 'Simpsons.pkl.gz'

  @property
  def num_train(self): return 6789

  @property
  def num_test(self): return 754

  @property
  def num_classes(self): return 10 # 35

  def download_data(self, force=False):
    self.compressed_file = DWL.download_from_kaggle('alexattia/the-simpsons-characters-dataset', 'simpsons_dataset.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

  def prepare_data(self):
    # TODO: only select classes with the most images
    self.extracted_files               = os.path.join(self.extracted_files, 'simpsons_dataset') + os.sep
    self.data_train, self.labels_train = DL.load_directory_as_dataset(self.extracted_files, self.shape, reshape=True, max_classes=10)

    # False, if not all classes were used
    self.validate_shape = False

  def clean_up(self):
    DS.file_delete('simpsons_dataset/')
    DS.file_delete('simpsons_dataset.zip')


if __name__ == '__main__':
  Simpsons().get_data(clean_up=False)
