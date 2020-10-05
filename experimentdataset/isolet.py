import os
import sys
import subprocess

from . import Data_Loader as DL
from . import Downloader as DWL
from . import Dataset as DS

class ISOLET(DS):
  ''' ISOLET (Isolated Letter Speech Recognition)
    source: https://archive.ics.uci.edu/ml/datasets/ISOLET

    Sources:
       Creators: Ron Cole and Mark Fanty
         Department of Computer Science and Engineering, Oregon Graduate Institute, Beaverton, OR 97006., cole@cse.ogi.edu, fanty@cse.ogi.edu
       Donor: Tom Dietterich
         Department of Computer Science, Oregon State University, Corvallis, OR 97331, tgd@cs.orst.edu
       September 12, 1994
  '''

  def __init__(self):
    super(ISOLET, self).__init__()

  @property
  def name(self): return 'ISOLET'

  @property
  def shape(self): return (617, 1, 1) # range -1.0 to 1.0.

  @property
  def pickle_file(self): return 'ISOLET.pkl.gz'

  @property
  def num_train(self): return 6238

  @property
  def num_test(self): return 1559

  @property
  def num_classes(self): return 10 # all = 25

  def download_data(self, force=False):
    self.compressed_file1 = DWL.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z', FORCE=force)
    self.compressed_file2 = DWL.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z', FORCE=force)

  def extract_data(self, force=False):
    if os.name == 'nt':
      def extract(filename):
        filename_no_ext = filename.rsplit('.', 1)[0]
        cmd = ['C:\\Program Files\\7-Zip\\7z.exe', '-y', 'x' if force else '', f'{filename}', f'-o{filename_no_ext}']
        with open(os.devnull, 'wb') as devnull: subprocess.check_call(cmd, stdout=devnull, stderr=subprocess.STDOUT)
      extract(self.compressed_file1)
      extract(self.compressed_file2)
    else: # Linux
      def extract(filename):
        filename_no_ext = filename.rsplit('.', 1)[0]
        os.system(f'mkdir {filename_no_ext}')
        os.system(f'mv {filename} {filename_no_ext}')
        os.system(f'uncompress {filename_no_ext}/*.Z')
      extract(self.compressed_file1)
      extract(self.compressed_file2)

  def prepare_data(self):
    data_train  = DL.load_sep_data(os.path.join(self.compressed_file1.rsplit('.', 1)[0], 'isolet1+2+3+4.data'), extract_classes=self.num_classes + 1) # +1 because class tag start at 1
    data_test   = DL.load_sep_data(os.path.join(self.compressed_file2.rsplit('.', 1)[0], 'isolet5.data'      ), extract_classes=self.num_classes + 1)

    label_train, data_train = data_train[:, -1], data_train[:, :-1]
    label_test , data_test  = data_test[:, -1] , data_test[:, :-1]

    self.data_train, self.labels_train = data_train, label_train
    self.data_test , self.labels_test  = data_test , label_test

    # True, if all present classes were created/used
    self.validate_shape = False

  def clean_up(self):
    DS.file_delete('isolet1+2+3+4.data.Z')
    DS.file_delete('isolet1+2+3+4.data')
    DS.file_delete('isolet5.data.Z')
    DS.file_delete('isolet5.data')

if __name__ == '__main__':
  ISOLET().get_data(clean_up=True)
