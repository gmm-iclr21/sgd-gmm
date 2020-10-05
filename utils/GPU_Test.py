'''
Test if a GPU is available, disable error messages
'''

import os
import sys
import tensorflow as tf
import warnings
from tensorflow.python.client import device_lib

warnings.filterwarnings('ignore', category=FutureWarning)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf debug messages

def check_GPU():
  devices = device_lib.list_local_devices()
  for device in devices:
    if 'GPU' in str(device) and device.IsInitialized(): return
  print('No GPU found or not initialized', file=sys.stderr)
  print(devices)

check_GPU()


