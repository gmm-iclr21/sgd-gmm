import time
import psutil
import os

def debug_variables(obj, ignore=[]):
  side = '-' * int(((80 - len('debug variables')) / 2))
  print(side + ' ' + 'debug variables' + ' ' + side)

  for k, v in sorted(vars(obj).items(), key=lambda x: x[0].lower()):
    if k in ignore: continue
    print('{:25}'.format(k), '=', v)
  print('-' * 80)


def timeing(method):
  ''' decorator for time (and memory) measuring of functions '''
  def measure(*args, **kwargs):
    start  = time.time()
    result = method(*args, **kwargs)
    if hasattr(psutil.Process(), 'memory_info'):
      mem = psutil.Process(os.getpid()).memory_info()[0] // (2 ** 20)
      print('{}...{:2.2f}s mem: {}MB'.format(method.__name__, time.time() - start, mem))
    elif hasattr(psutil.Process(), 'memory_full_info'):
      mem = psutil.Process(os.getpid()).memory_full_info()[0] // (2 ** 20)
      print('{}...{:2.2f}s mem: {}MB'.format(method.__name__, time.time() - start, mem))
    else:
      print('{}...{:2.2f}s '.format(method.__name__, time.time() - start))
    return result

  return measure
