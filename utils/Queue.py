
import numpy as np

class Queue(object):
  ''' a simple queue realized with numpy arrays'''

  def __init__(self, max_values, **kwargs):
    ''' initialize the queue

    @param max_values     : maximum size of the queue
    @param smoothing_alpha: average = old_average * (1 - smoothing_alpha) + new_value * smoothing_alpha
    @param reset          : define reset strategy:
      True  = all values are deleted
      False = no value is deleted
      (int) = the number of values to delete
    '''
    self.max_values     = max_values                         # size of queue (self.values)
    self.current_values = 0                                  # number of currently stored values
    self.values         = np.zeros(max_values)               # stores values for average calculation (queue)
    self.alpha          = kwargs.get('smoothing_alpha', 1.0) # smoothing alpha (default=1.0 (off))

    self._set_reset(**kwargs)


  def add(self, value):
    ''' add a value to the Regularizer, e.g., max_values=3 add(0) [1 2 3] -> [0 1 2] '''
    value = self._smooth(value)
    self._add(value)
    return self


  def _set_reset(self, **kwargs):
    ''' limit reach behavior (complete reset, no reset, partly reset) '''
    reset = kwargs.get('reset', True)
    if isinstance(reset, bool):
      if reset: self.reset_num_elements = 0   # complete reset
      else    : self.reset_num_elements = -1  # do not reset
    else:
      self.reset_num_elements = reset         # define the number of elements to delete


  def reset(self):
    ''' reset values '''
    if self.reset_num_elements == -1: return # no reset
    if self.reset_num_elements == 0 :        # complete reset
      self.values         = np.zeros(self.max_values)
      self.current_values = 0
    else:
      self.current_values = self.current_values - self.reset_num_elements


  def _data(self, start=-1, end=-1):
    ''' get (slice) data '''
    first = 0
    last  = self.current_values
    if start > -1 and start < self.current_values: first = start
    if end   > -1 and end  <= self.current_values: last  = end
    data = self.values[first: last]
    return data


  def _add(self, value):
    ''' add a value to queue '''
    self.values[1:]     = self.values[:-1]
    self.values[0]      = value
    self.current_values = min(self.current_values + 1, self.max_values)


  def _smooth(self, value):
    ''' sliding/incremental average '''
    try   : self.last = (1.0 - self.alpha) * self.last + self.alpha * value
    except: self.last = value
    return self.last

