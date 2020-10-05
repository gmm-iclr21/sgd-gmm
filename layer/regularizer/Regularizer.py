from experimentparser import ArgEnum

class Callback(object):

  def __init__(self, **kwargs):
    ''' initialize callback class '''
    callbacks          = kwargs.get('callback', None) # initial callback functions
    self.callbacks     = list()                       # stores callback function
    self.add_callback(callbacks)


  def add_callback(self, callback_function):
    ''' add a callback function (is called if limit is reached)
    @param callback_function: reference to callback function or list of callback functions
    '''
    if not callback_function              : return
    if isinstance(callback_function, list): self.callbacks += callback_function
    else                                  : self.callbacks += [callback_function]


  def delete_callback(self, callback_function=None, index=None):
    ''' delete a callback function
    @param callback_function: reference of the callback function
    @param index            : index of the callback function
    '''
    if callback_function and index: raise Exception('What now? Decide.')
    if index                      : return self.callbacks.pop(index)
    if callback_function          : self.callbacks = [ callback for callback in self.callback if callback != callback_function ]
    return self.callbacks


  def call_callbacks(self, *args, **kwargs):
    ''' execute all callback functions '''
    for callback in self.callbacks: callback(*args, **kwargs)


class Regularizer_Visualization():

  def __init__(self, **kwargs):
    self.visualization = kwargs.get('vis_reg_', False)

  def vis(self): pass


class Regularizer():

  def __init__(self, **kwargs):
    self._name         = kwargs.get('_name', 'Regularizer')
    self.V             = kwargs.get('V')
    self.vis_           = None


  def add(self, loss)                : pass
  def set(self, eps=None, sigma=None): pass
  def check_limit(self)              : self._check()
  def _check(self)                   : pass
  def __str__(self)                  : return self.__class__.__name__
  def vis(self)                      :
    if self.vis_: self.vis_.vis()


class Regularizer_Method(ArgEnum):
  DOUBLE_EXP    = 'double_exp'
  SINGLE_EXP    = 'single_exp'
  DOUBLE_WINDOW = 'double_window'
  TIME_DECAY    = 'time_decay'
  DUMMY_REG     = 'dummy_reg'
