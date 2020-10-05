
import fuckit
from functools import partial

DEBUG=False

def debug(msg): DEBUG and print(msg)

class Variables:
  def __init__(self):
    super().__setattr__('callbacks', dict())

  def __getattribute__(self, name):
    value = super().__getattribute__(name)
    debug(f'get {name} -> {value} callback')
    return value

  def changeable(self, name, to=None):
    # TODO: toggle if to is None, else use the boolean of to to set
    pass

  def __setattr__(self, name, value):
    # TODO: add parameter to stop overwrite (in combination with method changeable)
    debug(f'set {name} <- {value} callback')
    super().__setattr__(name, value)
    with fuckit: super().__getattribute__('callbacks')[name](value)

  def set_no_callback(self, **kwargs):
    for name, value in kwargs.items():
      debug(f'set {name} <- {value} no callback')
      super().__setattr__(name, value)

  def get_no_callback(self, name):
    value = super().__getattribute__(name)
    debug(f'get {name} -> {value} no callback')
    return value

  def _register(self, name, value, callback):
    debug(f'register {name} = {value} with {callback}')
    super().__setattr__(name, value)
    if callback: callback = partial(callback, name)
    super().__getattribute__('callbacks')[name] = callback

  def register(self, **kwargs):
    for name, value in kwargs.items():
      if name == 'callback': continue
      self._register(name, value, kwargs.get('callback'))

  def __str__(self):
    s = 'VARIABLES:'
    for variable_name, value in self.__dict__.items():
      if variable_name == 'callbacks': continue
      s +=  f'\n{variable_name} = {value}'
    return s

if __name__ == '__main__':
  # TESTING
  V = Variables()
  V.register(a=3, b='hallo',  callback=lambda x: print(x))

  V.b += ' welt'
  a = V.get_no_callback('a')
  V.set_no_callback(a=5)

