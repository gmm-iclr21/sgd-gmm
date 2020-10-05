'''
Created on 16.01.2020

@author: Benedikt
'''

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import CheckButtons
import fuckit

''' # TODO: (re)move for future hacks
def _update(variable, new_value):
  mutate(variable, new_value)

def mutate(obj, new_obj):
  import ctypes
  import sys
  mem = (ctypes.c_byte * sys.getsizeof(obj)).from_address(id(obj))
  new_mem = (ctypes.c_byte * sys.getsizeof(new_obj)).from_address(id(new_obj))
  for i in range(len(mem)):
    mem[i] = new_mem[i]
'''

class Parameter_Controller(object):

  def __new__(cls, *args, **kwargs):
    parameter_controller = super(Parameter_Controller, cls).__new__(cls)
    parameter_controller.__init__(*args, **kwargs) # if __new__ return something else than a instance, __init__ is not automatically invoked!
    return parameter_controller, lambda x, y: parameter_controller.update_value(x, y)

  def __init__(self, title='Parameter Controller', visualization_enabled=True):
    self.enabled = visualization_enabled
    if not self.enabled: return

    self.fig, self.ax = plt.subplots()
    self.fig.delaxes(self.ax)
    # self.ax.margins(x=0)

    self.axes         = dict()
    self.slider       = dict()
    self.buttons      = list()
    self.checkbox     = list()

    self._column      = 0.9

    plt.ion()                       # enable non blocking mode
    plt.draw()                      # start window
    plt.show()
    plt.gcf().canvas.set_window_title(title)

  def get_next_ax(self, name):
    if not self.enabled: return
    if name in self.axes: return self.axes[name]
    ax = plt.axes([0.3, self._column, 0.6, 0.03]) #  [left, bottom, width, height]
    self._column -= 0.05
    return ax

  def add_checkbox(self): # TODO: implement
    pass

  def add_button(self): # TODO: implement
    pass

  def add_slider(self, variables, name, valmin=-1, valmax=1, valsteps=40, valstep=None, checkbox=False): # TODO: add checkbox to stop value change
    if not self.enabled: return
    ax     = self.get_next_ax(name)
    value  = variables.get_no_callback(name)
    if not valstep: valstep = (valmax - valmin) / valsteps
    slider = Slider(
      ax      = ax     ,
      label   = name   ,
      valmin  = valmin ,
      valmax  = valmax ,
      valinit = value  ,
      valstep = valstep,
      )
    slider.on_changed(lambda x: variables.set_no_callback(**{name: x}))
    self.slider[name] = slider


  def update_value(self, name, value):
    if not self.enabled: return
    self.slider[name].set_val(value)
    self.fig.canvas.draw_idle()

  def visualize(self):
    if not self.enabled: return
    with fuckit: self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()

  def stop_vis(self):
    if not self.enabled: return
    plt.ioff() # disable the interactive mode
    plt.draw() # redraw
    plt.show() #
