
import matplotlib.pyplot as plt
import numpy as np
from utils.Queue import Queue
import warnings

class Visualizer(object):
  ''' a simple visualizer class '''

  def __init__(self, **kwargs):
    ''' initialize a visualization object
    @param enabled  : enable/disable the visualization                  (default = True)
    @param figsize  : shape of the figure                               (default = (16, 4))
    @param xlim     : left and right limit of x-axis, None = auto       (default = (None, None))
    @param ylim     : lower and upper limit of y-axis, None = auto      (default = (None, None))
    @param name     : name of the visualization                         (default = '')
    @param xlabel   : label of th x-axis                                (default = '')
    @param ylabel   : label of th y-axis                                (default = '')
    @param queue    : add a queue object that is visualized utils.Queue (default = None)
    @param linestyle: style of the line                                 (default = '-')
    @param color    : color of the line                                 (default = 'black')
    @param label    : label of the line                                 (default = name of Visualizer)
    '''
    self.enabled           = kwargs.get('enabled'   , True        )
    self.figsize           = kwargs.get('figsize'   , (16, 4)     )
    self.xlim              = kwargs.get('xlim'      , (None, None))
    self.ylim              = kwargs.get('ylim'      , (None, None))
    self.name              = kwargs.get('name'      , ''          )
    self.xlabel            = kwargs.get('xlabel'    , ''          )
    self.ylabel            = kwargs.get('ylabel'    , ''          )
    self.queue             = kwargs.get('queue'     , None        )

    if not self.enabled: return

    self.fig               = plt.figure(figsize=self.figsize)
    self.ax                = plt.gca()

    if self.queue and self.xlim == (None, None): self.xlim = (0, self.queue.max_values) # auto resize mechanism for queues

    self.ax.set_xlim(self.xlim[0] if self.xlim[0] is not None else 0, # set x-axis limits (left)
                     self.xlim[1] if self.xlim[1] is not None else 1) # set x-axis limits (right)
    self.ax.set_ylim(self.ylim[0] if self.ylim[0] is not None else 0, # set y-axis limits (lower)
                     self.ylim[1] if self.ylim[1] is not None else 1) # set y-axis limits (upper)

    self.init_plot(**kwargs)

    self.fig.legend(framealpha=1.0) # create legend
    plt.gcf().canvas.set_window_title(f'Visualization: {self.name}')
    plt.xlabel(self.xlabel)
    plt.ylabel(self.ylabel)
    plt.ion()                       # enable the interactive mode
    plt.draw()                      # activate window


  def init_plot(self, **kwargs):
    ''' create placeholder objects
    @param linestyle: style of the line (default = '-')
    @param color    : color of the line (default = 'black')
    @param label    : label of the line (default = name of Visualizer)
    '''
    linestyle = kwargs.get('linestyle', '-')
    color     = kwargs.get('color'    , 'black')
    label     = kwargs.get('label'    , self.name)

    if self.queue:
      label = f'{label} (smoothing factor={self.queue.alpha})'

    self.line, = self.ax.plot( # placeholder (is updated in redraw)
      [0]                   ,  # xs
      [0]                   ,  # ys
      linestyle = linestyle ,
      color     = color     ,
      label     = label     ,
      )

  def redraw(self, **kwargs):
    ''' update the visualization
    @param ys: values to visualize (default = None)
    @param xs: x-axis partition    (default = range(len(ys)))
    '''
    if not self.enabled: return

    ys = kwargs.get('ys')
    xs = kwargs.get('xs', np.arange(len(ys)) if ys else None)

    if self.queue: # use the data of the queue, if available
      ys = self.queue._data()[::-1]
      xs = np.arange(len(self.queue._data()[::-1]))

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      self.ax.set_xlim(self.ax.get_xlim()[0] if self.xlim[0] is not None else min(xs), # define x-axis limits (left limit)
                       self.ax.get_xlim()[1] if self.xlim[1] is not None else max(xs)) # define x-axis limits (right limit)
      self.ax.set_ylim(self.ax.get_ylim()[0] if self.ylim[0] is not None else min(ys), # define y-axis limits (lower limit)
                       self.ax.get_ylim()[1] if self.ylim[1] is not None else max(ys)) # define y-axis limits (upper limit)

    self.line.set_xdata(xs)
    self.line.set_ydata(ys)

    # redraw
    try:
      self.fig.canvas.draw_idle()
    except: # if window is closed by user
      self.enables = False
      return

    # self.fig.canvas.flush_events()

  def stop_vis(self, **kwargs):
    ''' deactivate the interactive mode '''
    if not self.enabled: return
    plt.ioff() # disable the interactive mode
    plt.draw() # redraw
    plt.show() #

class Queue_Visualizer(Visualizer, Queue):
  ''' combine a Queue and a Visualizer (auto queue inject)'''
  def __init__(self, max_values, **kwargs):
    Queue.__init__(self, max_values, **kwargs)
    Visualizer.__init__(self, queue=self, **kwargs)





