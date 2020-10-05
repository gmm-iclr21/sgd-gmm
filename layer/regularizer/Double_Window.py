'''
Created on 10.04.2020

@author: Benedikt
'''
import matplotlib.pyplot as plt
import numpy as np
from utils.Queue      import Queue

class Double_Window_Visualizer(object):
  ''' double window approach visualization '''

  def __init__(self, **kwargs):
    ''' initialize visualizations '''
    self.visualization = kwargs.get('visualization', True) # enable or disable visualization

    if not self.visualization: return # plot only one average
    self.first           = True
    self.fig             = plt.figure(figsize=(16, 4))
    self.ax              = plt.gca()
    self.texts           = list()                               # list for lables
    min_, max_           = 100, 700                             # min-max axis limit
    start_end_first      = [self.start_first , self.end_first]
    start_end_second     = [self.start_second, self.end_second]
    self.index           = np.arange(self.max_values)

    self.ax.set_xlim(0, self.max_values - 1)
    self.ax.set_ylim(min_, max_)
    plt.gcf().canvas.set_window_title(f'Visualize {self._name}')

    label                = kwargs.get('ylabel'         , 'log-likelihood')
    alpha                = kwargs.get('smoothing_alpha', 1.0)
    label                = label if alpha == 1.0 else f'{label} (smoothing factor = {alpha})'
    self._loglikelihood, = self.ax.plot([0]             , [0]   , linestyle='-' , color='black'  , label=label                     )
    self._avg_first    , = self.ax.plot(start_end_first , [0, 0], linestyle='-.', color='green'  , label='first window AVG'        )
    self._avg_second   , = self.ax.plot(start_end_second, [0, 0], linestyle='-.', color='blue'   , label='second window AVG (gray)')
    self._std_first      = self.ax.fill_between(   # standard deviation
      start_end_first                            , # The x coordinates of the nodes defining the curves.
      [0]                                        , # The y coordinates of the nodes defining the first curve.
      [0]                                        , # The y coordinates of the nodes defining the second curve.
      facecolor='red'                            ,
      alpha=0.3                                  ,
      label=f'first window STD {self.limit:+.0%}',
      )

    self._window_second = self.ax.fill_between( # highlight second window (gray background)
      [self.start_second, self.end_second], # The x coordinates of the nodes defining the curves.
      min_                                , # The y coordinates of the nodes defining the first curve.
      max_                                , # The y coordinates of the nodes defining the second curve.
      facecolor='gray'                    ,
      alpha=0.2                           ,
      )

    self.fig.legend(framealpha=1.0) # enable legend
    plt.ion()                       # enable non blocking mode
    plt.draw()                      # start window
    plt.ylabel('log-likelihood')
    plt.xlabel('value windows')


  def _vis(self):
    ''' a (faster) visualization of the regularizer '''
    if not self.visualization: return

    # full log-likelihood line
    self._loglikelihood.set_xdata(self.index[:self.current_values])
    self._loglikelihood.set_ydata(self._data()[::-1])
    # first window average
    self._avg_first.set_ydata([self.avg_first] * 2)
    # first window std
    self._std_first.get_paths()[0].vertices[:, 1]     = self.avg_first + self.std_first * (1 + self.limit) # upper limit
    self._std_first.get_paths()[0].vertices[[1,2], 1] = self.avg_first - self.std_first * (1 + self.limit) # lower limit
    # second window average
    self._avg_second.set_ydata([self.avg_second] * 2)
    # plot different backgrounds
    if self.current_values < self.max_values: background_color = 'yellow' # now enough values
    elif not self._check_limit()            : background_color = 'red'    # ready to trigger
    else                                    : background_color = 'green'  # regularize (trigger callbacks)

    # redraw
    self.fig.patch.set_facecolor(background_color)
    try   : self.fig.canvas.draw_idle()
    except: # if window is closed by user
      self.visualization = False
      return
    self.fig.canvas.flush_events()


  def _stop_vis(self):
    if not self.visualization: return
    plt.ioff()
    plt.draw()
    plt.show()


class Double_Window(Double_Window_Visualizer, Queue):
  ''' double window approach '''

  def __init__(self, **kwargs):
    ''' init double window approach<
      @param max_values  : number of previous values to store
      @param reset       : if True full reset else no reset or number of iterations (default = True)
      @param start_first : start of the first window        (default = 0)
      @param end_first   : end of the first window          (default = half of full window)
      @param start_second: start of the second window       (default = half of full window)
      @param end_seocnd  : end of the first window          (default = full window)
      @param limit       : percent SDT until regularization (default = 0.0 (0% of average))
    '''
    self.max_values   = kwargs.get('max_values'  , 100                 )
    self.start_first  = kwargs.get('start_first' , 0                   )
    self.end_first    = kwargs.get('end_first'   , self.max_values // 2)
    self.start_second = kwargs.get('start_second', self.max_values // 2)
    self.end_second   = kwargs.get('end_second'  , self.max_values - 1 )
    self.VALUE        = 0 # TODO: !

    self.avg_first    = 0
    self.std_first    = 0
    self.avg_second   = 0
    self.std_second   = 0

    self._set_limit(**kwargs)
    Queue.__init__(self, self.max_values, **kwargs)
    Double_Window_Visualizer.__init__(self, **kwargs)


  def _set_limit(self, **kwargs):
    ''' limit of x% (standard deviation must be lower then x% of average) '''
    self.limit = kwargs.get('limit', 0.0)


  def _check_limit(self):
    ''' two window approach to pull the trigger '''
    self.avg_first  = self._avg(self.start_second, self.end_second) # first and second are swapped because data are reversed stored
    self.std_first  = self._std(self.start_second, self.end_second) # e.g. [3,2,1] add (4) [4,3,2], so newest values are on the left side. visualization is reversed

    self.avg_second = self._avg(self.start_first, self.end_first)

    if self.current_values < self.max_values: return False # not enough values to check, window must be full

    if (self.avg_second > self.avg_first - self.std_first * (1 + self.limit) and
        self.avg_second < self.avg_first):
      try   : self.VALUE   += self.avg_second - self.prev_avg # or if hasattr(self, 'prev_avg'):
      except: self.prev_avg = self.avg_second
      return True

    if (self.avg_second < self.avg_first + self.std_first * (1 + self.limit) and
        self.avg_second > self.avg_first):
      try   : self.VALUE   += self.avg_second - self.prev_avg
      except: self.prev_avg = self.avg_second
      return True

    return False


  def _avg(self, start=-1, end=-1):
    ''' calculate the average '''
    data = self._data(start, end)
    return np.average(data)


  def _std(self, start=-1, end=-1):
    ''' calculate the standard deviation '''
    data = self._data(start, end)
    return np.std(data)

