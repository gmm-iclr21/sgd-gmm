# Annealing control used for NIPS experiments!!

import math
import matplotlib.pyplot as plt
from .Regularizer import Regularizer, Regularizer_Visualization

class Single_Exp_Visualizer(Regularizer_Visualization):

  def __init__(self, _regularizer, **kwargs):
    ''' initialize visualizations '''
    super().__init__(**kwargs)

    if not self.visualization: return # plot only one average
    self.regularizer    = _regularizer

    self.fig             = plt.figure(figsize=(6, 4))
    self.ax              = plt.gca()

    self.ax.set_xlim(0, 2)
    plt.gcf().canvas.set_window_title(f'Visualize {self.regularizer._name}')

    self.xy_         = self.ax.bar(0.35, 0, 0.35, label='verhaeltnis')

    self.limit_up_,  = self.ax.plot([0, 2], [0, 0], linestyle='-' , color='red', label='upper limit'  )
    self.limit_lo_,  = self.ax.plot([0, 2], [0, 0], linestyle='-' , color='red', label='lower limit'  )

    self._iter_count = self.ax.text(
      0., 1.1, f'',
      transform = self.ax.transAxes
      )

    self.fig.legend(framealpha=1.0) # enable legend
    plt.ion()                       # enable non blocking mode
    plt.draw()                      # start window
    plt.ylabel('log-likelihood')
    plt.xlabel('value')
    self.ax.get_xaxis().set_visible(False)


  def vis(self):
    ''' a (faster) visualization of the regularizer '''
    if not self.visualization: return

    limit_x = self.regularizer.avgLong - self.regularizer.lastAvg
    limit_y = self.regularizer.lastAvg - self.regularizer.l0
    limit   = limit_x / limit_y # verhaeltnis zu alten Wert (prozentuelle Zunahme)

    self.xy_[0].set_height(limit)

    self.limit_up_.set_ydata(self.regularizer.delta * -2)
    self.limit_lo_.set_ydata(self.regularizer.delta)

    text = f'mod = {self.regularizer.event_mod} count {self.regularizer.event_count} iterations {self.regularizer.iteration}'
    self._iter_count.set_text(text)
    try:
      self.fig.canvas.draw_idle()
    except: # if window is closed by user
      self.visualization = False
      return
    self.fig.canvas.flush_events()


class SingleExp(Regularizer):

  def __init__(self, **kwargs):
    kwargs['_name']  = kwargs.get('name'                   , 'SingleExpRegularizer')
    self.delta       = kwargs.get('regularizer_delta'      , 0.1                    )
    self.gamma       = kwargs.get('regularizer_gamma'      , 0.9                   )

    self.eps0        = kwargs.get('eps0'                                           )
    self.epsInf      = kwargs.get('epsInf'                                         )
    self.somSigma0   = kwargs.get('somSigma0'                                      )
    self.sigmaInf    = kwargs.get('somSigmaInf'                                    )

    self.reset_sigma = kwargs.get('regularizer_reset_sigma', self.sigmaInf         ) # default, reset to somSigmaInf, even if not yet completely reduced
    self.reset_eps   = kwargs.get('regularizer_reset_eps'  , self.epsInf           ) # default, reset to

    self.alpha       = self.epsInf / 1.
    self.avgLong     = 0.0
    self.l0          = 0
    self.lastAvg     = 0
    self.iteration   = 0
    self.event_mod   = 0
    self.event_count = 0

    super().__init__(**kwargs)

    self.vis_ = Single_Exp_Visualizer(self, **kwargs)


  def __str__(self):
    s = ''
    s += f'Regularizer: {self._name}\n'
    s += f' delta      : {self.delta}\n'
    s += f' gamma      : {self.gamma}\n'
    s += f' eps0       : {self.eps0}\n'
    s += f' epsInf     : {self.epsInf}\n'
    s += f' somSigma0  : {self.somSigma0}\n'
    s += f' sigmaInf   : {self.sigmaInf}\n'
    s += f' reset_sigma: {self.reset_sigma}\n'
    s += f' reset_eps  : {self.reset_eps}\n'
    s += f' alpha      : {self.alpha}'
    return s

  def add(self, loss):
    if self.iteration == 0: # init regularizer
      self.avgLong  = loss
      self.l0       = loss
    else:
      self.avgLong *= 1. - self.alpha
      self.avgLong += self.alpha * loss
    self.iteration += 1
    return self


  def set(self, eps=None, sigma=None):
    ''' reset the regularizer '''
    reset_eps            = eps   if eps   else self.reset_eps
    reset_sigma          = sigma if sigma else self.reset_sigma

    self.V.epsValue      = reset_eps
    self.V.somSigmaValue = reset_sigma
    self.iteration       = 0


  def _check(self):
    # do nothing during the first period just memorize initial loss value as a baseline
    max_iter    = math.ceil(1. / self.alpha) # round up
    event_count = self.iteration // max_iter
    event_mod   = self.iteration  % max_iter

    #print(self.iteration);
    if event_mod != 0  : # if we are not at the end of a period, do nothing
      return
    if event_count == 0: # first event: no lastAvg yet, so set it trivially, no action
      self.lastAvg = self.l0
      return
    if event_count == 1: # second event: lastAvg can be set non-trivially, no action
      self.lastAvg = self.avgLong
      return

    self.event_mod   = event_mod
    self.event_count = event_count


    # event later than second: check for sigma/eps reduction and do it (basis for decision: l0, lastAvg, avgLong)
    limit       = (self.avgLong - self.lastAvg) / (self.lastAvg - self.l0)

    if limit > -2 * self.delta and limit < self.delta: # if energy does not increase sufficiently --> reduce!
      if self.V.get_no_callback('epsValue')      * self.gamma >= self.epsInf  : self.V.epsValue      *= self.gamma
      if self.V.get_no_callback('somSigmaValue') * self.gamma >= self.sigmaInf: self.V.somSigmaValue *= self.gamma
    self.lastAvg = self.avgLong # update lastAvg for next event
