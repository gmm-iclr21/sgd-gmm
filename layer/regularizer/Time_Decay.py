'''
Created on 10.04.2020

@author: Benedikt
'''

import math
from . import Regularizer

class Time_Decay_Visualizer():
  def __init__(self, _regularizer, **kwargs):
    super().__init__(**kwargs)

    if not self.visualization: return # plot only one average
    self.regularizer    = _regularizer

  def _vis(self):
    # TODO: show the pre-defined curve (plotted only once)
    pass

class Time_Decay():
  ''' Time Decay Regularizer (no reset after task) '''

  def __init__(self, **kwargs):
    ''' init time (iteration) decay approach

    @param max_iterations: number of training iterations
    @param t0Frac        : start point of regularization (percentage of max_iterations) for both, leanrate and somSigma
    @param tInfFrac      : end point of regularization (percentage of max_iterations) for both, leanrate and somSigma
    @param eps0          : start value of learn rate
    @param epsInf        : end value of learn rate
    @param somSigma0     : start value of somSigma
    @param somSigmaInf   : end value of somSigma
    '''
    self.t0            = kwargs.get('t0Frac')   * kwargs.get('max_iterations')
    self.t1            = kwargs.get('tInfFrac') * kwargs.get('max_iterations')

    self.eps0          = kwargs.get('eps0')
    self.epsInf        = kwargs.get('epsInf')

    self.somSigma0     = kwargs.get('somSigma0')
    self.somSigmaInf   = kwargs.get('somSigmaInf')

    self.kappaEps      = math.log(self.eps0 / self.epsInf) / (self.t1 - self.t0)
    self.kappaSomSigma = math.log(self.somSigma0 / self.somSigmaInf) / (self.t1 - self.t0)

    self.iteration     = 0

    self._set_limit(**kwargs)


  def add(self, _):
    ''' add a iteration '''
    self.iteration += 1
    return self

  def _set_limit(self, **kwargs):
    ''' limit of x% (standard deviation must be lower then x% of average) '''
    self.limit = kwargs.get('limit', 0)

  def _check_limit(self):
    ''' check limit '''
    if self.iteration < self.t0:
      self.epsValue      = self.eps0
      self.somSigmaValue = self.somSigma0
      return self.epsValue, self.somSigmaValue
    elif self.iteration > self.t1:
      self.epsValue      = self.epsInf
      self.somSigmaValue = self.somSigmaInf
      return self.epsValue, self.somSigmaValue
    else:
      self.epsValue      = math.exp(-self.kappaEps      * (self.iteration - self.t0)) * self.eps0
      self.somSigmaValue = math.exp(-self.kappaSomSigma * (self.iteration - self.t0)) * self.somSigma0
      return self.epsValue, self.somSigmaValue
    return False

class TimeDecayReg(Regularizer):
  def __init__(self, max_iter, **kwargs):
    super().__init__(**kwargs)

    self.alpha         = kwargs.get('t0'       , 500)
    self.delta         = kwargs.get('tInf'     , 2000)
    self.cb            = kwargs.get('callbacks', list())
    self.eps0          = kwargs.get('eps0')
    self.epsInf        = kwargs.get('epsInf')

    self.somSigma0     = kwargs.get('somSigma0')
    self.somSigmaInf   = kwargs.get('somSigmaInf')
    self.max_iter      = max_iter
    self.t0            = kwargs.get('t0Frac') * self.max_iter
    self.t1            = kwargs.get('tInfFrac') * self.max_iter

    print('self.t0', self.t0)
    print('self.t1', self.t1)

    self.kappaEps      = math.log(self.eps0 / self.epsInf) / (self.t1 - self.t0)
    self.kappaSigma    = math.log(self.somSigma0 / self.somSigmaInf) / (self.t1 - self.t0)

    self.iteration     = 0


  def add(self, *args, **kwargs):
    self.iteration += 1
    return self


  def _check(self):
    if  self.iteration < self.t0 or self.iteration > self.t1: return

    self.V.set_no_callback(
      epsValue      = self.eps0      * math.exp(-self.kappaEps   * (self.iteration - self.t0)),
      somSigmaValue = self.somSigma0 * math.exp(-self.kappaSigma * (self.iteration - self.t0)),
      )
