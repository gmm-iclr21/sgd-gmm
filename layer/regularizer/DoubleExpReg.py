import matplotlib.pyplot as plt
from .Regularizer import Regularizer, Regularizer_Visualization

class Double_Exp_Visualizer(Regularizer_Visualization):

  def __init__(self, _regularizer, **kwargs):
    ''' initialize visualizations '''
    super().__init__(**kwargs)

    if not self.visualization: return # plot only one average
    self.regularizer     = _regularizer
    self.first           = True
    self.fig             = plt.figure(figsize=(6, 4))
    self.ax              = plt.gca()
    min_, max_           = -250, 100                             # min-max axis limit

    self.ax.set_xlim(0, 2)
    self.ax.set_ylim(min_, max_)
    plt.gcf().canvas.set_window_title(f'Visualize {self.regularizer._name}')

    self._avg_long ,     = self.ax.plot([0, 2], [0, 0], linestyle='-' , color='black', label='long AVG')
    self._avg_short,     = self.ax.plot([1, 2], [0, 0], linestyle='-' , color='green', label='short AVG')
    self._avg_delta,     = self.ax.plot([1, 2], [0, 0], linestyle='--', color='red'  , label='AVG delta')
    self._iter_count     = self.ax.text(0, max_ * 1.05 , f'')

    self._window_second = self.ax.fill_between( # highlight second window (gray background)
      [1, 2]            , # The x coordinates of the nodes defining the curves.
      min_              , # The y coordinates of the nodes defining the first curve.
      max_              , # The y coordinates of the nodes defining the second curve.
      facecolor = 'gray',
      alpha     = 0.2   ,
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

    # plot avg short and avg long
    self._avg_long.set_ydata([self.regularizer.avgLong] * 2)
    self._avg_short.set_ydata([self.regularizer.avgShort] * 2)
    self._avg_delta.set_ydata([self.regularizer.avgShort - self.regularizer.delta] * 2)

    sleep_iter = int(1. / self.regularizer.alpha)
    left_iter  = sleep_iter - self.regularizer.iteration
    if left_iter > 0: self._iter_count.set_text(f'iterations left until check: {left_iter}/{sleep_iter}')
    else            : self._iter_count.set_text(f'iterations over check: {-left_iter}')

    try:
      self.fig.canvas.draw_idle()
    except: # if window is closed by user
      self.visualization = False
      return
    self.fig.canvas.flush_events()


class DoubleExpReg(Regularizer, Double_Exp_Visualizer):
  def __init__(self, **kwargs):
    ''' Double Exp Regularizer
    @param _name                 :
    @param regularizer_alpha     :
    @param regularizer_delta     :
    @param regularizer_gamma     :
    @param somSigma0             :
    @param eps0                  :
    @param regularizer_rest_sigma: value to which the sigma is changed on reset (set function without parameters)
    @param regularizer_rest_eps  : value to which the epsilon is changed on reset (set function without parameters)
    '''
    kwargs['_name']  = kwargs.get('name'                   , 'DoubleExpRegularizer')
    self.alpha       = kwargs.get('regularizer_alpha'      , 0.002                 ) # TODO: start regularizing with 500 iterations delay
    self.delta       = kwargs.get('regularizer_delta'      , 4                     )
    self.gamma       = kwargs.get('regularizer_gamma'      , 0.9                   )

    self.epsInf      = kwargs.get('epsInf'                                         )
    self.sigmaInf    = kwargs.get('somSigmaInf'                                    )

    self.reset_sigma = kwargs.get('regularizer_reset_sigma', self.sigmaInf         ) # default, reset to somSigmaInf, even if not yet completely reduced
    self.reset_eps   = kwargs.get('regularizer_reset_eps'  , self.epsInf           ) # default, reset to

    self.avgLong   = 0.0
    self.avgShort  = 0.0
    self.iteration = 0

    super().__init__(**kwargs)
    self.vis_ = Double_Exp_Visualizer(self, **kwargs)


  def add(self, loss):
    if self.iteration == 0: self.avgShort = self.avgLong = self.l0 = loss # init regularizer
    self.avgLong   *= 1. - self.alpha
    self.avgLong   += self.alpha * loss

    self.avgShort  *= 1. - 2 * self.alpha
    self.avgShort  += 2 * self.alpha * loss
    self.iteration += 1
    return self


  def set(self, eps=None, sigma=None):
    reset_eps            = eps   if eps   else self.reset_eps
    reset_sigma          = sigma if sigma else self.reset_sigma

    self.V.epsValue      = reset_eps
    self.V.somSigmaValue = reset_sigma

    self.iteration       = 0


  def _check(self):
    if self.iteration > 1. / self.alpha:
      if (self.avgShort - self.avgLong) < self.delta:
        self.V.epsValue      *= self.gamma
        self.V.somSigmaValue *= self.gamma
        self.iteration  = 0

