import numpy as np
from scipy.interpolate import interp1d
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D # no not remove, used for 3d plots
import pickle
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Numpy_Visualizer(object):

  def __init__(self, filename, **kwargs):
    self.filename          = filename
    self.name              = filename
    self.x_plots           = kwargs.get('x_plots'          , 5)
    self.y_plots           = kwargs.get('y_plots'          , 5)
    self.w                 = kwargs.get('width'            , 28)
    self.h                 = kwargs.get('height'           , 28)
    self.c                 = kwargs.get('channels'         , 1)
    self.D                 = kwargs.get('dimensionality'   , 784)
    self.plot_only         = kwargs.get('plot_only'        , True)
    self.store_only_output = kwargs.get('store_only_output', True)


    self.data = np.load(filename)
    
    # for mu in self.data:
    #   print(mu.reshape(32,32))
    print('self.data.shape', self.data.shape)
    self.data = np.squeeze(self.data)
    # print(self.data)

    self.K                 = self.x_plots * self.y_plots
    self.fig, axes         = plt.subplots(self.y_plots, self.x_plots)

    empty                  = np.zeros([self.K, self.D])
    empty[:, 0]            = 1
    empty[:, -1]           = 1

    self.all_ax_img        = list()

    # 3d plot variables
    self.bar               = None
    self.cbar              = None
    self.ax                = None

    shape                  = (self.h, self.w) if self.c == 1 else (self.h, self.w, self.c)

    if type(axes) == np.ndarray: # multiple images
      axes = axes.ravel()
      for data, ax_ in zip(empty, axes):
        ax_img = ax_.imshow(data.reshape(shape))
        self.all_ax_img.append(ax_img)

      for ax_ in axes:
        ax_.tick_params( # disable labels and ticks
          axis        = 'both',
          which       = 'both',
          bottom      = False ,
          top         = False ,
          left        = False ,
          right       = False ,
          labelbottom = False ,
          labelleft   = False ,
          )
    else: # only one image
      empty  = np.eye(self.w, self.h)
      ax_img = axes.imshow(empty)
      self.all_ax_img.append(ax_img)
      axes.tick_params( # disable labels and ticks
          axis        = 'both',
          which       = 'both',
          bottom      = False ,
          top         = False ,
          left        = False ,
          right       = False ,
          labelbottom = False ,
          labelleft   = False ,
          )

    plt.gcf().canvas.set_window_title(f'Visualization: {self.name}')

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    plt.ion()
    plt.draw()
    plt.show()


  def visualize(self):
    data   = self.data
    min_ = np.min(data)
    max_ = np.max(data)
    print('min, nax', min_, max_)
    images = interp1d([min_, max_], [0., 1.])(data)
    shape  = (self.h, self.w) if self.c == 1 else (self.h, self.w, self.c)
    print('shape', shape)
    for i, ax_img in enumerate(self.all_ax_img):
      if len(images.shape) != 1: image = images[i].reshape(shape)
      else                     : image = images.reshape(shape)

      ax_img._A = cbook.safe_masked_invalid(image, copy=False)

    if self.plot_only:
      plt.pause(0.00000000000000001)
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

    self.fig.savefig(f'{self.filename}.pdf')



if __name__ == '__main__':
  Numpy_Visualizer('../degen.npz').visualize()
  Numpy_Visualizer('../multiComp.npz').visualize()

