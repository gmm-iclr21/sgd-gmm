
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D # no not remove, used for 3d plots
import os
import pickle
import cv2
import numpy as np
from scipy.interpolate import interp1d

class Visualization():

  visualizer = dict()

  def __init__(self, name, **kwargs):
    ''' construct an preconfigured visualization object  '''
    if not os.path.exists('./plots/'): os.makedirs('./plots/')

    self.name              = name
    self.x_plots           = kwargs.get('x_plots'                 )
    self.y_plots           = kwargs.get('y_plots'                 )
    self.w                 = kwargs.get('width'                   )
    self.h                 = kwargs.get('height'                  )
    self.c                 = kwargs.get('channels'                )
    self.D                 = kwargs.get('dimensionality'          )
    self.plot_only         = kwargs.get('plot_only'        , True )
    self.store_only_output = kwargs.get('store_only_output', False)
    self.grid              = kwargs.get('grid'             , True )
    self.enabled           = kwargs.get('enabled'          , False)

    if False: # debug
      print('create vis', name      )
      print('x-plots', self.x_plots )
      print('y-plots', self.y_plots )
      print('width', self.w         )
      print('height', self.h        )
      print('channels', self.c      )
      print('dimensionality', self.D)

    if not self.enabled: return

    if self.store_only_output:
      self.values = dict()
      self.values.update(kwargs)
      return

    if not self.plot_only: return

    self.fig               = plt.figure()
    self.axes              = plt.gca()

    self.origin_shape      = (self.y_plots, self.x_plots, self.h, self.w)   if self.c == 1 else (self.y_plots, self.x_plots, self.h, self.w, self.c)
    self.new_shape         = (self.y_plots * self.h, self.x_plots * self.w) if self.c == 1 else (self.y_plots * self.h, self.x_plots * self.w, self.c)
    empty                  = np.zeros(self.new_shape)
    if self.c == 1: empty[0, 0]    = 1. # grayscale: it feels like shit
    else          : empty[0, 0, :] = 1. # color
    empty                  = self.add_grid(empty)
    self.ax_img            = self.axes.imshow(empty, cmap='gray')

    # 3d plot variables
    self.bar               = None
    self.cbar              = None
    self.ax                = None
    self.txt               = None

    self.axes.tick_params( # disable labels and ticks
      axis        = 'both',
      which       = 'both',
      bottom      = False ,
      top         = False ,
      left        = False ,
      right       = False ,
      labelbottom = False ,
      labelleft   = False ,
      )

    self.fig.canvas.set_window_title(f'Visualization {self.name}')

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    self.start_vis()


  def add_grid(self, data, val=1.0):
    if self.grid: # add grid lines
      for i in range(self.x_plots + 1):
        data = np.insert(data, i * self.w + i, val, axis=1)
        data = np.insert(data, i * self.h + i, val, axis=0)
    return data


  def visualize_pis(self, pis, **kwargs):
    if not self.enabled: return

    it         = kwargs.get('iteration')
    min_, max_ = np.min(pis), np.max(pis)

    if self.store_only_output:
      self.values[it] = pis
      return
    if not self.plot_only: return

    np.save(f'./plots/pis_{it}.npy', pis) # store numpy files for recreation (e.g., rotate image)

    if self.bar: # clear figure after each update
      self.bar.remove()
      self.ax.remove()
      self.cbar.remove()
    else:
      self.fig.clf()
    self.ax = self.fig.add_subplot(111, projection='3d')

    # create bar values (sidewalls)
    _x        = np.arange(self.w)
    _y        = np.arange(self.h)
    _xx, _yy  = np.meshgrid(_x, _y)
    x  , y    = _xx.ravel(), _yy.ravel()
    depth     = 1
    width     = 1

    # create bar values (bottom and top)
    z         = interp1d([min_, max_], [0.0, 1.])(pis)
    bottom    = np.zeros_like(z)

    # define colors
    newcmp    = matplotlib.cm.get_cmap('Blues', 256)
    newcmp    = ListedColormap(newcmp(np.linspace(0.4, .9, 256))) # slice 'Blues' color map
    norm      = matplotlib.colors.Normalize(min_, max_)           # normalize color function
    colors    = newcmp(norm(z))                                   # apply color normalizer

    # create 3d plot
    self.bar  = self.ax.bar3d(x, y, bottom, width, depth, z, color=colors, shade=True, antialiased=True)

    self.ax.set_title('pis')

    # plot view orientation
    self.ax.view_init(25, 135)
    plt.subplots_adjust(top=0.5) # vertical compressing

    # define z axis limits
    self.ax.set_zlim3d(0, 1)
    z_scale   = np.linspace(0, 1, 5) # 5 ticks
    self.ax.set_zticks(z_scale)

    # disable x and y axis labels
    self.ax.set_yticklabels([])
    self.ax.set_xticklabels([])

    # define z axis lables
    z_label_vals = np.linspace(min_, max_, 5)
    z_label      = [ f'{float(val):.3}' for val in z_label_vals ] # only three decimal digits
    self.ax.set_zticklabels(z_label)

    # create colorbar
    m         = matplotlib.cm.ScalarMappable(cmap=newcmp)
    self.cbar = plt.colorbar(m, shrink=.7, ticks=z_scale, pad=0.1)
    self.cbar.ax.set_yticklabels(z_label)

    for rotation in range(0, 360, 45):
      self.ax.view_init(25, rotation)
      self.fig.savefig(f'./plots/pis_{it}_rot_{rotation}.pdf', format='pdf', bbox_inches='tight')
      self.fig.savefig(f'./plots/pis_{it}_rot_{rotation}.png', format='png', bbox_inches='tight')


  def vis_conv_mask(self, data, norm=True, **kwargs):
    if not self.enabled: return
    iteration = kwargs.get('iteration')

    if self.store_only_output:
      self.values[iteration] = data
      return
    if not self.plot_only: return

    if norm: heat = interp1d([np.min(data), np.max(data)], [0., 1.])(data)
    else   : heat = data

    heat = heat.reshape([5, 5, 5, 5])
    heat = heat.swapaxes(1, 2)
    heat = heat.reshape([25, 25])

    x_w, y_w       = (self.ax_img.get_array().shape[0:2])
    heat           = cv2.resize(heat, (x_w, y_w), interpolation=cv2.INTER_CUBIC) # cv2.INTER_NEAREST
    heat           = interp1d([np.min(heat), np.max(heat)], [0., 1.])(heat)

    if not hasattr(self, 'heat'):
      self.heat = self.axes.imshow(
        heat                     ,
        interpolation = 'bicubic',
        #cmap          = 'jet'    ,
        cmap          = 'gray'    ,
        alpha         = 1       ,
        )
      self.heat.set_zorder(-10)
      image                    = np.ones([140, 140, 4])
      image                    = self.add_grid(image, 0.0)
      image                    = 1 - image
      image[image[:,:,0] == 1] = 0, 0, 0, 1 # black frame
      self.ax_img.set_data(image)
      self.ax_img.set_alpha(1)

    self.heat.set_data(1 - heat)

    self.fig.canvas.flush_events()
    self.fig.canvas.draw_idle()
    self.fig.savefig(f'{self.name}_{iteration}.pdf')


  def _labels(self, labels):
    if labels is None: return
    labels       = labels.reshape(self.y_plots, self.x_plots)
    labels       = labels.T
    if self.txt:
      for txt in self.txt:
        txt.remove()
    self.txt = list()

    it = np.nditer(labels, flags=['multi_index'])
    while not it.finished:
      x, y = it.multi_index
      x = x * 28 + (x + 1)
      y = y * 28 + (y + 1) + 13
      self.txt += [self.axes.text(x, y, it[0], fontsize=15,  c='red')]
      it.iternext()


  def _heatmap(self, heat):
    if heat is None: return

    if not hasattr(self, 'chxbox'):
      self.activated = True
      ax             = self.fig.add_axes([.77, .82, 0.3, .2], frame_on=False)
      self.chxbox    = CheckButtons(ax, ['heatmap'], [self.activated])

      def callback(_):
        self.activated = not self.activated
      self.chxbox.on_clicked(callback)

    if not self.activated:
      if hasattr(self, 'heat'):
        self.heat.remove()
        delattr(self, 'heat')
      return

    min_, max_     = np.min(heat), np.max(heat)
    heat           = heat.reshape(self.x_plots, self.y_plots).T
    x_w, y_w       = (self.ax_img.get_array().shape)
    heat           = cv2.resize(heat, (x_w, y_w), interpolation=cv2.INTER_CUBIC) # cv2.INTER_NEAREST
    heat           = interp1d([np.min(heat), np.max(heat)], [0., 1.])(heat)

    if not hasattr(self, 'heat'):
      self.heat = self.axes.imshow(
        heat                     ,
        interpolation = 'bicubic',
        cmap          = 'jet'    ,
        alpha         = .5       ,
        )

    if not self.cbar:
      self.cbar = self.fig.colorbar(
        self.heat              ,
        ax        = self.axes ,
        drawedges = False      ,
        format    = '%.3f'     ,
        ticks     = [0, 0.5, 1],
        label     = 'pis'      ,
        )

    self.heat.set_data(heat) # update the heat map
    self.cbar.draw_all()
    self.cbar.set_alpha(1)   # avoid lines caused by transparency
    cbar_ticks     = [ float(f'{x:.4f}') for x in np.linspace(min_, max_, num=3, endpoint=True ) ]
    self.cbar.ax.set_yticklabels(cbar_ticks)


  def visualize(self, data, norm=True, labels=None, heatmap=None, **kwargs):
    if not self.enabled: return
    iteration = kwargs.get('iteration')

    if self.store_only_output:
      self.values[iteration] = data
      return
    if not self.plot_only: return
    # np.save(f'./plots/{self.name}_{iteration}.npy', data) # store numpy files for recreation (e.g., rotate image)
    if norm: images = interp1d([np.min(data), np.max(data)], [0., 1.])(data)
    else   : images = data

    # combine all images to one
    image = images.reshape(*self.origin_shape)
    image = image.swapaxes(1, 2)
    image = image.reshape(*self.new_shape)
    image = self.add_grid(image)
    image = 1 - image

    self.ax_img.set_array(image)
    self._labels(labels)
    # self._heatmap(heatmap)
    self.fig.canvas.flush_events()
    self.fig.canvas.draw_idle()

    self.fig.savefig(f'{self.name}_{iteration}.pdf')
    # self.fig.savefig(f'../plots/{self.name}_{iteration}.png')
    # self.fig.savefig('./plots/{name}_{iteration}.png'.format(name=self.name, iteration=iteration))

  def visualize_sample(self, sample, label=None):
    ''' visualize a single sample '''
    if not self.enabled: return
    image = sample.squeeze()
    image = image.reshape((self.h, self.w))
    self.ax_img.set_array(image)
    if label:
      if self.txt: self.txt.remove()
      self.txt = self.axes.text(-2, -2, label, fontsize=30,  c='red')
    self.fig.canvas.flush_events()
    self.fig.canvas.draw_idle()


  def start_vis(self):
    if not self.enabled: return
    if self.plot_only:
      plt.ion()
      plt.draw()
      plt.show()


  def stop_vis(self):
    if not self.enabled: return
    plt.ioff()
    plt.draw()
    plt.show()

    if self.store_only_output:
      with open(f'./plots/all_{self.name}.pkl.gz', 'wb') as file:
        pickle.dump(self.values, file, protocol=pickle.HIGHEST_PROTOCOL)

  @classmethod
  def stop_all_vis(cls):
    for visualizer in cls.visualizer.values():
      visualizer.stop_vis()

