"""
visualizes prototypes saved by main.py, plus the associated class
Three modes:
- no params, uses pis.npy, mus.npy, sigmas.npy etc
- 1 param: <iteration>, uses <iteration>pis.npy, <iteration>mus.npy, ...
- 6 params: <iteration> pifule mufile etc...
"""
import numpy as np
import matplotlib as mp
from matplotlib import cm

# so as to not try to invoke X on systems without it
mp.use("Agg")

import matplotlib.pyplot as plt, sys, math
from argparse import ArgumentParser

if __name__ == "__main__":
  plt.rc('text', usetex=False)
  plt.rc('font', family='serif')

  parser = ArgumentParser()
  parser.add_argument("--channels", required=False, default=1, type=int, help = "If u r visualizing centroids that come from color images (SVHN, Fruits), please specify 3 here!")
  parser.add_argument("--what", required=False, default="mus", choices=["mus","precs_diag"], type=str, help="Visualize centroids or precisions°")
  parser.add_argument("--vis_pis", required=False, default=True, type=eval, help="True or False depending on whether you want the weights drawn on each component")
  FLAGS = parser.parse_args()

  pifile = "pis.npy"
  mufile = "mus.npy"
  sigmafile = "sigmas.npy"

  channels = FLAGS.channels
  it = "" ;

  pis = np.load(it + pifile)[0, 0, 0]
  protos = np.load(it + mufile)
  sigmas = np.load(it + sigmafile)[0, 0, 0]
  protos = protos[0, 0, 0]

  n = int(math.sqrt(protos.shape[0]))
  d_ = int(math.sqrt(protos.shape[1] / channels))

  f, axes = plt.subplots(n, n)
  axes = axes.ravel()
  index = -1

  for (dir_, ax_, pi_, sig_) in zip(protos, axes, pis, sigmas):
    index += 1

    disp = dir_
    if FLAGS.what == "precs_diag":
      disp = sig_

    refmin = disp.min() ;
    refmax = disp.max() ;

    # This is interesting to see unconverged components
    print("minmax=", disp.min(), disp.max(), pi_)

    ax_.imshow(disp.reshape(d_, d_, channels) if channels == 3 else disp.reshape(d_, d_), vmin=refmin, vmax=refmax, cmap=cm.bone)

    if FLAGS.vis_pis == True:
      ax_.text(-5, 1, "%.03f" % (pi_), fontsize=18, c="black", bbox=dict(boxstyle="round", fc=(1, 1, 1), ec=(.5, 0.5, 0.5)))

    ax_.set_xticklabels([])
    ax_.set_yticklabels([])
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

  plt.savefig("mus.png")
  plt.tight_layout(pad=1, h_pad=.0, w_pad=-10)
  plt.savefig("mus.pdf")

