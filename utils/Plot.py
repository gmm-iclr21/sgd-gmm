import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from Visualization import Visualization as Vis
import pickle
import cv2
import os

plt.rcParams['text.usetex'] = True
class Plot():

  def plot_log_likelihood(self):
    log_likelihood_path      = 'log-likelihood.csv'
    real_log_likelihood_path = 'real_log-likelihood.csv'

    log_likelihood_file      = open(log_likelihood_path, 'r')
    real_log_likelihood_file = open(real_log_likelihood_path, 'r')

    log_likelihood           = np.array([ [float(val) for val in row] for row in csv.reader(log_likelihood_file) if row ])
    real_log_likelihood      = np.array([ [float(val) for val in row] for row in csv.reader(real_log_likelihood_file) if row ])

    iteration = log_likelihood[:, 0]
    #epochs = np.where(iteration == 0)[0][1]
    iteration[50:] = iteration[50:] + 5000
    #print(epochs)
    val  = log_likelihood[:, 1]
    val_  = real_log_likelihood[:, 1]

    plt.rc('font', family='serif')


    plt.figure(figsize=(6, 4))
    axis = plt.gca()
    axis.plot(iteration, val, '-',               # marker circle, with line
                  linewidth=2,
                  color='green',
                  label='SOM log-likelihood',
                  )

    axis.plot(iteration, val_, '-',               # marker circle, with line
                  linewidth=2,
                  color='blue',
                  label='real log-likelihood',
                  )

    # print gray background for retraining
    x = np.arange(5000, 2 * 5000 + 1, 1)
    min_ = min(min(val), min(val_))
    max_ = max(max(val), max(val_))
    axis.fill_between(x, min_, max_, where=(x > (x.shape[0] / 2)), facecolor='gray', alpha=0.2)

    axis.set_xlabel(r'iteration')
    axis.set_ylabel(r'log-likelihood')

    # axis.tick_params(labelsize=20)

    axis.legend(
      #fontsize=26,
      framealpha=1)
    axis.grid(True, which='both')
    #plt.tight_layout()
    plt.savefig('{}.pdf'.format('log-likelihood'))
    plt.show()

  def _plot_stored_data(self, file_name, vis_name):
    with open(f'../plots/{file_name}.pkl.gz', 'rb') as file   : data    = pickle.load(file)

    plot_only = False # True = visualize, False = only plot

    data_prop = { k:v for k, v in data.items() if type(k) != int }   # keep only parameters
    data      = { k:v for k, v in data.items() if type(k) == int }
    data_prop.update({'plot_only': plot_only, 'store_only_output': False})

    vis       = Vis(vis_name, **data_prop)
    for it, data in data.items():
      print('plot', it)
      vis.visualize(data, iteration=it)
    vis.stop_vis()

  def load_images(self, start, end, image_folder='../plots/'):
    images = [ img for img in os.listdir(image_folder) if img.startswith(start) and img.endswith(end) ]
    images = sorted(images, key=lambda x: float(x.split('_')[2].split('.')[0]))

    print('found', len(images), 'images with {}*{}'.format(start, end))
    return images

  def plot_stored_data(self):
    self._plot_stored_data('all_mus_0', 'mus_0')
    #self._plot_stored_data('all_mus_1', 'mus_1')
    #self._plot_stored_data('all_mus_2', 'mus_2')
    #self._plot_stored_data('all_mus_3', 'mus_3')

    # self._plot_stored_data('all_pis', 'pis_3')
    # self._plot_stored_data('all_sigmas', 'sigmas_3')


  def create_video(self, images, video_name):
    videos_folder = './videos/'
    if not os.path.exists(videos_folder): os.makedirs(videos_folder)

    frame = cv2.imread(os.path.join('../plots/', images[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(os.path.join(videos_folder, video_name), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

    for image in images:
      img = cv2.imread(os.path.join('../plots/', image))
      video.write(img)

    cv2.destroyAllWindows()
    video.release()

  def create_vid(self):
    self.create_video(self.load_images('mus_0', '.png'), 'mus_0.avi')
    self.create_video(self.load_images('0_D1-1-1-1-1-1-1-1-1-1_', '.png'), '0_D1-1-1-1-1-1-1-1-1-1.avi')
    self.combine_vids_vert('./videos/mus_0.avi', './videos/0_D1-1-1-1-1-1-1-1-1-1.avi', './videos/out.avi')
    self.add_audio('./videos/out.avi', './videos/lied2.mp3', output='./videos/out_m.avi')

  def create_videos(self):
    #self.create_video(self.load_images('mus_0'   , '.png')       , 'mus_0.avi')
    #self.create_video(self.load_images('mus_1'   , '.png')       , 'mus_1.avi')
    #self.create_video(self.load_images('mus_2'   , '.png')       , 'mus_2.avi')
    #self.create_video(self.load_images('mus_3'   , '.png')       , 'mus_3.avi')
    #self.create_video(self.load_images('sigmas'  , '.png')       , 'sigmas.avi')
    #self.create_video(self.load_images('pis'     , 'rot_0.png')  , 'pis_0.avi')
    #self.create_video(self.load_images('pis'     , 'rot_45.png') , 'pis_45.avi')
    #self.create_video(self.load_images('pis'     , 'rot_90.png') , 'pis_90.avi')
    #self.create_video(self.load_images('pis'     , 'rot_135.png'), 'pis_135.avi')
    #self.create_video(self.load_images('pis'     , 'rot_180.png'), 'pis_180.avi')
    #self.create_video(self.load_images('pis'     , 'rot_225.png'), 'pis_225.avi')
    #self.create_video(self.load_images('pis'     , 'rot_270.png'), 'pis_270.avi')
    #self.create_video(self.load_images('pis'     , 'rot_315.png'), 'pis_315.avi')
    #self.create_video(self.load_images('0_D1_', '.png'), '0_D1.avi')
    #self.create_video(self.load_images('0_D1-1-1-1-1-1-1-1-1-1_', '.png'), '0_D1-1-1-1-1-1-1-1-1-1.avi')

    #self.combine_vids_vert('./videos/mus_0.avi', './videos/0_D1-1-1-1-1-1-1-1-1-1.avi', './videos/out.avi')
    # self.combine_vids_hori('./videos/mus_2.avi', './videos/mus_3.avi','./videos/mus23.mp4')
    # self.combine_vids_vert('./videos/mus01.mp4', './videos/mus23.mp4','./videos/mus1234.mp4')
    #self.xstack('./videos/mus_0.avi', './videos/mus_1.avi', './videos/mus_2.avi', './videos/mus_3.avi', output='./videos/mus1234.mp4')
    self.add_audio('./videos/out.avi', './videos/lied2.mp3', output='./videos/out_m.avi')

  def combine_vids_vert(self, input1, input2, output):
    #ffmpeg_command = f'ffmpeg -i {input1} -i {input2}  -filter_complex vstack=inputs=2 -r 30 {output}'
    ffmpeg_command = f'ffmpeg -i {input1} -i {input2} -filter_complex "[0][1]scale2ref=iw:ow/mdar[2nd][ref];[ref][2nd]vstack[vid]" -map [vid] -c:v libx264 -crf 23 -preset veryfast {output}'

    os.system(ffmpeg_command)


  def combine_vids_hori(self, input1, input2, output):
    ffmpeg_command = f'ffmpeg -i {input1} -i {input2} -filter_complex -filter_complex hstack=inputs=2 -r 30 {output}'
    # ffmpeg_command = 'ffmpeg -i {} -i {} -filter_complex "[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]" -map [vid] -c:v libx264 -crf 23 -preset veryfast {}'.format(input1, input2, output)
    os.system(ffmpeg_command)

  def xstack(self, input0, input1, input2, input3, output):
    ''' https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-several-videos-using-ffmpeg '''
    ffmpeg_command = f'ffmpeg -i {input0} -i {input1} -i {input2} -i {input3} -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -r 60 {output}'
    os.system(ffmpeg_command)

  def add_audio(self, video_file, audio_file, output):
    ffmpeg_command = f'ffmpeg -i {video_file} -i {audio_file} -codec copy -shortest {output}'
    os.system(ffmpeg_command)

if __name__ == '__main__':
  #Plot().plot_log_likelihood()
  # Plot().plot_stored_data()
  Plot().create_vid()
  pass
