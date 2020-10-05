'''
SGD approach for GMMs. NIPS Submission 2020
'''

import os
import sys
import math
import time
import itertools
import utils.GPU_Test # import check if GPU is available
import numpy                       as np
import tensorflow                  as tf
import json                        as jason # who the fuck is Jason?
from enum                          import Enum, auto
from experimentdataset             import Dataset_Wrapper, Dataset_Type as DT
from experimentparser              import Parser
from utils.Metric                  import Metrics, Metric
from layer.GMM_Layer               import GMM_Layer
from layer.Folding_Layer           import Folding_Layer
from layer.Linear_Classifier_Layer import Linear_Classifier_Layer
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet as TF_DataSet

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

# enum abbreviations
En = GMM_Layer.Energies
MO = GMM_Layer.DiagMode
Me = Metrics

DEBUG = True # TODO: add logging package?

class Dataset_Type(Enum):
  TRAIN     = auto()
  TEST      = auto()
  GENERATED = auto()
  D_ALL     = auto()
  D_NOW     = auto()

class GMM(object):
  ''' GMM base class '''

  def __new__(cls, *args, **kwargs):
    if kwargs.get('parser_only', False): # only create parser for evaluation purposes
      parser = Parser()
      GMM._add_arguments(parser)
      return parser
    return super(GMM, cls).__new__(cls)

  def __init__(self, *args, **kwargs):
    ''' construct a fully configured GMM object '''
    self.parser = Parser(description='Gradient-Based Training of Gaussian Mixture Models in High-Dimensional Spaces')
    self.FLAGS  = GMM._add_arguments(self.parser)
    for k, v in vars(self.FLAGS).items(): self.__setattr__(k, v)   # offer command line parameters as instance variables
    self._init_dataset()
    self._init_variables()
    self._init_log_writer()
    self._init_tf()
    self._init_tf_variables()


  @staticmethod
  def _add_arguments(parser):
    parser.add_argument('--exp_id'                     , type=str  , default='0'                                          , help='unique experiment id (for experiment evaluation)')
    parser.add_argument('--tmp_dir'                    , type=str  , default='.'                                          , help='directory for output files (for experiment evaluation)')
    #-------------------------------------------------------------- SAVE/RESTORE
    parser.add_argument('--ckpt_dir'                   , type=str  , default='./checkpoints/'                             , help='directory for output files (for experiment evaluation)')
    # ------------------------------------------------------------------ DATASET
    parser.add_argument('--dataset_file'               , type=str  , default='MNIST'                          , help='load a compressed pickle file. If not present, a download attempt is made. This may take a while for large datasets such as SVHN')
    parser.add_argument('--slice'                      , type=int  , default=[-1, -1]         , nargs=2                   , help='replace all images in dataset by a NxM-patch cropped from the center of each image. if non negative, the two value represent N and M, otherwise the full image is used')
    parser.add_argument('--noise'                      , type=float, default=0.00                                         , help='add noise from normal (Gaussian) distribution [0, noise] to data once, before training')
    #------------------------------------------------------------------ TRAINING
    parser.add_argument('--epochs'                     , type=float, default=1.                                           , help='number of training epochs per taks')
    parser.add_argument('--batch_size'                 , type=int  , default=1                                            , help='size of mini-batches we feed from train dataSet.')
    #------------------------------------------------------------------- TESTING
    parser.add_argument('--test_batch_size'            , type=int  , default=100                                          , help='batch size for testing')
    parser.add_argument('--measuring_points'           , type=int  , default=10                                           , help='measure X points PER EPOCH during training loss on test set')
    parser.add_argument('--metrics'                    , type=str  , default='accuracy_score', nargs='+', choices=list(Me), help='select one or multiple measurement metrics')
    #------------------------------------------------------------- VISUALIZATION
    parser.add_argument('--vis_points'                 , type=int  , default=0                                            , help='X visualization points. X=0 --> off, X<0 --> generate plot files every |X| iterations, X> 0 --> display every X iterations. All generated plots are stored in subdirectory ./plots')
    parser.add_argument('--store_only_output'          , type=bool , default=False                                        , help='if True, no visualizations are created')
    #--------------------------------------------------------- SLT CONFIGURATION
    parser.add_argument('--task_epochs'               , type=float, nargs='*', default=[1.5, 5, 5]                       , help='number of training epochs for eqch task (ignores --epochs parameter)')
    parser.add_argument('--D1'                         , type=int  , default=[0,1, 2, 3, 4, 5, 6, 7, 8,9], nargs='*'          , help='classes for the specific task, D1, D2, ..., Dx, e.g. "--D1 0 1 2 3 4 5 6 7 8 9"')

    # create a 3 layer network (folding, gmm, linear classifier)
    parser.add_argument('--L1'                         , type=str  , default='folding'                                    , help='create a folding layer') # following parameters are possible
    parser.add_argument('--L1_name'                    , type=str  , default='folding_layer'                              , help='create a variable with reference to the tensor')
    #!!! The following parameters are possible
    #parser.add_argument('--L1_patch_height'           , type=int  , default=14                                           , help='set patch_Y value')
    #parser.add_argument('--L1_patch_width'            , type=int  , default=14                                           , help='set patch_X value')
    #parser.add_argument('--L1_stride_y'               , type=int  , default=14                                           , help='set stride_Y value')
    #parser.add_argument('--L1_stride_x'               , type=int  , default=14                                           , help='set stride_X value')

    parser.add_argument('--L2'                         , type=str  , default='gmm'                                        , help='create a gmm layer')
    parser.add_argument('--L2_name'                    , type=str  , default='gmm_layer'                                  , help='create a variable with reference to the tensor, i.e., self.gmm_layer')
    parser.add_argument('--L2_K'                       , type=int  , default=8 ** 2                                       , help='set number of prototypes')
    parser.add_argument('--L2_output_tensor'           , type=str  , default='norm_resp'                                  , help='set the output tensor of the layer')

    # The following parameters can be specified on the cmd line
    #parser.add_argument('--L2_mode'                   , type=str  , default='diag'                                       , help='~') # DiagMode.Diag
    #parser.add_argument('--L2_muInit'                  , type=float, default=0.1                                          , help='~')
    #parser.add_argument('--L2_sigmaUpperBound'         , type=float, default=20.                                          , help='~')
    #parser.add_argument('--L2_eps0'                    , type=float, default=0.1                                          , help='~')
    #parser.add_argument('--L2_epsInf'                  , type=float, default=0.001                                        , help='~')
    #parser.add_argument('--L2_energy'                 , type=str  , default='MC'                                         , help='~') # Energies.MC
    #parser.add_argument('--L2_lr_reduction'           , type=float, default=0.05                                         , help='~')
    #parser.add_argument('--L2_regularizer'            , type=str  , default='time_decay'                                 , help='~') # RM.DOUBLE_EXP
    #parser.add_argument('--L2_regularizer_limit'      , type=float, default=0.02                                         , help='~')
    #parser.add_argument('--L2_regularizer_alpha'       , type=float, default=0.001                                        , help='~')
    #parser.add_argument('--L2_regularizer_delta'       , type=float, default=0.05                                         , help='~')
    #parser.add_argument('--L2_regularizer_gamma'       , type=float, default=0.92                                         , help='~')
    #parser.add_argument('--L2_regularizer_t0Frac'     , type=float, default=0.3                                          , help='~')
    #parser.add_argument('--L2_regularizer_tInfFrac'   , type=float, default=0.8                                          , help='~')
    #parser.add_argument('--L2_regularizer_reset_sigma' , type=float, default=0.01                                         , help='reset sigma to default value (after each task) if no task specific reset value is given, if not set, somSigmaInf value is used to rest')
    #parser.add_argument('--L2_regularizer_reset_eps'   , type=float, default=0.001                                        , help='reset eps to default value (after each task) if no task specific reset value is given  , if not set, epsInf value is used to rest')

    FLAGS = parser.parse_all_args()
    return FLAGS


  def _init_dataset(self):
    ''' load dataset '''
    self.dataset           = Dataset_Wrapper(self.FLAGS)
    self.generated_dataset = None                                            # variable for the generated dataset object
    self.properties        = self.dataset.get_properties()
    self.w, self.h         = self.properties.get('dimensions')               # extract image properties or use slice patch size
    self.c                 = self.properties.get('num_of_channels')
    self.num_classes       = self.properties.get('num_classes')
    self.max_iterations    = int(self.properties.get('train_shape')[0] * self.epochs // self.batch_size) # needed iterations for the full dataset
    _, self.test_D_ALL     = self.dataset.get_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    _, self.iter_D_ALL     = self.dataset.get_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


  def _init_variables(self):
    ''' initialize variables '''
    self.D                = self.w * self.h * self.c             # data dimensionality (width x height x channels)
    #---------------------------------------------------------------- EVALUATION
    # self.metric           = Metric(self)
    #-------------------------------------------------------- TASK/DATASET LISTS
    self.num_tasks        = len(list(filter(lambda x: x.startswith('D'), vars(self.FLAGS))))
    self.tasks            = list() # stores classes for each task
    self.tasks_iterations = list() # stores number of (training, testing) iterations list(tuple)
    self.training_sets    = list() # stores task training datasets
    self.test_sets        = list() # stores task test datasets


  def _init_log_writer(self):
    ''' write all into one json string -.- (I hate JSON)'''
    self.log_path_name = f'{self.tmp_dir}/{self.exp_id}_gmm.json'
    self.log           = dict(
      parameters = dict(self.FLAGS.__dict__),
      ll         = list()                   ,
      created    = time.asctime()           ,
      )


  def _init_tf(self):
    ''' create TensorFlow session '''
    config                          = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement     = False
    self.sess                       = tf.InteractiveSession(config=config) # start an interactive session


  def _init_tf_variables(self):
    ''' define TensorFlow variables and placeholders '''
    self.X = tf.compat.v1.placeholder(name='X', shape=[None, self.h, self.w, self.c], dtype=tf.float64)  # placeholder for a batch of data
    self.Y = tf.compat.v1.placeholder(name='Y', shape=[None, self.num_classes]      , dtype=tf.float64)  # placeholder for labels


  def _load_task_dataset(self):
    ''' create sub-dataset and append to training and test lists '''
    task = getattr(self, f'D{self.task + 1}') # list of classes from command line parameter
    if isinstance(task, int): task = [task]   # turn single tasks (int) into a list (list(int))

    training     , testing        = self.dataset.get_dataset(task)
    training_iter, testing_iter   = self.dataset.get_iter(task)

    self.tasks                   += [task]
    self.tasks_iterations        += [(training_iter, testing_iter)]
    self.training_sets           += [training]
    self.test_sets               += [testing]

    if hasattr(self, 'test_now_tasks'):                                                    # load all-task-dataset only after the first task run (nasty)
      now_tasks                   = [ item for sublist in self.tasks for item in sublist ] # flatten
      self.now_tasks              = list(set(now_tasks))                                   # unique task list

      _, self.test_now_tasks      = self.dataset.get_dataset(self.now_tasks)
      _, self.test_now_tasks_iter = self.dataset.get_iter(self.now_tasks)
    else:
      self.test_now_tasks         = None # first definition, next time a combined test dataset based on all tasks is created


  def _build_graph(self):
    ''' build TensorFlow computation graph '''
    input_      = self.X
    self.layers = list()
    for i in itertools.count(start=1): # start layer loop
      layer_prefix = f'L{i}'
      param_prefix = f'L{i}_'
      if not hasattr(self, layer_prefix): break # stop layer loop if no more layer are specified by command line

      layer_type                     = self.__getattribute__(layer_prefix)                                                      # get layer type
      layer_params                   = { k[len(param_prefix):]: v for k,v in vars(self).items() if k.startswith(param_prefix) } # extract layer specific parameters
      layer_var_name                 = layer_params.get('name', 'layer')                                                        # get layer variable name (usage: self.<name>)
      layer_params['ys']             = self.Y                                                                                   # make label available as layer parameter
      layer_params['max_iterations'] = self.max_iterations                                                                      # set the maximum iterations (for the full dataset)

      if   layer_type == 'folding' : input_ =           Folding_Layer(input_, **layer_params)
      elif layer_type == 'gmm'     : input_ =               GMM_Layer(input_, **layer_params)
      elif layer_type == 'classify': input_ = Linear_Classifier_Layer(input_, **layer_params)
      else                         : Exception('invalid layer type')

      self.__setattr__(layer_var_name, input_) # create the variable (self.<name>)
      self.layers += [input_]                  # add layer to layers list (used for training all layers)

    if DEBUG:
      for layer in self.layers: print(layer)


  def _feed_dict(self, dataset=Dataset_Type.TRAIN, task=0, batch_size=None):
    ''' feed dictionary function for train and test datasets

    @param dataset  : select the dataset (Dataset_Type)
      TRAIN    : training dataset (current task) (default batch_size=1)  ,
      TEST     : testing dataset (current task)  (default batch_size=100),
      GENERATED: generated dataset (current GMM state) (default batch_size=1),
      D_ALL    : merged task dataset (all classes from the dataset) (default batch_size=100)
      D_NOW    : merged task dataset (all classes from all seen tasks, including the current) (default batch_size=100)
    @param task      : the current task id (start at 0)
    @param batch_size: use a specific batch size
    @return: feed dictionary (dict())
    '''
    if dataset == Dataset_Type.TRAIN:                                                              # load test data
      xs, ys = self.training_sets[task].next_batch(self.batch_size if not batch_size else batch_size)     # load train batch
      xs     = xs if self.noise == 0.0 else xs + np.random.normal(0, self.noise, size=xs.shape)           # add noise
    elif dataset == Dataset_Type.TEST:                                                             # load test data
      xs, ys = self.test_sets[task].next_batch(self.test_batch_size if not batch_size else batch_size)    # load test batch
    elif dataset == Dataset_Type.GENERATED:                                                        # load data from self generated dataset
      xs, ys = self.generated_dataset.next_batch(self.batch_size if not batch_size else batch_size)       # load generated batch
    elif dataset == Dataset_Type.D_NOW:                                                            # a combination of all tasks
      xs, ys = self.test_now_tasks.next_batch(self.test_batch_size if not batch_size else batch_size)     # load test batch
    elif dataset == Dataset_Type.D_ALL:                                                            # a combination of all seen tasks until now
      xs, ys = self.test_D_ALL.next_batch(self.test_batch_size if not batch_size else batch_size)         # load test batch
    else: raise Exception(f'invalid dataset type')

    xs = xs.reshape(-1, self.h, self.w, self.c)                                                    # reshape for convolution
    return {self.X: xs, self.Y: ys}


  def _generate_sample_dataset(self):
    ''' generate a dataset from gmm layer
    1. create samples from gmm
    2. use inference stage to classify the samples
    '''
    if self.task == 0: return # generate no dataset for first task (D1)

    num_samples            = math.ceil(self.training_sets[self.task].images.shape[0] * self.epochs)
    print(f'generate {num_samples} from gmm layer and labeling with the linear classifier layer')
    samples                = self.gmm_layer.generate_samples(num_samples)
    dummy_labels           = np.zeros([num_samples, self.num_classes])
    self.generated_dataset = TF_DataSet(255. * samples, dummy_labels, reshape=False)

    samples                = list()
    labels                 = list()

    iterations             = math.ceil(num_samples / self.test_batch_size)
    for _ in range(iterations):
      _feed_dict_  = self._feed_dict(Dataset_Type.GENERATED, batch_size=self.test_batch_size)
      samples     += [_feed_dict_.get(self.X)]
      labels      += [self.linear_layer._classify_batch(_feed_dict_)]

    samples                = np.concatenate(samples, axis=0)[:num_samples]
    labels                 = np.concatenate(labels , axis=0)[:num_samples]
    print(f'generated {num_samples} samples; distribution: {np.sum(labels, axis=0)}')
    self.generated_dataset = TF_DataSet(255 * samples, labels, reshape=False)

    if False:
      np.save('samples.npy', self.generated_dataset.images)
      np.save('labels.npy' , self.generated_dataset.labels)


  def _train_step(self, dataset=Dataset_Type.TRAIN):
    ''' training step '''
    _feed_dict_ = self._feed_dict(task=self.task, dataset=dataset)

    for layer in self.layers:
      layer.train_step(_feed_dict_)


  def _test_one_epoch(self, task_=-1, dataset_type=Dataset_Type.TEST):
    ''' calculate the log-likelihood and the classification accuracy

    @param task_   : the id of the current task (int)
    @param data_str: selects the dataset to be tested
    '''
    test_iterations_dict = {
      Dataset_Type.TEST  : self.tasks_iterations[task_][1]           ,
      Dataset_Type.TRAIN : self.tasks_iterations[task_][0]           ,
      Dataset_Type.D_NOW : getattr(self, 'test_now_tasks_iter', None),
      Dataset_Type.D_ALL : self.iter_D_ALL                           ,
      }
    test_iterations      = test_iterations_dict.get(dataset_type)
    if not test_iterations: return

    loglikelihood        = 0.0
    #y_pred               = list()
    #y_true               = list()

    for _ in range(test_iterations):
      _feed_dict_       = self._feed_dict(dataset_type, task=task_, batch_size=self.test_batch_size)

      loglikelihood    += self.gmm_layer.test_step(_feed_dict_)
      #_y_true, _y_pred  = self.linear_layer.test_step(_feed_dict_)
      #y_true           += [_y_true]
      #y_pred           += [_y_pred]
    loglikelihood       /= test_iterations

    #metric_values        = self.metric.eval(
    #  dict    = True                                     , # return a dictionary with metric values
    #  y_true  = np.concatenate(y_true)                   ,
    #  y_pred  = np.concatenate(y_pred)                   ,
    #  special = {Me.ACCURACY_SCORE: dict(normalize=True)}, # special parameter for different metrics
    #  )

    #accuracy             = metric_values.get('accuracy_score')
    accuracy = -1

    if   dataset_type == Dataset_Type.D_NOW:
      task_id              = f'D_NOW_TEST'
      classes              = ','.join(map(str, self.now_tasks))
      task_classes         = f'(classes: {classes})'
      task_                = -1 # D_NOW = 0
    elif dataset_type == Dataset_Type.D_ALL:
      task_id              = f'D_ALL_TEST'
      classes              = ','.join(map(str, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
      task_classes         = f'(classes: {classes})'
      task_                = -2 # D_ALL = -1
    else :
      task_id = f'D{task_ + 1}_{dataset_type.name}   '
      classes              = ','.join(map(str, self.tasks[task_]))
      task_classes         = f'(classes: {classes})'
      # D1 = 1, ...

    test_type_value      = f'{dataset_type.name:<10} log-likelihood: {loglikelihood:10.4f}'
    classification_acc   = f'(accuracy: {accuracy:8.3%})'
    test_step            = f'{"":>19}' if self.task != task_ else f'at step {self.iteration_task:>5} / {self.iterations_task_all - 1:>5}'
    progress             = f'({self.iteration_task / (self.iterations_task_all - 1):6.1%})' if self.task == task_ else ''
    print(f'{task_id:<5} {task_classes:<30} {test_type_value} {classification_acc} {test_step} {progress:<6}')

    if DEBUG or True:
      for layer in self.layers: layer.info()
    values = (task_ + 1, self.iteration_task, self.iteration_glob + self.iteration_task, loglikelihood) # , metric_values
    self.log['ll'].append(values)


  def _test(self):
    ''' test step (recording the max-log-likelihood and real-log-likelihood)
      1 epoch on: current task (D_x)
      1 epoch on: all single previous tasks (D_1, ..., D_x-1)
      1 epoch on: a combination/merge of all pervious tasks unitl now (D_1 u D_1 u ... u D_x = D_Now)
      1 epoch on: a combination/merge of all tasks (D_1 u D_1 u ... u D_x = D_ALL)
    '''
    if self.iteration_task != self.test_at_iteration[0]: return
    self.test_at_iteration.pop(0)

    for task_ in range(self.task + 1):
      self._test_one_epoch(task_, Dataset_Type.TEST) # test (testing dataset) the current and all previous tasks (D_1, D_2,..., D_x-1)

    if self.task > 0 and hasattr(self, f'D{self.task + 2}'): # start after the first task is finished and skip the last task (D_ALL is D_NOW for last task)
      self._test_one_epoch(dataset_type=Dataset_Type.D_NOW)  # testing on all merged (testing data) tasks until current (including)

    self._test_one_epoch(dataset_type=Dataset_Type.D_ALL)
    print('-' * 140)


  def _store_mus(self):
    data      = self.sess.run(self.mus)
    data      = data.reshape((-1, *data.shape[3:]))
    filename  = f'{self.exp_id}_{self.dataset_file}_{self.task}_last-mus_'
    filename += 'patch' if self.slice[0] > 0 else 'full'
    np.save(f'{filename}.npy', data)


  def _visualize(self):
    ''' call visualization steps '''
    if self.vis_points == 0: return
    if self.iteration_task % self.vis_all_n_batches != 0: return

    self.gmm_layer.visualize(self.iteration_glob + self.iteration_task)


  def _calc_test_points(self):
    ''' calculate the training and test iterations for test points (based on the number of measuring points) '''
    test_at_iteration      = np.array_split(np.arange(self.iterations_task_all), self.measuring_points - 1)
    test_at_iteration      = [ x[0] for x in test_at_iteration ]
    self.test_at_iteration = test_at_iteration if test_at_iteration[-1] == self.iterations_task_all - 1 else test_at_iteration + [self.iterations_task_all - 1]

    if 0 < self.start_task_iteration < self.test_at_iteration[-1]: # fix test_at_iteration list, if a checkpoint is loaded
      while self.start_task_iteration > self.test_at_iteration[0]: self.test_at_iteration.pop(0)

    test_all_n_batches      = self.iterations_task_all // self.measuring_points
    self.test_all_n_batches = test_all_n_batches if test_all_n_batches >= 1 else 1

    if self.vis_points != 0:
      vis_all_n_batches      = self.iterations_task_all // self.vis_points
      self.vis_all_n_batches = vis_all_n_batches if vis_all_n_batches >= 1 else 1


  def _save_model(self):
    ''' save the current model to disk based on the "save" command line parameter '''
    iteration = getattr(self, f'save_All', None) # save model for each task
    if iteration and not iteration < -1:
      iteration = self.iterations_task_all if iteration == -1 else iteration # if iteration is -1, checkpoint after last iteration
      if self.iteration_task == iteration - 1:
        print(f'create checkpoint for task D{self.task + 1} at iteration {iteration}/{self.iteration_task + 1}')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, f'gmm-{self.task + 1}-{iteration}.ckpt'))

    iteration = getattr(self, f'save_D{self.task + 1}', None) # save model for specific tasks
    if iteration and not iteration < -1:
      iteration = self.iterations_task_all if iteration == -1 else iteration # if iteration is -1, checkpoint after last iteration
      if self.iteration_task == iteration - 1:
        print(f'create checkpoint for task D{self.task + 1} at iteration {iteration}/{self.iteration_task + 1}')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, f'gmm-{self.task + 1}-{iteration}.ckpt'))


  def _load_model(self):
    if not hasattr(self, 'load'): self.load = [-1, -1] # do not load model
    if not os.path.isdir(self.ckpt_dir): os.mkdir(self.ckpt_dir)
    task, iteration = self.load

    def restore(filename):
      dir_filename = os.path.join(self.ckpt_dir, filename)
      if not os.path.isfile(dir_filename): raise FileNotFoundError(f'No checkpoint files found: {dir_filename}/gmm-{task}-{iteration}')
      print(f'load checkpoint file: {dir_filename}')
      saver        = tf.train.import_meta_graph(dir_filename)
      saver.restore(self.sess, tf.train.latest_checkpoint(self.ckpt_dir))

    try:
      if task == 0:                                                       # search for latest task/checkpoint
        files             = sorted(os.listdir(self.ckpt_dir))
        if len(files) == 0: raise FileNotFoundError(f'No checkpoint files found in: {self.ckpt_dir}')
        filename          = files[-1].rsplit('.', 1)[0]                   # remove ".meta" from filename
        task_, iteration_ = ( int(task_iter) for task_iter in filename.split('.')[0].split('-')[1:3] )
        if iteration == -1: restore(f'{filename}.meta')                   # use the latest checkpoint
        if iteration >=  0: restore(f'gmm-{task_}-{iteration}.ckpt.meta') # use the specified iteration checkpoint
      elif task > 0 and iteration == -1:                                  # search for latest iteration of a specified task
        files             = sorted([ file for file in os.listdir(self.ckpt_dir) if file != 'checkpoint' and int(file.split('-', 2)[1]) == task ])
        if len(files) == 0: raise FileNotFoundError(f'No checkpoint files found in {self.ckpt_dir} for task D{task}')
        filename          = files[-1].rsplit('.', 1)[0]                   # remove ".meta" from filename
        task_, iteration_ = ( int(task_iter) for task_iter in filename.split('.')[0].split('-')[1:3] )
        restore(f'{filename}.meta')                                       # use a specific checkpoint
      elif task > 0 and iteration >= 0:                                   # task and iteration specified
        task_, iteration_ = task, iteration
        restore(f'gmm-{task_}-{iteration_}.ckpt.meta')
      else: raise Exception('')                                           # load no checkpoint (default)

      self.start_task           = task_ - 1
      self.start_task_iteration = iteration_
    except Exception as ex:
      if len(str(ex)) > 0: print(f'restore error: {ex} (start from scratch)')
      self.start_task           = 0
      self.start_task_iteration = 0
      return

    for self.task in range(self.start_task): # initialize previous datasets and update global iteration counter if a checkpoint is loaded
      self._load_task_dataset()
      task_iterations_     = int(math.ceil(self.tasks_iterations[self.task][0]) * self.epochs) + 1
      self.iteration_glob += task_iterations_

    self.iteration_task = self.start_task_iteration
    print(f'start at task D{self.start_task + 1} (iteration: {self.start_task_iteration})')


  def train(self):
    ''' training and testing steps of GMM '''
    self.iteration_glob = 0 # global iteration counter
    self._build_graph()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.compat.v1.train.Saver()           # to store and restore models
    self._load_model()

    for self.task in range(self.start_task, self.num_tasks):
      print('-' * 60, f'Task D{self.task + 1}' ,'-' * 60)
      self._load_task_dataset() # load dataset for current task

      if hasattr(self, 'task_epochs'):
        self.epochs = self.task_epochs[self.task]

      self.iterations_task_all = int(math.ceil(self.tasks_iterations[self.task][0]) * self.epochs) + 1
      self._calc_test_points()
      # self._generate_sample_dataset() # for replay samples of previous tasks (current state of gmm)
      if self.task > 0: self.gmm_layer.reset_regularizer(self.task)

      if self.start_task_iteration == self.iterations_task_all: print(f'Nothing more to do for task D{self.task + 1}')

      if self.task > 0: self.gmm_layer.resetSigmaEps()

      for self.iteration_task in range(self.start_task_iteration, self.iterations_task_all):
        self._test()
        self._train_step()
        # if self.task > 0: self._train_step(Dataset_Type.GENERATED)
        self._visualize()
        self._save_model()
      # task done

      self.iteration_glob       = self.iteration_glob + self.iteration_task # update global iteration counter
      self.start_task_iteration = 0
    # all tasks done

    jason.dump(self.log, open(self.log_path_name, 'w')) # I hate json!
    # if self.vis_points != 0: Vis.stop_all_vis() # self._stop_vis()
    self.sess.close()

def main(_):
  GMM().train()

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])

