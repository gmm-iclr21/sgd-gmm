''' provide a GMM layer '''

import math
import fuckit
import numpy                  as np
import tensorflow             as tf
import tensorflow_probability as tfp # pip3 install tensorflow_probability==0.7

from .Layer                     import Layer
from .regularizer.DoubleExpReg  import DoubleExpReg
from .regularizer.SingleExp     import SingleExp
from .regularizer.Double_Window import Double_Window
from .regularizer.Time_Decay    import Time_Decay, TimeDecayReg
from .regularizer.Regularizer   import Regularizer, Regularizer_Method as RM
from experimentparser           import ArgEnum
from utils.Visualizer           import Queue_Visualizer
from utils.Visualization        import Visualization as Vis
from utils.Parameter_Controller import Parameter_Controller
from utils.Variables            import Variables

from tensorflow.python.framework.ops import Tensor

class GMM_Layer(Layer, Tensor):

  class DiagMode(ArgEnum):
    DIAG   = 'diag'
    FULL   = 'full'

  class Energies(ArgEnum):
    LOGLIK = 'loglik'
    MC     = 'mc'

  def __init__(self, tensor, **kwargs):
    ''' GMM_Layer

    @param tensor  : input tensor of the previous layer (shape: [batch_size=?, h_out, w_out, c_out], e.g., [?, 1, 1, 784] for MNIST image)
    '''
    self.tensor                  = tensor # input tensor

    # parameters with default value
    self.K                       = kwargs.get('K'                           , 5 ** 2            ) # see description from command line parameters (or copy them here)
    self.n                       = int(math.sqrt(self.K))                                         # edge length of the 2D grid
    self.mode                    = kwargs.get('mode'                        , self.DiagMode.DIAG)
    self.muInit                  = kwargs.get('muInit'                      , 0.01              )
    self.sigmaUpperBound         = kwargs.get('sigmaUpperBound'             , 10                )
    self.somSigma0               = kwargs.get('somSigma0'                   , self.n / 4.0      ) # only use auto initialization of somSigma0
    self.somSigmaInf             = kwargs.get('somSigmaInf'                 , 0.01              )
    self.eps0                    = kwargs.get('eps0'                        , 0.05              ) # initial learning rate. In the paper learning rate is always constant, here it can be decayed like sigma(t) for t>t0
    self.epsInf                  = kwargs.get('epsInf'                      , 0.001             )
    self.energy                  = kwargs.get('energy'                      , self.Energies.LOGLIK  )
    self.lr_reduction            = kwargs.get('lr_reduction'                , 0.05              )

    # regularizer parameter
    self.regularizer             = kwargs.get('regularizer'                 , RM.SINGLE_EXP     )
    self.regularizer_delta       = kwargs.get('regularizer_delta'           , 0.05             )
    self.regularizer_gamma       = kwargs.get('regularizer_gamma'           , 0.9               ) # reduction factor for somSigma regularizer
    self.regularizer_reset_sigma = kwargs.get('regularizer_reset_sigma'     , self.somSigmaInf  ) # default, reset to somSigmaInf, even if not yet completely reduced
    self.regularizer_reset_eps   = kwargs.get('regularizer_reset_eps'       , self.epsInf       ) # default, reset to epsInf, even if not yet completely reduced

    self.t0Frac                  = kwargs.get('t0Frac'                      , 0.3               ) # parameter for Time_Decay regularizer
    self.tInfFrac                = kwargs.get('tInfFrac'                    , 0.8               ) # parameter for Time_Decay regularizer

    # visualization parameter
    self.w = self.patchX         = kwargs.get('w'                           , 28                )
    self.h = self.patchY         = kwargs.get('h'                           , 28                )
    self.c                       = kwargs.get('c'                           , 1                 )
    self.vis_loglikelihood_      = kwargs.get('vis_loglikelihood'           , False             )
    self.vis_eps_                = kwargs.get('vis_eps'                     , False             )
    self.vis_convMask_           = kwargs.get('vis_convMask'                , False             )
    self.vis_mus_                = kwargs.get('vis_mus'                     , False             )
    self.vis_sigmas_             = kwargs.get('vis_sigmas'                  , False             )
    self.vis_pis_                = kwargs.get('vis_pis'                     , False             )
    self.vis_reg_                = kwargs.get('vis_reg'                     , False             )
    self.vis_samples_            = kwargs.get('vis_samples'                 , False             )
    self.vis_parmeter_           = kwargs.get('vis_parameter'               , False              )

    # sampling parameter
    self.num_samples_default     = kwargs.get('num_samples'                , 100                ) # create x samples per sampling step

    #------------------------------------------------------------- SOM VARIABLES
    self.V                             = Variables()                                                                 # variable wrapper: to invoke callbacks on value change
    self.param_controller, pc_callback = Parameter_Controller('variables', visualization_enabled=self.vis_parmeter_) # GUI for subsequent parameter modification

    self.V.register(epsValue=self.eps0          , callback=pc_callback)
    callback_wrapper = lambda x, y: [pc_callback(x, y), self._update_convMask_vis()]
    self.V.register(somSigmaValue=self.somSigma0, callback=callback_wrapper)

    self.param_controller.add_slider(self.V, 'epsValue'     , valmin=self.epsInf     , valmax=self.eps0      * 1.1)
    self.param_controller.add_slider(self.V, 'somSigmaValue', valmin=self.somSigmaInf, valmax=self.somSigma0 * 1.1)

    tensor_shape      = self._tensor_shape(tensor)
    self.batch_size   = tensor_shape[0]
    self.h_out        = tensor_shape[1]
    self.w_out        = tensor_shape[2]
    self.c_out        = tensor_shape[3]

    self.const        = -self.c_out / 2. * math.log(2. * math.pi)
    self.extraRegC    = 0.00001 # for reading out loss value and comparison to EM

    if isinstance(self.regularizer, Regularizer): self.reg = self.regularizer # custom regularizer
    if self.regularizer == RM.DOUBLE_EXP        : self.reg = DoubleExpReg(**vars(self))
    if self.regularizer == RM.DOUBLE_WINDOW     : self.reg = Double_Window(**vars(self))
    ''' # time decay regularizer need the knowledge of the current iteration (remove) '''
    max_iterations = kwargs.get('max_iterations')
    if self.regularizer == RM.TIME_DECAY        : self.reg = TimeDecayReg(max_iterations, **vars(self))
    if self.regularizer == RM.SINGLE_EXP        : self.reg = SingleExp(**vars(self))
    #------------------------------------------------------------------------------

    self._init_tf_variables()
    self._build_tf_graph(tensor)
    self._update()
    self._init_visualization()

    output_tensor_name = kwargs.get('output_tensor', 'logprobs')
    output_tensor      = getattr(self, output_tensor_name)
    self._name         = kwargs.get('name', 'gmm_layer')
    Tensor.__init__(self, output_tensor._op, output_tensor._value_index, output_tensor._dtype)


  def __str__(self):
    max_ = len(max(vars(self).keys(), key=len))
    s = ''
    s += f'GMM_Layer: {self._name}\n'
    for k, v in vars(self).items():
      if k not in ['sigmaUpperDiag', '_op', 'reg']: s += f' {k:<{max_}}:{v}\n'
    s += str(self.reg)
    return s


  def test_step(self, feed_dict):
    sess = tf.compat.v1.get_default_session()
    return sess.run(self.loglikelihood, feed_dict=feed_dict)


  def train_step(self, feed_dict, loss=None):
    sess = tf.compat.v1.get_default_session()
    feed_dict.update({
      self.eps             : self.V.get_no_callback('epsValue')     ,
      self.somSigma        : self.V.get_no_callback('somSigmaValue')
      #self.lambda_pi_factor: 0.0                                    ,
      })
    log_likelihood = sess.run([self.loglikelihood] + self.update1 + self.update + [ self.clipSigmas, self.sigmaUpperDiag], feed_dict=feed_dict)[0]

    self.reg.add(log_likelihood).check_limit()
    with fuckit: self.loglikelihood_visualizer.add(log_likelihood)


  def _init_tf_variables(self):
    self.eps                 = tf.compat.v1.placeholder_with_default(name='learn_rate'  , shape=[]               , input=tf.compat.v1.constant(self.epsInf, dtype=tf.float64)           )  # placeholder with default value for learning rate
    self.num_samples         = tf.compat.v1.placeholder_with_default(name='num_samples' , shape=[]               , input=tf.compat.v1.constant(self.num_samples_default, dtype=tf.int32))
    # place-holders for time-varying factors (covariance (sigma) factor, centroids (mu) factor and weights (pi) factor (selection probability)). These could be varied but are kept to 1.0 here...
    self.lambda_sigma_factor = tf.compat.v1.placeholder_with_default(name='lambda_sigma', shape=[]               , input=tf.compat.v1.constant(1.0, dtype=tf.float64)                   )
    self.lambda_mu_factor    = tf.compat.v1.placeholder_with_default(name='lambda_mu'   , shape=[]               , input=tf.compat.v1.constant(1.0, dtype=tf.float64)                   )
    self.lambda_pi_factor    = tf.compat.v1.placeholder_with_default(name='lambda_pi'   , shape=[]               , input=tf.compat.v1.constant(1.0, dtype=tf.float64)                   )

    self.somSigma            = tf.compat.v1.placeholder_with_default(name='somSigma'    , shape=[]               , input=tf.compat.v1.constant(self.somSigmaInf, dtype=tf.float64)      ) # place-holder for decaying eps and SOM sigma

    initializer_pi           = tf.compat.v1.constant_initializer(1. / self.K                                     , dtype=tf.float64                                                     )
    initializer_mu           = tf.compat.v1.initializers.random_uniform(-self.muInit, +self.muInit               , dtype=tf.float64                                                     )

    pis_shape                = (1, self.h_out, self.w_out, self.K)
    mus_shape                = (1, self.h_out, self.w_out, self.K, self.c_out)
    self.pis                 = tf.compat.v1.get_variable(name='pis'        , shape=pis_shape                     , dtype=tf.float64, initializer=initializer_pi                         ) # the raw pis, before use they are passed through a softmax
    self.pisTmp              = tf.compat.v1.get_variable(name='pisTMP'     , shape=pis_shape                     , dtype=tf.float64, initializer=initializer_pi                         )
    self.mus                 = tf.compat.v1.get_variable(name='mus'        , shape=mus_shape                     , dtype=tf.float64, initializer=initializer_mu                         )
    self.musTmp              = tf.compat.v1.get_variable(name='musTMP'     , shape=mus_shape                     , dtype=tf.float64, initializer=initializer_mu                         )

    if self.mode == self.DiagMode.DIAG: # full covariance matrices are initialized to diagonal ones with diagonal entries given by sigmaUpperBound
      sigmas_shape           = (1, self.h_out, self.w_out, self.K, self.c_out)
      initializer_sigma      = tf.compat.v1.constant_initializer(math.sqrt(self.sigmaUpperBound)                 , dtype=tf.float64                                                     )
    if self.mode == self.DiagMode.FULL: # full covariance matrices are initialized to diagonal ones with diagonal entries given by sigmaUpperBound
      npLowerDiagMat         = np.triu(np.ones([self.c_out, self.c_out])).reshape(1, 1, 1, 1, self.c_out, self.c_out) # use for enforcing that self.sigmas is upper triangular: this mask can be multiplied directly with self.sigmas
      self.lowerDiagMat      = tf.compat.v1.constant(name='enforceSigmaConstraint', value=npLowerDiagMat         , dtype=tf.float64)
      self.diagMask          = np.reshape(np.eye(self.c_out, self.c_out), (1, 1, 1, 1, self.c_out, self.c_out)) # used for extracting the diagonal from self.sigmas
      sigmas_shape           = None
      initializer_sigma      = math.sqrt(self.sigmaUpperBound) * np.reshape(np.stack([ np.eye(self.c_out, self.c_out) for _ in range(self.K) ], axis=0), sigmas_shape)

    self.sigmas              = tf.compat.v1.get_variable(name='sigmas'     , shape=sigmas_shape                  , dtype=tf.float64, initializer=initializer_sigma                      )
    self.sigmasTmp           = tf.compat.v1.get_variable(name='sigmasTMP'  , shape=sigmas_shape                  , dtype=tf.float64, initializer=initializer_sigma                      )

    #------------------------------------------------------ OUTPUT NORMALIZATION
    lr_shape                 = (1, self.w_out * self.h_out * self.K)
    self.lrMeans             = tf.compat.v1.get_variable(name='lrMeans'    , shape=lr_shape                      , dtype=tf.float64, initializer=self.initializer_zeros                 )
    self.lrVars              = tf.compat.v1.get_variable(name='lrVars'     , shape=lr_shape                      , dtype=tf.float64, initializer=self.initializer_ones                  )


  def _build_tf_graph(self, tensor):

    def annealing():
      ''' generate structures for efficiently computing the time-varying smoothing filter for the annealing process '''
      shift          = +1 if self.n % 2 == 1 else 0
      oneRow         = np.roll(np.arange(-self.n // 2 + shift, self.n // 2 + shift, dtype=np.float64), self.n // 2 + shift).reshape(self.n)
      npxGrid        = np.stack(self.n * [oneRow], axis=0)
      npyGrid        = np.stack(self.n * [oneRow], axis=1)
      npGrid         = np.array([ np.roll(npxGrid, x_roll, axis=1) ** 2 + np.roll(npyGrid, y_roll, axis=0) ** 2 for y_roll in range(self.n) for x_roll in range(self.n) ])
      xyGrid         = tf.constant(npGrid)
      cm             = tf.reshape(tf.exp(-xyGrid / (2.0 * self.somSigma ** 2.0)), (self.K, -1))
      self.convMasks = cm / tf.reduce_sum(cm, axis=1, keep_dims=True)
    annealing()

    def gmm():
      diffs                 = tf.expand_dims(tensor, 3) - self.mus # --> (N, pY, pX , D=w * h * c) - (1, 1,  1,  K, D=w * h) = N, pY, pX, K, D

      if self.mode == self.DiagMode.DIAG:
        self.logdet             = tf.reduce_sum(tf.log(self.sigmas), axis=4, keep_dims=False) + self.const # sigmas: 1, 1, 1, K, D --> 1, 1, 1, K, 1
        exp_pis                 = tf.exp(self.pis) # --> 1, 1, 1, K
        self.real_pis           = exp_pis / tf.reduce_sum(exp_pis) # obtain real pi values by softmax over the raw pis thus, the real pis are always positive and normalized
        sqDiffs                 = diffs ** 2.0
        self.logexp             = -0.5 * tf.reduce_sum(sqDiffs * (self.sigmas**2.), axis=4) # N, pY, pX, K, D --> N, pY, pX, K
      if self.mode == self.DiagMode.FULL:
        # TODO: self.sigmas ** 2 appropriate here? Rather: tf.dot(sigmas,sigmas)?
        self.logdet             = tf.reduce_sum(self.diagMask * tf.log(self.sigmas), axis=(4, 5)) + self.const # extraRegC is for avoiding infinities in the log, now ensured by clipping
        exp_pis                 = tf.exp(self.pis) # --> 1,1,1,K # obtain real pi values by softmax over the raw pis thus, the real pis are always positive and normalized
        self.real_pis           = exp_pis / tf.reduce_sum(exp_pis)
        diffs                   = tf.expand_dims(diffs, 4)
        self.logexp             = 0.5 * tf.reduce_sum(tf.matmul(diffs, self.sigmas) ** 2, axis=(4, 5))

      self.logprobs         = self.logdet + self.logexp
      self.responsibilities = self.pis

      if self.energy == self.Energies.LOGLIK: # standard log-likelihood loss using log-sum-exp-trick. Not possible without...
        logScores               = tf.expand_dims(tf.log(self.real_pis), axis=0) + self.logprobs
        maxLogs                 = tf.reduce_max(logScores, axis=3, keep_dims=True)
        normScores              = tf.exp(logScores - maxLogs)
        singleSampleSums        = tf.reduce_sum(maxLogs, axis=3) + tf.log(tf.reduce_sum(normScores, axis=3))
        self.loglikelihood      = tf.reduce_mean(singleSampleSums)
      if self.energy == self.Energies.MC:     # MC approximation. No tricks necessary
        self.responsibilities   = self.pis
        logpiprobs              = tf.expand_dims(tf.log(self.real_pis) + self.logprobs, axis=3)
        self.convMasks          = tf.expand_dims(self.convMasks, axis=0)
        self.convMasks          = tf.expand_dims(self.convMasks, axis=0)
        self.convMasks          = tf.expand_dims(self.convMasks, axis=0)
        convLogProbs            = tf.reduce_sum(logpiprobs * self.convMasks, axis=4)
        singleConvLogSampleSums = tf.reduce_max(convLogProbs, axis=3)
        self.loglikelihood      = tf.reduce_mean(singleConvLogSampleSums, axis=(0, 1, 2))
    gmm()
    print('self.responsibilities', self.responsibilities)
    def output_normalization(): # TODO: discuss: move to Linear_Classifier_Layer?
      normlogprobs        = self.logprobs # - tf.reduce_max(self.logprobs)
      resp                = tf.reshape(normlogprobs, (-1, int(self.w_out * self.h_out * self.K))) # / 700

      # rescales the logits to have temporal mean/var=0/1, keep learning rate for linreg constant across different GMM sizes and tasks
      self.lrVars        *= 1 - self.lr_reduction
      self.lrVars        += self.lr_reduction * tf.reduce_mean((resp - tf.expand_dims(self.lrMeans, 0)) ** 2, axis=0)

      batch_mean          = tf.reduce_mean(resp, axis=0)
      self.lrMeans       *= 1 - self.lr_reduction
      self.lrMeans       += self.lr_reduction * batch_mean

      self.norm_resp      = tf.stop_gradient((resp - self.lrMeans) / tf.sqrt(self.lrVars))
    output_normalization()

    def sampling():
      ''' sampling operator (create a batch of samples from the prototypes) '''
      real_pis            = tf.cast(self.real_pis , tf.float32)
      pis                 = tf.reshape(real_pis   , (1, self.K))
      means               = tf.squeeze(self.mus   , name='sample_squeeze_means' )
      sigmas              = tf.squeeze(self.sigmas, name='sample_squeeze_sigmas')

      selectors           = tf.squeeze(tf.multinomial(tf.log(pis), self.num_samples))
      means               = tf.gather(means, selectors)
      if self.mode == self.DiagMode.FULL:
        covariances_inv = tf.compat.v1.matrix_inverse(tf.matmul(tf.transpose(sigmas, perm=[0, 2, 1]), sigmas)) / 40.
        covariances     = tf.gather(covariances_inv, selectors)
        mvn             = tfp.distributions.MultivariateNormalFullCovariance(
          loc               = means      ,
          covariance_matrix = covariances,
          )
      if self.mode == self.DiagMode.DIAG:
        covariances = tf.gather(sigmas, selectors)
        covariances = (1. / (covariances + 0.00001) ** 2) / 40
        mvn         = tfp.distributions.MultivariateNormalDiag(
          loc        = means      ,
          scale_diag = covariances,
          )
      self.create_samples = mvn.sample()
    sampling()


  def _update(self):
    ''' build operations for update GMM '''
    (self.grad_pis   ,
     self.grad_protos,
     self.grad_sigmas,
     )            = tf.gradients(self.loglikelihood, [self.pis, self.mus, self.sigmas])

    # store new values for pis, sigmas and mus in temp variables because in GD, all variables are updated simultaneously
    updatePis1    = tf.assign(self.pisTmp   , self.pis    + self.lambda_pi_factor    * self.eps    * self.grad_pis   )
    updateProtos1 = tf.assign(self.musTmp   , self.mus    + self.lambda_mu_factor    * self.eps    * self.grad_protos)
    updateSigmas1 = tf.assign(self.sigmasTmp, self.sigmas + self.lambda_sigma_factor * self.eps    * self.grad_sigmas)

    # update ops for real variables. Our choice then which one we actually execute
    updatePis     = tf.assign(self.pis   , self.pisTmp   )
    updateProtos  = tf.assign(self.mus   , self.musTmp   )
    updateSigmas  = tf.assign(self.sigmas, self.sigmasTmp)

    self.update1  = [updatePis1, updateProtos1, updateSigmas1]
    with tf.control_dependencies(self.update1):
      self.update   = [updatePis , updateProtos , updateSigmas ]

    sigma_limit   = math.sqrt(self.sigmaUpperBound)

    if self.mode == self.DiagMode.FULL:
      self.sigmaUpperDiag = tf.assign(self.sigmas, self.sigmas * self.lowerDiagMat) # ensure that precisions matrices stay upper diagonal by simply erasing elements below the diag.
      self.clipSigmas     = tf.assign(self.sigmas, tf.clip_by_value(self.sigmas, -0 * sigma_limit, sigma_limit))

    if self.mode == self.DiagMode.DIAG:
      with tf.control_dependencies(self.update):
        self.clipSigmas     = tf.assign(self.sigmas, tf.clip_by_value(self.sigmas, -sigma_limit, sigma_limit))
        with tf.control_dependencies([self.clipSigmas]):
          self.sigmaUpperDiag = tf.no_op()


  def info(self, **kwargs):
    ''' print information '''
    sess = tf.compat.v1.get_default_session()
    ( np_pis     ,
     np_sigmas  ,
     np_resp    ,
     np_mus     ,
     ) = sess.run(
       [ self.real_pis,
        self.sigmas   ,
        self.pis      ,
        self.mus
        ])
        #use the second training dataset for evaluation (do not change the order of the training dataset)

    def _printf(name, value):
      ''' print formated '''
      if type(value) != np.ndarray: print(f'{name:<20} = {value:.4f}')
      else                        : print(f'{name:<20} = min {np.min(value):.4f} max {np.max(value):.4f}')

    if self.mode == self.DiagMode.FULL: diags = np.diagonal(np_sigmas, axis1=4, axis2=5)
    if self.mode == self.DiagMode.DIAG: diags = np_sigmas

    #iteration = self.iteration_glob + self.iteration_task

    #print(f'{"iteration":<20} = {iteration}/{self.iterations_task_all}')
    #print(f'{"epsValue":<20} = {self.V.epsValue:.4f} min = {self.epsInf:10.4f}')
    print(f'{"somSigmaValue":<20} = {self.V.somSigmaValue:.4f} min = {self.somSigmaInf:10.4f}')

    _printf('sigmas'          , np_sigmas  )
    _printf('pis'          , np_pis  )

    if True:
      np.save(f'pis.npy'              , np_pis   )
      np.save(f'mus.npy'              , np_mus   )
      np.save(f'sigmas.npy'           , np_sigmas)


  def print_sample_distribution(self):
    ''' use the (class) inference stage to label the prototypes (mus)
        and determine the probable class distribution of the generated samples based on the mus' real_pis
    '''
    sess                     = tf.compat.v1.get_default_session()
    pis                      = sess.run(self.real_pis).squeeze().reshape(self.n, self.n)
    prototypes               = sess.run(self.mus).squeeze().reshape(-1, self.w, self.h, self.c)
    pred                     = np.array([ sess.run(self.logits, feed_dict={self.X: np.expand_dims(prototyp, axis=0)}).argmax(axis=1) for prototyp in prototypes ]).reshape(self.n, self.n)
    class_sample_probability = [ np.sum(pis[pred == class_]) for class_ in range(self.num_classes) ]
    s = 'sample probability class: '
    for class_, probability in enumerate(class_sample_probability):
      s += f'{class_}={probability:.2%} '
    print(f'{s}(based on classified prototypes (mus) and their probabilities (real_pis))')


  def reset_regularizer(self, task=None):
    self.reg.set(
      sigma = getattr(self, f'D{task + 1}_regularizer_reset_somSigma0', None), # task specific reset parameter, if not set, default
      eps   = getattr(self, f'D{task + 1}_regularizer_reset_eps'      , None),
      )


  def generate_samples(self, num_samples):
    ''' uses the GMM layer to create x samples.
      TODO: (optional) performance: leave everything on the GPU

      @param num_samples: number of samples to create
      @return: samples (np.array)
    '''
    sess                = tf.compat.v1.get_default_session()
    sampling_iterations = math.ceil(num_samples / self.num_samples_default)
    samples             = [ sess.run(self.create_samples) for _  in range(sampling_iterations) ] # create self.test_batch_size (100) samples
    samples             = np.concatenate(samples, axis=0)
    samples             = samples[:num_samples]
    return samples


  def get_data(self, data, w=3, **kwargs):
    sess    = tf.compat.v1.get_default_session()
    data    = sess.run(data, **kwargs)
    reshape = data.reshape((-1, *data.shape[w:]))
    while True: yield reshape


  def _update_convMask_vis(self, **kwargs):
    if not self.vis_convMask_: return
    iteration_glob = kwargs.get('iteration_glob', 0)
    iteration_task = kwargs.get('iteration_task', 0)
    iteration      = iteration_glob + iteration_task
    conv_masks     = self.get_data(self.convMasks, w=4, feed_dict={self.somSigma: self.V.somSigmaValue})
    self.vis_convMask.vis_conv_mask(next(conv_masks), iteration=iteration)


  def visualize(self, iteration_glob):
    #if self.vis_points  == 0: return ;

    mus    = self.get_data(self.mus)
    sigmas = self.get_data(self.sigmas)
    pis    = self.get_data(self.real_pis)
    if self.vis_mus_:
      with fuckit: [ vis_mus.visualize(next(mus)[i], heatmap=next(pis)[i], iteration=iteration_glob) for i, vis_mus    in enumerate(self.vis_mus)    ]
    if self.vis_sigmas_:
      with fuckit: [ vis_sigmas.visualize(next(sigmas)[i]                , iteration=iteration_glob) for i, vis_sigmas in enumerate(self.vis_sigmas) ]

    if self.vis_pis:
      with fuckit: [ vis_pis.visualize_pis(next(pis)[i]                  , iteration=iteration_glob) for i, vis_pis    in enumerate(self.vis_pis)    ]

    if self.vis_parmeter_:
      with fuckit: self.param_controller.visualize()

    if self.vis_loglikelihood_:
      with fuckit: self.loglikelihood_visualizer.redraw()

    if self.vis_eps_:
      with fuckit: self.eps_visualizer.add(self.V.get_no_callback('epsValue')).redraw()

    if self.vis_convMask_:
      self._update_convMask_vis(iteration_glob=iteration_glob)

    if self.vis_reg_:
      self.reg.vis()

  def resetSigmaEps(self):
    self.V.epsValue = self.regularizer_reset_eps ; # ;
    self.V.somSigmaValue = self.regularizer_reset_sigma # ;

    self.reg.set(
      sigma = self.V.somSigmaValue,
      eps   = self.V.epsValue,
      )

  def _init_visualization(self):
    # if self.vis_points == 0: return ;
    self.vis_mus                  = list()
    self.vis_sigmas               = list()
    self.vis_pis                  = list()

    self.loglikelihood_visualizer = Queue_Visualizer(
      1000                                     ,
      enabled         = self.vis_loglikelihood_, # plot_only       ,
      name            = 'log-likelihood'       ,
      xlabel          = 'window'               ,
      ylabel          = 'log-likelihood'       ,
      ylim            = (None, None)           ,
      smoothing_alpha = 0.1                    ,
      )

    self.eps_visualizer = Queue_Visualizer(
      1000                                ,
      enabled = self.vis_eps_             , # plot_only
      name    = 'epsilon'                 ,
      xlabel  = 'window'                  ,
      ylabel  = 'epsilon'                 ,
      ylim    = (0, self.V.epsValue * 1.1),
      )

    self.vis_convMask = Vis(
      enabled           = self.vis_convMask_,
      name              = 'convMask'        ,
      x_plots           = self.n            ,
      y_plots           = self.n            ,
      width             = 28, # self.n            ,
      height            = 28, # self.n            ,
      channels          = 1                 ,
      dimensionality    = self.K            ,
      h_out             = 5        ,
      w_out             = 5        ,
      )
    if self.vis_convMask_: self._update_convMask_vis() # initial draw if active

    for conv_id in range(self.h_out * self.w_out):
      self.vis_mus += [ Vis(
        enabled           = self.vis_mus_                     ,
        name              = f'mus_{conv_id}'                  ,
        x_plots           = self.n                            ,
        y_plots           = self.n                            ,
        width             = self.w                            ,
        height            = self.h                            ,
        channels          = self.c                            ,
        dimensionality    = self.patchX * self.patchY * self.c,
        h_out             = self.h_out                        ,
        w_out             = self.w_out                        ,
        )]

      self.vis_sigmas += [ Vis(
        enabled        = self.vis_sigmas_                  ,
        name           = f'sigmas_{conv_id}'               ,
        x_plots        = self.n                            ,
        y_plots        = self.n                            ,
        width          = self.patchX                       ,
        height         = self.patchY                       ,
        channels       = self.c                            ,
        dimensionality = self.patchX * self.patchY * self.c,
        h_out          = self.h_out                        ,
        w_out          = self.w_out                        ,
        )]

      self.vis_pis += [ Vis(
        enabled        = self.vis_pis_   ,
        name           = f'pis_{conv_id}',
        x_plots        = 1               ,
        y_plots        = 1               ,
        width          = self.n          ,
        height         = self.n          ,
        channels       = 1               ,
        dimensionality = self.n          ,
        h_out          = self.h_out      ,
        w_out          = self.w_out      ,
        )]
