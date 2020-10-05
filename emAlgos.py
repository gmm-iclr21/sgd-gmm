'''
Implementation of EM and sEM. NIPS 2020 Submission, do not distrbute!
'''

import math
import sys, numpy as n, os, gzip, pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
import time

from experimentdataset          import Dataset_Wrapper, Dataset_Type as DT
import sklearn.mixture
import scipy.stats as stats
import scipy.special
import json


def findClasses(T, classes):
  nT = T.argmax(axis=1).astype("int32")
  acc = None

  for c in classes:
    if acc is None:
      acc = (nT == c).astype("bool")
    else:
      acc = np.logical_or (acc, (nT == c).astype("bool"))
  return acc


def selectClasses(classList, X, T):
  if len(classList) == 0 :
    return X, T
  else:
    indices = findClasses(T, classList)
    # print(indices)
    return X[indices, :], T[indices, :]


def norm(x):
  mn = x.min()
  mx = x.max()
  x -= mn
  x /= (mx - mn)
  return x


lookup_taskStr = dict(
  DAll = -1,
  DNow =  0,
  D1   =  1,
  D2   =  2,
  D3   =  3,
  )

def cfgParser(parser):
  parser.add_argument("--mode", type=str, required=False, choices=["EM","sEM"], default="sEM")
  parser.add_argument("--exp_id", type=str, required=False,default="exp")
  parser.add_argument("--dataset_file", type=str, required=False, default="MNIST.pkl.gz", help = "Dataset in pkl.gz format")
  parser.add_argument("--param", type=str, required=False, default="diag", choices = ["diag","full"],help="Precision matrix mode")
  parser.add_argument("--n", type=int, required=False, default=5, help = "Number of Gaussian components")
  parser.add_argument("--BS", type=int, required=False, default=1, help = "sEM batch size")
  parser.add_argument("--noise", type=float, required=False, default=0.0000001, help="Uniform noise amplitude added to data")
  parser.add_argument("--regC", type=float, required=False, default=0.00001)
  parser.add_argument("--alpha", type=float, required=False, default=0.001, help="sEM alpha, see paper")
  parser.add_argument("--alpha0", type=float, required=False, default=0.1, help="sEM alpha0, see paper")
  parser.add_argument("--initPrecs", type=float, required=False, default=1., help="init value & upper bound for diag sigma entries")
  parser.add_argument("--initMus", type=float, required=False, default=1., help="init value & upper bound for diag sigma entries")
  parser.add_argument("--rhoMin", type=float, required=False, default=0.0000000001, help="lower limit for sEM learning rate")
  parser.add_argument("--taskEpochs", type=float, nargs="*", required=True, help = "How many epochs for each sub-task?")
  parser.add_argument("--nrSamples", type=int, required=False, default=-1, help="how many samples to use from train data? -1 = all")
  parser.add_argument("--nrTestSamples", type=int, required=False, default=-1, help="how many test samples to use from train data? -1 = all")
  parser.add_argument("--initMode", type=str, required=False, default="random", choices=["random", "kmeans"], help = "obvious")
  parser.add_argument("--D1", type=int, nargs="*", required=False, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help = "classes for task D1")
  parser.add_argument("--D2", type=int, nargs="*", required=False, default=[])
  parser.add_argument("--D3", type=int, nargs="*", required=False, default=[])
  parser.add_argument("--D4", type=int, nargs="*", required=False, default=[])
  parser.add_argument("--nrTasks", type=int, required=True)

  parser.add_argument("--DAll", type=int, nargs="*", required=False, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help = "classes for DAll")
  parser.add_argument("--slice", type=int, nargs=2, required=False, default=[-1, -1], help = "Use central patch of size X,Y; [-1,-1] means use full image")
  parser.add_argument("--tmp_dir", type=str, required=False, default="./", help="where to store logfiles")


def processData(X, T, D, nrSamples):
  X = norm(X.reshape(-1, D))

  # shuffling
  # print("SSSHH",traind.shape, mode, nrSamples, D, P)
  indices = np.arange(0, X.shape[0])
  np.random.shuffle(indices)
  X = X[indices]
  T = T[indices]

  X = X[0:nrSamples]
  T = T[0:nrSamples]

  # add noise to train and test
  X += np.random.normal(0, noise, size=[X.shape[0], D])
  print("!!!!!XSAPE", X.shape)
  return X, T


def createInitializers(P, D, FLAGS):
  if FLAGS.initMode == "kmeans":

    # m = sklearn.mixture.GaussianMixture(n_components=P, covariance_type=param, reg_covar=0.000001, max_iter=maxIter, init_params='kmeans')
    initProtos = sklearn.cluster.KMeans(n_clusters=P).fit(traind).cluster_centers_
    means_init = initProtos
    precisions_init = np.random.uniform(0.99999, 1, (P, D)) * FLAGS.initPrecs
    weights_init = np.random.uniform(0.9999, 1, (P,))
    weights_init /= weights_init.sum()
  elif FLAGS.initMode == "random":
    print ('random init')
    means_init = np.random.uniform(-1, 1 , [P, D]) * FLAGS.initMus
    weights_init = np.random.uniform(0.9999, 1, [P])
    weights_init /= weights_init.sum()
    precisions_init = np.random.uniform(1, 1, [P, D]) * FLAGS.initPrecs

  return weights_init, means_init, precisions_init


if __name__ == "__main__":
  # print("MM=",testd.min(),testd.max())

  parser = argparse.ArgumentParser()
  cfgParser(parser)
  FLAGS = parser.parse_args()
  log = {}
  log["parameters"] = dict(FLAGS.__dict__)
  log["created"] = time.asctime()
  log['ll'] = []
  logname = os.path.join(FLAGS.tmp_dir, FLAGS.exp_id + '_sem.json')
  # json.dump(log, open(FLAGS.logfile, "w"))

  DS = Dataset_Wrapper(FLAGS)
  prop = DS.get_properties()
  print(FLAGS)
  _xDim, _yDim = prop.get('dimensions') # extract image properties or use slice pano atch size
  channels = prop.get('num_of_channels')
  nrRawClasses = prop.get('num_classes')
  xDim = _xDim if FLAGS.slice[0] <= 0 else FLAGS.slice[0] # if slice x is positive, change width
  yDim = _yDim if FLAGS.slice[1] <= 0 else FLAGS.slice[1] # if slice y is positive, change height

  mode = FLAGS.mode
  param = FLAGS.param
  n = FLAGS.n
  d = xDim # assume square pics
  nrSamples = FLAGS.nrSamples
  noise = FLAGS.noise
  initMode = FLAGS.initMode
  BS = FLAGS.BS
  alpha = FLAGS.alpha
  alpha0 = FLAGS.alpha0
  regC = FLAGS.regC
  # do not change
  D = yDim * xDim * channels
  P = n * n
  # N = testd.shape[0]

  # load&process
  _, tst = DS.get_dataset(FLAGS.DAll)
  testd, testl = processData(tst.images, tst.labels, D, FLAGS.nrTestSamples)

  # testing the sklearn gmm without kmeans!!
  if FLAGS.mode == "EM":

    protos = None

    covtype = param # diag, full, spehrical
    weights_init, means_init, precisions_init = createInitializers(P, D, FLAGS)

    m = sklearn.mixture.GaussianMixture(n_components=P, covariance_type=covtype, reg_covar=0.05, max_iter=1, weights_init=weights_init, means_init=means_init, n_init=1, precisions_init=precisions_init, verbose=0, warm_start=True)
    trall,tstall = DS.get_dataset(FLAGS.DAll) ;
    emtestd,emtestl = tstall.images, tstall.labels ;
    for taskI in range(1, FLAGS.nrTasks + 1):
      # load and process traind,trainl
      task = None
      if taskI == 1:
        task = FLAGS.D1
      if taskI == 2:
        task = FLAGS.D2
      if taskI == 3:
        task = FLAGS.D3
      if taskI == 4:
        task = FLAGS.D4

      print("Task=", task)
      tr, _ = DS.get_dataset(task)
      traind, trainl = processData(tr.images, tr.labels, D, FLAGS.nrSamples)

      print('Data shape', traind.shape)
      for it_ in range(0, int(FLAGS.taskEpochs[taskI - 1])):
        m.fit(traind.reshape(-1, D))
        # print (m.get_params().keys())
        m.set_params(precisions_init=np.clip(m.precisions_, 0, FLAGS.initPrecs))
        m.precisions_cholesky_ = np.clip(m.precisions_cholesky_, 0, math.sqrt(FLAGS.initPrecs))

        ll = m.score(emtestd)
        print(it_, "LL=", ll, " XX")
        log["ll"].append((taskI, it_, ll))
        # X = m._estimate_weighted_log_prob(testd)
        # print("xsh",X.shape)
        # ll = -scipy.special.logsumexp(X,axis=1).mean()
        # print("ownLL",ll)

    protos = m.means_
    if True:
      np.save(FLAGS.exp_id + "_mus.npy".format(x=covtype), protos[np.newaxis, np.newaxis, np.newaxis])
      np.save(FLAGS.exp_id + "_sigmas.npy".format(x=covtype), m.precisions_[np.newaxis, np.newaxis, np.newaxis])
      np.save(FLAGS.exp_id + "_pis.npy", m.weights_[np.newaxis, np.newaxis, np.newaxis])
      np.save("mus.npy".format(x=covtype), protos[np.newaxis, np.newaxis, np.newaxis])
      np.save("sigmas.npy".format(x=covtype), m.precisions_[np.newaxis, np.newaxis, np.newaxis])
      np.save("pis.npy", m.weights_[np.newaxis, np.newaxis, np.newaxis])

  # MEIN online EM using sEM
  # can compute normal EM or EM along proto-proto axes
  elif FLAGS.mode == "sEM":

    def compute_logprobs(diffs, sigmaSqs, regC):
      # compute log probabilities
      # --> P
      logdet = 0.5 * np.log(sigmaSqs).sum(axis=1)
      # --> (N, P, D).sum(axis=2) = (N,P)
      logprobs = (logdet[np.newaxis, :]) - 0.5 * ((diffs ** 2) * (sigmaSqs[np.newaxis, :, :])).sum(axis=2)
      return logprobs

    def compute_diffs(batch, protos):
      return batch[:, np.newaxis, :] - protos[np.newaxis, :, :]

    def compute_ll_test(data, pis, protos, sigmaSqs, regC):
      diffs = compute_diffs(data, protos)
      logprobs = compute_logprobs(diffs, sigmaSqs, regC)
      return compute_ll(logprobs, pis)

    def compute_ll(logprobs, pis):
      # estimate, faster and simpler!
      return np.mean((np.log(pis[np.newaxis, :]) + logprobs).max(axis=1)) - D / 2 * math.log(2 * 3.14159265)
      return scipy.special.logsumexp(np.log(pis[np.newaxis, :]) + logprobs, axis=1).mean() - D / 2 * math.log(2 * 3.14159265)

    def getTaskClasses(taskI, FLAGS):
      task = None
      if taskI == 1:
        task = FLAGS.D1
      if taskI == 2:
        task = FLAGS.D2
      if taskI == 3:
        task = FLAGS.D3
      if taskI == 4:
        task = FLAGS.D4
      return task

    def logTest(classList, log, taskStr):
      _testd, _testl = selectClasses(classList, testd, testl)
      #print("selected classes");
      accTest = 0.0 ;
      testBS = 10 ;
      nrTestBatches =_testd.shape[0]//testBS ;
      for testBatchIndex in range(0,nrTestBatches):
        mbatch = _testd[testBatchIndex*testBS:(testBatchIndex+1)*testBS,:] ;
        #print(_testd.shape[0], '=_testd, bs = ', mbatch.shape[0]);
        ll = testBS*compute_ll_test(mbatch, pis, protos, sigmaSqs, regC)
        accTest += ll ;
      processedSamples = (testBS*nrTestBatches)  ;
      #print("processed ", processedSamples, "acc =", accTest );
      accTest /= processedSamples ;

      if log.get("ll-" + taskStr, None) == None:
        log["ll-" + taskStr] = []

      log["ll-" + taskStr].append((lookup_taskStr[taskStr], it_, accTest))
      print('test ll on ' + taskStr + " = ", accTest)

    def saveForVis(protos,sigmaSqs,pis):
      np.save("mus.npy", protos[np.newaxis, np.newaxis, np.newaxis])
      #np.save("protoAcc.npy", protoAcc)
      np.save("sigmas.npy", sigmaSqs[np.newaxis, np.newaxis, np.newaxis])
      #np.save("sigmaSqAcc.npy", sigmaSqAcc)
      #np.save("respAcc.npy", respAcc)
      np.save("pis.npy", pis[np.newaxis, np.newaxis, np.newaxis])

    def logging(log, taskI, FLAGS, protos, sigmaSqs, pis, it_):
      print (it_, "-"*50)
      logTest([], log, "DAll")
      #print("Tested on DAll");

      classesNow = []
      for j in range(1, taskI + 1):
        taskStr = "D" + str(j)
        #print ("testing on ", taskStr);
        taskClasses = getTaskClasses (j, FLAGS)
        classesNow.extend(taskClasses)
        logTest(taskClasses, log, taskStr)

      logTest(classesNow, log, "DNow")
      saveForVis(protos, sigmaSqs, pis) ;



    # alloc
    pis = np.zeros([P])
    protos = np.zeros([P, D])
    sigmaSqs = np.zeros([P, D])
    protoAcc = np.ones([P, D]) * 0.0000
    sigmaSqAcc = np.ones([P, D]) * 0
    respAcc = np.random.uniform(1, 1, [P]) * 0

    # init
    weights_init, means_init, precisions_init = createInitializers(P, D, FLAGS)
    protos[:] = means_init
    sigmaSqs[:] = precisions_init
    pis[:] = weights_init

    it_ = -1
    classesNow = [] #  !!#
    for taskI in range(1, FLAGS.nrTasks + 1):
      # load and process traind,trainl
      task = getTaskClasses(taskI, FLAGS)
      classesNow.append(task)

      tr, tst = DS.get_dataset(task)
      traind, trainl = processData(tr.images, tr.labels, D, FLAGS.nrSamples)

      N = traind.shape[0]
      nrBatches = N // BS

      maxIter = int(FLAGS.taskEpochs[taskI - 1] * N / BS)
      initPeriod = int (N*0.1) ;

      for it in range(0, maxIter):
        it_ += 1
        # log probabilities
        batch = traind[(it_ % nrBatches) * BS:(it_ % nrBatches) * BS + BS]
        diffs = compute_diffs(batch, protos)
        # sqDiffs = (np.power(diffs, 2.))
        logprobs = compute_logprobs(diffs, sigmaSqs, regC)

        # compute responsibilities
        normlps = logprobs - logprobs.max(axis=1, keepdims=True)
        normps = np.exp(normlps) * pis[np.newaxis, :]
        resp = (normps / (normps.sum(axis=1, keepdims=True)))

        # step size update
        rho_ = alpha0 * BS * math.pow((it_ + 1.), -(0.5 + alpha))
        if rho_ < FLAGS.rhoMin:
          rho_ = FLAGS.rhoMin

        # update suff. statistics for the weights
        respAcc *= (1. - rho_)
        respAcc += (rho_ * resp.mean(axis=0))

        # update sufficient statstics for the mus
        protoAcc *= (1. - rho_)
        protoAcc += (rho_ * (resp[:, :, np.newaxis] * batch[:, np.newaxis, :]).mean(axis=0))

        """
        early M-step bec in order to have a robust estimate of sigma
        we need to use the diffs 2 the NEW means, see "Welford's online algorithm" on wikipedia
        """

        # check whether init period is over
        stillInInit = it_ * BS < initPeriod ;

        # Removing the if makes protos much better, but that is cheating
        if not stillInInit:
        # if True: <-- this hack achieves very fast conv at const learning rates
        # but of course a hack
          pis [:] = respAcc
          protos [:, :] = protoAcc / (pis[:, np.newaxis] + regC)

        # update suff statistics for the variances
        # need to update sigmaSqAcc as rho (x-mu(it-1))(x-mu(it))
        newdiffs = batch[:, np.newaxis, :] - protos[np.newaxis, :, :]
        sigmaSqAcc *= (1. - rho_)
        sigmaSqAcc += (rho_ * (resp[:, :, np.newaxis] * (diffs * newdiffs)).mean(axis=0))

        if stillInInit:
          continue

        # M-Step for sigmas, inverted because they are precisions
        sigmaSqs[:, :] = (pis[:, np.newaxis]) / (sigmaSqAcc + regC)
        sigmaSqs = np.clip(sigmaSqs, 0, FLAGS.initPrecs)

        # L-Step: compute loglik on test
        # D(ebug)-step
        if it_ % (1000 // BS) == 0:
          logging(log, taskI, FLAGS, protos, sigmaSqs, pis, it_);
      # ---------------------------------it_
    # ----------------------------------taskI
    pass
    logging(log, taskI, FLAGS, protos, sigmaSqs, pis, it_)


# ----------------------

  print ("logging to ", logname)
  json.dump(log, open(logname, "w"))

