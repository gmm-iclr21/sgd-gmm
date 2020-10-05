import sklearn.metrics
import inspect
from experimentparser import ArgEnum


class Metrics(ArgEnum):
  ACCURACY_SCORE                  = 'accuracy_score',                  # metrics.accuracy_score(y_true, y_pred)           TESTED # Accuracy classification score. e.g., y_true=[1,2,3], y_pred=[1,1,3]
  AUC                             = 'auc',                             # metrics.auc(x, y)                                       # Compute Area Under the Curve (AUC) using the trapezoidal rule
  AVERAGE_PRECISION_SCORE         = 'average_precision_score',         # metrics.average_precision_score(y_true, y_score)        # Compute average precision (AP) from prediction scores
  BALANCED_ACCURACY_SCORE         = 'balanced_accuracy_score',         # metrics.balanced_accuracy_score(y_true, y_pred)  TESTED # Compute the balanced accuracy
  BRIER_SCORE_LOSS                = 'brier_score_loss',                # metrics.brier_score_loss(y_true, y_prob)                # Compute the Brier score.
  CLASSIFICATION_REPORT           = 'classification_report',           # metrics.classification_report(y_true, y_pred)    TESTED # Build a text report showing the main classification metrics;
  COHEN_KAPPE_SCORE               = 'cohen_kappa_score',               # metrics.cohen_kappa_score(y1, y2)                       # Cohens kappa: a statistic that measures inter-annotator agreement.
  CONFUSION_MATRIX                = 'confusion_matrix',                # metrics.confusion_matrix(y_true, y_pred)                # Compute confusion matrix to evaluate the accuracy of a classification
  F1_SCORE                        = 'f1_score',                        # metrics.f1_score(y_true, y_pred)                 TESTED # Compute the F1 score, also known as balanced F-score or F-measure
  FBETA_SCORE                     = 'fbeta_score',                     # metrics.fbeta_score(y_true, y_pred, beta)               # Compute the F-beta score
  HAMMING_LOSS                    = 'hamming_loss',                    # metrics.hamming_loss(y_true, y_pred)             TESTED # Compute the average Hamming loss.
  HINGE_LOSS                      = 'hinge_loss',                      # metrics.hinge_loss(y_true, pred_decision)               # Average hinge loss (non-regularized)
  JACCARD_SCORE                   = 'jaccard_score',                   # metrics.jaccard_score(y_true, y_pred)            TESTED # Jaccard similarity coefficient score
  LOG_LOSS                        = 'log_loss',                        # metrics.log_loss(y_true, y_pred)                        # Log loss, aka logistic loss or cross-entropy loss.
  MATTHEWS_CORRCOEF               = 'matthews_corrcoef',               # metrics.matthews_corrcoef(y_true, y_pred)        TESTED # Compute the Matthews correlation coefficient (MCC)
  PRECISION_RECALL_CURVE          = 'precision_recall_curve',          # metrics.precision_recall_curve(y_true)                  # Compute precision-recall pairs for different probability thresholds
  PRECISION_RECALL_FSCORE_SUPPORT = 'precision_recall_fscore_support', # metrics.precision_recall_fscore_support()        TESTED # Compute precision, recall, F-measure and support for each class
  PRECISION_SCORE                 = 'precision_score',                 # metrics.precision_score(y_true, y_pred)          TESTED # Compute the precision
  RECALL_SCORE                    = 'recall_score',                    # metrics.recall_score(y_true, y_pred)             TESTED # Compute the recall
  ROC_AUC_SCORE                   = 'roc_auc_score',                   # metrics.roc_auc_score(y_true, y_score)                  # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
  ROC_CURVE                       = 'roc_curve',                       # metrics.roc_curve(y_true, y_score)               TESTED # Compute Receiver operating characteristic (ROC)
  ZERO_ONE_LOSS                   = 'zero_one_loss',                   # metrics.zero_one_loss(y_true, y_pred)            TESTED # Zero-one classification loss.
  # Regression metrics
  EXPLAINED_VARIANCE_SCORE        = 'explained_variance_score',        # metrics.explained_variance_score(y_true, y_pred)        # Explained variance regression score function
  MAX_ERROR                       = 'max_error',                       # metrics.max_error(y_true, y_pred)                       # max_error metric calculates the maximum residual error.
  MEAN_ABSOLUTE_ERROR             = 'mean_absolute_error',             # metrics.mean_absolute_error(y_true, y_pred)             # Mean absolute error regression loss
  MEAN_SQUARED_ERROR              = 'mean_squared_error',              # metrics.mean_squared_error(y_true, y_pred)              # Mean squared error regression loss
  MEAN_SQUARED_LOG_ERROR          = 'mean_squared_log_error',          # metrics.mean_squared_log_error(y_true, y_pred)          # Mean squared logarithmic error regression loss
  MEDIAN_ABSOLUTE_ERROR           = 'median_absolute_error',           # metrics.median_absolute_error(y_true, y_pred)           # Median absolute error regression loss
  R2_SCORE                        = 'r2_score',                        # metrics.r2_score(y_true, y_pred)                        # R^2 (coefficient of determination) regression score function.


class Metric(object):

  def __init__(self, parameter):
    ''' metric class provide metric functions from scikit-learn (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) '''
    if not isinstance(parameter.metrics, list):
      parameter.metrics = [parameter.metrics]
    self.metrics    = parameter.metrics
    self.csv_header = self.build_csv_header()
    self.print_metric_parameter()


  def build_csv_header(self):
    header  = ['task', 'iteration']
    header += [f'{metric}' for metric in self.metrics]
    return header


  def print_metric_parameter(self):
    s = ''
    for metric in self.metrics:
      s                += f'metric {metric}: '
      metric_function   = getattr(sklearn.metrics, str(metric))
      params            = inspect.getfullargspec(metric_function)
      args , varargs    = params[0], params[1]
      varkw, defaults   = params[2], params[3]
      num_args          = len(args)
      num_default_args  = len(defaults) if defaults else num_args
      arg_list          = ', '.join([ arg                       for arg          in args[:num_args - num_default_args] ])
      default_arg_list  = ', '.join([''] + [ f'{arg}={default}' for arg, default in zip(args[num_args - num_default_args:], defaults) ])
      s                += arg_list
      s                += default_arg_list    if num_default_args != num_args else ''
      s                += f', args={varargs}' if varargs                      else ''
      s                += f', kwargs={varkw}' if varkw                        else ''
      s                += '\n'                if metric != self.metrics[-1]   else ''
    print(s)


  def eval(self, **kwargs):
    ''' calculate the specified metric(s)

    @param print_pre : is printed befor the metric output
    @param print_post: is printed after the metric output
    @param kwargs    : parameter for metric calculation (look at the metric function documentation)
      https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics, e.g.:
        y_true (ground truth) (np.array)                                                                                  ,
        y_pred (predicted labels) (np.array)                                                                         ,
        special (metric specific parameter (dict )), e.g. special={Metrics.ACCURACY_SCORE: dict(normalize=True)},
        list: if True, a list of measurement values is returned (list(values, ...)) (default if both False)
          AND
        dict: if True, a dict of measurement values is returned (dict(Metrics=value)) (default=False)
    @return: the return values of the metric function(s) ((optional)list, (optional)dict)
    '''
    values      = list()
    values_dict = dict()
    s           = f'{kwargs.get("print_pre", ""):<35}'
    for metric in self.metrics:
      metric_function   = getattr(sklearn.metrics, str(metric))                                                       # load the function
      param_list        = inspect.getfullargspec(metric_function)[0]                                                  # get the valid parameter names of the function
      filled_param_list = { param_name: kwargs.get(param_name) for param_name in param_list if param_name in kwargs } # build dict with parameter name and values
      filled_param_list.update(kwargs.get('special', {}).get(metric, {}))                                             # update with special parameter
      metric_value      = metric_function(**filled_param_list)                                                        # call metric function
      values           += [metric_value]                                                                              # combine all metric values
      values_dict.update({metric: metric_value})
      # TODO: format multiline metrics for csv files
      if str(metric) in ['classification_report', 'confusion_matrix']:  # for metrics with multiple output lines
        s += f'\n{metric}\n{metric_value}\n'
      else:
        s += f'| {metric}={metric_value} '
    s      += f'{kwargs.get("print_post", ""):<35}'
    #print(s)
    list_, dict_ = [ kwargs.get(key, False) for key in ['list', 'dict'] ]
    if list_ and dict_: return values, values_dict
    if dict_          : return values_dict
    return values
