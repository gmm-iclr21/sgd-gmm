import argparse
from enum import Flag

class ArgEnum(Flag):
  def __str__(self):
    return self.name.lower()

  def __repr__(self):
    return str(self)

  @staticmethod
  def arg(s):
    try            : return ArgEnum[s.upper()]
    except KeyError: return s



class Parser():

  def __init__(self, **kwargs):
    self.default_parser = argparse.ArgumentParser(**kwargs)

  def print_flags(self, flags):
    return self._print_flags(flags)

  def _print_flags(self, flags):
    ''' print all flags '''
    s = ''
    for k, v, in sorted(vars(flags).items()):
      if v is not None:
        s += '\n    * {:25}: {}'.format(k, v)
    return s

  def _printFlags(self, flags):
    ''' print all flags '''
    side = '-' * int(((80 - len('Flags')) / 2))
    print(side + ' ' + 'Flags' + ' ' + side)
    print(self._print_flags(flags))
    print('-' * 80)

  def add_argument(self, *args, **kwargs):
    self.default_parser.add_argument(*args, **kwargs)


  def _parse_unknown(self, unparsed, FLAGS):
    remove_unparsed = unparsed
    for unknown in unparsed:
      if unknown.startswith('--'):   # create new parameter
        parameter_name = unknown # get parameter name (with trailing "--"), dont know why...
        values = list()
        remove_unparsed = remove_unparsed[1:]
        for following_values in remove_unparsed:
          if following_values.startswith('--'): break
          values.append(following_values)

        if not values: continue
        if len(values) == 1: # if a single value
          try: # if a single int
            int_value = int(values[0])
            self.default_parser.add_argument(parameter_name, default=int_value, type=int, help='~')
            continue
          except: pass
          try: # if a single float
            float_value = float(values[0])
            self.default_parser.add_argument(parameter_name, default=float_value, type=float, help='~')
            continue
          except: pass
          self.default_parser.add_argument(parameter_name, default=values[0], type=str, help='~') # if a single string
        else: # if multiple values
          try: # if multiple ints
            int_values = [ int(x) for x in values ]
            self.default_parser.add_argument(parameter_name, default=int_values, type=int, nargs='*', help='~')
            continue
          except: pass
          try: # if multiple floats
            float_values = [ float(x) for x in values ]
            self.default_parser.add_argument(parameter_name, default=float_values, type=float, nargs='*', help='~')
            continue
          except: pass
          self.default_parser.add_argument(parameter_name, default=values, type=str, nargs='*', help='~',) # if multiple strings
      else:
        remove_unparsed = remove_unparsed[1:]

    self.default_parser.parse_known_args(namespace=FLAGS)

  def parse_all_args(self, print_flags=False):
    ''' parse all arguments and add automatic additional arguments

    @return: an object with all parameter (argparse.Namespace)
    '''
    try:
      FLAGS, unparsed = self.default_parser.parse_known_args()
    except Exception as ex:
      print(ex)
      self.default_parser.print_help()
      exit(0)

    self._parse_unknown(unparsed, FLAGS)
    if print_flags: self._printFlags(FLAGS)
    return FLAGS
