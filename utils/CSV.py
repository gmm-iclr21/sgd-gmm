import csv
import fuckit

class CSV():

  writer = dict()

  def __init__(self, *args, **kwargs):
    self.path       = kwargs.get('path', 'output.csv')
    self.name       = kwargs.get('name', 'default')
    #self.writer     = csv.writer(open(self.path, 'w')) # clear the opened file!!!
    self.write_head = False
    self.read_head  = False
    self._iterator  = self._get_iterator()

    CSV.writer[self.name] = self

  def write(self, row):
    try:
      self.writer.writerow(row)
    except:
      self.file   = open(self.path, 'w')
      self.writer = csv.writer(self.file) # clear the opened file!!!
      self.writer.writerow(row)

  def write_header(self, header):
    if not self.write_head:
      self.write(header)
      self.write_head = True

  def separator(self):
    self.writer.writerow('-' * 80)

  @classmethod
  def write_(cls, writer, row):
    writer.get(writer).write(row)

  def _get_iterator(self):
    with open(self.path) as file:
      reader = csv.reader(file)
      for row in reader:
        if not row: continue
        if not self.read_head:
          row = [ str(head) for head in row ]
          row = { name: column for column, name in enumerate(row) }
        else:
          values = list()
          for value in row:
            try: value = float(value)
            except:
              if '[' in value and ']' in value:
                value = float(''.join(value.split('[')[1]).split(']')[0])
            values += [value]
            row = values
        yield row


  def read_header(self):
    header         = next(self._iterator)
    self.header    = header
    self.read_head = True
    return header

  def read(self):
    if not self.read_head:
      return self.read_header()
    return self._iterator

  @classmethod
  def flush_all(cls):
    for csv_write in cls.writer.values(): csv_write.flush()

  @classmethod
  def close_all(cls):
    for csv_write in cls.writer.values(): csv_write.close()

  def flush(self):
    with fuckit: self.file.flush()

  def close(self):
    self.flush()
    with fuckit: self.file.close()
