import numpy as np
import csv

class VectorStream(object):
    def __init__(self, filename, dtype=float, missing='', header=False):
        self._filename = filename
        self._dtype = dtype
        self._header = header
        self._missing = missing
        with open(self._filename, 'rb') as f:
            reader = csv.reader(f)
            if header:
                reader.next()
            line = reader.next()
            self._vecsize = len(line)
        self._len = 0
        with open(self._filename, 'rb') as f:
            for line in f:
                self._len += 1
            if self._header:
                self._len -= 1
        self.shape = (self._len, self._vecsize)

    def __iter__(self):
        with open(self._filename, 'rb') as f:
            reader = csv.reader(f)

            if self._header:
                reader.next()

            for line in reader:
                if hasattr(self._dtype, "__len__"):
                    x = np.array([d(x) if x != self._missing else np.nan for d,x in zip(self._dtype, line)])
                else:
                    x = np.array([self._dtype(x) if x != self._missing else np.nan for x in line])

                yield np.ma.masked_array(x, mask=np.isnan(x))

    def __getitem__(self, idx):
        if idx > len(self):
            raise Exception('index greater than length of file')
        for i,line in enumerate(self):
            if i == idx:
                return line

    def __len__(self):
        return self._len


