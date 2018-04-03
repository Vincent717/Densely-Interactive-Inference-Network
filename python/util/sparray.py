
"""
Module for sparse arrays using dictionaries. Inspired in part 
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart
Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)
https://github.com/jesolem/sparray/blob/master/sparray.py
"""

import numpy

class Sparray3D(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, data, default=0, dtype=float):
        
        self.__default = default #default value of non-assigned elements
        self.ndim = 3
        sp0 = len(data)
        sp1 = len(data[0])
        sp2 = len(data[0][0])
        self.shape = (sp0, sp1, sp2)
        self.dtype = dtype
        self.__data = {}
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                for k, val in enumerate(col):
                    if val != 0:
                        self.__data[(i,j,k)] = val

    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.__data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self.__data.get(index,self.__default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if self.__data.has_key(index):
            del(self.__data[index])
            
    def __str__(self):
        return str(self.dense())

    def get_data(self):
        return self.__data

    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.__default * numpy.zeros(self.shape)
        for ind in self.__data:
            out[ind] = self.__data[ind]
        return out


if __name__ == "__main__":
    
    #test cases
    1