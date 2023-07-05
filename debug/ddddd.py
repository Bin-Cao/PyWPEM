import sys
import pandas as pd
Yourdir = '/Users/jacob/Documents/GitHub/'

sys.path.append(Yourdir)

from PyWPEM import WPEM


WPEM.SubstitutionalSearch('wpem.csv','Mn2O3.cif',wavelength='CuKa',SolventAtom = 'Mn3+', 
                          SoluteAtom= 'Ru',max_iter = 100)