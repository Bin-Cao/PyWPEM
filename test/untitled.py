import sys
import pandas as pd
Yourdir = '/Users/jacob/Documents/GitHub/'
sys.path.append(Yourdir)
from PyWPEM import WPEM

latt, AtomCoordinates = WPEM.CIFpreprocess(filepath='Mn2O3.cif',two_theta_range=(20, 80),show_unitcell=True)