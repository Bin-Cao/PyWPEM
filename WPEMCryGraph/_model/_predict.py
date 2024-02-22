from ase.io import read,write
import numpy as np
from ase.db import connect
import os
import pkg_resources

class CSNN_train(object):
    def __init__(self, folder_path):
        extract_to_folder = pkg_resources.get_distribution('WPEMCryGraph').location
        _loc = os.path.join(extract_to_folder,'WPEMCryGraph')
        loc = os.path.join(_loc,'_database')
        dbpath = connect(os.path.join(loc,'CSNNpreData.db'))
        if os.path.exists(dbpath):
            os.remove(dbpath)
        cif_file_paths = find_cif_files(folder_path)

        for cif_file_path in cif_file_paths:
            try:
                atoms = read(cif_file_path)
                dbpath.write(atoms, format='cif')
            except:
                print('Parse error of cif file %s' % cif_file_path)
        self.db = dbpath  # save the predicted data 






def find_cif_files(directory):
    """
    Walk through the given directory and return a list of paths to .cif files.
    """
    cif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(os.path.join(root, file))
    return cif_files