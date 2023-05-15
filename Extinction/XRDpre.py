# Calculation of Diffraction Conditions and Extinction via Crystal Symmetry
# Author: Bin CAO <binjacobcao@gmail.com>

import os
import warnings
from sympy import *
import copy
import numpy as np
import pandas as pd
import re
from .pymatgen_cif import CifFile
from ..EMBraggOpt.BraggLawDerivation import BraggLawDerivation
from ..XRDSimulation.Simulation import cal_atoms

class profile:
    def __init__(self, wavelength='CuKa',two_theta_range=(10, 90)):
        """
        Args:
            wavelength: The wavelength can be specified as either a
                float or a string. If it is a string, it must be one of the
                supported definitions in the dict of WAVELENGTHS.
                Defaults to "CuKa", i.e, Cu K_alpha radiation.
        """
        warnings.filterwarnings('ignore')
        if isinstance(wavelength, (float, int)):
            self.wavelength = wavelength
        elif isinstance(wavelength, str):
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        else:
            raise TypeError("'wavelength' must be either of: float, int or str")
        self.two_theta_range = two_theta_range

    def generate(self, filepath = None,):
        """
        for a single crystal
        Computes the XRD pattern and save to csv file
        Args:
            filepath (str): file path of the cif file to be calculated
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
        return 
        latt: lattice constants : [a, b, c, al1, al2, al3]
        structure_factor : [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
        """
        if filepath == None:
            pass
        else:
            latt, _, structure_factor = read_cif(filepath)
            print('cif file parse completed')
        
        StructureFactor = Bravais_grid(copy.deepcopy(structure_factor))
        system = det_system(latt)
    
        grid, d_list = Diffraction_index(system,latt,self.wavelength,self.two_theta_range)
        print('retrieval of all reciprocal vectors satisfying the diffraction geometry is done')
      
        res_HKL, ex_HKL, d_res_HKL, d_ex_HKL = cal_extinction(structure_factor, latt,grid,d_list,self.wavelength)
        print('extinction peaks are distinguished')
        print('There are {} extinction peaks'.format(len(d_ex_HKL)) )

        difc_peak = pd.DataFrame(res_HKL,columns=['H','K','L'])
        difc_peak['Distance'] = d_res_HKL
        difc_peak['2theta/TOF'] = 2 * np.arcsin(self.wavelength /2/np.array(difc_peak['Distance'])) * 180 / np.pi
        difc_peak['Mult'] = mult_dic(res_HKL,system)
        difc_peak.sort_values(by=['2theta/TOF'], ascending=True, inplace=True)
        os.makedirs('output_xrd/', exist_ok=True)
        difc_peak.to_csv('output_xrd/{}HKL.csv'.format(filepath[-11:-4]),index=False)
        # difc_peak.to_csv('output_xrd/{}_extinction.csv'.format(filepath[-11:-4]))

        ex_peak = pd.DataFrame(ex_HKL,columns=['H','K','L'])
        ex_peak['Distance'] = d_ex_HKL
        ex_peak['2theta/TOF'] = 2 * np.arcsin(self.wavelength /2/np.array(ex_peak['Distance'])) * 180 / np.pi
        ex_peak['Mult'] = mult_dic(ex_HKL,system)
        ex_peak.sort_values(by=['2theta/TOF'], ascending=True, inplace=True)
        os.makedirs('output_xrd/', exist_ok=True)
        ex_peak.to_csv('output_xrd/{}_Extinction_peak.csv'.format(filepath[-11:-4]),index=False)
        print('Diffraction condition judgment end !')
        
        return latt, StructureFactor

########################################################################
def getFloat(s):
    return float(re.sub(u"\\(.*?\\)", "", s))

def read_cif(cif_dir):
    cif = CifFile.from_file(cif_dir)
    for k in cif.data:
        v = cif.data[k].data

        a = getFloat(v['_cell_length_a'])
        b = getFloat(v['_cell_length_b'])
        c = getFloat(v['_cell_length_c'])
        alpha = getFloat(v['_cell_angle_alpha'])
        beta = getFloat(v['_cell_angle_beta'])
        gamma = getFloat(v['_cell_angle_gamma'])
        latt = [a, b, c, alpha, beta, gamma]

        if '_symmetry_space_group_name_H-M' in v:
            symbol = v['_symmetry_space_group_name_H-M'][0]
        elif '_symmetry_space_group_name_Hall' in v:
            symbol = v['_symmetry_space_group_name_Hall'][0]
        else:
            raise Exception('symmetry_space_group_name not found in {}'.format(cif_dir))
        
        sites = [symbol]
        for i, s in enumerate(v['_atom_site_type_symbol']):
            sites.append([s, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
    return latt, symbol, sites
########################################################################

def get_float(f_str, n):
    f_str = str(f_str)      
    a, _, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       
    return float(".".join([a, c]))
    
def det_system(Lattice_constants):
    # Lattice_constants is a list
    ini_a = Lattice_constants[0]
    ini_b = Lattice_constants[1]
    ini_c = Lattice_constants[2]
    ini_la1 = Lattice_constants[3]
    ini_la2 = Lattice_constants[4]
    ini_la3 = Lattice_constants[5]
    crystal_sys = 7
    if ini_la1 == ini_la2 and ini_la1 == ini_la3:
        if ini_la1 == 90:
            if ini_a == ini_b and ini_a == ini_c:
                crystal_sys = 1
            elif ini_a == ini_b and ini_a != ini_c:
                crystal_sys = 3
            elif ini_a != ini_b and ini_a != ini_c and ini_b != ini_c:
                crystal_sys = 4
        elif ini_la1 != 90 and ini_a == ini_b and ini_a == ini_c:
            crystal_sys = 5
    elif ini_la1 == ini_la2 and ini_la1 == 90 and ini_la3 == 120 and ini_a == ini_b and ini_a != ini_c:
        crystal_sys = 2
    elif ini_la1 == ini_la3 and ini_la1 == 90 and ini_la2 \
            != 90 and ini_a != ini_b and ini_a != ini_c and ini_c != ini_b:
        crystal_sys = 6
    return crystal_sys

def de_redundant(grid, d_list):
    """
    Multiplicity due to spatial symmetry
    """
    # input is DataFrame
    grid = np.array(grid.iloc[:,[0,1,2]])
    d_list = np.array(d_list.iloc[:,0])

    res_HKL = []
    res_d = []

    index = -1
    for i in d_list:
        item = get_float(i,4)
        index += 1
        if item not in res_d:
            res_d.append(item)
            res_HKL.append(grid[index])
    return res_HKL, res_d

def Diffraction_index(system,latt,cal_wavelength,two_theta_range):
    """
    Calculation of Diffraction Peak Positions by Diffraction Geometry
    (S-S') = G*, where G* is the reciprocal lattice vector
    """
    hh, kk, ll = np.mgrid[0:21:1, 0:21:1,0:21:1]
    grid = np.c_[hh.ravel(), kk.ravel(),ll.ravel()]
    d_f = BraggLawDerivation().d_spcing(system)
    sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')
    d_list = [1e-10] # HKL=000

    for i in range(len(grid)-1):
        peak = grid[i+1]
        d_list.append(
            float(d_f.subs({sym_h: peak[0], sym_k: peak[1], sym_l: peak[2], sym_a: latt[0], sym_b: latt[1],
                            sym_c: latt[2], angle1: latt[3]*np.pi/180, angle2: latt[4]*np.pi/180, angle3:latt[5]*np.pi/180}))
                            )
        
    # Satisfied the Bragg Law
    # 2theta = 2 * arcsin (lamda / 2 / d)
    bragg_d = cal_wavelength /2/np.array(d_list)
    index0 = np.where(bragg_d > 1)
    # avoid null values
    _d_list = pd.DataFrame(d_list).drop(index0[0])
    _grid = pd.DataFrame(grid).drop(index0[0])

    # recover the index of DataFrame
    for i in [_d_list, _grid]:
        i.index = range(i.shape[0])

    two_theta = 2 * np.arcsin(cal_wavelength /2/np.array(_d_list.iloc[:,0])) * (180 / np.pi)
    index = np.where((two_theta <= two_theta_range[0]) | (two_theta >= two_theta_range[1]))

    d_list = _d_list.drop(index[0])
    grid = _grid.drop(index[0])

    # return all HKL which are satisfied Bragg law
    res_HKL, res_d = de_redundant(grid, d_list)
    return res_HKL, res_d

def translation(ori_loc, tran_path):
    # ori_loc = [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  m atoms
    # tran_path = [[1/2,1/2,1/2],[1/2,1/2,0],.....] t times translations
    # return the autom locations after translation, len = (t+1)m
    # step_a, step_b, step_c
    ori_atom = copy.deepcopy(ori_loc)
    for serial in range(len(tran_path)):
        step_a, step_b, step_c = tran_path[serial][0],tran_path[serial][1],tran_path[serial][2]
        # traverse all atoms
        for j in range(len(ori_atom)): # j = 0,1,...m
            atom = []
            atom.append(ori_atom[j][0])
            atom.append(ori_atom[j][1]+step_a)
            atom.append(ori_atom[j][2]+step_b)
            atom.append(ori_atom[j][3]+step_c)
            # add a new location of atom
            ori_loc.append(atom)
    return ori_loc

def det_Bravais(structure_factor):
    index = []
    # structure_factor ---> [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] 
    for j in range(0, len(structure_factor)):
        if 0 <= structure_factor[j][1] <= 1 and 0 <= structure_factor[j][2] <= 1 and 0 <= structure_factor[j][3] <= 1:
            pass
        else:  
            index.append(j)
    index.reverse()
    for i in index:
        structure_factor.pop(i)
    return structure_factor

def Bravais_grid(structure_factor):
    """
    Find all atomic positions in the unit cell
    """
    # structure_factor ---> ['P',['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] 
    _type = structure_factor[0]
    structure_factor.pop(0)
    # symmetry structures
    if _type == 'P' or _type == 'R':
        res =  structure_factor
    elif _type == 'I': # body center
        tran_path = [
                     [1/2, 1/2,1/2],
                     [-1/2,1/2,1/2],[1/2,-1/2,1/2],[1/2, 1/2,-1/2],
                     [-1/2,-1/2,1/2],[-1/2, 1/2,-1/2],[1/2, -1/2,-1/2],
                     [-1/2,-1/2,-1/2]
                     ]
        res = translation(structure_factor,tran_path)
    elif _type == 'C': # bottom center
        tran_path = [
                     [1/2, 1/2,0],
                     [-1/2, 1/2,0],[1/2, -1/2,0],
                     [-1/2,-1/2,0]
                     ]
        res = translation(structure_factor,tran_path)
    elif _type == 'F': # face center
        tran_path = [
                     [1/2, 1/2,0],
                     [-1/2, 1/2,0],[1/2, -1/2,0],
                     [-1/2, -1/2,0],
                     [1/2, 0, 1/2],
                     [-1/2, 0, 1/2],[1/2, 0, -1/2],
                     [-1/2, 0, -1/2]
                     [0, 1/2, 1/2],
                     [0,-1/2, 1/2],[0, 1/2, -1/2],
                     [0, -1/2, -1/2]
                     ]
        # delete atoms outside the unit cell
        res = translation(structure_factor,tran_path)
    else:  
        print('Error type -symbol- only following are allowed:')
        print('\'P\',\'R\',\'I\',\'C\',\'F\'')
        raise ValueError

    # v_res = det_Bravais(res)
    # del function det_Bravais
    return res

# def fun for calculating extinction
# del the peak extincted
def cal_extinction(structure_factor, latt,HKL_list,dis_list,wavelength):
    system = det_system(latt)
    HKL_list = np.array(HKL_list).tolist()
    # structure_factor ---> ['P',['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
    uni_atom_loc = Bravais_grid(structure_factor)
    # uni_atom_loc ---> [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] after transformation
    res_HKL = []
    d_res_HKL = []

    ex_HKL = []
    d_ex_HKL = []
    
    for angle in range(len(HKL_list)):
        FHKL_square_left = 0
        FHKL_square_right = 0
        for atom in range(len(uni_atom_loc)):
            d_f = BraggLawDerivation().d_spcing(system)
            sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')
            plane_d = float(d_f.subs({sym_h: HKL_list[angle][0], sym_k: HKL_list[angle][1], sym_l: HKL_list[angle][2], sym_a: latt[0], sym_b: latt[1],
                            sym_c: latt[2], angle1: latt[3]*np.pi/180, angle2: latt[4]*np.pi/180, angle3:latt[5]*np.pi/180}))
            mu = 2 * np.arcsin(wavelength /2/plane_d) * 180 / np.pi
            fi = cal_atoms(uni_atom_loc[atom][0], mu, wavelength)
    
            FHKL_square_left +=  fi * np.cos(2 * np.pi * (uni_atom_loc[atom][1] * HKL_list[angle][0] +
                                                uni_atom_loc[atom][2] * HKL_list[angle][1] + uni_atom_loc[atom][3] * HKL_list[angle][2]))
            FHKL_square_right += fi * np.sin(2 * np.pi * (uni_atom_loc[atom][1] * HKL_list[angle][0] +
                                                uni_atom_loc[atom][2] * HKL_list[angle][1] + uni_atom_loc[atom][3] * HKL_list[angle][2]))
        FHKL_square = FHKL_square_left ** 2 + FHKL_square_right ** 2
        if FHKL_square <= 1e-5:
            ex_HKL.append(HKL_list[angle])
            d_ex_HKL.append(dis_list[angle])
        else:
            res_HKL.append(HKL_list[angle])
            d_res_HKL.append(dis_list[angle])
   
    return res_HKL, ex_HKL, d_res_HKL, d_ex_HKL

def mult_rule(H, K,L,system):
    """
    Define the multiplicity factor resulting from crystal symmetry
    """
    if system == 1: # Cubic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif (H == K !=  0 and L == 0) or (H == L != 0 and K == 0) or (K == L != 0 and H == 0):
            mult = 12
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 48
        elif H == K == L != 0:
            mult = 8
        else:
            mult = 24
            
    elif system == 2 : # Hexagonal
        if (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif H == K == 0 and L != 0:
            mult = 2
        elif H == K !=  0 and L == 0:
            mult = 6
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 24
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 12
    
    elif system == 5: # Rhombohedral
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 6
        elif (H == K !=  0 and L == 0) or (H == L != 0 and K == 0) or (K == L != 0 and H == 0):
            mult = 6
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0  :
            mult = 24
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 12

    elif system == 3: # Tetragonal
        if (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 4
        elif H == K == 0 and L != 0:
            mult = 2
        elif H == K !=  0 and L == 0:
            mult = 4
        elif  H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 16
        elif H == K == L != 0:
            mult = 1
        else:
            mult = 8

    elif system == 4: # Orthorhombic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 2
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 8
        elif (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        else:
            mult = 4
        
    elif system == 6: # Monoclinic
        if (H == K == 0 and L != 0) or (H == L == 0 and K != 0 ) or  (K == L == 0 and H != 0):
            mult = 2
        elif H != K and K != L and H != L and H!=0 and K!=0 and L!=0 :
            mult = 4
        elif (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        elif H != L and H!=0 and L!=0 and K==0:
            mult = 2
        else:
            mult = 4
       
    elif system == 7: # Triclinic
        if (H == K == L != 0) or (H == K != 0 and L == 0) or (H == K != L and H != 0 and L != 0):
            mult = 1
        else:
            mult = 2
    else:
        raise ValueError
    return mult


def mult_dic(HKL_list,system):
    mult = []
    for i in range(len(HKL_list)):
          mult.append(mult_rule(HKL_list[i][0],HKL_list[i][1],HKL_list[i][2],system))
    return mult


# XRD wavelengths in angstroms
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.544414,
    "CuKa1": 1.540593,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}
