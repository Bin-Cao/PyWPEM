# Calculation of Diffraction Conditions and Extinction via Crystal Symmetry
# Author: Bin CAO <binjacobcao@gmail.com>

import os
import warnings
from sympy import *
import copy
import numpy as np
import pandas as pd
import re
from .wyckoff import wyckoff_dict
from .CifReader import CifFile
from ..XRDSimulation.DiffractionGrometry.atom import atomics
from ..EMBraggOpt.BraggLawDerivation import BraggLawDerivation
from ..Plot.UnitCell import plotUnitCell

class profile:
    def __init__(self, wavelength='CuKa',two_theta_range=(10, 90),show_unitcell=False):
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
        self.show_unitcell = show_unitcell

    def generate(self, filepath ,latt = None, Asymmetric_atomic_coordinates = None,):
        """
        for a single crystal
        Computes the XRD pattern and save to csv file
        Args:
            filepath (str): file path of the cif file to be calculated
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.
                Asymmetric_atomic_coordinates : ['22',['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
                latt: lattice constants : [a, b, c, al1, al2, al3]
        return 
        latt: lattice constants : [a, b, c, al1, al2, al3]
        AtomCoordinates : [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
        """
        if type(filepath) != str:
            print('Need to specify the file (.cif) path to be processed')
        else:
            try:
                latt, space_g, Asymmetric_atomic_coordinates,Point_group= read_cif(filepath)
                print('the space group of input crystal is :',space_g )
                print('cif file parse completed')
            except:
                print('cif file parse failed with error')
                print('Please replace another cif file, or enter manually input lattice constants and  structure factor')
        
        AtomCoordinates= UnitCellAtom(copy.deepcopy(Asymmetric_atomic_coordinates))
        system = det_system(latt)

        grid, d_list = Diffraction_index(system,latt,self.wavelength,self.two_theta_range)
        print('retrieval of all reciprocal vectors satisfying the diffraction geometry is done')
      
        res_HKL, ex_HKL, d_res_HKL, d_ex_HKL = cal_extinction(Point_group, grid,d_list,system,AtomCoordinates,self.wavelength)
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

        if self.show_unitcell == True:
            plotUnitCell(AtomCoordinates,latt,).plot()
        else: pass
        
        return latt, AtomCoordinates

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
        space_group_code = int(v['_symmetry_Int_Tables_number'])
        latt = [a, b, c, alpha, beta, gamma]

        if '_symmetry_space_group_name_Hall' in v:
            symbol = v['_symmetry_space_group_name_Hall'][0]
            spaceG = v['_symmetry_space_group_name_Hall']
        elif '_symmetry_space_group_name_H-M' in v:
            symbol = v['_symmetry_space_group_name_H-M'][0]
            spaceG = v['_symmetry_space_group_name_H-M']
        else:
            raise Exception('symmetry_space_group_name not found in {}'.format(cif_dir))
        
        sites = [space_group_code]
        for i, name in enumerate(v['_atom_site_type_symbol']):
            sites.append([name, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
    return latt, spaceG, sites,symbol
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

def unit_cell_range(ori_atom):
    """
    Atoms within a unit cell are retained
    """
    # ori_loc = [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  Atoms after symmetry operations
    # Count the atoms in a unit cell
    unit_cell_atom = []
    for atom in range(len(ori_atom)): 
        x_ = ori_atom[atom][1]
        y_ = ori_atom[atom][2]
        z_ = ori_atom[atom][3]
        if 0 <= x_ <=1 and 0 <= y_ <=1 and 0 <= z_ <=1:
            unit_cell_atom.append(ori_atom[atom])
        elif 0 <= (x_-1) <=1 and 0 <= (y_-1) <=1 and 0 <= (z_-1) <=1:
            move_atom = ori_atom[atom]
            move_atom[1] -= 1
            move_atom[2] -= 1
            move_atom[3] -= 1
            unit_cell_atom.append(move_atom)
        elif 0 <= (x_+1) <=1 and 0 <= (y_+1) <=1 and 0 <= (z_+1) <=1:
            move_atom = ori_atom[atom]
            move_atom[1] += 1
            move_atom[2] += 1
            move_atom[3] += 1
            unit_cell_atom.append(move_atom)
        else: pass
    
    unique_data = []
    for item in unit_cell_atom:
        if item not in unique_data:
            unique_data.append(item)
    return unique_data


def UnitCellAtom(Asymmetric_atomic_coordinates):
    """
    Find all atomic positions in the unit cell
    """
    # Asymmetric_atomic_coordinates ---> [22,['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] 

    spg = Asymmetric_atomic_coordinates[0]
    Asymmetric_atomic_coordinates.pop(0)
    atom_loc = trans_atom(Asymmetric_atomic_coordinates,spg)

    return unit_cell_range(atom_loc)

# def fun for calculating extinction
# del the peak extincted
def cal_extinction(Point_group,HKL_list,dis_list,system,AtomCoordinates,wavelength):
    HKL_list = np.array(HKL_list).tolist()
    # Diffraction crystal plat
    res_HKL = []
    # interplanar spacing
    d_res_HKL = []

    # extinction crystal plat
    ex_HKL = []
    # interplanar spacing
    d_ex_HKL = []
    for angle in range(len(HKL_list)):
        two_theta = 2 * np.arcsin(wavelength /2/dis_list[angle]) * 180 / np.pi
        l_extinction = lattice_extinction(Point_group,HKL_list[angle],system)
        if l_extinction == True:
            ex_HKL.append(HKL_list[angle])
            d_ex_HKL.append(dis_list[angle])
        else:
            s_extinction = structure_extinction(AtomCoordinates,HKL_list[angle],two_theta,wavelength)
            if s_extinction == True:
                ex_HKL.append(HKL_list[angle])
                d_ex_HKL.append(dis_list[angle])
            else:
                res_HKL.append(HKL_list[angle])
                d_res_HKL.append(dis_list[angle])
   
    return res_HKL, ex_HKL, d_res_HKL, d_ex_HKL

def lattice_extinction(lattice_type,HKL,system):
    extinction = False
    # symmetry structures
    if lattice_type == 'P' or lattice_type == 'R':
        pass
    elif lattice_type == 'I': # body center
        if (HKL[0]+HKL[1]+HKL[2]) % 2 == 1:
            extinction = True
        else: pass
    elif lattice_type == 'C': # bottom center
        if system == 1 or system == 5: 
            if ((HKL[0]+HKL[1]) % 2 == 1) or ((HKL[1]+HKL[2]) % 2 == 1) or ((HKL[0]+HKL[2]) % 2 == 1): 
                extinction = True
            else: pass
        else:
            if (HKL[0]+HKL[1]) % 2 == 1: 
                extinction = True
            else: pass
    elif lattice_type == 'F': # face center
        if (HKL[0] % 2 == 1 and HKL[1] % 2 == 1 and HKL[2] % 2 == 1) or (HKL[0] % 2 == 0 and HKL[1] % 2 == 0 and HKL[2] % 2 == 0):
            pass
        else: extinction = True
    return extinction

def structure_extinction(AtomCoordinates,HKL,two_theta,wavelength):
    # AtomCoordinates = [['Cu2+',0.5,0.5,0.5],[],..]
    extinction = False

    FHKL_square_left = 0
    FHKL_square_right = 0
    for atom in range(len(AtomCoordinates)):
        fi = cal_atoms(AtomCoordinates[atom][0],two_theta, wavelength)
        FHKL_square_left += fi * np.cos(2 * np.pi * (AtomCoordinates[atom][1] * HKL[0] +
                                            AtomCoordinates[atom][2] * HKL[1] + AtomCoordinates[atom][3] * HKL[2]))
        FHKL_square_right += fi * np.sin(2 * np.pi * (AtomCoordinates[atom][1] * HKL[0] +
                                            AtomCoordinates[atom][2] * HKL[1] + AtomCoordinates[atom][3] * HKL[2]))
    FHKL_square = (FHKL_square_left ** 2 + FHKL_square_right ** 2)

    if FHKL_square <= 1e-5:
        extinction = True
    else: pass
    return extinction

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


def apply_operation(expression, variable, value):
    # Replace variable with given value
    for i in range(len(variable)):
        expression = expression.replace(variable[i], str(value[i]))
    # Use the eval function to evaluate the result of an expression
    result = eval(expression)
    return result


def trans_atom(atom_coordinate,sp_c):
    """
    atom_coordinate is the list of atoms in the shape of [['Cu2+',a0,b0,c0],['O2-',a1,b1,c1],...]
    sp_c is the code of space group, a int, e.g., 121
    """
    atom_loc = copy.deepcopy(atom_coordinate)
    wyckoff_site = wyckoff_dict.load()
    # Read in the wyckoff coordinates a
    opt_list = eval(np.array(wyckoff_site.iloc[sp_c,:])[0])

    # i.e., [['x', 'y', 'z', '-x', '-y', '-z',],[],]
        
    for atom in range(len(atom_coordinate)):
        # read in the asymmetric atoms coordinates
        a = atom_coordinate[atom][1]
        b = atom_coordinate[atom][2]
        c = atom_coordinate[atom][3]
        equivalent_pos = check_notations(a, b, c, opt_list) 
        # perform symmetric operations on atomic repeatedly according to wyckoff 
        for k in range(len(equivalent_pos)):
            new_loc = [atom_coordinate[atom][0]]
            # Determine the type of operation
            variable = ['x','y','z']
            value = [a,b,c]
            loc = apply_operation(expression=equivalent_pos[k], variable=variable, value=value)
            new_loc.append(get_float(loc[0],5))
            new_loc.append(get_float(loc[1],5))
            new_loc.append(get_float(loc[2],5))
            atom_loc.append(new_loc)
    return atom_loc

def check_notations(a, b, c, opt_list) :
    wyckoff_notation_num  = len(opt_list)
    for i in range(wyckoff_notation_num):
        # search from special location
        operators = opt_list[wyckoff_notation_num-1-i]
        coordinate = operators[0]
        # e.g., operators = ['0, 1/2, z', '1/2, 0, z+1/2']
        # operators[0] = '0, 1/2, z'
        variable = ['x','y','z']
        value = [a,b,c]
        real_value_coor = apply_operation(expression=coordinate, variable=variable, value=value)
        # march the coordinate
        if float(a) == float(real_value_coor[0]) and float(b) == float(real_value_coor[1]) and float(c) == float(real_value_coor[2]):
            res =  opt_list[wyckoff_notation_num-1-i]
            break
    return res


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

# functions defined in Simulatiuon module
def cal_atoms(ion, angle, wavelength,):
    """
    ion : atomic type, i.e., 'Cu2+' 
    angle : 2theta
    returns : form factor at diffraction angle 
    """
    dict =  atomics()
    # in case errors 
    try:
        # read in the form factor
        dict[ion]
    except:
        # atomic unionized forms
        # Planned replacement with Thomas-Fermi method
        ion = getHeavyatom(ion)
    loc = np.sin(angle / 2 * np.pi/180) / wavelength 
    floor_ = get_float(loc,1)
    roof_ = get_float((floor_+ 0.1),1)
    if floor_ == 0.0:
        floor_ = 0
    down_key = '{}'.format(floor_)
    up_key = '{}'.format(roof_)

    down = dict[ion][down_key]
    up = dict[ion][up_key]
    # linear interpolation
    # interval = 0.1 defined in form factor table
    fi = (loc - floor_) / 0.1 * (up-down) + down 
    return fi

def getHeavyatom(s):
    """
    Some atomic ionization forms not defined in the table are replaced by their unionized forms
    """
    # Define a function called getHeavyatom that takes one parameter: s, a string that contains letters and/or non-letter characters.
    return re.sub(r'[^A-Za-z]+', "", s)
    # Use the re.sub() function to replace all non-letter characters in s with an empty string. Return the modified string.