# Calculation of Diffraction Conditions and Extinction via Crystal Symmetry
# Author: Bin CAO <binjacobcao@gmail.com>

import os
import warnings
import sympy
from sympy import symbols, cos, sin
import copy
import numpy as np
import pandas as pd
import re
import sys
from .wyckoff import wyckoff_dict
from .CifReader import CifFile
from ..XRDSimulation.DiffractionGrometry.atom import atomics
from ..EMBraggOpt.BraggLawDerivation import BraggLawDerivation
from ..Plot.UnitCell import plotUnitCell
from ..EMBraggOpt.WPEMFuns.SolverFuns import cal_system
from .Relaxer import _Relaxer

class profile:
    def __init__(self, wavelength='CuKa',two_theta_range=(10, 90),show_unitcell=False,cal_extinction = True,relaxation=False):
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
        self.cal_extinction = cal_extinction
        self.show_unitcell = show_unitcell
        self.relaxation = relaxation

    # Calculate the volume of the unit cell
    def LatticVolume(self, crystal_system):
        # imput the number of crystal_system
        crystal_system = crystal_system
        sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_a sym_b sym_c angle1 angle2 angle3')
        if crystal_system == 1:  # Cubic
            Volume = sym_a ** 3
        elif crystal_system == 2:  # Hexagonal
            Volume = sym_a ** 2 * sym_c * sympy.sqrt(3) / 2
        elif crystal_system == 3:  # Tetragonal
            Volume = sym_a * sym_a * sym_c
        elif crystal_system == 4:  # Orthorhombic
            Volume = sym_a * sym_b * sym_c
        elif crystal_system == 5:  # Rhombohedral
            Volume = sym_a ** 3 * sympy.sqrt(1 - 3 * cos(angle1) ** 2 + 2 * cos(angle2) ** 3)
        elif crystal_system == 6:  # Monoclinic
            Volume = sym_a * sym_b * sym_c * sin(angle2)
        elif crystal_system == 7:  # Triclinic
            Volume = sym_a * sym_b * sym_c * sympy.sqrt(1 - cos(angle1) ** 2 - cos(angle2) **2
                                              - cos(angle3) ** 2 + 2 * cos(angle1) * cos(angle2) * cos(angle3))
        else:
            Volume = -1

        return Volume
    
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
            latt, space_g, Asymmetric_atomic_coordinates,Point_group, symmetric_operation = read_cif(filepath)
            if space_g != None:
                print('the space group of input crystal is :',space_g )
            print('cif file parse completed')

            # check 
            if latt == None or Point_group == None or Asymmetric_atomic_coordinates == None:
                print('cif file parse failed with error')
                print('Please replace another cif file, or enter manually the essential params!')
                if latt == None:
                    __latt = input("Please input lattice constants (i.e., 3.12,3.12,3.12,90,90,90): ")
                    latt = list(eval(__latt))
                if Point_group == None:
                    __Point_group = input("Please input point_group singal (i.e., F):")
                    Point_group = str(__Point_group)
                if Asymmetric_atomic_coordinates == None:
                    __Asymmetric_atomic_coordinates = input("Please input asymmetric_atomic_coordinates, contains space group code (i.e., 72,['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....):")
                    Asymmetric_atomic_coordinates = list(eval(__Asymmetric_atomic_coordinates))
            else: pass
        
        AtomCoordinates= UnitCellAtom(copy.deepcopy(Asymmetric_atomic_coordinates),symmetric_operation)
        system = det_system(latt)

        grid, d_list = Diffraction_index(system,latt,self.wavelength,self.two_theta_range)
        print('retrieval of all reciprocal vectors satisfying the diffraction geometry is done')

        res_HKL, ex_HKL, d_res_HKL, d_ex_HKL = cal_extinction(Point_group, grid,d_list,system,AtomCoordinates,self.wavelength,self.cal_extinction)
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

        lattic_mass = cal_lattic_mass(AtomCoordinates)
        VolumeFunction = self.LatticVolume(cal_system([latt])[0])
        sym_a, sym_b, sym_c, angle1, angle2, angle3 = symbols('sym_a sym_b sym_c angle1 angle2 angle3')
        lattic_volume = (float(VolumeFunction.subs(
            {sym_a: latt[0], sym_b: latt[1], sym_c:latt[2],
            angle1: latt[3] * np.pi/180 , angle2: latt[4] * np.pi/180, angle3: latt[5] * np.pi/180}))
            )

        lattic_density = lattic_mass / lattic_volume

        # relaxition
        if self.relaxation == True:
            print('M2GNET is applied for calculating the fromation energy per atom')
            lattice, final_energy_per_atom = _Relaxer(system,latt,AtomCoordinates)
        else:pass


        return latt, AtomCoordinates,lattic_density

########################################################################
def getFloat(s):
    return float(re.sub(u"\\(.*?\\)", "", s))

def read_cif(cif_dir):
    cif = CifFile.from_file(cif_dir)
    for k in cif.data:
        v = cif.data[k].data
        try:
            a = getFloat(v['_cell_length_a'])
            b = getFloat(v['_cell_length_b'])
            c = getFloat(v['_cell_length_c']) 
            alpha = getFloat(v['_cell_angle_alpha'])
            beta = getFloat(v['_cell_angle_beta'])
            gamma = getFloat(v['_cell_angle_gamma'])
            latt = [a, b, c, alpha, beta, gamma]
        except: latt = None
        try:
            space_group_code = int(v['_symmetry_Int_Tables_number'])
        except KeyError:
            try:
                space_group_code = int(v['_space_group_IT_number'])
            except:
                space_group_code = -1

        sites =[space_group_code]
        try:
            for i, name in enumerate(v['_atom_site_type_symbol']):
                sites.append([name, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
        except KeyError:
            try:
                for i, name in enumerate(v['_atom_site_label']):
                    sites.append([name, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
            except:
                sites = None
       
        try:
            symmetric_operation = list(v['_space_group_symop_operation_xyz'])
        except KeyError:
            try:
                symmetric_operation = list(v['_symmetry_equiv_pos_as_xyz'])
            except KeyError:
                symmetric_operation = None

        if '_symmetry_space_group_name_H-M' in v:
            symbol = v['_symmetry_space_group_name_H-M'][0]
            spaceG = v['_symmetry_space_group_name_H-M']
        elif '_symmetry_space_group_name_Hall' in v:
            symbol = v['_symmetry_space_group_name_Hall'][0]
            spaceG = v['_symmetry_space_group_name_Hall']
        else:
            symbol = None
            spaceG = None
        
    return latt, spaceG, sites, symbol,symmetric_operation
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

    grid = grid_atom()
    index_of_origin = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] == 0))[0][0]
    grid[[0, index_of_origin]] = grid[[index_of_origin, 0]]
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

        if x_ < 0 : x_ += 1 
        elif x_ >= 1 : x_ -= 1
        else : pass 

        if y_ < 0 : y_ += 1 
        elif y_ >= 1 : y_ -= 1
        else : pass

        if z_ < 0 : z_ += 1 
        elif z_ >= 1 : z_ -= 1
        else : pass

        x_ = get_float(x_,3)
        y_ = get_float(y_,3)
        z_ = get_float(z_,3)

        if 0 <= x_ <=1 and 0 <= y_ <=1 and 0 <= z_ <=1:
            ori_atom[atom][1] = x_  
            ori_atom[atom][2] = y_
            ori_atom[atom][3] = z_
            unit_cell_atom.append(ori_atom[atom])  
        else: pass
    
    unique_data = []
    for item in unit_cell_atom:
        # delete repeating points 
        if 1 not in item:
            trans_item = [item]
        else:
            trans_item = trans_(item)
        add = True
        for d in range(len(trans_item)):
            if trans_item[d] in unique_data: add = 0
        if add == 1:
            unique_data.append(item)
    return unique_data


def trans_(lst):
    result = []
    indices = [i for i, x in enumerate(lst) if x == 1]
    num_ones = len(indices)
    
    for i in range(2**num_ones):
        binary = bin(i)[2:].zfill(num_ones)
        new_lst = [0 if j in indices and binary[indices.index(j)] == '1' else x for j, x in enumerate(lst)]
        result.append(new_lst)
    
    return result

"""
unique_data = []
for item in unit_cell_atom:
    # delete repeating points 
    if item not in unique_data: # x,y,z
        coordinate = copy.deepcopy(item)
        print('coordinate',coordinate)
        # Coordinate rotation
        axis_x = coordinate[1]
        axis_y = coordinate[2]
        axis_z = coordinate[3]
        # y,z,x
        rot_first = [coordinate[0]]
        rot_first.append(axis_y) , 
        rot_first.append(axis_z) 
        rot_first.append(axis_x)
        # z,x,y
        rot_second = [coordinate[0]]
        rot_second.append(axis_z)
        rot_second.append(axis_x)
        rot_second.append(axis_y)
        print(rot_first)
        if rot_first not in unique_data and rot_second not in unique_data:
            unique_data.append(item)
"""


def UnitCellAtom(Asymmetric_atomic_coordinates,symmetric_operation):
    """
    Find all atomic positions in the unit cell
    """
    # Asymmetric_atomic_coordinates ---> [22,['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] 

    spg = Asymmetric_atomic_coordinates[0]
    Asymmetric_atomic_coordinates.pop(0)
    atom_loc = trans_atom(Asymmetric_atomic_coordinates,spg,symmetric_operation)
    return unit_cell_range(atom_loc)

# def fun for calculating extinction
# del the peak extincted
def cal_extinction(Point_group,HKL_list,dis_list,system,AtomCoordinates,wavelength,cal_extinction=True):
    if cal_extinction == False:
        return HKL_list,[],dis_list,[]
    else:
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
        if abs((HKL[0]+HKL[1]+HKL[2])) % 2 == 1:
            extinction = True
        else: pass
    elif lattice_type == 'C': # bottom center
        if system == 1 or system == 5: 
            if abs((HKL[0]+HKL[1])) % 2 == 1 or abs((HKL[1]+HKL[2])) % 2 == 1 or abs((HKL[0]+HKL[2])) % 2 == 1: 
                extinction = True
            else: pass
        else:
            if abs((HKL[0]+HKL[1])) % 2 == 1: 
                extinction = True
            else: pass
    elif lattice_type == 'F': # face center
        if (abs(HKL[0]) % 2 == 1 and abs(HKL[1]) % 2 == 1 and abs(HKL[2]) % 2 == 1) or (abs(HKL[0]) % 2 == 0 and abs(HKL[1]) % 2 == 0 and abs(HKL[2]) % 2 == 0):
            pass
        else: extinction = True
    else:
        print('ERROR : unknown lattice type: %s' % lattice_type)  
        sys.exit()
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


def trans_atom(atom_coordinate,sp_c,symmetric_operation):
    """
    atom_coordinate is the list of atoms in the shape of [['Cu2+',a0,b0,c0],['O2-',a1,b1,c1],...]
    sp_c is the code of space group, a int, e.g., 121
    """
    atom_loc = copy.deepcopy(atom_coordinate)
    if type(symmetric_operation) == list:
        print('atom locations claculated by parsed cif file')
        # move atom directly
        for atom in range(len(atom_coordinate)):
            # read in the asymmetric atoms coordinates
            a = atom_coordinate[atom][1]
            b = atom_coordinate[atom][2]
            c = atom_coordinate[atom][3]
            for k in range(len(symmetric_operation)):
                new_loc = [atom_coordinate[atom][0]]
                # Determine the type of operation
                variable = ['x','y','z']
                value = [a,b,c]
                loc = apply_operation(expression=symmetric_operation[k], variable=variable, value=value)
                new_loc.append(loc[0])
                new_loc.append(loc[1])
                new_loc.append(loc[2])
                atom_loc.append(new_loc)
        
    else:
        print('atom locations claculated by wyckoff site')
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
                new_loc.append(loc[0])
                new_loc.append(loc[1])
                new_loc.append(loc[2])
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



def grid_atom():
    hh, kk, ll = np.mgrid[-11:13:1, -11:13:1, -11:13:1]
    grid = np.c_[hh.ravel(), kk.ravel(), ll.ravel()]
    distances = np.linalg.norm(grid, axis=1)
    grid = grid[distances < 13]
    distances = distances[distances < 13]
    index_of_origin = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] == 0))[0][0]
    # Place the origin at the first position in the array
    grid[[0, index_of_origin]] = grid[[index_of_origin, 0]]
    distances[[0, index_of_origin]] = distances[[index_of_origin, 0]]
    sorted_indices = np.argsort(distances)
    grid = grid[sorted_indices]
    return grid


def cal_lattic_mass(structure_factor):
    # structure_factor:  ==> [['atom1',0,0,0],['atom2',0.5,1,1],.....]
    mass = 0
    for atom in structure_factor:
        _a = re.sub(r'[^A-Za-z]+', "", atom[0])
        result = find_atomic_mass(_a)
        if result is None:
            print(f"Element with symbol {_a} not found.")
            result = 0
        mass += result
    return mass


def find_atomic_mass(element_symbol):
    # def the relative atomic mass
    atomic_masses = {
        'H': 1.008,
        'He': 4.0026,
        'Li': 6.94,
        'Be': 9.0122,
        'B': 10.81,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'F': 18.998,
        'Ne': 20.180,
        'Na': 22.990,
        'Mg': 24.305,
        'Al': 26.982,
        'Si': 28.085,
        'P': 30.974,
        'S': 32.06,
        'Cl': 35.45,
        'K': 39.098,
        'Ar': 39.948,
        'Ca': 40.078,
        'Sc': 44.956,
        'Ti': 47.867,
        'V': 50.942,
        'Cr': 51.996,
        'Mn': 54.938,
        'Fe': 55.845,
        'Ni': 58.693,
        'Cu': 63.546,
        'Zn': 65.38,
        'Ga': 69.723,
        'Ge': 72.630,
        'As': 74.922,
        'Se': 78.971,
        'Br': 79.904,
        'Kr': 83.798,
        'Rb': 85.468,
        'Sr': 87.62,
        'Y': 88.906,
        'Zr': 91.224,
        'Nb': 92.906,
        'Mo': 95.95,
        'Tc': 98.0,
        'Ru': 101.07,
        'Rh': 102.91,
        'Pd': 106.42,
        'Ag': 107.87,
        'Cd': 112.41,
        'In': 114.82,
        'Sn': 118.71,
        'Sb': 121.76,
        'Te': 127.60,
        'I': 126.90,
        'Xe': 131.29,
        'Cs': 132.91,
        'Ba': 137.33,
        'La': 138.91,
        'Ce': 140.12,
        'Pr': 140.91,
        'Nd': 144.24,
        'Pm': 145.0,
        'Sm': 150.36,
        'Eu': 151.96,
        'Gd': 157.25,
        'Tb': 158.93,
        'Dy': 162.50,
        'Ho': 164.93,
        'Er': 167.26,
        'Tm': 168.93,
        'Yb': 173.04,
        'Lu': 174.97,
        'Hf': 178.49,
        'Ta': 180.95,
        'W': 183.84,
        'Re': 186.21,
        'Os': 190.23,
        'Ir': 192.22,
        'Pt': 195.08,
        'Au': 196.97,
        'Hg': 200.59,
        'Tl': 204.38,
        'Pb': 207.2,
        'Bi': 208.98,
        'Th': 232.04,
        'Pa': 231.04,
        'U': 238.03,
        'Np': 237.0,
        'Pu': 244.0,
        'Am': 243.0,
        'Cm': 247.0,
        'Bk': 247.0,
        'Cf': 251.0,
        'Es': 252.0,
        'Fm': 257.0,
        'Md': 258.0,
        'No': 259.0,
        'Lr': 262.0,
        'Rf': 267.0,
        'Db': 270.0,
        'Sg': 271.0,
        'Bh': 270.0,
        'Hs': 277.0,
        'Mt': 276.0,
        'Ds': 281.0,
        'Rg': 280.0,
        'Cn': 285.0,
        'Nh': 284.0,
        'Fl': 289.0,
        'Mc': 288.0,
        'Lv': 293.0,
        'Ts': 294.0,
        'Og': 294.0,
        'Co': 58.933,
    }

    element_symbol = element_symbol.capitalize()

    if element_symbol in atomic_masses:
        return atomic_masses[element_symbol]
    else:
        return None  

