# XRD simulation for a sigle crystal 
# Author: Bin CAO <binjacobcao@gmail.com>

import sympy
from sympy import symbols, cos, sin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import re
import random
import math
import itertools
from scipy.special import wofz
from ..Extinction.XRDpre import profile
from .DiffractionGrometry.atom import atomics
from ..EMBraggOpt.WPEMFuns.SolverFuns import theta_intensity_area
        
class XRD_profile(object):
    def __init__(self,filepath,wavelength='CuKa',two_theta_range=(10, 90,0.01),SuperCell=False,PeriodicArr=[3,3,3],ReSolidSolution = None, RSSratio=0.1,  
                 GrainSize = None, LatticCs = None,PeakWidth=True, CSWPEMout = None,work_dir=None):
        """
        # filepath: The path of the CIF file
        
        # GrainSize: The default value is 'none,' or you can input a float representing the grain size within a range of 5-30 nanometers.

        # CSWPEMout: Crystal System WPEMout file

        # PeakWidth=False: The peak width of the simulated peak is set to 0.
        
        # ReSolidSolution : list, default None
            If not None, should contain the original atom type and replace atom type
            e.g., ReSolidSolution = ['Ru4+', 'Na2+'], means 'Na2+' replaces the 'Ru4+' atom locations

        # PeakWidth=True: The peak width of the simulated peak is set to the peak obtained by WPEM.

        # LatticCs: The lattice constants after WPEM refinement. The default is None. If set to None, WPEM reads lattice constants from an input CIF file. Read parameters from CIF by using ..Extinction.XRDpre.

        """
        if work_dir is None:
            work_dir = os.getcwd()
        self.ReSolidSolution = ReSolidSolution
        self.RSSratio = RSSratio
        if type(ReSolidSolution) == list:
            _range = (two_theta_range[0],two_theta_range[1])
            if LatticCs == None:
                LatticCs, Atom_coordinate,_ = profile(wavelength,_range,cal_extinction = False).generate(filepath)
            elif type(LatticCs) == list and len(LatticCs) == 6 : 
                _, Atom_coordinate = profile(wavelength,_range,cal_extinction = False).generate(filepath)
            else : print('Type Error of Param LatticCs')
        else:
            _range = (two_theta_range[0],two_theta_range[1])
            if LatticCs == None:
                LatticCs, Atom_coordinate,_ = profile(wavelength,_range).generate(filepath)
            elif type(LatticCs) == list and len(LatticCs) == 6 : 
                _, Atom_coordinate = profile(wavelength,_range).generate(filepath)
            else : print('Type Error of Param LatticCs')
        print('\n')

        self.two_theta_range = two_theta_range 
        self.filepath = filepath
        self.crystal_system = det_system(LatticCs)
       
        self.PeriodicArr = PeriodicArr
        if SuperCell:
            print('-----SuperCell is cofigured-----')
            self.Atom_coordinate = generate_super_cell(Atom_coordinate,PeriodicArr[0],PeriodicArr[1],PeriodicArr[2])
            a = LatticCs[0] * PeriodicArr[0]
            b = LatticCs[1] * PeriodicArr[1]
            c = LatticCs[2] * PeriodicArr[2]
            self.LatticCs = [a,b,c,LatticCs[3],LatticCs[4],LatticCs[5]]
        else :
            self.LatticCs = LatticCs
            self.Atom_coordinate = Atom_coordinate   
            #  i.e., [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....] 

        if isinstance(wavelength, (float, int)):
            self.wavelength = wavelength
        elif isinstance(wavelength, str):
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        else:
            raise TypeError("'wavelength' must be either of: float, int or str")
        
        # generate delta function
        if PeakWidth==False:
            peak = pd.read_csv(os.path.join(work_dir,'output_xrd','{}HKL.csv'.format(filepath[-11:-4])))
            self.mu_list = peak['2theta/TOF'].tolist()
            self.Mult = peak['Mult'].tolist()
            self.HKL_list = np.array(peak[['H','K','L']]).tolist()
            print('Initilized witout peak\'s shape')
        elif PeakWidth==True:
            if type(CSWPEMout) != str:
                print('WPEM simulates the peaks as the default Voigt functions')
                peak = pd.read_csv(os.path.join(work_dir,'output_xrd','{}HKL.csv'.format(filepath[-11:-4])))
                self.mu_list = peak['2theta/TOF'].tolist()
                self.Mult = peak['Mult'].tolist()
                self.HKL_list = np.array(peak[['H','K','L']]).tolist()
                if GrainSize is not None:
                    Γ = 0.888*self.wavelength/(GrainSize*np.cos(np.radians(np.array(self.mu_list)/2)))
                    self.gamma_list = Γ / 2 + 1e-10
                    self.sigma2_list = Γ**2 / (8*np.sqrt(2)) + 1e-10
                else:
                    self.gamma_list = 0.1 * np.ones(len(self.mu_list))
                    self.sigma2_list = np.zeros(len(self.mu_list))
                print('Initilized with default peak\'s shape')

            else:
                # readin the refined parameters
                print('WPEM simulates the peaks by the decomposed peak shapes')
                peak = pd.read_csv(os.path.join(work_dir,'output_xrd','HKL.csv'.format(filepath[-11:-4])))
                data = pd.read_csv(CSWPEMout)
                self.mu_list = data['mu_i'].tolist() 
                self.gamma_list = data['L_gamma_i'].tolist() 
                self.sigma2_list = data['G_sigma2_i'].tolist()
                self.Mult = peak['Mult'].tolist()
                self.HKL_list = np.array(peak[['H','K','L']]).tolist()
                print('Initilized with decomposed peak\'s shape')

        self.PeakWidth = PeakWidth
        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        
        self.Simfolder = os.path.join(work_dir, 'Simulation_WPEM')

        os.makedirs(self.Simfolder, exist_ok=True)


    def Simulate(self,plot=True, write_in = True, Vacancy=False, Vacancy_atom = None, Vacancy_ratio = None,orientation=None,thermo_vib=None,zero_shift = None, bacI=False,seed=10):
        """
        orientation: The default value is 'none,' or you can input a list such as [-0.2, 0.3], adjusting intensity within the range of (1-20%)I to (1+30%)I.

        thermo_vib: The default is 'none,' or you can input a float, for example, thermo_vib=0.05, representing the variability in the average atom position. It is recommended to use values between 0.05 and 0.5 angstrom.

        zero_shift: The default is 'none,' or you can input a float, like zero_shift=1.5, which represents the instrument zero shift. It is recommended to use values between 2θ = -3 and 3 degrees.

        bacI: The default is False. If bacI = True, a three-degree polynomial function is applied to simulate the background intensity.
        """

        if self.ReSolidSolution == None: 
            _Atom_coordinate = self.Atom_coordinate
            Latticematrix = lattice_parameters_to_matrix(self.LatticCs[0], self.LatticCs[1], self.LatticCs[2], self.LatticCs[3], self.LatticCs[4], self.LatticCs[5])
            print("The nearest neighbor atoms are :",find_closest_atoms(group_elements_by_first_element(_Atom_coordinate), Latticematrix))
            write_vasp_file(Latticematrix, group_elements_by_first_element(_Atom_coordinate), os.path.join(self.Simfolder,'CrySits_{}.vasp'.format(self.filepath[-11:-4])))
            
        elif type(self.ReSolidSolution) == list: 
            _Atom_coordinate = ReplaceAtom(self.Atom_coordinate,self.ReSolidSolution,self.RSSratio,Vacancy, Vacancy_atom, Vacancy_ratio,self.LatticCs,seed)
            Latticematrix = lattice_parameters_to_matrix(self.LatticCs[0]*self.PeriodicArr[0], self.LatticCs[1]*self.PeriodicArr[1], self.LatticCs[2]*self.PeriodicArr[2], self.LatticCs[3], self.LatticCs[4], self.LatticCs[5])
            print("The nearest neighbor atoms are :",find_closest_atoms(group_elements_by_first_element(_Atom_coordinate), Latticematrix))
            write_vasp_file(Latticematrix, group_elements_by_first_element(_Atom_coordinate), os.path.join(self.Simfolder,'CrySits_{}.vasp'.format(self.filepath[-11:-4])),self.PeriodicArr)

        else: print("ReSolidSolution is not a list")
        FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
        if self.ReSolidSolution == None: 
            for angle in range(len(self.HKL_list)):
                FHKL_square_left = 0
                FHKL_square_right = 0
                for atom in range(len(_Atom_coordinate)):
                    fi = cal_atoms(_Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength)
                    FHKL_square_left += fi * np.cos(2 * np.pi * (_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                        _Atom_coordinate[atom][2] * self.HKL_list[angle][1] + _Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                    FHKL_square_right += fi * np.sin(2 * np.pi * (_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                        _Atom_coordinate[atom][2] * self.HKL_list[angle][1] + _Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)
        
        elif type(self.ReSolidSolution) == list:
            for angle in range(len(self.HKL_list)):
                FHKL_square_left = 0
                FHKL_square_right = 0
                for atom in range(len(_Atom_coordinate)):
                    fi = cal_atoms(_Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength)
                    FHKL_square_left += fi * np.cos(2 * np.pi * (_Atom_coordinate[atom][1] * self.HKL_list[angle][0]*self.PeriodicArr[0] +
                                                        _Atom_coordinate[atom][2] * self.HKL_list[angle][1]*self.PeriodicArr[1] + _Atom_coordinate[atom][3] * self.HKL_list[angle][2]*self.PeriodicArr[2]))
                    FHKL_square_right += fi * np.sin(2 * np.pi * (_Atom_coordinate[atom][1] * self.HKL_list[angle][0]*self.PeriodicArr[0] +
                                                        _Atom_coordinate[atom][2] * self.HKL_list[angle][1]*self.PeriodicArr[1] + _Atom_coordinate[atom][3] * self.HKL_list[angle][2]*self.PeriodicArr[2]))
                FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)
        # cal unit cell volume
        VolumeFunction = LatticVolume(self.crystal_system)
        sym_a, sym_b, sym_c, angle1, angle2, angle3 = symbols('sym_a sym_b sym_c angle1 angle2 angle3')
        Volume = (float(VolumeFunction.subs(
            {sym_a: self.LatticCs[0], sym_b: self.LatticCs[1], sym_c: self.LatticCs[2],
                angle1: self.LatticCs[3] * np.pi/180 , angle2: self.LatticCs[4] * np.pi/180, angle3: self.LatticCs[5] * np.pi/180})))

        # I = C / (V0 ** 2) * F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta))
        # without considering the temperature and line absorption factor
        _Ints = []
        for angle in range(len(FHKL_square)):
            _Ints.append(float(FHKL_square[angle] * self.Mult[angle] / Volume ** 2
                        * (1 + np.cos(self.mu_list[angle] * np.pi/180) ** 2) / (np.sin(self.mu_list[angle] / 2 * np.pi/180) **2 * np.cos(self.mu_list[angle] / 2 * np.pi/180))))
        
        # augmentation
        if orientation is not None and thermo_vib is None:
            Ints = []
            for k in range(len(_Ints)):
                Ints.append(_Ints[k] * np.clip(np.random.normal(loc=1, scale=0.2), 1-orientation[0], 1+orientation[0]))

        elif orientation is not None and thermo_vib is not None:
            Ints = []
            for k in range(len(_Ints)):
                Ori_coe = np.clip(np.random.normal(loc=1, scale=0.2), 1-orientation[0], 1+orientation[0])
                M = 8/3 * np.pi**2*thermo_vib**2 * (np.sin(np.radians(self.mu_list[k]/2)) / self.wavelength)**2
                Deb_coe = np.exp(-2*M)
                Ints.append(_Ints[k] * Ori_coe * Deb_coe)
        else:
            Ints = _Ints
        
        if self.PeakWidth == True:
            x_sim = np.arange(self.two_theta_range[0],self.two_theta_range[1],self.two_theta_range[2])
            y_sim = 0
            for num in range(len(Ints)):
                _ = draw_peak_density(x_sim, Ints[num], self.mu_list[num], self.gamma_list[num], self.sigma2_list[num])
                y_sim += _
            # normalize the profile
            nor_y = y_sim / theta_intensity_area(x_sim,y_sim)
        elif self.PeakWidth == False:
            _x_sim = np.arange(self.two_theta_range[0],self.two_theta_range[1],self.two_theta_range[2])
            x_sim,y_sim = cal_delta_peak(self.mu_list,Ints,_x_sim)
            # normalize the profile
            nor_y = y_sim / y_sim.sum()

       
        if zero_shift is not None:
            x_sim += zero_shift
            self.mu_list = np.array(self.mu_list) + zero_shift
        if bacI == True:
            random_polynomial = generate_random_polynomial(degree=6)
            _bac = random_polynomial(x_sim)
            _bac -= _bac.min()
            _bacI = _bac / _bac.max() * nor_y.max() * 0.1
            nor_y +=  np.flip(_bacI)
       
        nor_y = scale_list(nor_y)
        if plot == True:
            # save simulation results
            fig, ax = plt.subplots()
            ax.plot(x_sim, nor_y, '-k', label= "WPEM simulation", )
            ax.scatter(x_sim, nor_y, s=3,c='r', alpha=0.5,label= "signals", )
            for x in self.mu_list:
                ax.vlines(x=x, ymin=-10, ymax=-5, color='b', linestyle='-',)
            plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            ax.legend()
            plt.savefig(os.path.join(self.Simfolder,'{}_Simulation_profile.png'.format(self.filepath[-11:-4])), dpi=800)
            plt.savefig(os.path.join(self.Simfolder,'{}_Simulation_profile.svg'.format(self.filepath[-11:-4])), dpi=800)
            plt.show()
            plt.clf()
        else: pass
        

        if write_in == True:
            # Save the simulated peak
            res = []
            for i in range(len(Ints)):
                res.append([i+1, self.HKL_list[i][0], self.HKL_list[i][1], self.HKL_list[i][2], self.Mult[i], self.mu_list[i],Ints[i]])
            res.insert(0, ['No.', 'H', 'K', 'L', 'Mult', '2theta/','Ints/'])
            save_file = os.path.join(self.Simfolder,'{}_Bragg_peaks.csv'.format(self.filepath[-11:-4]))
            dataFile = open(save_file, 'w')
            dataWriter = csv.writer(dataFile)
            dataWriter.writerows(res)
            dataFile.close()

            profile = []
            for i in range(len(x_sim)):
                profile.append([i+1, x_sim[i], nor_y[i]])
            profile.insert(0, ['No.', 'x_simu', 'y_simu'])
            save_file = os.path.join(self.Simfolder,'{}_Simu_profile.csv'.format(self.filepath[-11:-4]))
            dataFile = open(save_file, 'w')
            dataWriter = csv.writer(dataFile)
            dataWriter.writerows(profile)
            dataFile.close()
            print('XRD simulation process of WPEM is completed !')
        else: pass

        return FHKL_square,x_sim,nor_y
    


def get_float(f_str, n):
    # Define a function called get_float with two parameters: f_str (a string or float value to be processed), and n (an integer representing the number of decimal places to keep).
    f_str = str(f_str)
    # Convert f_str to a string, in case it was initially a float.
    a, _, c = f_str.partition('.')
    # Partition f_str into three parts: the integer part (a), the decimal point ('.'), and the fractional part (c). If there is no decimal point, c will be an empty string.
    c = (c+"0"*n)[:n]
    # Add zeros to the end of the fractional part of the number until it has n digits, then slice off any extra digits beyond the nth digit.
    return float(".".join([a, c]))
    # Combine the integer part and the modified fractional part into a new string with a decimal point, and convert this new string to a float. Return the result.

# Normal distribution
def normal_density( x, mu, sigma2):
    """
    :param x: sample data (2theta)
    :param mu: mean (μi)
    :param sigma2: variance (σi^2)
    :return: Return the probability density of Normal distribution x~N(μi,σi^2)
    """
    density = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
    return density

# Lorenz distribution
def lorenz_density(x, mu, gamma):
    """
    :param x: sample data (2theta)
    :param mu: mean (μi)
    :param gamma: FWHM of Lorenz distribution
    :return: Return the probability density of Lorenz distribution
    """
    density = (1 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
    return density

def draw_peak_density(x, Weight, mu, gamma, sigma2):
    if sigma2 == 0:
        peak_density = Weight * lorenz_density(x, mu, gamma)
    else:
        z = ((x-mu) + 1j * gamma) / (np.sqrt(sigma2) * np.sqrt(2))
        Voigt = np.real(wofz(z) / (np.sqrt(sigma2) * np.sqrt(2 * np.pi)))
        peak_density = Weight * Voigt
    return peak_density

def getHeavyatom(s):
    """
    Some atomic ionization forms not defined in the table are replaced by their unionized forms
    """
    # Define a function called getHeavyatom that takes one parameter: s, a string that contains letters and/or non-letter characters.
    return re.sub(r'[^A-Za-z]+', "", s)
    # Use the re.sub() function to replace all non-letter characters in s with an empty string. Return the modified string.

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
        # Plan to replaces with Thomas-Fermi method
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

def cal_delta_peak(mu_list,Ints_list,_x_sim):
    # find peak's location
    nearest_indices = []
    for num in mu_list:
        nearest_index = np.abs(_x_sim - num).argmin()
        nearest_indices.append(nearest_index)

    # cal intensity
    peak_inten = np.zeros_like(_x_sim)
    for i, index in enumerate(nearest_indices):
        peak_inten[index] = Ints_list[i]
    return _x_sim,peak_inten

def LatticVolume(crystal_system):
        """
        returns the unit cell volume
        """
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
            Volume = sym_a ** 3 * sympy.sqrt(1 - 3 * cos(angle1) ** 2 + 2 * cos(angle1) ** 3)
        elif crystal_system == 6:  # Monoclinic
            Volume = sym_a * sym_b * sym_c * sin(angle2)
        elif crystal_system == 7:  # Triclinic
            Volume = sym_a * sym_b * sym_c * sympy.sqrt(1 - cos(angle1) ** 2 - cos(angle2) **2
                                              - cos(angle3) ** 2 + 2 * cos(angle1) * cos(angle2) * cos(angle3))
        else:
            Volume = -1
        return Volume

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

def scale_list(lst):
    max_value = max(lst)
    if max_value == 0:
        return [0] * len(lst)
    scaled_list = [x * (100 / max_value) for x in lst]
    return scaled_list


def generate_random_polynomial(degree):
    coefficients = np.random.randn(degree + 1)
    return np.poly1d(coefficients)



def generate_super_cell(input_coordinates, x_cells, y_cells, z_cells):
    super_cell = []

    for i in range(x_cells):
        for j in range(y_cells):
            for k in range(z_cells):
                for atom in input_coordinates:
                    atom_type = atom[0]
                    x_original, y_original, z_original = atom[1], atom[2], atom[3]

                    x_new = x_original + i
                    y_new = y_original + j
                    z_new = z_original + k

                    super_cell.append([atom_type, x_new, y_new, z_new])
    return super_cell



def ReplaceAtom(Atom_coordinate, ReSolidSolution, RSSratio,Vacancy=False, Vacancy_atom = None, Vacancy_ratio = None,latticC=None,seed=10):
    """
    For a single type of vacancy atom
    """
    random.seed(seed)
    if Vacancy == True and type(Vacancy_atom) == str and type(Vacancy_ratio) == int:
        latticeMatrix = lattice_parameters_to_matrix(latticC[0], latticC[1], latticC[2], latticC[3], latticC[4], latticC[5])
        # Count the number of atoms to be replaced
        num_atoms_to_replace = round(len([atom for atom in Atom_coordinate if atom[0] == ReSolidSolution[0]]) * RSSratio)

        # Count the number of vacancy atom
        num_vacancy_atoms = num_atoms_to_replace * Vacancy_ratio
        
        # Get the indices of the atoms to be replaced
        indices_to_replace = [i for i, atom in enumerate(Atom_coordinate) if atom[0] == ReSolidSolution[0]]
        
        # Randomly select indices to replace
        indices_replaced = random.sample(indices_to_replace, num_atoms_to_replace)

        indices_removed = []
        # Replace the atoms at the selected indices
        for index in indices_replaced:
            Atom_coordinate[index][0] = ReSolidSolution[1]
            indices_removed.append(find_nearest_vacancy_index(Atom_coordinate, Atom_coordinate[index],Vacancy_atom,num_vacancy_atoms,latticeMatrix))
        Atom_coordinate = [x for i, x in enumerate(Atom_coordinate) if i not in indices_removed[0]]
        
    else : 
        # Count the number of atoms to be replaced
        num_atoms_to_replace = round(len([atom for atom in Atom_coordinate if atom[0] == ReSolidSolution[0]]) * RSSratio)
        
        # Get the indices of the atoms to be replaced
        indices_to_replace = [i for i, atom in enumerate(Atom_coordinate) if atom[0] == ReSolidSolution[0]]
        
        # Randomly select indices to replace
        indices_replaced = random.sample(indices_to_replace, num_atoms_to_replace)
        
        # Replace the atoms at the selected indices
        for index in indices_replaced:
            Atom_coordinate[index][0] = ReSolidSolution[1]
    
    return Atom_coordinate


def distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


def find_nearest_vacancy_index(Atom_coordinate, target_atom,vacancy_atom,num,latticeMatrix):
    distances = []
    for i, atom in enumerate(Atom_coordinate):
        if atom[0] == vacancy_atom:
            dist = distance(np.dot(atom[1:],latticeMatrix), np.dot(target_atom[1:],latticeMatrix))
            distances.append((dist, i))
    
    distances.sort()  # Sort distances
    return [index for _, index in distances[:num]] 


def lattice_parameters_to_matrix(a, b, c, alpha, beta, gamma):
    a1 = a
    b1 = b * math.cos(math.radians(gamma))
    b2 = b * math.sin(math.radians(gamma))
    c1 = c * math.cos(math.radians(beta))
    c2 = c *(math.cos(math.radians(alpha)) - math.cos(math.radians(beta)) * math.cos(math.radians(gamma))) /  math.sin(math.radians(gamma))
    al2 = math.cos(math.radians(alpha)) **2 
    bet2 = math.cos(math.radians(beta)) **2 
    gam2 = math.cos(math.radians(gamma)) **2 
    c3 = c * math.sqrt(1 + 2 * math.cos(math.radians(alpha))*math.cos(math.radians(beta))*math.cos(math.radians(gamma)) - al2 - bet2 - gam2) / math.sin(math.radians(gamma))
    
    lattice_matrix = [[round(a1,8), 0, 0],
                      [round(b1,8),round(b2,8), 0],
                      [round(c1,8), round(c2,8), round(c3,8)]]
    
    return lattice_matrix



def lattice_matrix_to_lattice_constants(lattice_vectors):
    lattice_vectors = np.array(lattice_vectors)
    a = np.linalg.norm(lattice_vectors[0])
    b = np.linalg.norm(lattice_vectors[1])
    c = np.linalg.norm(lattice_vectors[2])

    alpha = np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c))
    beta = np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c))
    gamma = np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b))

    return a, b, c, alpha, beta, gamma


def write_vasp_file(matrix, atom, filename,PeriodicArr=[1,1,1]):
    with open(filename, 'w') as file:
        atom_count = count_atoms(atom)
        file.write('WPEM developed by BinCAO (HKUST(GZ)) https://github.com/Bin-Cao/WPEM' + '\n')
        file.write(' 1.0000000000000000' + '\n')

        for row in matrix:
            file.write(f'    {row[0]}    {row[1]}    {row[2]}\n')

        file.write(' '.join(atom_count.keys()) + '\n')
        file.write(' '.join(str(value) for value in atom_count.values()) + '\n')
        
        file.write('Direct' + '\n')

        for atom_info in atom:
            file.write(f'  {atom_info[1]/PeriodicArr[0]} {atom_info[2]/PeriodicArr[1]} {atom_info[3]/PeriodicArr[2]}\n')

def count_atoms(atom):
    atom_count = {}
    for atom_info in atom:
        atom_type = getHeavyatom(atom_info[0])
        if atom_type in atom_count:
            atom_count[atom_type] += 1
        else:
            atom_count[atom_type] = 1
    return atom_count

def group_elements_by_first_element(input_list):
    result = {}
    for item in input_list:
        key = item[0]
        if key in result:
            result[key].append(item)
        else:
            result[key] = [item]
    output_list = []
    for key in result:
        output_list += result[key]
    return output_list

def find_closest_atoms(atom_data, latticeMatrix):
    min_distance = float('inf')
    closest_atoms = []

    for pair in itertools.combinations(atom_data, 2):
        atom1, coord1 = pair[0][0], pair[0][1:]
        atom2, coord2 = pair[1][0], pair[1][1:]
       
        dist = distance(np.dot(coord1,latticeMatrix), np.dot(coord2,latticeMatrix) )
        if dist < min_distance:
            min_distance = dist
            closest_atoms = [atom1, atom2, np.round(dist,4)]

    return closest_atoms