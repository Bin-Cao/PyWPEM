# XRD simulation for a sigle crystal 
# Author: Bin CAO <binjacobcao@gmail.com>

from sympy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import re
from ..Extinction.XRDpre import profile
from .DiffractionGrometry.atom import atomics
from ..EMBraggOpt.WPEMFuns.SolverFuns import theta_intensity_area
        
class XRD_profile(object):
    def __init__(self,filepath,wavelength='CuKa',two_theta_range=(10, 90,0.01), LatticCs = None,PeakWidth=False, CSWPEMout = None):
        # filepath : the path of the cif file
        # CSWPEMout : Crystal System WPEMout file
        # PeakWidth=False, The peak width of the simulated peak is 0
        # PeakWidth=True, The peak width of the simulated peak is set to the peak obtained by WPEM
        # LatticCs : the lattic constants after WPEM refinement, default = None, if None,
        # WPEM reads lattice constants from an input cif file 
        # read parameters from cif by ..Extinction.XRDpre
        _range = (two_theta_range[0],two_theta_range[1])
        if LatticCs == None:
            LatticCs, Atom_coordinate = profile(wavelength,_range).generate(filepath)
        elif type(LatticCs) == list and len(LatticCs) == 6 : 
            _, Atom_coordinate = profile(wavelength,_range).generate(filepath)
        else : print('Type Error of Param LatticCs')
        print('\n')
        self.LatticCs = LatticCs
        self.two_theta_range = two_theta_range 

        self.crystal_system = det_system(LatticCs)
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
            peak = pd.read_csv('./output_xrd/{}HKL.csv'.format(filepath[-11:-4]))
            self.mu_list = peak['2theta/TOF'].tolist()
            self.Mult = peak['Mult'].tolist()
            self.HKL_list = np.array(peak[['H','K','L']]).tolist()
            print('Initilized witout peak\'s shape')
        elif PeakWidth==True:
            if type(CSWPEMout) != str:
                print('Please provide the decomposed peak parameters of WPEM')
            else:
                peak = pd.read_csv('./output_xrd/{}HKL.csv'.format(filepath[-11:-4]))
                data = pd.read_csv(CSWPEMout)
                self.mu_list = data['mu_i'].tolist() 
                self.gamma_list = data['L_gamma_i'].tolist() 
                self.sigma2_list = data['G_sigma2_i'].tolist()
                self.Mult = peak['Mult'].tolist()
                self.HKL_list = np.array(peak[['H','K','L']]).tolist()
                print('Initilized with peak\'s shape')
        self.PeakWidth = PeakWidth
        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 15
        os.makedirs('Simulation_WPEM', exist_ok=True)

    def Simulate(self,plot=True, write_in = True):
        FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
        for angle in range(len(self.HKL_list)):
            FHKL_square_left = 0
            FHKL_square_right = 0
            for atom in range(len(self.Atom_coordinate)):
                fi = cal_atoms(self.Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength)
                FHKL_square_left += fi * np.cos(2 * np.pi * (self.Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                    self.Atom_coordinate[atom][2] * self.HKL_list[angle][1] + self.Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                FHKL_square_right += fi * np.sin(2 * np.pi * (self.Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                    self.Atom_coordinate[atom][2] * self.HKL_list[angle][1] + self.Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
            FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)

        # cal unit cell volume
        VolumeFunction = LatticVolume(self.crystal_system)
        sym_a, sym_b, sym_c, angle1, angle2, angle3 = symbols('sym_a sym_b sym_c angle1 angle2 angle3')
        Volume = (float(VolumeFunction.subs(
            {sym_a: self.LatticCs[0], sym_b: self.LatticCs[1], sym_c: self.LatticCs[2],
                angle1: self.LatticCs[3] * np.pi/180 , angle2: self.LatticCs[4] * np.pi/180, angle3: self.LatticCs[5] * np.pi/180})))

        # I = C / (V0 ** 2) * F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta))
        # without considering the temperature and line absorption factor
        Ints = []
        for angle in range(len(FHKL_square)):
            Ints.append(float(FHKL_square[angle] * self.Mult[angle] / Volume ** 2
                        * (1 + np.cos(self.mu_list[angle] * np.pi/180) ** 2) / (np.sin(self.mu_list[angle] / 2 * np.pi/180) **2 * np.cos(self.mu_list[angle] / 2 * np.pi/180))))
        
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

        if plot == True:
            # save simulation results
            plt.plot(x_sim, nor_y, '-g', label= "Simulated Profile (crystal)", )
            plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            plt.legend()
            plt.savefig('./Simulation_WPEM/Simulation_profile.png', dpi=800)
            plt.show()
            plt.clf()
        else: pass
        

        if write_in == True:
            # Save the simulated peak
            res = []
            for i in range(len(Ints)):
                res.append([i+1, self.HKL_list[i][0], self.HKL_list[i][1], self.HKL_list[i][2], self.Mult[i], self.mu_list[i],Ints[i]])
            res.insert(0, ['No.', 'H', 'K', 'L', 'Mult', '2theta/','Ints/'])
            save_file = 'Simulation_WPEM/Bragg_peaks.csv'
            dataFile = open(save_file, 'w')
            dataWriter = csv.writer(dataFile)
            dataWriter.writerows(res)
            dataFile.close()

            profile = []
            for i in range(len(x_sim)):
                profile.append([i+1, x_sim[i], nor_y[i]])
            profile.insert(0, ['No.', 'x_simu', 'y_simu'])
            save_file = 'Simulation_WPEM/Simu_profile.csv'
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
    peak_density = Weight * (lorenz_density(x, mu, gamma) + normal_density(x, mu, sigma2))
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
            Volume = sym_a ** 2 * sym_c * np.sqrt(3) / 2
        elif crystal_system == 3:  # Tetragonal
            Volume = sym_a * sym_a * sym_c
        elif crystal_system == 4:  # Orthorhombic
            Volume = sym_a * sym_b * sym_c
        elif crystal_system == 5:  # Rhombohedral
            Volume = sym_a ** 3 * np.sqrt(1 - 3 * cos(angle1) ** 2 + 2 * cos(angle1) ** 3)
        elif crystal_system == 6:  # Monoclinic
            Volume = sym_a * sym_b * sym_c * sin(angle2)
        elif crystal_system == 7:  # Triclinic
            Volume = sym_a * sym_b * sym_c * np.sqrt(1 - cos(angle1) ** 2 - cos(angle2) **2
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







