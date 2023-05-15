# XRD simulation
# Author: Bin CAO <binjacobcao@gmail.com>

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
from ..Extinction.XRDpre import det_system
from .DiffractionGrometry.atom import atomics
        
class XRD_profile(object):
    def __init__(self,structure_factor,mu_list,gamma_list, sigma2_list, Mult, HKL_list,  LatticCs, Wavelength=1.54184):
        self.crystal_system = det_system(LatticCs)
        self.structure_factor = structure_factor #  ==> [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
        # len(Theta_list) = len(Mult) = len(HKL_list)
        self.mu_list = mu_list # ==> calculated mui
        self.gamma_list = gamma_list # ==> calculated gamma
        self.sigma2_list = sigma2_list # ==> calculated sigma2
        self.Mult = Mult
        self.HKL_list = HKL_list # [H,K,L] in shape of n*3
        self.LatticCs = LatticCs # [a,b,c,alpha,beta,gamma]
        self.wavelength = Wavelength
        """
        atomic_dif = pd.read_csv('./Atom_f.csv',index_col=0)
        self.atomic_dif = atomic_dif.to_dict('index')
        print(self.atomic_dif)
        """
        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 18 
    
        os.makedirs('Simulation_WPEM', exist_ok=True)

    def Simulate(self, two_theta_range=(0, 90,0.02)):
        """
        two_theta_range : The range of the generated simulation pattern
        """
        FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
        
        for angle in range(len(self.HKL_list)):
            FHKL_square_left = 0
            FHKL_square_right = 0
            for atom in range(len(self.structure_factor)):
                fi = cal_atoms(self.structure_factor[atom][0],self.mu_list[angle], self.wavelength)
                FHKL_square_left += fi * np.cos(2 * np.pi * (self.structure_factor[atom][1] * self.HKL_list[angle][0] +
                                                    self.structure_factor[atom][2] * self.HKL_list[angle][1] + self.structure_factor[atom][3] * self.HKL_list[angle][2]))
                FHKL_square_right += fi * np.sin(2 * np.pi * (self.structure_factor[atom][1] * self.HKL_list[angle][0] +
                                                    self.structure_factor[atom][2] * self.HKL_list[angle][1] + self.structure_factor[atom][3] * self.HKL_list[angle][2]))
            FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)


        # cal unit cell volume
        VolumeFunction = self.LatticVolume()
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

        x_sim = np.arange(two_theta_range[0],two_theta_range[1],two_theta_range[2])

        y_sim = 0
        for num in range(len(Ints)):
            _ = draw_peak_density(x_sim, Ints[num], self.mu_list[num], self.gamma_list[num], self.sigma2_list[num])
            y_sim += _

        # save simulation results
        plt.plot(x_sim, y_sim, '-g', label= "Simulated  Profile (crystal)", )
        plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig('./Simulation_WPEM/Simulation_profile.png', dpi=800)
        plt.show()
        plt.clf()
        
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
            profile.append([i+1, x_sim[i], y_sim[i]])
        profile.insert(0, ['No.', 'x_simu', 'y_simu'])
        save_file = 'Simulation_WPEM/Simu_profile.csv'
        dataFile = open(save_file, 'w')
        dataWriter = csv.writer(dataFile)
        dataWriter.writerows(profile)
        dataFile.close()
        print('XRD simulation process of WPEM is completed !')
        return FHKL_square
    
    def LatticVolume(self, ):
        """
        returns the unit cell volume
        """
        sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_a sym_b sym_c angle1 angle2 angle3')
        if self.crystal_system == 1:  # Cubic
            Volume = sym_a ** 3
        elif self.crystal_system == 2:  # Hexagonal
            Volume = sym_a ** 2 * sym_c * np.sqrt(3) / 2
        elif self.crystal_system == 3:  # Tetragonal
            Volume = sym_a * sym_a * sym_c
        elif self.crystal_system == 4:  # Orthorhombic
            Volume = sym_a * sym_b * sym_c
        elif self.crystal_system == 5:  # Rhombohedral
            Volume = sym_a ** 3 * np.sqrt(1 - 3 * cos(angle1) ** 2 + 2 * cos(angle1) ** 3)
        elif self.crystal_system == 6:  # Monoclinic
            Volume = sym_a * sym_b * sym_c * sin(angle2)
        elif self.crystal_system == 7:  # Triclinic
            Volume = sym_a * sym_b * sym_c * np.sqrt(1 - cos(angle1) ** 2 - cos(angle2) **2
                                              - cos(angle3) ** 2 + 2 * cos(angle1) * cos(angle2) * cos(angle3))
        else:
            Volume = -1
        return Volume


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










