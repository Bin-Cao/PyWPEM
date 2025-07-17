# Defining the effect of the structure factor on the diffraction intensity
# Author: Bin CAO <binjacobcao@gmail.com>

from sympy import *
import numpy as np
import pandas as pd
import os
import re
from ..XRDSimulation.Simulation import cal_atoms
from ..Extinction.XRDpre import find_atomic_mass

class VandMFraction:
    def __init__(self, timename,work_dir=None): 
        timename = timename
        self.namey = timename.tm_year
        self.nameM = timename.tm_mon
        self.named = timename.tm_mday
        self.nameh = timename.tm_hour
        self.namem = timename.tm_min
        if work_dir is None:
            self.FRfolder = 'WPEMFittingResults'
        else:
            self.FRfolder = os.path.join(work_dir, 'WPEMFittingResults')
    
    # Calculate the volume of the unit cell
    def LatticVolume(self, crystal_system):
        # imput the number of crystal_system
        crystal_system = crystal_system
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

    def get_Volome_Fraction(self, structure_factor, crystal_sys_list, Multi_list, HKL_list, Theta_list, Intensity_list, LatticCs_list,Wavelength):
        """
        Calculation of the crystal mass involved in the diffraction by the structure factor
        Input the information of the first few strongest peaks, the input format is as follows
        """
        # structure_factor:  ==> [ [['atom1',0,0,0],['atom2',0.5,1,1],.....], [['atom1',0,0,0],['atom2',0.5,1,1],.....]] ==> [System1, System2,...]  fi is the form factor
        # Mult_list: [ [mult1,mult2...],[mult1,mult2...], ...]  ==> [[System1], [System2],...]
        # HKL_list: [ [[H1,K1,L1],[H2,K2,L2]...] , [[H1,K1,L1],[H2,K2,L2]...],...]   ==> [[System1], [System2],...]
        # Theta_list: [ [theta1,theta2...], [theta1,theta2...],...]  ==> [[System1], [System2],...]
        # Intensity_list: [[int1,int2...],[int1,int2...],...] ==> [[System1, System2,...],...]
        # LatticCs_list: [[a,b,c,alpha,beta,gamma],[a,b,c,alpha,beta,gamma]...]  ==> [System1, System2,...]
        if len(Wavelength) == 1:
            Wavelength = Wavelength
        else:
            Wavelength = Wavelength[0]

        Total_Fraction = np.zeros(len(Multi_list))
        lattic_mass_set = cal_lattic_mass(structure_factor)
        # for a single peak, calculate the ratio of each peak and take the average
        # for a single crystal system
        for peak in range(len(Multi_list[0])):
            FHKL_square = [] # [System1, System2,...]   
            for system in range(len(structure_factor)):
                FHKL_square_left = 0
                FHKL_square_right = 0
                
                for atom in range(len(structure_factor[system])):
                    # calculate the form factor 
                    fi = cal_atoms(structure_factor[system][atom][0],Theta_list[system][peak], Wavelength)
                    FHKL_square_left += fi * np.cos(2 * np.pi * (structure_factor[system][atom][1] * HKL_list[system][peak][0] +
                                                    structure_factor[system][atom][2] * HKL_list[system][peak][1] + structure_factor[system][atom][3] * HKL_list[system][peak][2]))
                    FHKL_square_right += fi * np.sin(2 * np.pi * (structure_factor[system][atom][1] * HKL_list[system][peak][0] +
                                                    structure_factor[system][atom][2] * HKL_list[system][peak][1] + structure_factor[system][atom][3] * HKL_list[system][peak][2]))
                FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)

            Volume = [] # [System1, System2,...]
            for system in range(len(crystal_sys_list)):
                VolumeFunction = self.LatticVolume(crystal_sys_list[system])
                sym_a, sym_b, sym_c, angle1, angle2, angle3 = symbols('sym_a sym_b sym_c angle1 angle2 angle3')
                Volume.append(float(VolumeFunction.subs(
                    {sym_a: LatticCs_list[system][0], sym_b: LatticCs_list[system][1], sym_c: LatticCs_list[system][2],
                    angle1: LatticCs_list[system][3] * np.pi/180 , angle2: LatticCs_list[system][4] * np.pi/180, angle3: LatticCs_list[system][5] * np.pi/180})))
            destiny_arr = np.array(lattic_mass_set) / np.array(Volume)
            # without considering the temperature and line absorption factor
            # I = C * V / (V0 ** 2) * F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta))
            # Coef = F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta)) / (V0 ** 2)
            Coef = []
            for system in range(len(crystal_sys_list)):
                Coef.append(float(FHKL_square[system] * Multi_list[system][peak] / Volume[system] ** 2
                            * (1 + np.cos(Theta_list[system][peak] * np.pi/180) ** 2) / (np.sin(Theta_list[system][peak] / 2 * np.pi/180) **2 * np.cos(Theta_list[system][peak] / 2 * np.pi/180))))

            # V  = I / (C * Coef)
            """
            Sum = 0.0
            for system in range(len(crystal_sys_list)):
                Sum += float(Intensity_list[system][peak]/Coef[system])
            """
            Fraction = []  # len == system num
            # cal multi-peaks
            for system in range(len(crystal_sys_list)):
                Fraction.append((Intensity_list[system][peak] / Coef[system]) )
            Total_Fraction += np.array(Fraction)  
        
        Average_Fraction = []
        Sum = 0.0
        for system in range(len(Total_Fraction)):
            Sum += float(Total_Fraction[system]* destiny_arr[system])
        for system in range(len(Total_Fraction)):
            Average_Fraction.append(Total_Fraction[system] * destiny_arr[system]/Sum * 100)

        print('Mass fraction based on structure factor in % :', str(Average_Fraction),'\n Saved at the result documents')
        with open(os.path.join(self.FRfolder, 'MassFraction_accurate_{year}.{month}.{day}_{hour}.{minute}.txt'.format(year=self.namey,
                   month=self.nameM, day=self.named, hour=self.nameh,minute=self.namem)), 'w') as wfid:
            print('The accurately determined Mass fraction in % :', file=wfid)
            print(str(Average_Fraction), file=wfid)


    def Volome_Fraction_Cal(self, crystal_sys_list, num, density_set, EXACT):
        """
        This function calculates the volume fraction through the structure factor
        Calculate by selecting the strongest peaks, num is the number of peaks
        if EXACT == False:
        Only consider the effect of Polarization factor and Lorentz factor on the intensity of diffraction peaks
        if EXACT == False:
        Consider the effect of structure factor
        """
        if EXACT == False:
            Intensity_list = []
            for system in range(len(crystal_sys_list)):
                data = pd.read_csv(os.path.join(self.FRfolder,'CrystalSystem{Task}_WPEMout_{year}.{month}.{day}_{hour}.{minute}.csv'.format(
                    Task=system, year=self.namey, month=self.nameM, day=self.named, hour=self.nameh,minute=self.namem)))
                int = np.array(data.intensity)
                two_theta = np.array(data.mu_i)
                # index = heapq.nlargest(1, enumerate(int), key=lambda x: x[1])[0][0]
                index = np.argpartition(int, -num)[-num:]
                # for a single crystal system 
                Total_int = 0
                for i in range(len(index)):
                    peak_int = int[index[i]]
                    anglefactor = (1 + np.cos(two_theta[index[i]] * np.pi / 180) ** 2) / (
                            np.sin(two_theta[index[i]] / 2 * np.pi / 180) ** 2 * np.cos(two_theta[index[i]] / 2 * np.pi / 180))
                    Total_int += peak_int / anglefactor 
                Intensity_list.append(Total_int / num)

            Sum = 0.0
            for system in range(len(crystal_sys_list)):
                Sum += float(Intensity_list[system] * density_set[system])

            Fraction = []
            for system in range(len(crystal_sys_list)):
                Fraction.append(Intensity_list[system] * density_set[system] / Sum * 100)

            print('Mass fraction without structure factor estimate in % :', str(Fraction), '\n Saved at the result documents')
            with open(os.path.join(self.FRfolder,
                                   'MassFraction_estimate_{year}.{month}.{day}_{hour}.{minute}.txt'.format(year=self.namey,
                                        month=self.nameM,day=self.named,hour=self.nameh,minute=self.namem)),'w') as wfid:
                print('The estimated Mass fraction in % :', file=wfid)
                print(str(Fraction), file=wfid)

        elif EXACT == True:
            Mult_list = []
            HKL_list = []
            Theta_list = []
            Intensity_list = []
            for system in range(len(crystal_sys_list)):
                data = pd.read_csv(os.path.join(self.FRfolder,'CrystalSystem{Task}_WPEMout_{year}.{month}.{day}_{hour}.{minute}.csv'.format(
                    Task=system, year=self.namey, month=self.nameM, day=self.named, hour=self.nameh,minute=self.namem)))
                mult = np.array(data.Mult)
                H = np.array(data.H)
                K = np.array(data.K)
                L = np.array(data.L)
                theta = np.array(data.mu_i)
                int = np.array(data.intensity)
                # index = heapq.nlargest(1, enumerate(int), key=lambda x: x[1])[0][0]
                index = np.argpartition(int, -num)[-num:]
                # for a single system 
                _Mult_list = []
                _HKL_list = []
                _Theta_list = []
                _Intensity_list = []
                for i in range(num):
                    _Mult_list.append(mult[index[i]] )
                    _HKL_list.append([H[index[i]], K[index[i]], L[index[i]]])
                    _Theta_list.append(theta[index[i]])
                    _Intensity_list.append(int[index[i]])

                Mult_list.append(_Mult_list)
                HKL_list.append(_HKL_list)
                Theta_list.append(_Theta_list)
                Intensity_list.append(_Intensity_list)

            return Mult_list, HKL_list,Theta_list, Intensity_list
        else:
            print('Type Error \'EXACT\'')



def cal_lattic_mass(structure_factor):
    # structure_factor:  ==> [ [['atom1',0,0,0],['atom2',0.5,1,1],.....], [['atom1',0,0,0],['atom2',0.5,1,1],.....]] ==> [System1, System2,...]
    total_mass = []
    for system in range(len(structure_factor)):
        mass = 0
        for atom in structure_factor[system]:
            _a = re.sub(r'[^A-Za-z]+', "", atom[0])
            result = find_atomic_mass(_a)
            if result is None:
                print(f"Element with symbol {_a} not found.")
                result = 0
            mass += result
        total_mass.append(mass)
    return total_mass


