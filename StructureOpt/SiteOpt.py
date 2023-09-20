# Atom Information Searching for a sigle crystal 
# Author: Bin CAO <binjacobcao@gmail.com>

import re
import os
import random
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
from ..Extinction.XRDpre import profile
from ..Plot.UnitCell import plotUnitCell
from ..XRDSimulation.Simulation import det_system,cal_atoms,symbols,LatticVolume,WAVELENGTHS

class BgolearnOpt(object):
    def __init__(self, xrd_pattern, cif_file,random_num, wavelength='CuKa',search_cap = 100,cal_extinction=True) :
        # xrd_pattern : the path of experimental xrd diffraction pattern of a signal crystal (2theta, intensity)
        # cif_file : the path of the cif file
        # WPEM reads lattice constants from an input cif file 
        # read parameters from cif by ..Extinction.XRDpre

        os.makedirs('WPEMSitOpt', exist_ok=True)
        self.random_num = random_num
        self.search_cap = search_cap

        data = pd.read_csv(xrd_pattern)
        self.mu_list = data['mu_i'].tolist() 
        self.intensity = data['intensity'].tolist()
        self.Mult = data['Mult'].tolist()
        self.HKL_list = np.array(data[['H','K','L']]).tolist()

        # the parameters of peak's shapes
        self.W =  data['wi'].tolist() 
        self.A = data['Ai'].tolist()
        self.gamma = data['L_gamma_i'].tolist() 
        self.sigma2 = data['G_sigma2_i'].tolist()

        first_row = data.columns[:] 
        matches = re.findall(r'(\d+\.\d+)', ','.join(first_row)) 
        LatticCs = [float(match) for match in matches] 
        if type(wavelength) == list:
            _, Atom_coordinate = profile(wavelength=wavelength[0],cal_extinction=cal_extinction).generate(cif_file)
        else:_, Atom_coordinate = profile(wavelength=wavelength,cal_extinction=cal_extinction).generate(cif_file)

        self.LatticCs = LatticCs
        self.crystal_system = det_system(LatticCs)
        self.Atom_coordinate = Atom_coordinate   
        self.code =  data['code'].tolist()
 
        if isinstance(wavelength, (float, int, list)):
            self.wavelength = wavelength
        elif isinstance(wavelength, str):
            self.wavelength = [WAVELENGTHS[wavelength]]
        else:
            raise TypeError("'wavelength' must be either of: float, int, list or str")
        
        if 2 in data['code'].tolist() and len(self.wavelength) != 2:
            raise RuntimeError("wavelength error") 


    # Substitutional solid solutions are those in which 
    # the atoms of the minor component (solute) are substituted 
    # for the atoms of the major component (solvent) on the 
    # lattice positions normally occupied by the solvent atoms.
    def Substitutional_SS(self,SolventAtom, SoluteAtom,max_iter ):
        # SolventAtom : solvent atoms, str, default is None. i.e., SolventAtom = 'Cu'
        # SoluteAtom : solute atoms, str, default is None. i.e., SoluteAtom = 'Ti'

        feature_names = set([item[0] for item in self.Atom_coordinate]) | {SoluteAtom}
        feature_names = list(feature_names)

        feature_matrix = []
        response_vector = []    
        feature_matrix.append(label_encode(self.Atom_coordinate,feature_names))
        # for one_hot code : feature_matrix.append(label_encode(self.Atom_coordinate,feature_names).tolist())

        # cal the siumlation pattern
        FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
        for angle in range(len(self.code)):
            FHKL_square_left = 0
            FHKL_square_right = 0
            for atom in range(len(self.Atom_coordinate)):
                fi = cal_atoms(self.Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength[int(self.code[angle])])
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
        Ints_zero = []
        for angle in range(len(FHKL_square)):
            Ints_zero.append(float(FHKL_square[angle] * self.Mult[angle] / Volume ** 2
                        * (1 + np.cos(self.mu_list[angle] * np.pi/180) ** 2) / (np.sin(self.mu_list[angle] / 2 * np.pi/180) **2 * np.cos(self.mu_list[angle] / 2 * np.pi/180))))
        
        del FHKL_square 
        Intensity_list = [Ints_zero]
        score = mark_fun(Ints_zero,self.intensity,'mse')
        response_vector.append(score)
        print('initial error is', score)

        
        if self.random_num >= max_iter: 
            print('WPEM uses a random optimization method based on the current settings')
            save_resource = True
        else: 
            print('\n')
            save_resource = False
            import Bgolearn.BGOsampling as BGOS
            search_space = generate_sp(self.Atom_coordinate,SolventAtom,SoluteAtom,feature_names,self.search_cap)
            print('\n')

        """
        for one hot
        unique_labels = list(set([item[0] for item in self.Atom_coordinate]))
        atom_num = len(unique_labels) + 1
        """
        
        for iter in range(max_iter):
            if iter <= self.random_num:
                print('WPEM Site optimization | {}-th | random '.format(iter+1))
                new_Atom_coordinate = random_substitute(self.Atom_coordinate,SolventAtom,SoluteAtom,None)
                feature_matrix.append(label_encode(new_Atom_coordinate,feature_names))
                # for one hot code : feature_matrix.append(label_encode(new_Atom_coordinate,feature_names).tolist())
                # cal the siumlation pattern
                FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
                for angle in range(len(self.HKL_list)):
                    FHKL_square_left = 0
                    FHKL_square_right = 0
                    for atom in range(len(new_Atom_coordinate)):
                        fi = cal_atoms(new_Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength[int(self.code[angle])])
                        FHKL_square_left += fi * np.cos(2 * np.pi * (new_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                            new_Atom_coordinate[atom][2] * self.HKL_list[angle][1] + new_Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                        FHKL_square_right += fi * np.sin(2 * np.pi * (new_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                            new_Atom_coordinate[atom][2] * self.HKL_list[angle][1] + new_Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                    FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)

                # I = C / (V0 ** 2) * F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta))
                # without considering the temperature and line absorption factor
                Ints = []
                for angle in range(len(FHKL_square)):
                    Ints.append(float(FHKL_square[angle] * self.Mult[angle] / Volume ** 2
                                * (1 + np.cos(self.mu_list[angle] * np.pi/180) ** 2) / (np.sin(self.mu_list[angle] / 2 * np.pi/180) **2 * np.cos(self.mu_list[angle] / 2 * np.pi/180))))
                del FHKL_square 
                score = mark_fun(Ints,self.intensity,'mse')
                """
                save_resource == True
                The sits are searched in a random way, in which I only save the better structures
                save_resource == False
                The sits are searched by Bgolearn, I saves all the structures for providing a larger training dataset to Bgolearn
                """
                if save_resource == True:
                    # this structure is better than the last one
                    if score <= response_vector[-1]:
                        response_vector.append(score)
                        Intensity_list.append(Ints)
                    else: 
                        # don't save the information of worse structure : response_vector,  Intensity_list
                        # Delete the stored structure information, the structure is not a satisfactory structure
                        feature_matrix.pop()
                        print('feature_matrix',feature_matrix)
                else: response_vector.append(score)
                
            else:
                print('WPEM Site optimization | {}-th | Bgolearn '.format(iter+1))
                Bgolearn = BGOS.Bgolearn() 
                Mymodel = Bgolearn.fit(data_matrix = np.array(feature_matrix), Measured_response = response_vector, virtual_samples = np.array(search_space))
                _, data = Mymodel.EI()
                print('\n')
                feature_matrix.append(data[0])
                search_space.remove(data[0].tolist())
                if len(search_space) == 0:
                    print("All potential structures were compared by Bgolearn !")
                    break

                new_Atom_coordinate = label_decode(data[0].tolist(),feature_names)
                # for one hot: 
                """
                new_Atom_coordinate = label_decode(convert_to_2d_list(data[0],4),feature_names)
                new_Atom_coordinate = label_decode(convert_to_2d_list(data[0],atom_num+3),feature_names)
                """

                # cal the siumlation pattern
                FHKL_square = [] # [FHKL2_1, FHKL2_2,...] a list has the same length with HKL_list
                for angle in range(len(self.HKL_list)):
                    FHKL_square_left = 0
                    FHKL_square_right = 0
                    for atom in range(len(new_Atom_coordinate)):
                        fi = cal_atoms(new_Atom_coordinate[atom][0],self.mu_list[angle], self.wavelength[int(self.code[angle])])
                        FHKL_square_left += fi * np.cos(2 * np.pi * (new_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                            new_Atom_coordinate[atom][2] * self.HKL_list[angle][1] + new_Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                        FHKL_square_right += fi * np.sin(2 * np.pi * (new_Atom_coordinate[atom][1] * self.HKL_list[angle][0] +
                                                            new_Atom_coordinate[atom][2] * self.HKL_list[angle][1] + new_Atom_coordinate[atom][3] * self.HKL_list[angle][2]))
                    FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)

                # I = C / (V0 ** 2) * F2HKL * P * (1 + cos(2*theta) ** 2) / (sin(theta) **2 * cos(theta))
                # without considering the temperature and line absorption factor
                Ints = []
                for angle in range(len(FHKL_square)):
                    Ints.append(float(FHKL_square[angle] * self.Mult[angle] / Volume ** 2
                                * (1 + np.cos(self.mu_list[angle] * np.pi/180) ** 2) / (np.sin(self.mu_list[angle] / 2 * np.pi/180) **2 * np.cos(self.mu_list[angle] / 2 * np.pi/180))))
                del FHKL_square 
                Intensity_list.append(Ints)
                score = mark_fun(Ints,self.intensity,'mse')
                response_vector.append(score)

        # search the best structure
        opt_index = response_vector.index(min(response_vector))
        structure = feature_matrix[opt_index]
        Int_opt = Intensity_list[opt_index]
        real_structure = label_decode(structure,feature_names)

        print('optimized error is',response_vector[opt_index])
         
        """
        # for one hot :
        real_structure = label_decode(convert_to_2d_list(structure,4),feature_names)
        real_structure = label_decode(convert_to_2d_list(structure,atom_num+3),feature_names)
        """
        
        plotUnitCell(real_structure,self.LatticCs,).plot()
       
        # Define the font of images
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10 

        # Plot the result 
        o_x = np.arange(min(self.mu_list)-2, max(self.mu_list)+2, 0.01)
        ori_peak = self.plot_peak(o_x, self.intensity,self.intensity)
        simulation_ori_peak = self.plot_peak(o_x, scale_peak(self.intensity),scale_peak(Ints_zero))
        simulation_opt_peak = self.plot_peak(o_x, scale_peak(self.intensity),scale_peak(Int_opt))
        plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.plot(o_x, ori_peak,'k',label="Decomposed experimental profile")
        plt.plot(o_x, simulation_ori_peak,'--',label="Initial lattice structure")
        plt.plot(o_x, simulation_opt_peak,'-.',label="Searched lattice structure - Bgolearn")
        plt.legend()
        plt.savefig('./WPEMSitOpt/substitutional.png', dpi=800)
        plt.show()
    
        return real_structure
    
    def plot_peak(self,o_x, ori_intensity,changed_intensity):
        peak_intens = []
        k = len(self.mu_list)
        for p in range(k):
            w_l = self.W[p] * self.A[p]
            w_n = self.W[p] * (1 - self.A[p])
            cal_peak = np.array(draw_peak_density(o_x, w_l, w_n, self.mu_list[p], self.gamma[p], self.sigma2[p]))
            peak_intens.append(cal_peak * changed_intensity[p] / ori_intensity[p])
        total_intens = np.zeros(len(o_x))
        for j in range(k):
            total_intens += np.array(peak_intens[j])
        return total_intens
            
def random_substitute(Atom_coordinat,SolventAtom,SoluteAtom,n_atom=None):
    acoord = copy.deepcopy(Atom_coordinat)
    indices = [i for i, item in enumerate(acoord) if item[0] == SolventAtom]
    random.shuffle(indices) 

    if n_atom is None:
        n_atom = random.randint(1, len(indices)-1)
    else:pass
    indices_to_replace = indices[:n_atom] 

    for i in indices_to_replace:
        acoord[i][0] = SoluteAtom
    return acoord

################################################################
# Using category coding, in order to reduce the feature dimension
def label_encode(atom_coordinates, feature_categories):
    category_mapping = {category: index for index, category in enumerate(feature_categories)}
    encoded_features = []
    for item in atom_coordinates:
        encoded_item = [category_mapping[item[0]]] + item[1:]
        encoded_features.extend(encoded_item)
    return encoded_features


def label_decode(encoded_features, feature_categories):
    category_mapping = {index: category for index, category in enumerate(feature_categories)}
    atom_coordinates = []
    # one label + three coordinates (x,y,z) = 4
    for i in range(0, len(encoded_features), 4):
        category = category_mapping[encoded_features[i]]
        coordinates = encoded_features[i+1:i+4]
        atom_coordinates.append([category] + coordinates)
    return atom_coordinates
################################################################

################################################################
# One-hot encoding does not distinguish atom types（better）
def one_hot_code(atom_coordinat,feature_categories):
    encoder = OneHotEncoder(categories=[list(feature_categories)], sparse=False)
    feature_columns = np.array(atom_coordinat)[:, 0].reshape(-1, 1)
    encoded_features = encoder.fit_transform(feature_columns)
    encoded_features_with_coordinates = np.hstack((encoded_features, np.array(atom_coordinat)[:, 1:])).flatten().astype(float)
    return encoded_features_with_coordinates

def recover_atoms_coordinate(encoded_features,unique_labels):
    restored_features = []
    for encoded_feature in encoded_features:
        label = unique_labels[np.argmax(encoded_feature[:len(unique_labels)])]
        try:
            coordinates = encoded_feature[len(unique_labels):].tolist()
        except : coordinates = encoded_feature[len(unique_labels):]
        feature = [label] + coordinates
        restored_features.append(feature)
    return restored_features
################################################################

def scale_peak(y):
    max_value = max(y)  
    scaling_factor = 100 / max_value 
    scaled_list = [value * scaling_factor for value in y] 
    return scaled_list

def mark_fun(cal_y,decompose_y,measure):
    scaled_list1 = scale_peak(decompose_y)
    scaled_list2 = scale_peak(cal_y) 
    if measure == 'mse':
        return np.sqrt(np.sum((np.array(scaled_list1) - np.array(scaled_list2))**2)/len(scaled_list1))

def generate_sp(Atom_coordinat,SolventAtom,SoluteAtom,feature_names,search_cap):
    search_spaces = []
    indices = [i for i, item in enumerate(Atom_coordinat) if item[0] == SolventAtom] 
    random.shuffle(indices)

    all_combinations = []
    for r in range(1, len(indices) + 1):
        combinations_r = combinations(indices, r)
        all_combinations.extend(list(combinations_r))
    random.shuffle(all_combinations)
    for k in range(min(len(all_combinations),search_cap)):
        backup = copy.deepcopy(Atom_coordinat)
        combination = all_combinations[k]
        for rep in combination:
            backup[rep][0] = SoluteAtom
        new_feature = label_encode(backup,feature_names)
        try: list_format = new_feature.tolist()
        except: list_format = new_feature
        search_spaces.append(list_format)

    return search_spaces
        

def convert_to_2d_list(lst,num):
    return [lst[i:i+num] for i in range(0, len(lst), num)]

def draw_peak_density(x, w_l, w_g, mu, gamma, sigma2):
    peak_density = w_l * lorenz_density(x, mu, gamma) + w_g * normal_density(x, mu, sigma2)
    return peak_density

# Normal distribution
def normal_density( x, mu, sigma2):
    density = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
    return density

# Lorenz distribution
def lorenz_density( x, mu, gamma):
    density = (1 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
    return density