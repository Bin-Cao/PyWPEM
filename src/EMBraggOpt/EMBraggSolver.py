# The EM-Bragg Solver defined in WPEM
# Author: Bin CAO <binjacobcao@gmail.com>

import copy
import csv
from tqdm import tqdm
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import  ProcessPoolExecutor
from .WPEMFuns.SolverFuns import *
from .BraggLawDerivation import BraggLawDerivation
from ..Refinement.VolumeFractionDertermination import VandMFraction
import time
import os

timename = time.localtime(time.time())
namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min
BLD = BraggLawDerivation()


"""
This class is the core solver of WPEM

Please feel free to contact Bin Cao (bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code.

Contribution and suggestions are always welcome. You can also contact the authors for research collaboration.
"""
class WPEMsolver(object):
    def __init__(
        self,wavelength,Var,asy_C,s_angle,subset_number, low_bound,up_bound,
        lattice_constants,density_list,singal,no_bac_intensity_file,original_file,bacground_file,two_theta_range,
        initial_peak_file,bta, bta_threshold,limit, iter_limit,w_limit,iter_max,
        lock_num,structure_factor, MODEL, InitializationEpoch,Macromolecule,cpu,num, EXACT,Cu_tao,
        loadParams, ZeroShift,wk_dir = None

    ):
        self.VFD = VandMFraction(timename,wk_dir)
        self.wavelength = wavelength # wavelength
        self.Var = Var # standard deviation of background intensity
        self.asy_C = asy_C # asymmetric parameter for descripting the asymmetric peak
        self.s_angle = s_angle # diffraction angles below s_angle are considered asymmetric
        self.subset_number = subset_number # number of peaks unsed in Bragg step
        self.low_bound = low_bound # the lower boundary of the selected peaks at Bragg step
        self.up_bound = up_bound # the upper boundary of the selected peaks at Bragg step
        self.Lattice_constants = lattice_constants # lattice constants
        if density_list is not None:
            self.density_list= density_list # the densities of crytal, can be calculated by fun. WPEM.CIFpreprocess()
        else:
            self.density_list = np.ones(len(lattice_constants))
        self.singal = singal # whether to allow Bragg step adjustment
        self.no_bac_intensity_file = no_bac_intensity_file # crystal diffraction intensity 
        self.original_file = original_file # experimental observation intensity
        self.bacground_file = bacground_file # fitted bacground intensity
        self.two_theta_range = two_theta_range # theta range measured in the experiment 
        self.initial_peak_file = initial_peak_file # the estimated peak's locations based on (S-S') = G* (diffraction conditions)
        self.bta = bta # the ratio of Lorentzian components in PV function
        self.bta_threshold = bta_threshold # a preset lower boundary of bta
        self.limit = limit #  a preset lower boundary of sigma2
        self.iter_limit = iter_limit #  a preset threshold iteration promotion (likelihood) 
        self.w_limit = w_limit # a preset lower boundary of peak weight
        self.iter_max = iter_max # maximum number of iterations
        self.lock_num = lock_num # in case of loglikelihood iterations continously decrease 
        self.structure_factor = structure_factor # structure factor
        self.MODEL = MODEL # str default = 'REFINEMENT' for lattice constants REFINEMENT; 'ANALYSIS' for components ANALYSIS
        self.IniEpoch = InitializationEpoch # Initialization epoch
        self.Macromolecule = Macromolecule # Boolean default = False, for profile fitting of crystals. True for macromolecules.
        self.cpu = cpu # parallel computatis CPU core numerus
        self.num = num # the number of the strongest peaks used in calculating mass fraction
        self.EXACT = EXACT # Boolean default = False, True for exact calculation of mass fraction by diffraction intensity theory
        self.Cu_tao = Cu_tao # for limitting the relationship between the ray diffraction intensities of copper Ka1 and Ka2
        self.loadParams = loadParams # if read in the initial parameters
        self.ZeroShift = ZeroShift # If a standard sample is available, the instrument offset can be calibrated
        self.wk_dir = wk_dir

        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        
        # mike dir
        if wk_dir is None:
            self.DCfolder = 'DecomposedComponents'
            self.FRfolder = 'WPEMFittingResults'
        else:
            self.DCfolder = os.path.join(wk_dir, 'DecomposedComponents')
            self.FRfolder = os.path.join(wk_dir, 'WPEMFittingResults')

        os.makedirs(self.DCfolder, exist_ok=True)
        os.makedirs(self.FRfolder, exist_ok=True)

        print('Initialization')
        print('-'*80)
    
    # Whole pattern decomposition execution function, implemented by data reading and EM-Bragg iteration
    def cal_output_result(self,):
        lmd = len(self.wavelength)
        try:
            # Read non-background data, and the data format is 2theta-intensity/X-Y data.
            data = pd.read_csv(self.no_bac_intensity_file, header=None, names=['two_theta', 'intensity'])
            # Read background data, and the data format is 2theta-intensity/X-Y data.
            bac_data = pd.read_csv(self.bacground_file, header=None, names=['two_theta', 'intensity'])
        
        except FileNotFoundError as e:
            # print(f"File not found: {e}")
            # Read non-background data, and the data format is 2theta-intensity/X-Y data.
            data = pd.read_csv(os.path.join('ConvertedDocuments',self.no_bac_intensity_file), header=None, names=['two_theta', 'intensity'])
            # Read background data, and the data format is 2theta-intensity/X-Y data.
            bac_data = pd.read_csv(os.path.join('ConvertedDocuments',self.bacground_file), header=None, names=['two_theta', 'intensity'])

        except pd.errors.EmptyDataError:
            print("The file is empty!")
            
        # Read raw/original data, and the data format is 2theta-intensity/X-Y data.
        in_data = read_data_file(self.original_file,)
        
        # Detect whether the user restricts the fitting range, and cut the diffranction range 
        if type(self.two_theta_range) == tuple:
            index = np.where((data.two_theta < self.two_theta_range[0]) | (data.two_theta > self.two_theta_range[1]))
            data = data.drop(index[0])
            in_data = in_data.drop(index[0])
            bac_data = bac_data.drop(index[0])
            pass
        elif self.two_theta_range == None:
            pass
        else:
            print('Type Error -two_theta_range-')

        # Read initial {Hi, Ki, Li}
        for task in range(len(self.initial_peak_file)):
            read_ini_peak(self.initial_peak_file[task], task,self.FRfolder)

        o_x = np.array(in_data.two_theta)
        f_x = np.array(data.two_theta)
        b_x = np.array(bac_data.two_theta)

        # Check if the data matches
        if len(o_x) == len(f_x) and len(f_x) == len(b_x):
            hkl_data = []
            for task in range(len(self.Lattice_constants)):
                hkl_data_task = pd.read_csv(os.path.join(self.FRfolder,'hkl{task}_{year}.{month}.{day}_{hour}.{minute}.csv'.format(task=task,
                    year=namey,month=nameM,day=named, hour=nameh, minute=namem)),header=0)
                hkl_list = [np.array(hkl_data_task.H), np.array(hkl_data_task.K), np.array(hkl_data_task.L)]
                hkl_data.append(hkl_list)
                # hkl_data = [[array1,array2,array3],[array1,array2,array3],....]
                # hkl_data is a list
            
            # experimental diffraction data
            theta_data = np.array(data.two_theta)
            __intensity = np.array(data.intensity)
            
            # estimated background data
            bac = np.array(bac_data.intensity)
            # estimated crystal diffraction data
            i_obser = np.array(in_data.intensity)

            # sliding window
            fwhm_len = (theta_data[1] - theta_data[0]) / 20

            inten = copy.deepcopy(__intensity)
            min_i = min(__intensity)
            # Subtract the DC signal
            inten -= min_i
            # Calculation of integrated probability of measure function, crystal diffraction pattern
            area = theta_intensity_area(theta_data, inten)

            print('\n')
            # Update parameters via EM-Bragg process
            w_list, p1_list, p2_list, i_ter, flag, \
            ini_CL, mui_abc_set, i_out, bac_up, Rp, Rwp, Rsquare,crystal_sys_set = self.up_parameter(hkl_data,theta_data, inten, area, min_i, i_obser, bac,lmd)

            csv_file = os.path.join(self.FRfolder, "modelparams.csv")
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(w_list)
                writer.writerow(p1_list)
                writer.writerow(p2_list)

            i_calc = i_out + min_i

            """
            w_list saves the weights of peaks
            w_i * deta [termed as A], w_i * (1-deta)[termed as B]
            """ 
            # save the values of A + B 
            i_w_cal = []
            # save the values of A / ( A + B )
            a_n_list = []
            """
            w_i * deta = i_w_cal[i] * a_n_list[i]
            w_i * (1-deta) = i_w_cal[i] * (1 - a_n_list[i])
            """
            for i in range(len(p1_list)):
                i_ln = 2 * i
                i_w_cal.append(w_list[i_ln] + w_list[i_ln + 1])     
                a_n_list.append(w_list[i_ln] / i_w_cal[i])

            if self.ZeroShift == True:
                print('='*60)
                top_indices = np.argpartition(i_w_cal, -6)[-6:]
                peaks = np.array(p1_list)[top_indices]
                peak_array = pd.read_csv('peak0.csv')['2theta/TOF']
                ref_peaks = peak_array[top_indices]
                offset = sum(ref_peaks - peaks) / 6
                print('The instrument offset is:', offset)
                print('='*60)

            y1 = [] # γi
            y2 = [] # σi^2
            for i in range(len(p1_list)):
                i_l = i * 2
                i_n = i_l + 1
                y1.append(float(p2_list[i_l]))
                y2.append(float(p2_list[i_n]))

            y = [] # WPEM fitted pattern
            for j in range(len(theta_data)):
                y.append(bac_up[j] + i_calc[j])

            # intensity of each peak
            peak_intensity_h = []
            for i in range(len(p1_list)):
                i_ln = i * 2
                peak_intensity_h.append(
                    mix_normal_lorenz_density_cal_fwhm(p1_list[i], w_list[i_ln: i_ln + 2], p1_list[i],
                                                            p2_list[i_ln: i_ln + 2])
                                                            )
            # fwhm 
            peak_fwhm_list = []
            for i in range(len(p1_list)):
                i_ln = i * 2
                inten_h = (peak_intensity_h[i] / 2)
                for j in range(1, len(theta_data) * 20):
                    fwhm_l = p1_list[i] - fwhm_len * j
                    # here each is ideal PV function
                    intensity_fwhm = mix_normal_lorenz_density_cal_fwhm(fwhm_l, w_list[i_ln: i_ln + 2], p1_list[i],
                                                                             p2_list[i_ln: i_ln + 2])
                    if intensity_fwhm <= inten_h:
                        peak_fwhm_list.append(fwhm_len * j * 2)
                        break

            # save result files 
            with open(os.path.join(self.FRfolder,'WPEMfittingProfile_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                                       month=nameM,day=named, hour=nameh,minute=namem)),'w') as wfid:
                for j in range(len(theta_data)):
                    print(theta_data[j], end=', ', file=wfid)
                    print(float(y[j]), file=wfid)

            with open(os.path.join(self.DCfolder, 'fitting_profile.csv'),'w') as wfid:
                for j in range(len(theta_data)):
                    print(theta_data[j], end=', ', file=wfid)
                    print(float(y[j]), file=wfid)

            with open(os.path.join(self.FRfolder,'WPEMPeakParas_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM,
                                                   day=named, hour=nameh,minute=namem)),'w') as wfid:
                print('wi', end=',', file=wfid)
                print('Ai', end=',', file=wfid)
                print('mu_i', end=',', file=wfid)
                print('L_gamma_i', end=',', file=wfid)
                print('G_sigma2_i', end=',', file=wfid)
                print('fwhm', end=',', file=wfid)
                print('Rwp: %f' % Rwp, end=',', file=wfid)
                print('Rp: %f ' % Rp, end=',',file=wfid)
                print('Rsquare: %f ' % Rsquare, file=wfid)
                for j in range(len(i_w_cal)):
                    print(i_w_cal[j], end=',', file=wfid)
                    print(a_n_list[j], end=',', file=wfid)
                    print(p1_list[j], end=',', file=wfid)
                    print(y1[j], end=',', file=wfid)
                    print(y2[j], end=',', file=wfid)
                    print(peak_fwhm_list[j], file=wfid)

            # Prepare for visualization module 
            with open(os.path.join(self.DCfolder, 'sub_peaks.csv'),'w') as wfid:
                print('wi', end=',', file=wfid)
                print('Ai', end=',', file=wfid)
                print('mu_i', end=',', file=wfid)
                print('L_gamma_i', end=',', file=wfid)
                print('G_sigma2_i', end=',', file=wfid)
                print('fwhm', end=',', file=wfid)
                print('Rwp: %f' % Rwp, end=',', file=wfid)
                print('Rp: %f ' % Rp, end=',', file=wfid)
                print('Rsquare: %f ' % Rsquare, file=wfid)
                for j in range(len(i_w_cal)):
                    print(i_w_cal[j], end=',', file=wfid)
                    print(a_n_list[j], end=',', file=wfid)
                    print(p1_list[j], end=',', file=wfid)
                    print(y1[j], end=',', file=wfid)
                    print(y2[j], end=',', file=wfid)
                    print(peak_fwhm_list[j], file=wfid)

            with open(os.path.join(self.FRfolder, 'LatticeConstances_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                                         month=nameM,day=named,hour=nameh,minute=namem)), 'w') as wfid:
                print('system', end=',', file=wfid)
                print('a', end=',', file=wfid)
                print('b', end=',', file=wfid)
                print('c', end=',', file=wfid)
                print('alpha', end=',', file=wfid)
                print('beta', end=',', file=wfid)
                print('gamma', file=wfid)
                for task in range(len(self.Lattice_constants)):
                    print(crystal_sys_set[task], end=',', file=wfid)
                    print(ini_CL[task][0], end=',', file=wfid)
                    print(ini_CL[task][1], end=',', file=wfid)
                    print(ini_CL[task][2], end=',', file=wfid)
                    print(ini_CL[task][3], end=',', file=wfid)
                    print(ini_CL[task][4], end=',', file=wfid)
                    print(ini_CL[task][5], file=wfid)

            # Store files according to crystal system category in polycrystal
            for task in range(len(self.Lattice_constants)):
                w_list_match = []
                a_list_match = []
                gam_list_match = []
                sig_list_match = []
                task_intensity = []
                for i in range(len(mui_abc_set[task])):
                    for j in range(len(p1_list)):
                        if mui_abc_set[task][i] == p1_list[j]:
                            w_list_match.append(i_w_cal[j])
                            a_list_match.append(a_n_list[j])
                            gam_list_match.append(y1[j])
                            sig_list_match.append(y2[j])
                            cal_sub_peak_intensity = i_w_cal[j] * a_n_list[j] / (y1[j] * np.pi) \
                                                     + i_w_cal[j] * (1 - a_n_list[j]) / np.sqrt(2 * np.pi * y2[j])
                            task_intensity.append(cal_sub_peak_intensity)
                            break

                peakdata = pd.read_csv(os.path.join(self.wk_dir,'peak{task}.csv'.format(task=task)))
                Mult = np.array(peakdata.Mult)
                
                if len(self.wavelength) == 2:
                    with open(os.path.join(self.FRfolder,'CrystalSystem{Task}_WPEMout_{year}.{month}.{day}_{hour}.{minute}.csv'.format(
                                            Task=task, year=namey, month=nameM, day=named, hour=nameh,minute=namem)), 'w') as wfid:
                        print('code', end=',', file=wfid)
                        print('H', end=',', file=wfid)
                        print('K', end=',', file=wfid)
                        print('L', end=',', file=wfid)
                        print('Mult', end=',', file=wfid)
                        print('wi', end=',', file=wfid)
                        print('Ai', end=',', file=wfid)
                        print('mu_i', end=',', file=wfid)
                        print('intensity', end=',', file=wfid)
                        print('L_gamma_i', end=',', file=wfid)
                        print('G_sigma2_i', end=',', file=wfid)
                        print("a=%f" % ini_CL[task][0], end=',', file=wfid)
                        print("b=%f" % ini_CL[task][1], end=',', file=wfid)
                        print("c=%f" % ini_CL[task][2], end=',', file=wfid)
                        print('alpha=%f' % ini_CL[task][3], end=',', file=wfid)
                        print('beta=%f' % ini_CL[task][4], end=',', file=wfid)
                        print('gamma=%f' % ini_CL[task][5], file=wfid)
                        for j in range(len(mui_abc_set[task])):
                            t = int(j / 2)
                            print((j % 2)*(len(self.wavelength)-1), end=',', file=wfid)
                            print(hkl_data[task][0][t], end=',', file=wfid)
                            print(hkl_data[task][1][t], end=',', file=wfid)
                            print(hkl_data[task][2] [t], end=',', file=wfid)
                            print(Mult[t], end=',', file=wfid)
                            print(w_list_match[j], end=',', file=wfid)
                            print(a_list_match[j], end=',', file=wfid)
                            print(mui_abc_set[task][j], end=',', file=wfid)
                            print(task_intensity[j], end=',', file=wfid)
                            print(gam_list_match[j], end=',', file=wfid)
                            print(sig_list_match[j], file=wfid)

                    if len(self.Lattice_constants) != 1:
                        # rewrite for Decomposing
                        with open(os.path.join(self.DCfolder,
                                            'System{Task}.csv'.format(Task=task)), 'w') as wfid:
                            print('code', end=',', file=wfid)
                            print('H', end=',', file=wfid)
                            print('K', end=',', file=wfid)
                            print('L', end=',', file=wfid)
                            print('Mult', end=',', file=wfid)
                            print('wi', end=',', file=wfid)
                            print('Ai', end=',', file=wfid)
                            print('mu_i', end=',', file=wfid)
                            print('intensity', end=',', file=wfid)
                            print('L_gamma_i', end=',', file=wfid)
                            print('G_sigma2_i', end=',', file=wfid)
                            print("a=%f" % ini_CL[task][0], end=',', file=wfid)
                            print("b=%f" % ini_CL[task][1], end=',', file=wfid)
                            print("c=%f" % ini_CL[task][2], end=',', file=wfid)
                            print('alpha=%f' % ini_CL[task][3], end=',', file=wfid)
                            print('beta=%f' % ini_CL[task][4], end=',', file=wfid)
                            print('gamma=%f' % ini_CL[task][5], file=wfid)
                            for j in range(len(mui_abc_set[task])):
                                t = int(j / 2)
                                print((j % 2)*(len(self.wavelength)-1), end=',', file=wfid)
                                print(hkl_data[task][0][t], end=',', file=wfid)
                                print(hkl_data[task][1][t], end=',', file=wfid)
                                print(hkl_data[task][2][t], end=',', file=wfid)
                                print(Mult[t], end=',', file=wfid)
                                print(w_list_match[j], end=',', file=wfid)
                                print(a_list_match[j], end=',', file=wfid)
                                print(mui_abc_set[task][j], end=',', file=wfid)
                                print(task_intensity[j], end=',', file=wfid)
                                print(gam_list_match[j], end=',', file=wfid)
                                print(sig_list_match[j], file=wfid)
                # for single X-ray
                if len(self.wavelength) == 1:
                    with open(os.path.join(self.FRfolder,'CrystalSystem{Task}_WPEMout_{year}.{month}.{day}_{hour}.{minute}.csv'.format(
                                            Task=task, year=namey, month=nameM, day=named, hour=nameh,minute=namem)), 'w') as wfid:
                        print('H', end=',', file=wfid)
                        print('K', end=',', file=wfid)
                        print('L', end=',', file=wfid)
                        print('Mult', end=',', file=wfid)
                        print('wi', end=',', file=wfid)
                        print('Ai', end=',', file=wfid)
                        print('mu_i', end=',', file=wfid)
                        print('intensity', end=',', file=wfid)
                        print('L_gamma_i', end=',', file=wfid)
                        print('G_sigma2_i', end=',', file=wfid)
                        print("a=%f" % ini_CL[task][0], end=',', file=wfid)
                        print("b=%f" % ini_CL[task][1], end=',', file=wfid)
                        print("c=%f" % ini_CL[task][2], end=',', file=wfid)
                        print('alpha=%f' % ini_CL[task][3], end=',', file=wfid)
                        print('beta=%f' % ini_CL[task][4], end=',', file=wfid)
                        print('gamma=%f' % ini_CL[task][5], file=wfid)
                        for j in range(len(mui_abc_set[task])):
                            print(hkl_data[task][0][j], end=',', file=wfid)
                            print(hkl_data[task][1][j], end=',', file=wfid)
                            print(hkl_data[task][2] [j], end=',', file=wfid)
                            print(Mult[j], end=',', file=wfid)
                            print(w_list_match[j], end=',', file=wfid)
                            print(a_list_match[j], end=',', file=wfid)
                            print(mui_abc_set[task][j], end=',', file=wfid)
                            print(task_intensity[j], end=',', file=wfid)
                            print(gam_list_match[j], end=',', file=wfid)
                            print(sig_list_match[j], file=wfid)

                    if len(self.Lattice_constants) != 1:
                        # rewrite for Decomposing
                        with open(os.path.join(self.DCfolder,
                                            'System{Task}.csv'.format(Task=task)), 'w') as wfid:
                            print('H', end=',', file=wfid)
                            print('K', end=',', file=wfid)
                            print('L', end=',', file=wfid)
                            print('Mult', end=',', file=wfid)
                            print('wi', end=',', file=wfid)
                            print('Ai', end=',', file=wfid)
                            print('mu_i', end=',', file=wfid)
                            print('intensity', end=',', file=wfid)
                            print('L_gamma_i', end=',', file=wfid)
                            print('G_sigma2_i', end=',', file=wfid)
                            print("a=%f" % ini_CL[task][0], end=',', file=wfid)
                            print("b=%f" % ini_CL[task][1], end=',', file=wfid)
                            print("c=%f" % ini_CL[task][2], end=',', file=wfid)
                            print('alpha=%f' % ini_CL[task][3], end=',', file=wfid)
                            print('beta=%f' % ini_CL[task][4], end=',', file=wfid)
                            print('gamma=%f' % ini_CL[task][5], file=wfid)
                            for j in range(len(mui_abc_set[task])):
                                print(hkl_data[task][0][j], end=',', file=wfid)
                                print(hkl_data[task][1][j], end=',', file=wfid)
                                print(hkl_data[task][2][j], end=',', file=wfid)
                                print(Mult[j], end=',', file=wfid)
                                print(w_list_match[j], end=',', file=wfid)
                                print(a_list_match[j], end=',', file=wfid)
                                print(mui_abc_set[task][j], end=',', file=wfid)
                                print(task_intensity[j], end=',', file=wfid)
                                print(gam_list_match[j], end=',', file=wfid)
                                print(sig_list_match[j], file=wfid)
        
            if self.EXACT == False:
                self.VFD.Volome_Fraction_Cal(crystal_sys_set,self.num, self.density_list, EXACT=False)
            elif self.EXACT == True:
                if self.structure_factor != None:
                    Mult_list, HKL_list, Theta_list, Intensity_list = VFD.Volome_Fraction_Cal(crystal_sys_set,self.num,self.density_list, EXACT=True)
                    self.VFD.get_Volome_Fraction(self.structure_factor, crystal_sys_set, Mult_list, HKL_list, Theta_list,
                                            Intensity_list, ini_CL,self.wavelength,)
                else:
                    print('If you want to get the accurately determined of mass fraction, Please input the structure factor!')
            else:
                print('Type Error \'EXACT\'')
            return Rp, Rwp, i_ter, flag,ini_CL

        else:
            return -1, -1, -1, -1,-1

    # Update parameters. / Determine the latent parameters.
    # The main function of EM-Bragg Solver
    # ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO formula (2.12-2.25)
    def up_parameter(self, hkl_data,two_theta, intensity,area, min_i, i_obser, bac,lmd,):
        # two_theta, intensity represent the user-selected range
        # number of mixted components, int
        multitasks = len(self.Lattice_constants)
        # determine the systems[[],[],...]
        crystal_sys_set = cal_system(self.Lattice_constants)
        # calculate the interplanar distances[[],[],...]
        d_list_set = BLD.get_d_space_multitask(crystal_sys_set, hkl_data, self.Lattice_constants)
        # calculate the peak locations [[],[],...]
        ini_p_set = BLD.get_new_mui_multitask(d_list_set, self.wavelength)
     
        # read in the calculated peak locations by Bragg law, a list
        # prepare documents required for EM
        ini_p = []
        for task in range(multitasks):
            for j in range(len(ini_p_set[task])):
                ini_p.append(ini_p_set[task][j])

        if np.array(ini_p).min() <= two_theta.min():
            print('The smallest measuring angle is :', two_theta.min(),
                  ', Which is larger than the angle corresponding to min(H,K,L) : ',np.array(ini_p).min())
            print('Imported HKL file has errors')
            raise NameError('NOTE! H K L matching error')
        
        if np.array(ini_p).max() >= two_theta.max():
            print('The largest measuring angle is :', two_theta.max(),
                  ', Which is lower than the angle corresponding to max(H,K,L) : ',np.array(ini_p).max())
            print('Imported HKL file has errors')
            raise NameError('NOTE! H K L matching error')
        
        if self.loadParams == True:
            ini_w_list, ini_p1_list, ini_p2_list = readfile(os.path.join(self.FRfolder,"modelparams.csv"))
            print('—————————— Loaded parameters successfully ——————————')
        else:
            # Initialize the parameters
            print('—————————— Initilize the parameters by WPEM ——————————')
            if len(self.wavelength) == 1:
                ini_w_list, ini_p1_list, ini_p2_list = self.initial_p_single(ini_p, area,two_theta,intensity)
            elif len(self.wavelength) == 2:
                ini_w_list, ini_p1_list, ini_p2_list = self.initial_p(ini_p, area,two_theta,intensity)
            else:
                print('Your input wavelength number is unsupported in current version of WPEM')
        print('Parameter initialization has been completed \n')

        # total number of peaks
        k_ln = int(len(ini_p1_list))
        # total number of sub-peaks, one PV peaks contains two sub-peaks
        k = k_ln * 2
        # the total value of all position's intensity
        n_all = sum(intensity)
        # for iterating 
        new_w_list = copy.deepcopy(ini_w_list)
        new_p1_list = copy.deepcopy(ini_p1_list)
        new_p2_list = copy.deepcopy(ini_p2_list)

        # EM iteration
        i_ter = 0
        log_likelihood = [1e6]
        Rp = [30]
        Rwp = [30]
        # the Bragg peaks in two different data formats
        mui_abc = copy.deepcopy(new_p1_list) # a list []
        mui_abc_set = copy.deepcopy(ini_p_set) # a 2-d list [[],[],]
        # the inputs lattice constants [[],[],]
        ini_LC = copy.deepcopy(self.Lattice_constants)

        # Initilize the lock 
        # If the likelihood value drops multiple times in a row, stop the iteration
        lock = 0
        # Enter the EM-Bragg iteration
        for iteration in tqdm(range(self.iter_max)):
            w_list = copy.deepcopy(new_w_list)
            p1_list = copy.deepcopy(mui_abc)
            p2_list = copy.deepcopy(new_p2_list)
            i_ter += 1
            # E step, calculate the distribution of latent variables
            gamma_ji, gamma_ji_l, p_as_ji = gamma_ji_list(two_theta, w_list, p1_list, p2_list,self.s_angle, self.asy_C)
            # M step, update parameters by Q function
            # update w_list
            denominator = []
            denominator_l = []

            for i_l in range(k_ln):
                denominator_l.append(np.multiply(gamma_ji_l[:, i_l:i_l + 1].T, intensity).sum())

            for i in range(k):
                denominator.append(np.multiply(gamma_ji[:, i:i + 1].T, intensity).sum())
                new_w_list[i] = denominator[i] / n_all * area
            
            if self.Cu_tao == None:
                pass
            else:
                new_w_list = Cu_ab(new_w_list,self.wavelength,self.Cu_tao)
            
            # update mu
            denominator_mu = []
            for mu_d in range(k_ln):
                mu_k = mu_d * 2
                de = (p2_list[mu_k + 1] / p2_list[mu_k]) * (2 * np.pi)
                denominator_mu.append(denominator_l[mu_d] + denominator[mu_k + 1] / de)
                denominator_mu.append(denominator_l[mu_d] * de + denominator[mu_k + 1])

            numerator_mu = []
            for mu_n in range(k_ln):
                mu_half = mu_n * 2
                numerator_mu.append(
                    np.multiply(np.multiply(gamma_ji_l[:, mu_n:mu_n + 1].T, two_theta), intensity).sum())
                numerator_mu.append(np.multiply(np.multiply(gamma_ji[:, mu_half + 1:mu_half + 2].T, two_theta),intensity).sum())

            if type(self.IniEpoch) != int:
                print('type error, InitializationEpoch must be an integer')
            elif i_ter >= self.IniEpoch :
                for i_ln in range(k_ln):
                    mu_k = 2 * i_ln
                    new_p1_list[i_ln] = (numerator_mu[mu_k] / (denominator_mu[mu_k])) + \
                                        (numerator_mu[mu_k + 1] / (denominator_mu[mu_k + 1] ))

            else: new_p1_list = copy.deepcopy(p1_list)
            __new_mu_set = Distribute_mu(new_p1_list, mui_abc, hkl_data, lmd)
            # a set of mu [[mu1,mu2,mu3....],[],..]
            

            # Computing Bragg steps in parallel
            pool = ProcessPoolExecutor(self.cpu)
            outlist = []
            for task in range(multitasks):
                out = pool.submit(self.BraggIteration, __new_mu_set, mui_abc_set, ini_LC, hkl_data, crystal_sys_set, task,i_ter)
                outlist.append(out.result())
            pool.shutdown(True)

            for task in range(multitasks):
                ini_LC[task] = np.round(outlist[task][0][task],5)
                mui_abc_set[task] = outlist[task][1][task]
                __new_mu_set[task] = outlist[task][2][task]

            mui_abc = []
            for task in range(multitasks):
                for j in range(len(mui_abc_set[task])):
                    mui_abc.append(mui_abc_set[task][j])
                # mui_abc is the renew p1_list

            print("WPEM %s-th iteration" % i_ter)
            print(ini_LC)

            read_mui = []
            for task in range(multitasks):
                for j in range(len(__new_mu_set[task])):
                    read_mui.append(__new_mu_set[task][j])
            # peaks in list form
            p1_list = copy.deepcopy(read_mui)

            for i_ln in range(k_ln):
                i_l = i_ln * 2
                i_n = i_l + 1
                # update the parameter of lorenz distribution, γi
                new_p2_list[i_l] = np.sqrt(
                    solve_sigma2(two_theta, gamma_ji_l[:, i_ln:i_ln + 1], intensity,
                                      p1_list[i_ln], denominator_l[i_ln],self.limit))
                # update the parameter of normal distribution, σi^2
                new_p2_list[i_n] = solve_sigma2(two_theta, gamma_ji[:, i_n:i_n + 1], intensity,
                                                     p1_list[i_ln], denominator[i_n],self.limit)

            # calculate the log likelihood value
            log_likelihood.append(np.multiply(np.log(mix_normal_lorenz_density(two_theta,
                                                                                    new_w_list, p1_list, new_p2_list,
                                                                                    p_as_ji)).T, intensity).sum())

            # calculate the WPEM fitted crystal diffraction intensity
            i_epoch = mix_normal_lorenz_density(two_theta, new_w_list, p1_list, new_p2_list, p_as_ji)

            # add the small constant 
            i_epoch = i_epoch + min_i

            # calculate the the fitting effect at this iteration
            # ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO formula (3.1)
            p_error = [] 
            wp_error = []  
            _Rsquare = [] # R2
            y = []

            for j in range(len(two_theta)):
                y.append(bac[j] + i_epoch[j])
                p_error.append(float(abs(y[j] - i_obser[j])))
                wp_error.append((p_error[j] ** 2) / max(float(i_obser[j]),1))
            obs = sum(i_obser)
            p_error_sum = sum(p_error)
            wp_error_sum = sum(wp_error)
            Rp.append(p_error_sum / obs * 100)
            Rwp.append(np.sqrt(wp_error_sum / obs) * 100)
            _Rsquare.append(Rsquare(y, i_obser))

            print('Rp:%.3f' % (p_error_sum / obs * 100) + ' | Rwp:%.3f' % (np.sqrt(wp_error_sum / obs) * 100) + ' | Rsquare:%.3f' % (Rsquare(y, i_obser)))

            # lock by detecting log likelihood
            if log_likelihood[-1] < log_likelihood[-2]:
                lock += 1

            flag = -1
            # Determine the convergence type and return
            if abs(log_likelihood[-1] - log_likelihood[-2]) <= self.iter_limit:
                flag = 1
                break
           
            if min(new_w_list) <= self.w_limit:
                flag = 2
                break

            if i_ter > self.iter_max:
                flag = 3
                break

            if lock >= self.lock_num:
                flag = 4
                break
        
        # WPEM fitted pattern 
        __i_out = mix_normal_lorenz_density(two_theta, new_w_list, p1_list, new_p2_list, p_as_ji)

        i_calc = list(chain(*__i_out))
        # __i_out = [[],[],...,[]]  --- > [ , , , ]
        
        # Model = 'REFINEMENT', the background strength is updated once after fitting according to the distribution variance
        if self.MODEL == 'REFINEMENT':
            # tolerance
            bacerr = i_calc - intensity    # ypre - ytrue

            if type(self.Var) == float :
                bacerrup1 = np.where((bacerr >= -self.Var) & (bacerr <= self.Var), 0, bacerr)
                bacerrup2 = np.where(bacerrup1 <= -self.Var, bacerrup1 + self.Var, bacerrup1)
                bacerrup3 = np.where(bacerrup2 >= self.Var, bacerrup2 - self.Var, bacerrup2)
                
                # updata the no_bac and no_con intensity
                inten_upbac = i_calc - bacerrup3           
                # bac_old + inten + minc = bac_new + inten_upbac + minc
                bac_up = bac + intensity - inten_upbac
                # Save update background intensity at WPEMFittingResults(FRfolder)
                with open(os.path.join(self.FRfolder,
                    'upbackground_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM,
                    day=named, hour=nameh, minute=namem)),'w') as wfid:
                    for j in range(len(two_theta)):
                        print(two_theta[j], end=', ', file=wfid)
                        print(float(bac_up[j]), file=wfid)
                with open(os.path.join(self.DCfolder,'upbackground.csv'),'w') as wfid:
                    for j in range(len(two_theta)):
                        print(two_theta[j], end=', ', file=wfid)
                        print(float(bac_up[j]), file=wfid)
            else:
                bacerrup = []
                for angle in range(len(bacerr)):
                    if bacerr[angle] <= -self.Var[angle]:
                        bacerrup.append(bacerr[angle] + self.Var[angle])
                    elif bacerr[angle] >= self.Var[angle]:
                        bacerrup.append(bacerr[angle] - self.Var[angle])
                    else :
                        bacerrup.append(0)
                inten_upbac = list(map(lambda x: abs(x[0]-x[1]), zip(i_calc,bacerrup)))

                # bac_old + inten + minc = bac_new + inten_upbac + minc
                bac_up = bac + intensity - inten_upbac
                                
                # Save update background intensity
                with open(os.path.join(self.FRfolder, 'upbackground_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                        month=nameM,day=named, hour=nameh, minute=namem)), 'w') as wfid:
                    for j in range(len(two_theta)):
                        print(two_theta[j], end=', ', file=wfid)
                        print(float(bac_up[j]), file=wfid)
                with open(os.path.join(self.DCfolder,'upbackground.csv'),'w') as wfid:
                    for j in range(len(two_theta)):
                        print(two_theta[j], end=', ', file=wfid)
                        print(float(bac_up[j]), file=wfid)
        
        # Model = 'ANALYSIS', the background strength is frozen
        elif self.MODEL == 'ANALYSIS':
            bac_up = bac
            Other_peaks_diff = bac + intensity + min_i - i_calc
            # Smooth out any negative intensities that appears
            Other_peaks = np.where(Other_peaks_diff < 0, 0, Other_peaks_diff)
            with open(os.path.join(self.FRfolder,'Other_Phase_peaks{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                                          month=nameM, day=named, hour=nameh, minute=namem)),'w') as wfid:
                for j in range(len(two_theta)):
                    print(two_theta[j], end=', ', file=wfid)
                    print(float(Other_peaks[j]), file=wfid)
            
            plt.plot(two_theta, Other_peaks, '-g', label="residual", )
            plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            plt.legend()
            plt.savefig(os.path.join(self.FRfolder, 'Residual_Intensity_{year}.{month}.{day}_{hour}.{minute}.png'.format(year=namey,
                                 month=nameM, day=named,hour=nameh,minute=namem)),dpi=800)
            plt.show()
            plt.clf()

        # Recalculate Rwp after updating the background
        y = []
        p_error = []
        wp_error = []

        for j in range(len(two_theta)):
            y.append(bac_up[j] + i_calc[j] + min_i)
            p_error.append(float(abs(y[j] - i_obser[j])))
            # in case the real intensity equal to zero
            wp_error.append((p_error[j] ** 2) / max(float(i_obser[j]),1))

        obs = sum(i_obser)
        p_error_sum = sum(p_error)
        wp_error_sum = sum(wp_error)
        Rp.append(p_error_sum / obs * 100)
        Rwp.append(np.sqrt(wp_error_sum / obs) * 100)
        _Rsquare.append(Rsquare(y, i_obser))
        print('After update the background :  ',
              'Rp = %.3f' % (p_error_sum / obs * 100) + ' | Rwp = %.3f' % (np.sqrt(wp_error_sum / obs) * 100) + ' | Rsquare = %.3f' % (Rsquare(y, i_obser)))

        plt.plot(two_theta, inten_upbac+bac_up, '-k', label="Experimental Profile (crystal)", )
        plt.plot(two_theta, i_calc+bac_up, 'g', linestyle='--',label="WPEM fitting profile",)
        plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.FRfolder, 'WPEM_fittingresult_{year}.{month}.{day}_{hour}.{minute}.png'.format(year=namey,
                    month=nameM,day=named, hour=nameh, minute=namem)),dpi=800)
        plt.show()
        plt.clf()

        plt.plot([i for i in range(1, len(log_likelihood))], log_likelihood[1:], '-p', color='gray',
                 markersize=10, linewidth=2, markerfacecolor='white',markeredgecolor='gray', markeredgewidth=1,
                 label='log_likelihood')
        plt.xlabel('Iterations', )
        plt.ylabel('Log-likelihood', )
        plt.legend()
        plt.savefig(os.path.join(self.FRfolder, 'WPEM_log_likelihood_{year}.{month}.{day}_{hour}.{minute}.png'.format(year=namey,
                    month=nameM,day=named, hour=nameh,minute=namem)),dpi=800)
        plt.show()
        plt.clf()

        plt.plot([i for i in range(1, len(Rp))], Rp[1:], '-p', color='gray',
                 markersize=10, linewidth=2, markerfacecolor='white',markeredgecolor='gray', markeredgewidth=1,
                 label='Rp%', )
        plt.plot([i for i in range(1, len(Rwp))], Rwp[1:], '-o', color='black',
                 markersize=10, linewidth=2, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1,
                 label='Rwp%', )
        plt.xlabel('Iterations', )
        plt.ylabel('R factors', )
        plt.legend()
        plt.savefig( os.path.join(self.FRfolder, 'WPEM_Rfactor_{year}.{month}.{day}_{hour}.{minute}.png'.format(year=namey, month=nameM,
                    day=named, hour=nameh, minute=namem)), dpi=800)
        plt.show()
        plt.clf()

        if flag == -1:
            flag = 3

        return new_w_list, p1_list, new_p2_list, i_ter, flag, ini_LC, mui_abc_set, i_calc, bac_up, Rp[-1], Rwp[-1], _Rsquare[-1], crystal_sys_set


    # Initialization parameters for single ray. 
    # Define the initial value of the EM process
    def initial_p_single(self, rawpeak, i_obs,two_theta,intensity):
        """
        This is a method for initializing the parameters of a single peak in the EM process.
        When bta>=bta_threshold, the initial values of γi and σi^2 are set according to the original XRD pattern.
        When bta<bta_threshold, the initial values of γi and σi^2 are set to bta, and all of Ai are 0.5.
        rawpeak : the raw peak locations
        i_obs : Integrated area of whole pattern
        k : number of peaks
        """
        k = len(rawpeak)
        # k is the number of peak i_obs is the measure of profile
        Ai = 0.5

        # store peak locations
        p1_list = []
        for i in range(k):
            p1_list.append(rawpeak[i])

        # store peak intensity values
        i_list = []
        # store peak intensity at fwhm 
        fwhm_list = []
        # store peak indexes
        p = []
        for i in range(k):
            p.append(p_index(two_theta, rawpeak[i]))
            i_list.append(intensity[p[i]])
            fwhm_list.append(i_list[i] / 2)
            # half intensity of each peak
        i_sum = sum(i_list)
        # store peak weights
        w_list = []
        # store peak's γi and σi^2
        p2_list = []

        # define the list of peak weights
        # assuming that the broadening of the diffraction peaks is small
        if self.bta >= self.bta_threshold:
            for i_ln in range(k):
                i_w = i_list[i_ln] / i_sum
                w = i_obs * i_w
                w_list.append(w * self.bta)
                w_list.append(w * (1 - self.bta))

            n = p[0] - 1

            # this part is defined for searching fwhm for the first peak 
            ### star
            for i in range(p[0]):
                # ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO table (1.2)
                # gamma roughly equal to fwhm and sigma2 equal to the gamma2
                if intensity[n - i] <= (intensity[p[0]] / 2):
                    p2_list.append(2 * (two_theta[p[0]] - two_theta[n - i]) * self.bta) # γi
                    p2_list.append((2 * (two_theta[p[0]] - two_theta[n - i]) * (1 - self.bta)) ** 2) # σi^2
                    break
            # in case not found fwhm from the left side of the peak 
            if len(p2_list) == 0:
                p2_list.append(2 * (two_theta[p[0]] - two_theta[0]) * self.bta)
                p2_list.append((2 * (two_theta[p[0]] - two_theta[0]) * (1 - self.bta)) ** 2)
            ### end

            # this part is defined for searching fwhm for the middle peak 
            ### star
            for i_l in range(1, k - 1):
                fwhm_i = fwhm_find(p[i_l], p[i_l - 1], fwhm_list[i_l], intensity, two_theta)
                p2_list.append(fwhm_i * self.bta)
                p2_list.append((fwhm_i * (1 - self.bta)) ** 2)
            ### end

            # this part is defined for searching fwhm for the last peak 
            ### star
            n = p[-1] + 1
            for i in range((len(two_theta) - p[-1] - 1)):
                if intensity[n + i] <= (intensity[p[-1]] / 2):
                    p2_list.append(2 * (two_theta[n + i] - two_theta[p[k - 1]]) * self.bta)
                    p2_list.append((2 * (two_theta[n + i] - two_theta[p[k - 1]]) * (1 - self.bta)) ** 2)
                    break

            if len(p2_list) == (k - 1) * 2:
                p2_list.append(2 * (two_theta[-1] - two_theta[p[-1]]) * self.bta)
                p2_list.append((2 * (two_theta[-1] - two_theta[p[-1]]) * (1 - self.bta)) ** 2)
            ### end

            # incase extremely unreasonable values
            for i in range(k):
                p2_ln = 2 * i
                # at initial the gamma value must above 0.5
                if p2_list[p2_ln] == 0:
                    p2_list[p2_ln] = 0.5
                    p2_list[p2_ln + 1] = 0.5

        # bta<bta_threshold, the initial values of γi and σi^2 are set to bta_threshold, and all of Ai are 0.5.
        elif self.bta < self.bta_threshold:
            for i_ln in range(k):
                i_w = i_list[i_ln] / i_sum
                w = i_obs * i_w
                w_list.append(w * Ai)
                w_list.append(w * Ai)

            for i in range(k):
                p2_list.append(self.bta_threshold)
                p2_list.append(self.bta_threshold)
            
            print("Pay Attention! The ratio of Lorentzian components defined in PV function is too small")
            print("That may cause a reasonable result")

        return w_list, p1_list, p2_list

    
    # Initialization parameters for mixed rays. 
    # Define the initial value of the EM process
    # see function initial_p_single defined above
    def initial_p(self, rawpeak, i_obs,two_theta,intensity):
        k = len(rawpeak)
        Ai = 0.5

        p1_list = []
        for i in range(k):
            p1_list.append(rawpeak[i])

        i_list = []
        fwhm_list = []
        p = []
        for i in range(k):
            p.append(p_index(two_theta, rawpeak[i]))
            i_list.append(intensity[p[i]])
            fwhm_list.append(i_list[i] / 2)
        i_sum = sum(i_list)
        w_list = []
        p2_list = []

        # Only take two lambda into consideration, the intensity of lambda 2 is roughly equal to 1/2 of lambda 1
        if self.bta >= self.bta_threshold:
            for i_ln in range(k):
                if i_ln % 2 == 0:
                    i_w = (i_list[i_ln] + i_list[i_ln + 1]) / i_sum
                    w = i_obs * i_w * 2 / 3
                    w_list.append(w * self.bta)
                    w_list.append(w * (1 - self.bta))
                    w_2 = i_obs * i_w / 3
                    w_list.append(w_2 * self.bta)
                    w_list.append(w_2 * (1 - self.bta))
            
            # first peak
            n = p[0] - 1
            for i in range(p[0]):
                if intensity[n - i] <= (intensity[p[0]] / 2):
                    p2_list.append(2 * (two_theta[p[0]] - two_theta[n - i]) * self.bta)
                    p2_list.append((2 * (two_theta[p[0]] - two_theta[n - i]) * (1 - self.bta)) ** 2)
                    break
            if len(p2_list) == 0:
                p2_list.append(2 * (two_theta[p[0]] - two_theta[0]) * self.bta)
                p2_list.append((2 * (two_theta[p[0]] - two_theta[0]) * (1 - self.bta)) ** 2)

            # middle peaks
            for i_l in range(1, k - 1):
                fwhm_i = fwhm_find(p[i_l], p[i_l - 1], fwhm_list[i_l], intensity, two_theta)
                p2_list.append(fwhm_i * self.bta)
                p2_list.append((fwhm_i * (1 - self.bta)) ** 2)
            
            # last peak
            n = p[-1] + 1
            for i in range((len(two_theta) - p[-1] - 1)):
                if intensity[n + i] <= (intensity[p[-1]] / 2):
                    p2_list.append(2 * (two_theta[n + i] - two_theta[p[k - 1]]) * self.bta)
                    p2_list.append((2 * (two_theta[n + i] - two_theta[p[k - 1]]) * (1 - self.bta)) ** 2)
                    break
            if len(p2_list) == (k - 1) * 2:
                p2_list.append(2 * (two_theta[-1] - two_theta[p[-1]]) * self.bta)
                p2_list.append((2 * (two_theta[-1] - two_theta[p[-1]]) * (1 - self.bta)) ** 2)

            for i in range(k):
                p2_ln = 2 * i
                if p2_list[p2_ln] == 0:
                    p2_list[p2_ln] = 0.5
                    p2_list[p2_ln + 1] = 0.5

        elif self.bta < self.bta_threshold:
            for i_ln in range(k):
                if i_ln % 2 == 0:
                    i_w = (i_list[i_ln] + i_list[i_ln + 1]) / i_sum
                    w = i_obs * i_w * 2 / 3
                    w_list.append(w * Ai)
                    w_list.append(w * Ai)
                    w_2 = i_obs * i_w / 3
                    w_list.append(w_2 * Ai)
                    w_list.append(w_2 * Ai)

            for i in range(k):
                p2_list.append(self.bta)
                p2_list.append(self.bta)
            
            print("Pay Attention! The ratio of Lorentzian components defined in PV function is too small")
            print("That may cause a reasonable result")

        return w_list, p1_list, p2_list
    
    def BraggIteration(self, new_mu_set, mui_abc_set, ini_LC, hkl_data, crystal_sys_set,task,i_ter):
        
        """
        This function defines the Bragg-step in EM_Bragg Solver
        For a single crystal system
        """
        mui_sort = copy.deepcopy(new_mu_set[task])
        mui_sort.sort()
        # fun get_angle_sort matches the peaks obtained by the EM process with the theoretical peaks in order
        mui_cal_em_match = BLD.get_angle_sort(mui_sort, mui_abc_set[task])
        
        # read in the lattice constants
        ini_a = (ini_LC[task][0])
        ini_b = (ini_LC[task][1])
        ini_c = (ini_LC[task][2])
        ini_la1 = (ini_LC[task][3])
        ini_la2 = (ini_LC[task][4])
        ini_la3 = (ini_LC[task][5])

        lattice_h = hkl_data[task][0]
        lattice_k = hkl_data[task][1]
        lattice_l = hkl_data[task][2]
        # check if lattice constant changes are allowed
        if self.singal[task] == 0:
            fixed = False
        elif self.singal[task] == 1:
            fixed = True


        if type(self.IniEpoch) != int:
            print('type error, InitializationEpoch must be an integer')
        elif self.IniEpoch >= i_ter:
            return ini_LC, new_mu_set, new_mu_set
        
        # associate the peak's locations by Bragg law
        lattice_a, lattice_b, lattice_c, lattice_ang1, lattice_ang2, lattice_ang3, mui_abc_part = BLD.OptmialLatticeConstant(
            crystal_sys_set[task], mui_abc_set[task], mui_cal_em_match,self.subset_number,self.low_bound, self.up_bound, 
            lattice_h,lattice_k, lattice_l, ini_a, ini_b, ini_c,ini_la1, ini_la2, ini_la3, self.wavelength,fixed, tao=0.05)

        # write and update
        C_ini_LC = copy.deepcopy(ini_LC)
        C_mui_abc_set = copy.deepcopy(mui_abc_set)
        C_new_mu_set = copy.deepcopy(new_mu_set)
        C_ini_LC[task][0] = copy.deepcopy(lattice_a)
        C_ini_LC[task][1] = copy.deepcopy(lattice_b)
        C_ini_LC[task][2] = copy.deepcopy(lattice_c)
        C_ini_LC[task][3] = copy.deepcopy(lattice_ang1)
        C_ini_LC[task][4] = copy.deepcopy(lattice_ang2)
        C_ini_LC[task][5] = copy.deepcopy(lattice_ang3)
        C_mui_abc_set[task] = mui_abc_part
        C_new_mu_set[task] = BLD.get_mui_sort(new_mu_set[task], mui_cal_em_match, mui_abc_part)

        return C_ini_LC, C_mui_abc_set, C_new_mu_set
    

################################################################
# Functions
# Get the initial {Hi, Ki, Li}
# Supports two read-in formats 
# Supports fullprof import files and WPEM simulation module generated files, the latter is recommended.
def read_ini_peak(data_file, Num, FRfolder):
    peak_data = pd.read_csv(data_file, header=0)
    h_all = np.array(peak_data.H)
    k_all = np.array(peak_data.K)
    l_all = np.array(peak_data.L)

    try:
        # read in fullprof files
        code_n = np.array(peak_data.Code)
        h1, k1, l1 = [], [], []
        # contains two wavelength values
        if code_n[0] == 2 or code_n[1] == 2:
            code_n = np.array(peak_data.Code)
            for i in range(len(h_all)):
                if code_n[i] == 1:
                    h1.append(h_all[i])
                    k1.append(k_all[i])
                    l1.append(l_all[i])
            # save organized files 
            with open(os.path.join(FRfolder,
                                'hkl{Num}_{year}.{month}.{day}_{hour}.{minute}.csv'.format(Num=Num, year=namey,
                                        month=nameM, day=named, hour=nameh,minute=namem)),
                    'w') as wfid:
                print('H', end=',', file=wfid)
                print('K', end=',', file=wfid)
                print('L', file=wfid)
                for j in range(len(h1)):
                    print(h1[j], end=',', file=wfid)
                    print(k1[j], end=',', file=wfid)
                    print(l1[j], file=wfid)
            print('Diffraction indexs have been obtained by WPEM')

        else:
            # contains one wavelength value
            with open(os.path.join(FRfolder,
                                'hkl{Num}_{year}.{month}.{day}_{hour}.{minute}.csv'.format(Num=Num, year=namey,
                                    month=nameM, day=named,hour=nameh, minute=namem)),
                    'w') as wfid:
                print('H', end=',', file=wfid)
                print('K', end=',', file=wfid)
                print('L', file=wfid)
                for j in range(len(h_all)):
                    print(float(h_all[j]), end=',', file=wfid)
                    print(float(k_all[j]), end=',', file=wfid)
                    print(float(l_all[j]), file=wfid)
            print('Diffraction indexs have been obtained by WPEM')

    except :
        # read in WPEM simulation module generatedfiles
        print('The input HKL document is matched with WPEM')
        with open(os.path.join(FRfolder,
                                'hkl{Num}_{year}.{month}.{day}_{hour}.{minute}.csv'.format(Num=Num, year=namey,
                                    month=nameM, day=named,hour=nameh, minute=namem)),
                    'w') as wfid:
                print('H', end=',', file=wfid)
                print('K', end=',', file=wfid)
                print('L', file=wfid)
                for j in range(len(h_all)):
                    print(float(h_all[j]), end=',', file=wfid)
                    print(float(k_all[j]), end=',', file=wfid)
                    print(float(l_all[j]), file=wfid)
        print('Diffraction indexs have been obtained by WPEM')

def readfile(loadParams):
    # readin the initial parameters from file
    with open(loadParams, 'r') as file:
        reader = csv.reader(file)
        w_read = [float(item) for item in next(reader)]
        p1_read = [float(item) for item in next(reader)]
        p2_read = [float(item) for item in next(reader)]
    return w_read, p1_read, p2_read


def try_read(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, header=None, names=['two_theta', 'intensity'])
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(path, header=None, names=['two_theta', 'intensity'])
    else:
        raise ValueError(f"Unsupporting formate: {ext}")


def read_data_file(filepath):
    tried_paths = []
    try:
        return try_read(filepath)
    except Exception as e1:
        tried_paths.append(filepath)

    base = os.path.splitext(filepath)[0]
    csv_path = base + '.csv'
    try:
        return try_read(csv_path)
    except Exception as e2:
        tried_paths.append(csv_path)

    xlsx_path = base + '.xlsx'
    try:
        return try_read(xlsx_path)
    except Exception as e3:
        tried_paths.append(xlsx_path)
    raise FileNotFoundError(f"Fail to read in:{tried_paths}")
        
################################################################