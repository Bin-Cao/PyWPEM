# X-ray photoelectron spectroscopy decomposition Module
# Author: Bin CAO <binjacobcao@gmail.com>

import re
import copy
import csv
from itertools import chain
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..EMBraggOpt.WPEMFuns.SolverFuns import *
import time
import os

timename = time.localtime(time.time())
namey, nameM, named, nameh, namem = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min

"""
This class is the core solver of XPS module in WPEM 

Please feel free to contact Bin Cao (bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code.

Contribution and suggestions are always welcome. You can also contact the authors for research collaboration.
"""

class XPSsolver(object):
    def __init__(
        self, Var,asy_C, s_energy,atomIdentifier,SatellitePeaks, no_bac_df,original_df,bacground_df,energy_range,
        bta, bta_threshold,limit, iter_limit, w_limit,iter_max, lock_num, InitializationEpoch, loadParams,tao,ratio,work_dir=None

    ):
      
        self.Var = Var # variance of background intensity
        self.asy_C = asy_C # asymmetric parameter for descripting the asymmetric peak
        self.s_energy = s_energy # asymmetric peak's range, a float
        self.atomIdentifier = atomIdentifier 
        # A list of atom identifiers, please 
        # please put the split electron descriptor together
        # i.e., [['Cu2+','2p3/2',935.6],['Cu2+','2p1/2',938.4],['Cu+','2p1/2',934.6]]
        self.SatellitePeaks = SatellitePeaks # list of satellite peaks
        # i.e., [['Cu2+','2p3/2',934.8],['Cu2+','2p3/2',934.9],['Cu+','2p1/2',933.7]]
        self.no_bac_df = no_bac_df # electron binding energy
        self.original_df = original_df # experimental observations
        self.bacground_df = bacground_df # fitted bacground energy
        self.energy_range = energy_range # energy range studied in the spectrum 
        self.bta = bta # the ratio of Lorentzian components in PV function
        self.bta_threshold = bta_threshold # a preset lower boundary of bta
        self.limit = limit #  a preset lower boundary of sigma2
        self.iter_limit = iter_limit #  a preset threshold iteration promotion (likelihood) 
        self.w_limit = w_limit # a preset lower boundary of peak weight
        self.iter_max = iter_max # maximum number of iterations
        self.lock_num = lock_num # in case of loglikelihood iterations continously decrease  
        self.IniEpoch = InitializationEpoch # Initialization epoch
        self.loadParams = loadParams # if read in the initial parameters
        self.tao = tao
        self.ratio = ratio
        self.total_p_num = len(atomIdentifier) + len(SatellitePeaks)
        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        
        # mike dir
        if work_dir is None:
            self.XPScomfolder = 'XPScomponents'
            self.XPSFRfolder = 'XPSFittingProfile'
        else:
            self.XPScomfolder = os.path.join(work_dir, 'XPScomponents')
            self.XPSFRfolder = os.path.join(work_dir, 'XPSFittingProfile')

        os.makedirs(self.XPScomfolder, exist_ok=True)
        os.makedirs(self.XPSFRfolder, exist_ok=True)

        print('Initialization')
        print('-'*80)
    
    # Whole pattern decomposition execution function, implemented by data reading and XPSEM iteration
    def cal_output_result(self,):
       
        # Read non-background data, and the data format is energy-intensity/X-Y data.
        data = pd.read_csv(self.no_bac_df, header=None, names=['Benergy', 'intensity'])
        # Read raw/original data, and the data format is energy-intensity/X-Y data.
        in_data = pd.read_csv(self.original_df, header=None, names=['Benergy', 'intensity'])
        # Read background data, and the data format is energy-intensity/X-Y data.
        bac_data = pd.read_csv(self.bacground_df, header=None, names=['Benergy', 'intensity'])

        # Detect whether the user restricts the fitting range, and cut the diffranction range 
        if type(self.energy_range) == tuple:
            index = np.where((data.Benergy < self.energy_range[0]) | (data.Benergy > self.energy_range[1]))
            data = data.drop(index[0])
            in_data = in_data.drop(index[0])
            bac_data = bac_data.drop(index[0])
            pass
        elif self.energy_range == None:
            pass
        else:
            print('Type Error -energy_range-')


        o_x = np.array(in_data.Benergy)
        f_x = np.array(data.Benergy)
        b_x = np.array(bac_data.Benergy)

        # Check if the data matched
        if len(o_x) == len(f_x) and len(f_x) == len(b_x):
            
            # Pure electron binding energy
            EBenergy = np.array(data.Benergy)
            intensity = np.array(data.intensity)
            EBenergy, intensity = reorder_vector(EBenergy, intensity)
            # estimated background data
            bac = np.array(bac_data.intensity)
            # experimental XPS data
            i_obser = np.array(in_data.intensity)

            # sliding window
            fwhm_len = (EBenergy[1] - EBenergy[0]) / 20

            inten = copy.deepcopy(intensity)
            min_i = min(intensity)
            # Subtract the DC signal
            inten -= min_i
            # Calculation of integrated probability of measure function, XPS pattern
            area = theta_intensity_area(EBenergy, inten)

            print('\n')
            # Update parameters via XPSEM process
            w_list, p1_list, p2_list, i_ter, flag, \
            i_out, bac_up, Rp, Rwp, Rsquare, = self.up_parameter(EBenergy, inten, area, min_i, i_obser, bac)
 
            # This part is reserved due to historical version reasons
            # To speed up, the R factor is not calculated in the iteration
            # I consider one day to implement these part again
        
            csv_file = os.path.join(self.XPSFRfolder,"XPSparams.csv")
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

            y1 = [] # γi
            y2 = [] # σi^2
            for i in range(len(p1_list)):
                i_l = i * 2
                i_n = i_l + 1
                y1.append(float(p2_list[i_l]))
                y2.append(float(p2_list[i_n]))

            # WPEMXPS fitted pattern
            y= bac_up + i_calc

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
                for j in range(1, len(EBenergy) * 20):
                    fwhm_l = p1_list[i] - fwhm_len * j
                    # here each is ideal PV function
                    intensity_fwhm = mix_normal_lorenz_density_cal_fwhm(fwhm_l, w_list[i_ln: i_ln + 2], p1_list[i],
                                                                             p2_list[i_ln: i_ln + 2])
                    if intensity_fwhm <= inten_h:
                        peak_fwhm_list.append(fwhm_len * j * 2)
                        break

            # save result files 
            with open(os.path.join(self.XPSFRfolder,'XPSfittingProfile{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                                       month=nameM,day=named, hour=nameh,minute=namem)),'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
                    print(float(y[j]), file=wfid)

            with open(os.path.join(self.XPScomfolder, 'FittingProfile.csv'),'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
                    print(float(y[j]), file=wfid)

            with open(os.path.join(self.XPSFRfolder,'XPSPeakParas_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM,
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
            with open(os.path.join(self.XPScomfolder, 'QuantumState.csv'),'w') as wfid:
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

            return Rp, Rwp, i_ter, flag

        else:
            return -1, -1, -1, -1

    # Update parameters. 
    # The main function of XPSEM Solver
    def up_parameter(self, EBenergy, inten, area, min_i, i_obser, bac,): 
        OriPeaks, SatPeaks = SpinOrbitCoupling(self.atomIdentifier,self.SatellitePeaks)
        _OriPeaks = copy.copy(OriPeaks)
        _SatPeaks = copy.copy(SatPeaks)
        """
        OriPeaks:
        [['Cu', '2p3/2', 932.6,[2,2,3/2]], ['Cu', '2p1/2', 933.6,[2,2,1/2]],] # [n,l,j]
        [['Cu2+', '2p3/2', 933.4, [2,2,3/2]],]
        [['Zn', '2p3/2', 932.4,[2,2,3/2]],]

        SatPeaks:
        [['Cu2+','2p3/2',934.8],['Cu2+','2p3/2',934.9]]
        [['Cu+','2p1/2',933.7]]
        """
        # initilize the peak locations [...]
        ini_p = []
        for k in self.atomIdentifier:
            ini_p.append(k[2])
        for p in self.SatellitePeaks:
            ini_p.append(p[2])

        if self.loadParams == True:
            ini_w_list, ini_p1_list, ini_p2_list = readfile("./WPEMFittingResults/modelparams.csv")
            print('—————————— Loaded parameters successfully ——————————')
        else:
            # Initialize the parameters
            print('—————————— Initilize the parameters by WPEM ——————————')
            ini_w_list, ini_p1_list, ini_p2_list = self.initial_p_single(ini_p, area, EBenergy, inten,)
            
        print('Parameters of XPS are initialized has been completed \n')

        # total number of peaks
        k_ln = int(len(ini_p1_list))
        # total number of sub-peaks, one PV peaks contains two sub-peaks
        k = k_ln * 2
        # the total value of all position's intensity
        n_all = sum(inten)
        # for iterating 
        new_w_list = copy.deepcopy(ini_w_list)
        new_p1_list = copy.deepcopy(ini_p1_list)
        new_p2_list = copy.deepcopy(ini_p2_list)
      
        # EM iteration
        i_ter = 0
        log_likelihood = [1e6]
        Rp = [30]
        Rwp = [30]
        # The energies corresponding to electrons in different quantum states
        mui_set = copy.deepcopy(new_p1_list) # a list []

        # Initilize the lock 
        # If the likelihood value drops multiple times in a row, stop the iteration
        lock = 0
        # Enter the EM-Bragg iteration
        for iteration in tqdm(range(self.iter_max)):
            print("WPEM %s-th iteration" % i_ter)

            w_list = copy.deepcopy(new_w_list)
            p1_list = copy.deepcopy(mui_set)
            p2_list = copy.deepcopy(new_p2_list)
            i_ter += 1
            # E step, calculate the distribution of latent variables
            gamma_ji, gamma_ji_l, p_as_ji = gamma_ji_list(EBenergy, w_list, p1_list, p2_list, self.s_energy, self.asy_C,model='XPS')
            # M step, update parameters by Q function
            # update w_list
            denominator = []
            denominator_l = []

            for i_l in range(k_ln):
                denominator_l.append(np.multiply(gamma_ji_l[:, i_l:i_l + 1].T, inten).sum())

            for i in range(k):
                denominator.append(np.multiply(gamma_ji[:, i:i + 1].T, inten).sum())
                new_w_list[i] = denominator[i] / n_all * area
            
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
                    np.multiply(np.multiply(gamma_ji_l[:, mu_n:mu_n + 1].T, EBenergy), inten).sum())
                numerator_mu.append(np.multiply(np.multiply(gamma_ji[:, mu_half + 1:mu_half + 2].T, EBenergy),inten).sum())

            for i_ln in range(k_ln):
                mu_k = 2 * i_ln
                new_p1_list[i_ln] = (numerator_mu[mu_k] / (denominator_mu[mu_k])) + \
                                    (numerator_mu[mu_k + 1] / (denominator_mu[mu_k + 1] ))
            
            
            if i_ter <= self.IniEpoch:
                pass
            else:
                p1_list, new_w_list, _child_peaks= SLCoupling(new_p1_list,ini_p1_list, new_w_list,OriPeaks, SatPeaks,self.tao,self.ratio,p2_list)
            

            order = 0
            # just for showing, the peak locations in OriPeaks, SatPeaks are not used in the updating process
            for d in range(len(_OriPeaks)):
                for p in range(len(_OriPeaks[d])):
                    _OriPeaks[d][p][2]=round(p1_list[order],3)
                    order += 1
            print("The energies of electrons are:")
            print(_OriPeaks)

            print("The energies of satellite peak are:")
            for d in range(len(_SatPeaks)):
                for p in range(len(_SatPeaks[d])):
                    _SatPeaks[d][p][2]=round(p1_list[order],3)
                    order += 1
            print(_SatPeaks)


            for i_ln in range(k_ln):
                i_l = i_ln * 2
                i_n = i_l + 1
                # update the parameter of lorenz distribution, γi
                new_p2_list[i_l] = np.sqrt(
                    solve_sigma2(EBenergy, gamma_ji_l[:, i_ln:i_ln + 1], inten,
                                      p1_list[i_ln], denominator_l[i_ln],self.limit))
                # update the parameter of normal distribution, σi^2
                new_p2_list[i_n] = solve_sigma2(EBenergy, gamma_ji[:, i_n:i_n + 1], inten,
                                                     p1_list[i_ln], denominator[i_n],self.limit)


            # constrains the peak's shape
            shape_params = copy.deepcopy(new_p2_list)
            for _num in range(int(len(new_p2_list)/2)):
                _gamma = shape_params[2*_num]
                _sigma2 = shape_params[2*_num+1]
                Γ = (2*_gamma + np.sqrt(8*np.log(2)*_sigma2)) /2
                new_p2_list[2*_num]= Γ /2 * 0.9 + 0.1 *_gamma
                new_p2_list[2*_num+1]= Γ**2 / (8*np.log(2)) * 0.9 + 0.1 *_sigma2

            # constrains of the peak broadening of satellite peaks
            try:
                for _k in range(len(_child_peaks)):
                    if len(_child_peaks[_k]) != 1:
                        Γ_list = []
                        for _sp in _child_peaks[_k]:
                            Γ_list.append((2*new_p2_list[2*_sp] + np.sqrt(8*np.log(2)*new_p2_list[2*_sp+1]))/2) 
                        _Γ = np.array(Γ_list).mean()
                        for __sp in _child_peaks[_k]:
                            new_p2_list[2*__sp]= _Γ /2
                            new_p2_list[2*__sp+1]= _Γ**2 / (8*np.log(2))
            except : pass

            # calculate the log likelihood value
            log_likelihood.append(np.multiply(np.log(mix_normal_lorenz_density(EBenergy,
                                                                                    new_w_list, p1_list, new_p2_list,
                                                                                    p_as_ji)).T, inten).sum())

            # calculate the WPEM fitted XPS profile
            i_epoch = mix_normal_lorenz_density(EBenergy, new_w_list, p1_list, new_p2_list, p_as_ji)

            # add the small constant 
            i_epoch = i_epoch + min_i

            # calculate the the fitting effect at this iteration
            p_error = [] 
            wp_error = []  
            _Rsquare = [] # R2
            y = []

            for j in range(len(EBenergy)):
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
            print('\n')
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

        # ending the loops
        # WPEM fitted XPS pattern 
        __i_out = mix_normal_lorenz_density(EBenergy, new_w_list, p1_list, new_p2_list, p_as_ji)

        i_calc = list(chain(*__i_out))
        # __i_out = [[],[],...,[]]  --- > [ , , , ]
        
      
        # tolerance update the background
        bacerr = i_calc - inten    # ypre - ytrue
        if type(self.Var) == float :
            bacerrup1 = np.where((bacerr >= -self.Var) & (bacerr <= self.Var), 0, bacerr)
            bacerrup2 = np.where(bacerrup1 <= -self.Var, bacerrup1 + self.Var, bacerrup1)
            bacerrup3 = np.where(bacerrup2 >= self.Var, bacerrup2 - self.Var, bacerrup2)
            
            # updata the no_bac and no_con intensity
            inten_upbac = i_calc - bacerrup3           
            # bac_old + inten + minc = bac_new + inten_upbac + minc
            bac_up = bac + inten - inten_upbac
            # Save update background intensity at WPEMFittingResults
            with open(os.path.join(self.XPSFRfolder,
                'upbackground_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey, month=nameM,
                day=named, hour=nameh, minute=namem)),'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
                    print(float(bac_up[j]), file=wfid)
            with open(os.path.join(self.XPScomfolder,'upbackground.csv'),'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
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
            bac_up = bac + inten - inten_upbac
                            
            # Save update background intensity
            with open(os.path.join(self.XPSFRfolder, 'upbackground_{year}.{month}.{day}_{hour}.{minute}.csv'.format(year=namey,
                    month=nameM,day=named, hour=nameh, minute=namem)), 'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
                    print(float(bac_up[j]), file=wfid)
            with open(os.path.join(self.XPScomfolder,'upbackground.csv'),'w') as wfid:
                for j in range(len(EBenergy)):
                    print(EBenergy[j], end=', ', file=wfid)
                    print(float(bac_up[j]), file=wfid)
        
       
        # Recalculate Rwp after updating the background
        y = []
        p_error = []
        wp_error = []

        for j in range(len(EBenergy)):
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
        

    
        plt.plot(EBenergy, inten_upbac + bac_up, '-k', label="Experimental XPS Profile", )
        plt.plot(EBenergy, i_calc + bac_up, 'g',linestyle='--', label="WPEM-XPS fitting profile",)
        for com in range(self.total_p_num):
            index = 2 * com
            y_com = (new_w_list[index] * lorenz_density(EBenergy,p1_list[com],new_p2_list[index]) + new_w_list[index+1] *
                                    normal_density(EBenergy, p1_list[com],new_p2_list[index+1])) * p_as_ji[:,com]
            # plt.plot(EBenergy, y_com, label="electron{num}".format(num = com))
            plt.plot(EBenergy, y_com+ bac_up,)
            plt.fill_between(EBenergy, y_com+ bac_up,bac_up, alpha=0.3)
        plt.xlabel('Binding Energy(eV)')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig(os.path.join(self.XPSFRfolder,'WPEM_fittingresult.png'),dpi=800)
        plt.show()
        plt.clf()

        plt.plot([i for i in range(1, len(log_likelihood))], log_likelihood[1:], '-p', color='gray',
                 markersize=10, linewidth=2, markerfacecolor='white',markeredgecolor='gray', markeredgewidth=1,
                 label='log_likelihood')
        plt.xlabel('Iterations', )
        plt.ylabel('Log-likelihood', )
        plt.legend()
        plt.savefig(os.path.join(self.XPSFRfolder,'WPEMXPS_log_likelihood.png'),dpi=800)
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
        plt.savefig( os.path.join(self.XPSFRfolder,'WPEMXPS_Rfactor.png'), dpi=800)
        plt.show()
        plt.clf()

        if flag == -1:
            flag = 3

        return new_w_list, p1_list, new_p2_list, i_ter, flag, i_calc, bac_up, Rp[-1], Rwp[-1], _Rsquare[-1],



    # Define the initial value of the XPSEM process
    def initial_p_single(self, ini_p, i_obs, EBenergy, inten):
        """
        This is a method for initializing the parameters in the XPSEM process.
        When bta>=bta_threshold, the initial values of γi and σi^2 are set according to the original XRD pattern.
        When bta<bta_threshold, the initial values of γi and σi^2 are set to bta, and all of Ai are 0.5.
        rawpeak : the raw peak locations
        i_obs : Integrated area of whole pattern
        """
        k = len(ini_p)
        # k is the number of peak i_obs is the XPS profile

        # store peak locations / energies
        p1_list = copy.deepcopy(ini_p)

        # store peak intensity values
        i_list = []
        # store peak intensity at fwhm 
        fwhm_list = []
        # store peak indexes
        p = []
        for i in range(k):
            p.append(p_index(EBenergy, ini_p[i]))
            i_list.append(inten[p[i]])
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
                if inten[n - i] <= (inten[p[0]] / 2):
                    p2_list.append(2 * (EBenergy[p[0]] - EBenergy[n - i]) * self.bta) # γi
                    p2_list.append((2 * (EBenergy[p[0]] - EBenergy[n - i]) * (1 - self.bta)) ** 2) # σi^2
                    break
            # in case not found fwhm from the left side of the peak 
            if len(p2_list) == 0:
                p2_list.append(2 * (EBenergy[p[0]] - EBenergy[0]) * self.bta)
                p2_list.append((2 * (EBenergy[p[0]] - EBenergy[0]) * (1 - self.bta)) ** 2)
            ### end

            # this part is defined for searching fwhm for the middle peak 
            ### star
            for i_l in range(1, k - 1):
                fwhm_i = fwhm_find(p[i_l], p[i_l - 1], fwhm_list[i_l], inten, EBenergy)
                p2_list.append(fwhm_i * self.bta)
                p2_list.append((fwhm_i * (1 - self.bta)) ** 2)
            ### end

            # this part is defined for searching fwhm for the last peak 
            ### star
            n = p[-1] + 1
            for i in range((len(EBenergy) - p[-1] - 1)):
                if inten[n + i] <= (inten[p[-1]] / 2):
                    p2_list.append(2 * (EBenergy[n + i] - EBenergy[p[k - 1]]) * self.bta)
                    p2_list.append((2 * (EBenergy[n + i] - EBenergy[p[k - 1]]) * (1 - self.bta)) ** 2)
                    break

            if len(p2_list) == (k - 1) * 2:
                p2_list.append(2 * (EBenergy[-1] - EBenergy[p[-1]]) * self.bta)
                p2_list.append((2 * (EBenergy[-1] - EBenergy[p[-1]]) * (1 - self.bta)) ** 2)
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
                w_list.append(w * 0.5)
                w_list.append(w * 0.5)

            for i in range(k):
                p2_list.append(self.bta_threshold)
                p2_list.append(self.bta_threshold)
            
            print("Pay Attention! The ratio of Lorentzian components defined in PV function is too small")
            print("That may cause a reasonable result")

        return w_list, p1_list, p2_list

    

################################################################
# Functions
def SpinOrbitCoupling(AtomIdentifier,SatellitePeaks):
    """
    Handle input format

    input:
        AtomIdentifier : A list of atom identifiers
        i.e.,
        [['Cu2+','2p3/2',932.9],['Cu2+','2p1/2',932.4],['Cu1+','2p3/2',933.6]]

        SatellitePeaks : A list of satellite peaks
        i.e., 
        [['Cu2+','2p3/2',934.8],['Cu2+','2p3/2',934.9],['Cu+','2p1/2',933.7]]
    output:
        OriPeaks:
        [['Cu', '2p3/2', 932.6,[2,2,3/2]], ['Cu', '2p1/2', 933.6,[2,2,1/2]],] # [n,l,j]
        [['Cu2+', '2p3/2', 933.4, [2,2,3/2]],]
        [['Zn', '2p3/2', 932.4,[2,2,3/2]],]

        SatPeaks:
        [['Cu2+','2p3/2',934.8],['Cu2+','2p3/2',934.9]]
        [['Cu+','2p1/2',933.7]]

    """
    # Group by element type
    element_dict = {}
    for item in AtomIdentifier:
        first_element = item[0]
        n, l , j = parse_orbital(item[1])
        # for a given atom, when n and l number are same, they are split states.
        # Only if the initial state is the same can it be considered the same atom.
        key = (first_element, n,l) 
        item.append([n,l,j])
        if key in element_dict:
            element_dict[key].append(item)
        else:
            element_dict[key] = [item]

    OriPeaks = []
    for key, value in element_dict.items():
        OriPeaks.append(value)

    _element_dict = {}
    for k in SatellitePeaks:
        first_element = k[0]
        n, l , _ = parse_orbital(k[1])
        key = (first_element, n,l) 
        if key in _element_dict:
            _element_dict[key].append(k)
        else:
            _element_dict[key] = [k]
    
    SatPeaks = []
    for key, value in _element_dict.items():
        SatPeaks.append(value)

    
    return OriPeaks, SatPeaks, 

def parse_orbital(orbital):
  
    match = re.match(r'(\d+)([spdfl])(.*)', orbital)
    
    if match:
        n = int(match.group(1))
        l = match.group(2)
        j = eval(match.group(3))
        if l == 's':
            l = 0
        elif l == 'p':
            l = 1
        elif l == 'd':
            l = 2
        elif l == 'f':
            l = 3
        return n, l, j
    else:
        raise ValueError(f"Invalid orbital format: {orbital}")
    
def readfile(loadParams):
    # readin the initial parameters from file
    with open(loadParams, 'r') as file:
        reader = csv.reader(file)
        w_read = [float(item) for item in next(reader)]
        p1_read = [float(item) for item in next(reader)]
        p2_read = [float(item) for item in next(reader)]
    return w_read, p1_read, p2_read


def reorder_vector(EBenergy, inten):
    # Create a list of tuples containing the EBenergy and inten values
    data = list(zip(EBenergy, inten))
    # Sort the data based on the EBenergy values in ascending order
    sorted_data = sorted(data, key=lambda x: x[0])
    # Extract the sorted EBenergy and inten values into separate lists
    sorted_EBenergy, sorted_inten = zip(*sorted_data)
    return np.array(sorted_EBenergy), np.array(sorted_inten)

def SLCoupling(_mu_list, _ori_mu_list, _w_list, OriPeaks, SatPeaks,tao, ratio,p2_list):
    """
    mu_list = [632,654,752,815]
    w_list = [60000,56000,72000,63000,]
   
    OriPeaks:
    [['Cu', '2p3/2', 932.6,[2,2,3/2]], ['Cu', '2p1/2', 933.6,[2,2,1/2]],] # [n,l,j]
    [['Cu2+', '2p3/2', 933.4, [2,2,3/2]],]
    [['Zn', '2p3/2', 932.4,[2,2,3/2]],]

    SatPeaks:
    [['Cu2+','2p3/2',934.8],['Cu2+','2p3/2',934.9],['Cu2+','2p1/2',935.9]]
    [['Cu+','2p1/2',933.7]]
    """

    new_mu_list = copy.deepcopy(_mu_list)
    new_w_list = copy.deepcopy(_w_list)
    ori_mu_list = copy.deepcopy(_ori_mu_list)


    # fine-tuning binding energy 
    for peak in range(len(new_mu_list)):
        if abs(new_mu_list[peak] - ori_mu_list[peak]) < tao:
            pass
        else:
            new_mu_list[peak] = ratio * ori_mu_list[peak] + (1-ratio) * _mu_list[peak]
    
    
    
    total_index = 0 # remember the index of the current peak 
    orbit_names = [] # a list save the electronic orbit name, e.g., [['Cu', '2p3/2'],['Cu', '2p1/2'],...]
    for k in range(len(OriPeaks)):
        if OriPeaks[k] == 1: # no SL coupling
            total_index += 1
            orbit_names.append(OriPeaks[k][0], OriPeaks[k][1])
            pass
        else:
            related_peaks = []
            quantum_num = []
            for v in range(len(OriPeaks[k])):
                orbit_names.append([OriPeaks[k][v][0], OriPeaks[k][v][1]])
                related_peaks.append(total_index)
                quantum_num.append(OriPeaks[k][v][3])
                total_index += 1
                atom_name = OriPeaks[k][v][0]
            # At most two peaks are related by SL coupling
            # return x1, y1, x2, y2
            new_mu_list, new_w_list = calib_energy_fun(new_mu_list,new_w_list,related_peaks,quantum_num,atom_name)

    # this part of the code updating the peaks locations and intensity inducted by SL coupling
    # the recorded total_index represents the total number of peaks contained in OriPeaks right now
            
    for k in range(len(SatPeaks)):
        if SatPeaks[k] == 1:
            # Satellite peaks are related only if n and l are the same
            total_index += 1
            pass
        else:
            _orbit_name_list = []
            SatPeaks_loc = []
            for v in range(len(SatPeaks[k])):
                _orbit_name = [SatPeaks[k][v][0],SatPeaks[k][v][1]]
                _orbit_name_list.append(_orbit_name)
                SatPeaks_loc.append(total_index)
                total_index += 1
            new_w_list, _child_peaks= cal_SatPeaks(new_mu_list,new_w_list,orbit_names,_orbit_name_list,SatPeaks_loc,p2_list)
                  
    return new_mu_list,new_w_list,_child_peaks

def calZeffect(total_energy, n,l,atom_symbol):
    """
    total_energy : list of splitting peak's energy
    """
    deta_U = abs(total_energy[1] - total_energy[0]) 
    Zeff = (deta_U * n**3 * l * (l+1) / 7.25e-4)** (1/4)
    Zeff = np.round(Zeff,3)
    energy_gap = Zeff**4 * 7.25e-4/ (n**3 * l * (l+1))

    low_energy_electron = (total_energy[1] + total_energy[0]) / 2 - energy_gap / 2
    high_energy_electron = (total_energy[1] + total_energy[0]) / 2 + energy_gap / 2
    if total_energy[1] >= total_energy[0]:
        res = [low_energy_electron,high_energy_electron]
    else:res = [high_energy_electron,low_energy_electron]

    name,_ = split_string(atom_symbol)
    print('Under the central field approximation, the effective charge of {} is:'.format(name), Zeff)
    return res  



def calib_energy_fun(new_mu_list,new_w_list,related_peaks,quantum_num,atom_symbol):
    """
    new_mu_list : peak location list
    new_w_list : peak intensity list, w*beta, w*(1-beta) list

    related_peaks : list contains the index of SL coupling peaks
    quantum_num : list contains the quantum number of splitting peaks
    """
    _mu_list = copy.deepcopy(new_mu_list)
    _w_list = copy.deepcopy(new_w_list)
    ratio = (quantum_num[0][2] *2 +1 )/ (quantum_num[0][2] *2 +1 + quantum_num[1][2] *2 +1) # 2j +1 
    

    w_peak1 = new_w_list[2 * related_peaks[0]] + new_w_list[2 * related_peaks[0]+1]
    w_peak2 = new_w_list[2 * related_peaks[1]] + new_w_list[2 * related_peaks[1]+1]
    ori_ratio = w_peak1 / (w_peak1 + w_peak2) 

    practice_ratio = ori_ratio + (ratio - ori_ratio) * 0.5
    int_p1 = (w_peak1 + w_peak2) * practice_ratio
    int_p2 = (w_peak1 + w_peak2) * (1 - practice_ratio)

    _w_list[2 * related_peaks[0]] = int_p1 * new_w_list[2 * related_peaks[0]] / (new_w_list[2 * related_peaks[0]] + new_w_list[2 * related_peaks[0]+1])
    _w_list[2 * related_peaks[0]+1] = int_p1 * new_w_list[2 * related_peaks[0]+1] / (new_w_list[2 * related_peaks[0]] + new_w_list[2 * related_peaks[0]+1])
   
    _w_list[2 * related_peaks[1]] = int_p2 * new_w_list[2 * related_peaks[1]] / (new_w_list[2 * related_peaks[1]] + new_w_list[2 * related_peaks[1]+1])
    _w_list[2 * related_peaks[1]+1] = int_p2 * new_w_list[2 * related_peaks[1]+1] / (new_w_list[2 * related_peaks[1]] + new_w_list[2 * related_peaks[1]+1])


    peak_list = calZeffect([new_mu_list[related_peaks[0]],new_mu_list[related_peaks[1]]], quantum_num[0][0],quantum_num[0][1],atom_symbol)
    _mu_list[related_peaks[0]] = peak_list[0]
    _mu_list[related_peaks[1]] = peak_list[1]
    return _mu_list, _w_list


def cal_SatPeaks(new_mu_list,new_w_list,orbit_names,_orbit_name_list,SatPeaks_loc,p2_list):

    """
    After the previous processing, the different electron orbitals entering this function are all related. 

    orbit_names, name list of each peaks in electron orbit
    _orbit_name_list, name list of satellite peaks
        the name is in form of [['Cu2+', '2p3/2'],...]
    SatPeaks_loc, index of satellite peaks
    """
    _w_list = copy.deepcopy(new_w_list)

    dict = {}
    for k, key in enumerate(_orbit_name_list):
        key = tuple(key)
        if key in dict:
            dict[key].append(SatPeaks_loc[k])
        else:
            dict[key] = [SatPeaks_loc[k]]

    _mother_peaks = []
    _child_peaks = []
    __name = []
    for key, value in dict.items():
        __name.append(key)
        _mother_peaks.append(orbit_names.index(list(key)))
        _child_peaks.append(value)

    ratio_list = []
    for k in range(len(_mother_peaks)):
        p = _mother_peaks[k]
        mo_integral_energy = integral_PV_single([new_w_list[2*p],new_w_list[2*p+1]], new_mu_list[p],
                                                                             [p2_list[2*p],p2_list[2*p+1]])
        
        # each individual splitting peak may have several satellite peaks
        chi_integral_energy = []
        
        for sp in _child_peaks[k]:
            chi_integral_energy.append(integral_PV_single([new_w_list[2*sp],new_w_list[2*sp+1]], new_mu_list[sp],
                                                                             [p2_list[2*sp],p2_list[2*sp+1]]))
            

        ratio = np.sum(chi_integral_energy) / mo_integral_energy
        ratio_list.append(ratio)

    base_line = np.array(ratio_list).mean()

    
    for k in range(len(_mother_peaks)):
        ori_ratio = ratio_list[k]
        practice_ratio = ori_ratio + (base_line - ori_ratio) * 0.5

        p = _mother_peaks[k]

        _w_list[2*p] = new_w_list[2*p] / (1-ratio_list[k]) * (1-practice_ratio)
        _w_list[2*p+1] = new_w_list[2*p+1] / (1-ratio_list[k]) * (1-practice_ratio)

        for sp in _child_peaks[k]:
            _w_list[2*sp] = new_w_list[2*sp] / ratio_list[k] * practice_ratio
            _w_list[2*sp+1] = new_w_list[2*sp+1] / ratio_list[k] * practice_ratio

    

    print('The Integrated Energy ratio for satellite peak of {} is {}'.format(__name,base_line) )   

    return _w_list,_child_peaks

"""
def splt_energy():
    n1,l1,j1,n2,l2,j2 = symbols('n1 l1 j1 n2 l2 j2')

def calZeffect(total_energy, QNset, j_list):
    d_f =  self.d_spcing(crystal_sys)

"""

def split_string(string):
    match = re.match(r'([A-Za-z]+)(\d*[\+\-]*)', string)
    if match:
        letter_part = match.group(1)
        other_part = match.group(2)
        return letter_part, other_part
    else:
        return None, None
    

def integral_PV_single(w_list, mu, p2_list):
    """
    :param w_list: list of weight (Ai)
    :param mu: a single peak μ1
    :param p2_list: list of γi and σi^2
    :return: Return integral density of a single PV peak 
    """
    x = np.arange(mu-20,mu+20,0.1)
    eare =w_list[0] * lorenz_density(x, mu, p2_list[0]) + \
                        w_list[1] * normal_density(x, mu, p2_list[1])
    return eare.sum()*0.1

################################################################


