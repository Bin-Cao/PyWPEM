# Raman spectroscopy Qualitative Description Module
# Author: Bin CAO <binjacobcao@gmail.com>

import math
import heapq
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fit( mix_component, Raman_file,bacground_file,fix_peaks = None, wavenumbers_range = None, sigma2_coef = 0.5, max_iter = 5000,
        threshold=1000,component_show=False, wavenumbers_location = None,):
    """
    Raman spectroscopy Qualitative Description Module

    :param mix_component : the number of peaks 

    :param Raman_file : cryastal Raman Signal 

    :param bacground_file : bacground signal 

    :param fix_peaks, default is None
        a list, frozen any number of peaks

    :param wavenumbers_range : default is None
        wavenumber range of study

    :param sigma2_coef : default is 0.5
        sigma2 of gaussian peak
    
    :param max_iter : default is 5000
        the maximum number of iterations of solver
    
   
        
        
    """

    # mike dir
    os.makedirs('RamanProfile', exist_ok=True)

    # range: (0,90) angle range
    # Read raw/original data, and the data format is wavenumbers-absorbance/X-Y data.
    Ex_Raman = pd.read_csv(Raman_file, header=None, names=['wavenumbers', 'absorbance'])
    Raman = copy.deepcopy(Ex_Raman)
    # Read background data, and the data format is wavenumbers-absorbance/X-Y data.
    bac_data = pd.read_csv(bacground_file, header=None, names=['wavenumbers', 'absorbance'])
    # subtract background
    Raman['absorbance'] = Ex_Raman['absorbance'] - bac_data['absorbance']
    
    full_wavenmb = copy.deepcopy(np.array(Raman.wavenumbers))

    if wavenumbers_range == None:
        pass 
    else:
        x_list = np.array(Raman.wavenumbers)
        y_list = np.array(Raman.absorbance)
        index = np.where( (Raman.wavenumbers < wavenumbers_range[0]) | (Raman.wavenumbers > wavenumbers_range[1]) )
        Raman = Raman.drop(index[0])

    x_list = np.array(Raman.wavenumbers)
    y_list = np.array(Raman.absorbance)
    
    # remove constant
    NAa = y_list.min()
    y_list -= NAa

    singal = False
    # initializ the  parameters
    sigma2_list = sigma2_coef * np.ones((mix_component, 1), dtype=float)
    w_list = np.ones((mix_component, 1), dtype=float)/mix_component
    if type(wavenumbers_location) == list:
        if wavenumbers_location[-1] == 'fixed':
            singal = True
            mu_list = np.array(wavenumbers_location[:-1])
        elif type(wavenumbers_location[-1]) == float or int:
            mu_list = np.array(wavenumbers_location)
        else:
            print('Type Error - only \'fixed\' is allowed')

    elif wavenumbers_location == None:
        mu_list = initalize(mix_component,x_list,y_list)
    else:
        print('type error! : wavenumbers_location, please input as a list')


    new_w_list = w_list
    new_mu_list = mu_list
    new_sigma2_list = sigma2_list
    gamma_list = gamma_ji_list(x_list, w_list, mu_list, sigma2_list)
    denominator = denominator_list(w_list, gamma_list,y_list)

    int_area = theta_intensity_area(x_list, y_list)

    i_ter = 0
    while (True):
        i_ter += 1
        w_list = new_w_list
        mu_list = new_mu_list
        sigma2_list = new_sigma2_list
        if singal == False:
            up_mu_list = solve_mu_list(x_list, gamma_list, denominator,y_list)
            new_mu_list = replace_mu_list(up_mu_list,fix_peaks)
        else:
            pass 
        new_sigma2_list = solve_sigma2_list(x_list, gamma_list, w_list, mu_list, denominator,y_list)
        new_w_list = solve_w_list(y_list, denominator,int_area)
        gamma_list = gamma_ji_list(x_list, new_w_list, new_mu_list, new_sigma2_list)
        denominator = denominator_list(new_w_list, gamma_list,y_list)

        if i_ter % 200 == 0:
            print("Number of Iterations: %s" % i_ter)
            print("W_list: %s" % new_w_list)
            print("mu_list: %s" % new_mu_list)
            print("sigma2_list: %s" % new_sigma2_list)
            print('------------------------------------------------------------------','\n')
        if compare_list(w_list, new_w_list, 1e-6):
            if compare_list(mu_list, new_mu_list, 1e-6):
                if compare_list(sigma2_list, new_sigma2_list, 1e-6):
                    print("Convergence get at %s iterations!" % i_ter)
                    print('------------------------------------------------------------------','\n')
                    break

        if i_ter > max_iter:
            break
    
    print("W_list: %s" % new_w_list)
    print("mu_list: %s" % new_mu_list)
    print("sigma2_list: %s" % new_sigma2_list)
    
    new_w_list, new_mu_list, new_sigma2_list, part_drop_cal, drop_cal= fluorescence_G(new_w_list, new_mu_list, new_sigma2_list, x_list, full_wavenmb,  threshold )
    y_list = y_list - part_drop_cal
    Up_original_int = np.array(Ex_Raman.absorbance) - drop_cal
    Up_bc_int = np.array(bac_data.absorbance) + drop_cal

    # update background and Intensity files
    with open(os.path.join('RamanProfile', 'Up_intensity.csv'), 'w') as wfid:
        for j in range(len(full_wavenmb)):
            print(full_wavenmb[j], end=', ', file=wfid)
            print(float(Up_original_int[j]), file=wfid)
    with open(os.path.join('RamanProfile', 'Up_background.csv'), 'w') as wfid:
        for j in range(len(full_wavenmb)):
            print(full_wavenmb[j], end=', ', file=wfid)
            print(float(Up_bc_int[j]), file=wfid)


    part_y_cal = np.array(mixture_normal_density(x_list, new_w_list, new_mu_list, new_sigma2_list))

    # cal the fitting goodness
    error_p = []
   
    for i in range(len(y_list)):
        error_p.append(abs(y_list[i] - part_y_cal[i]))
    error_p_sum = sum(error_p)
    y_sum = sum(y_list)

    Rp = error_p_sum / y_sum * 100
    print("Rp = ", error_p_sum / y_sum * 100)

    # cal intensities on entire diffraction range
    y_cal = np.array(mixture_normal_density(full_wavenmb, new_w_list, new_mu_list, new_sigma2_list))

    with open(os.path.join('RamanProfile', 'Raman_total_peaks.csv'), 'w') as wfid:
            print('wi', end=',', file=wfid)
            print('mu_i', end=',', file=wfid)
            print('sigma2_i', end=',', file=wfid)
            print('Rp: %f ' % Rp, file=wfid)
            for j in range(len(new_w_list)):
                print(new_w_list[j], end=',', file=wfid)
                print(new_mu_list[j], end=',', file=wfid)
                print(new_sigma2_list[j], file=wfid)
            


    # write Raman fitting profile
    with open(os.path.join('RamanProfile', 'Raman_fitting.csv'), 'w') as wfid:
        for j in range(len(full_wavenmb)):
            print(full_wavenmb[j], end=', ', file=wfid)
            print(float(y_cal[j]), file=wfid)
    
    
   
    plt.xlabel('Raman shift/(cm-1)', size=15)
    plt.ylabel('Intensity/(a.u.)', size=15)
    plt.title('RamanProfile', size=15)
    plt.plot(x_list,y_list, label="Experimental Obs.")
   
    show_fit_profile = np.zeros(len(x_list)) 
    for com in range(len(new_mu_list)):
        y_com = new_w_list[com] * np.array(normal_density(x_list,new_mu_list[com], new_sigma2_list[com]))
        if component_show == True:
            plt.plot(x_list, y_com,)
        else:
            pass
        show_fit_profile += y_com

        # write amorphous components  
        entire_y_com = new_w_list[com] * np.array(normal_density(full_wavenmb,new_mu_list[com], new_sigma2_list[com]))  
        with open(os.path.join('RamanProfile', 'Raman_components{num}.csv'.format(num = com)), 'w') as wfid:
            for j in range(len(full_wavenmb)):
                print(full_wavenmb[j], end=', ', file=wfid)
                print(float(entire_y_com[j]), file=wfid)

    
    plt.plot(x_list,show_fit_profile, c='k',label="WPEM fitting Raman profile")
    plt.legend()
    plt.savefig('./RamanProfile/FittingRamanProfile.png', dpi=800)
    plt.show()
    plt.clf()

    Rama_shift = np.array(new_mu_list)
    Rama_shift.sort()
    print('peaks loc : ',np.round(Rama_shift,0))


def fluorescence_G(w_list, mu_list, var_list, x_list,full_wavenmb, threshold):
    # treat big variance peak as flurescence and subtract it into the background
    save_peak_w = []
    save_peak_mu = []
    save_peak_var = []
    drop_peak_w = []
    drop_peak_mu = []
    drop_peak_var = []
    for i in range(len(var_list)):
        if var_list[i] < threshold:
            save_peak_w.append(w_list[i])
            save_peak_mu.append(mu_list[i])
            save_peak_var.append(var_list[i])
        else:
            drop_peak_w.append(w_list[i])
            drop_peak_mu.append(mu_list[i])
            drop_peak_var.append(var_list[i])
    part_drop_cal = np.array(mixture_normal_density(x_list, drop_peak_w, drop_peak_mu, drop_peak_var))
    drop_cal = np.array(mixture_normal_density(full_wavenmb, drop_peak_w, drop_peak_mu, drop_peak_var))

    return save_peak_w, save_peak_mu, save_peak_var,part_drop_cal, drop_cal

def replace_mu_list(up_mu_list,fix_peaks):
    if fix_peaks == None:
        raw_mu_list = copy.deepcopy(up_mu_list)
    elif type(fix_peaks) == list:
        raw_mu_list = copy.deepcopy(up_mu_list)
        for j in range(len(fix_peaks)):
            Mht_dis = abs(up_mu_list - fix_peaks[j])
            _min_index = np.argmin(Mht_dis) 
            raw_mu_list[_min_index] = fix_peaks[j]
    else:
        print('Type error -fix_peaks-')
    return raw_mu_list
    

def normal_density(x, mu, sigma2):
    """
    :param x: sample data
    :param mu: mean
    :param sigma2:
    :return: variance
    Return the probability density of Normal distribution x~N(mu,sigma2)
    """
    density = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
    return density


def mixture_normal_density(x, w_list, mu_list, sigma2_list):
    """
    Input paramters:
    x_list     -> sample data from generated sample j, start from 0
    w_list     -> list of mixture coefficients
    mu_list    -> list of mu
    sigma2_list-> list of sigma2
    Return the mixture probability density of Normal distribution sum(x~N(mu,sigma2))
    """
    k = len(w_list)  # the number of mixture normal distributions
    mix_density = 0
    for i in range(k):
        mix_density += w_list[i] * normal_density(x, mu_list[i], sigma2_list[i])
    return mix_density


def gamma_ji_list(x_list, w_list, mu_list, sigma2_list):
    """
    :param x_list: sample data
    :param w_list: list of mixture coefficients
    :param mu_list: list of mu
    :param sigma2_list: list of sigma2
    :return: the matrix of post-probability of x_j at distribution i, i,j start from 0
    """
    m = len(x_list)  # number of data
    k = len(w_list)  # the number of mixture normal distributions
    gamma_ji = np.ones((m, k), dtype=float)  # data * peak
    for j in range(m):
        denominator = 0.0
        numerator = np.linspace(0., 0., k)
        for i_m in range(k):
            denominator += w_list[i_m] * normal_density(x_list[j], mu_list[i_m], sigma2_list[i_m])
            numerator[i_m] = w_list[i_m] * normal_density(x_list[j], mu_list[i_m], sigma2_list[i_m])
        for i in range(k):
            # numerator = w_list[i] * normal_density(x_list[j], mu_list[i], sigma2_list[i])
            gamma_ji[j][i] = numerator[i] / (denominator + 1e-10)
        # gamma_ji[j][:] = numerator / denominator
    return gamma_ji  # m*k


def denominator_list(w_list, gamma_list,int_list):
    # int_list : array in the shape of x_list (m, 1)
    k = len(w_list)  # the number of mixture normal distributions
    denominator_i = np.linspace(0., 0., k)
    for i in range(k):
        denominator_i[i] = np.multiply(int_list,gamma_list[:, i]).sum()
    return denominator_i  # k * 1


def solve_mu_list(x_list, gamma_list, denominator, int_list):
    """
    Return new_mu_list which makes likelihood reaches maximum
    """
    k = len(denominator)  # the number of mixture normal distributions
    numerator = np.linspace(0., 0., k)
    for i in range(k):
        numerator[i] = np.multiply(np.multiply(int_list,gamma_list[:, i]), x_list).sum()
    new_mu_list = numerator / (denominator + 1e-12)
    return new_mu_list


def solve_sigma2_list(x_list, gamma_list, w_list, mu_list, denominator,int_list):
    """
    mu_list maybe the new_mu_list
    Return new_sigma2_list which makes likelihood reaches maximum
    """
    k = len(w_list)
    numerator = np.linspace(0., 0., k)
    for i in range(k):
        x_list_i = (x_list - mu_list[i])
        x_list_i2 = np.multiply(x_list_i, x_list_i)  # m*1
        numerator[i] = np.multiply(np.multiply(int_list,gamma_list[:, i]), x_list_i2).sum()
    new_sigma2_list = (numerator / (denominator + 1e-12))
    return new_sigma2_list


def solve_w_list(int_list, denominator,int_area):
    """
    Return new_sigma2_list which makes likelihood reaches maximum
    """
    new_w_list = int_area/int_list.sum() * denominator
    return new_w_list


def compare_list(old_list, new_list, tor):
    length = len(old_list)
    tot_error = 0
    for i in range(length):
        tot_error += (abs(old_list[i] - new_list[i])) / old_list[i]
    tot_error /= length
    if tot_error <= tor:
        return True
    else:
        return

def theta_intensity_area(x_list, int_list):
    n = len(x_list) - 1
    __area = 0
    for i in range(n):
        __h = (int_list[i] + int_list[i + 1]) / 2
        __l = x_list[i + 1] - x_list[i]
        __area += __h * __l
    return __area

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def initalize(mix_component,x_list,y_list):
    peak_index = []
    split_angle_part = chunks(y_list,mix_component)
    for peak in range(mix_component):
        peak_index.append(
                (heapq.nlargest(1, enumerate(split_angle_part[peak]),key=lambda x: x[1]))[0][0]
                 + peak * len(split_angle_part[0])
                  )
    mu_list = []
    for j in range(len(peak_index)):
        mu_list.append(x_list[peak_index[j]])

    return np.array(mu_list)

