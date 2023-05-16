# Amorphous Qualitative Description Module
# Author: Bin CAO <binjacobcao@gmail.com>

import math
import heapq
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sin, pi

def Amorphous_fitting(mix_component, ang_range = None, sigma2_coef = 0.5, max_iter = 5000, peak_location = None,Wavelength = 1.54184):
    """
    Amorphous Qualitative Description Module

    :param mix_component : the number of amorphous peaks 

    :param ang_range : default is None
        two theta range of study

    :param sigma2_coef : default is 0.5
        sigma2 of gaussian peak
    
    :param max_iter : default is 5000
        the maximum number of iterations of solver
    
    : param peak_location : default is None
        the initial peak position of the amorphous peaks
        can input as a list, e.g.,
        peak_location = [20,30,40]
        the peak position can be frozen by the assigned input,
        peak_location = [20,30,40,'fixed']

    : param Wavelength : Wavelength of ray, default is 1.54184 (Cu)
    """

    # range: (0,90) angle range
    data = pd.read_csv("./DecomposedComponents/upbackground.csv", header=None, names=['ang','int'])
    
    full_ang = copy.deepcopy(np.array(data.ang))
    ori_int = copy.deepcopy(np.array(data.int))
    if ang_range == None:
        pass 
    else:
        x_list = np.array(data.ang)
        y_list = np.array(data.int)
        index = np.where( (data.ang < ang_range[0]) | (data.ang > ang_range[1]) )
        data = data.drop(index[0])

    x_list = np.array(data.ang)
    y_list = np.array(data.int)
    
    # remove constant
    NAa = y_list.min()
    y_list -= NAa

    singal = False
    # initializ the  parameters
    sigma2_list = sigma2_coef * np.ones((mix_component, 1), dtype=float)
    w_list = np.ones((mix_component, 1), dtype=float)/mix_component
    if type(peak_location) == list:
        if peak_location[-1] == 'fixed':
            singal = True
            mu_list = np.array(peak_location[:-1])
        elif type(peak_location[-1]) == float or int:
            mu_list = np.array(peak_location)
        else:
            print('Type Error - only \'fixed\' is allowed')

    elif peak_location == None:
        mu_list = initalize(mix_component,x_list,y_list)
    else:
        print('type error! : peak_location, please input as a list')

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
            new_mu_list = solve_mu_list(x_list, gamma_list, denominator,y_list)
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

        if compare_list(w_list, new_w_list, 1e-6):
            if compare_list(mu_list, new_mu_list, 1e-6):
                if compare_list(sigma2_list, new_sigma2_list, 1e-6):
                    print("Convergence get at %s iterations!" % i_ter)
                    break

        if i_ter > max_iter:
            break

    print("W_list: %s" % new_w_list)
    print("mu_list: %s" % new_mu_list)
    print("sigma2_list: %s" % new_sigma2_list)

    # estimated 
    d = 1.23 * Wavelength / 2 / sin(new_mu_list[0]/2 * pi / 180)
    print('estimated interatomic distances : %f' % d)
  
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
    y_cal = np.array(mixture_normal_density(full_ang, new_w_list, new_mu_list, new_sigma2_list))

    with open(os.path.join('DecomposedComponents', 'M_Amorphous_peaks.csv'), 'w') as wfid:
            print('wi', end=',', file=wfid)
            print('mu_i', end=',', file=wfid)
            print('sigma2_i', end=',', file=wfid)
            print('Rp: %f ' % Rp, file=wfid)
            for j in range(mix_component):
                print(new_w_list[j], end=',', file=wfid)
                print(new_mu_list[j], end=',', file=wfid)
                print(new_sigma2_list[j], file=wfid)
            
    # updata background
    # (amorphous) up_bac  = y_list + constant
    # up_up_bac = y_list + constant - y_fit = up_bac - y_fit
    up_up_bac = ori_int - y_cal
    if up_up_bac.min() < 0:
        print('warning! the fitting profile of amorphous is overflow \n ','please choise another set of reasonable paras. in model AmorphousFitting ')
    else:
        pass
    
    # write up_up_bac
    with open(os.path.join('DecomposedComponents', 'M_background_amorphous_stripped.csv'), 'w') as wfid:
        for j in range(len(full_ang)):
            print(full_ang[j], end=', ', file=wfid)
            print(float(up_up_bac[j]), file=wfid)
    

    # write amorphous fitting profile
    with open(os.path.join('DecomposedComponents', 'Amorphous.csv'), 'w') as wfid:
        for j in range(len(full_ang)):
            print(full_ang[j], end=', ', file=wfid)
            print(float(y_cal[j]), file=wfid)
    
   # Define the font of the image
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 15
    
    plt.xlabel('2\u03b8\u00B0')
    plt.ylabel('I (a.u.)')
    plt.title('Amorphous decomposited components')
    plt.plot(x_list,y_list, label="Real intensity")
   
    show_fit_profile = np.zeros(len(x_list)) 
    for com in range(mix_component):
        y_com = new_w_list[com] * np.array(normal_density(x_list,new_mu_list[com], new_sigma2_list[com]))
        plt.plot(x_list, y_com,label="components {num}".format(num = com))
        show_fit_profile += y_com

        # write amorphous components  
        entire_y_com = new_w_list[com] * np.array(normal_density(full_ang,new_mu_list[com], new_sigma2_list[com]))  
        with open(os.path.join('DecomposedComponents', 'M_Amorphous_components{num}.csv'.format(num = com)), 'w') as wfid:
            for j in range(len(full_ang)):
                print(full_ang[j], end=', ', file=wfid)
                print(float(entire_y_com[j]), file=wfid)

    plt.plot(x_list,show_fit_profile, label="WPEM fitting intensity")
    plt.xlabel('2\u03b8\u00B0')
    plt.ylabel('I (a.u.)')
    plt.legend()
    plt.savefig('./DecomposedComponents/M_Amorphous_components.png', dpi=800)
    plt.show()
    plt.clf()

################################################################

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
