# Some functions for auxiliary calculation
# Author: Bin CAO <binjacobcao@gmail.com>

import copy
import numpy as np


def Cu_ab(w_list, wavelength, Cu_tao):
    """
    Cu_ab is a function to limit the relationship between the ray diffraction intensities of copper Ka1 and Ka2 
    input:
    w_list: list of wavelengths (float)
    wavelength: list of input wavelengths (float)
    Cu_tao: tolerance value (float)
    """
    # Create a copy of the input wavelength list
    new_w_list = copy.deepcopy(w_list)
    # If there is only one input wavelength, return the original wavelength list
    if len(wavelength) == 1:
        return w_list
    # If there are two input wavelengths, adjust the intensities of Ka1 and Ka2 accordingly
    elif len(wavelength) == 2:   
        # Iterate over pairs of wavelengths in the wavelength list
        for p in range( int(len(w_list)/2) ):  
            a = 2*p
            b = a +1   
            # If the ratio of the two intensities is less than (2-Cu_tao), adjust the intensities
            if w_list[a]/w_list[b] < (2-Cu_tao):
                new_w_list[a] = (2-Cu_tao)/(3-Cu_tao)*(w_list[a]+ w_list[b]) 
                new_w_list[b] = 1/(3-Cu_tao)*(w_list[a]+ w_list[b]) 
            # If the ratio of the two intensities is greater than (2+Cu_tao), adjust the intensities
            elif  w_list[a]/w_list[b] > (2+Cu_tao):
                new_w_list[a] = (2+Cu_tao)/(3+Cu_tao)*(w_list[a]+ w_list[b]) 
                new_w_list[b] = 1/(3+Cu_tao)*(w_list[a]+ w_list[b]) 
            # If the ratio of the two intensities is between (2-Cu_tao) and (2+Cu_tao), do not adjust the intensities
            else:
                new_w_list[a] = w_list[a]
                new_w_list[b] = w_list[b]
        # Return the adjusted wavelength list
        return new_w_list
    # If there are more than two input wavelengths, print an error message
    else:
        print('The input wavelength exhibits more than two distinct values, which lacks of support for WPEM')


def cal_system(Lattice_constants):
    """
    This function takes a list of lattice constants as input and returns a list of crystal systems.
    """
    crystal_sys_set = []
    # Iterate through each set of lattice constants
    for task in range(len(Lattice_constants)):
        # Extract the lattice constants for the current task
        ini_a = Lattice_constants[task][0]
        ini_b = Lattice_constants[task][1]
        ini_c = Lattice_constants[task][2]
        ini_la1 = Lattice_constants[task][3]
        ini_la2 = Lattice_constants[task][4]
        ini_la3 = Lattice_constants[task][5]
        # Initialize the crystal system to Triclinic 
        crystal_sys = 7
        # Check the conditions to determine the crystal system
        if ini_la1 == ini_la2 and ini_la1 == ini_la3:
            if ini_la1 == 90:
                if ini_a == ini_b and ini_a == ini_c:
                    crystal_sys = 1 # Cubic
                elif ini_a == ini_b and ini_a != ini_c:
                    crystal_sys = 3 # Tetragonal
                elif ini_a != ini_b and ini_a != ini_c and ini_b != ini_c:
                    crystal_sys = 4 # Orthorhombic
            elif ini_la1 != 90 and ini_a == ini_b and ini_a == ini_c:
                crystal_sys = 5 # Rhombohedral
        elif ini_la1 == ini_la2 and ini_la1 == 90 and ini_la3 == 120 and ini_a == ini_b and ini_a != ini_c:
            crystal_sys = 2 # Hexagonal
        elif ini_la1 == ini_la3 and ini_la1 == 90 and ini_la2 \
                != 90 and ini_a != ini_b and ini_a != ini_c and ini_c != ini_b:
            crystal_sys = 6 # Monoclinic
         # Append the crystal system to the list
        crystal_sys_set.append(crystal_sys) 
        # print  crystal parameters
        __name = ['Cubic', 'Hexagonal', 'Tetragonal', 'Orthorhombic', 'Rhombohedral', 'Monoclinic', 'Triclinic']
        print("The input crystal system is:", __name[crystal_sys - 1],' | ', "The initial lattice constants :", ini_a, ini_b, ini_c,
                ini_la1, ini_la2, ini_la3 )
    return crystal_sys_set

# Determination Coefficient
def Rsquare(X, Y):
        # X : Cal
        # Y : Obs
        mean = np.mean(Y)
        SStot = 0
        SSres = 0
        for i in range(len(X)):
            SStot += (Y[i] - mean) ** 2
            SSres += (Y[i] - X[i]) ** 2
        return SSres * 100 / SStot

# Normal distribution
def normal_density(x, mu, sigma2):
    """
    :param x: sample data (2theta)
    :param mu: mean (μi)
    :param sigma2: variance (σi^2)
    :return: Return the probability density of Normal distribution x~N(μi,σi^2)
    """
    __density = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
    return __density

# Lorenz distribution
def lorenz_density(x, mu=0, gamma=1):
    """
    :param x: sample data (2theta)
    :param mu: mean (μi)
    :param gamma: FWHM of Lorenz distribution
    :return: Return the probability density of Lorenz distribution
    """
    __density = (1 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
    return __density

# Mixtrue distribution
def mix_normal_lorenz_density(x_list, w_list, p1_list, p2_list, asy_list):
    """
    The PV probability density function for the Mixtrue distribution:
    ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO formula (2.13-2.15)

    :param x_list: list of sample data (2theta)
    :param w_list: list of weight (Ai)
        Note : w_list = [wi*deta, wi*(1-deta)]
    :param p1_list: list of mean (μi)
    :param p2_list: list of γi and σi^2
    :param asy_list: list of the parameters of profile asymmetry models
    :return: Return the probability density of the mixtrue distribution
    """
    k_ln = int(len(w_list) / 2)
    m = len(x_list)
    __mix_density = np.zeros((m, 1), dtype=float)
    for j in range(m):
        for i in range(k_ln):
            i_l = 2 * i
            __mix_density[j] += (w_list[i_l] * lorenz_density(x_list[j],p1_list[i],
                                                                        p2_list[i_l]) + w_list[i_l + 1] *
                                    normal_density(x_list[j], p1_list[i], p2_list[i_l + 1])) * \
                                asy_list[j][i]
    return __mix_density

# Probability density of mixtrue distribution at peak positions
def mix_normal_lorenz_density_cal_fwhm(x0, w_list, p1, p2_list):
    """
    :param x0: list of sample data (2theta)
    :param w_list: list of weight (Ai)
    :param p1: a single peak μ1
    :param p2_list: list of γi and σi^2
    :return: Return the probability density of the mixtrue distribution when x = μi
    """
    __mix_density_h =w_list[0] * lorenz_density(x0, p1, p2_list[0]) + \
                        w_list[1] * normal_density(x0, p1, p2_list[1])
    return __mix_density_h

# The estimation of total diffraction intensity, that is, the area under the XRD pattern.
def theta_intensity_area(theta_data, intensity):
    n = len(theta_data) - 1
    __area = 0
    for i in range(n):
        __h = (intensity[i] + intensity[i + 1]) / 2
        __l = theta_data[i + 1] - theta_data[i]
        __area += __h * __l
    return __area

# Find the index of the peak position in the list of 2Theta/x
def p_index(two_theta, peak):
    n = len(two_theta)
    p = 0
    for j in range(n):
        if two_theta[j] >= peak:
            p = j
            break
    return p

# Find the approximate value of FWHM of the peak in the original data
def fwhm_find(p, p_1, intensity_i, intensity, two_theta):
    """
    It returns the approximate value of the full width at half maximum (FWHM) of the peak in the original data. 
    """
    # p_1 located at the previous peak
    n = p - 1  # index of previous peak
    fwhm_h = 0  # initialize FWHM / 2 
    for i in range(p):
        # seek form the left side
        if intensity[n - i] == intensity[p_1]:  # if intensity at n-i matches intensity at p_1, it's overlapped
            # seek form the right side
            for j in range(2, p):
                if (n + j) >= len(intensity):  # if n+j is beyond intensity array length
                    # searching task failed
                    fwhm_h = 0.1  # set FWHM to arbitrary small value
                    break
                if intensity[n + j] <= intensity_i:  # if intensity at n+j is below the intensity/2 
                    fwhm_h = two_theta[n + j] - two_theta[p]  # calculate FWHM
                    break
            break
        elif intensity[n - i] <= intensity_i:  # if intensity at n-i is below the intensity/2
            fwhm_h = two_theta[p] - two_theta[n - i]  # calculate FWHM
            break
        else: pass
    return fwhm_h * 2 # FWHM

    
# The asymmetry model
def intensity_as(p1, theta, asy_C,model):
    """
    p1 : peak location
    theta : x-axis
    """
    if model == 'XRD':
        # Calculate the difference between theta and p1
        t = (theta - p1)
        # Determine the sign of t
        if t < 0:
            sign = -1
        elif t > 0:
            sign = 1
        else:
            sign = 0 
    elif model == 'XPS':
        t = (theta - p1)
        # Determine the sign of t
        if t > 0:
            sign = -1
        elif t < 0:
            sign = 1
        else:
            sign = 0 
    # Calculate the asymmetry correction factor
    p_as = 1 - asy_C * sign * (t ** 2) 
    return p_as


#  Based on Bayesian theory, the posterior probability of latent variables (αji and βji) are calculated.
#  Calculate the coefficient as asymmetry correction
def gamma_ji_list(x_list, w_list, p1_list, p2_list,s_angle,asy_C,model='XRD'):
    """
    ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO formula (2.23)

    :s_angle the lowest threshold add asymmetry
    :param x_list: sample data (2Theta)
    :param w_list: list of mixture coefficients (wi*Ai)
    :param p1_list: list of mean (μi)
    :param p2_list: list of γi and σi^2
    :return: The probability of observing diffraction peak i at angle xj
    """

    m = len(x_list)
    k = len(w_list)  # the number of distributions
    k_ln = int(k / 2)  # the number of peak
    gamma_ji = np.ones((m, k), dtype=float)
    gamma_ji_l = np.ones((m, k_ln), dtype=float)
    p_as_ji = np.ones((m, k_ln), dtype=float)  # matrix of asymmetry coefficient
    numerator = np.linspace(0., 0., k)
    numerator_l = np.linspace(0., 0., k_ln)
    for i in range(k_ln):
        pi_index = p_index(x_list, p1_list[i])
        for j in range(m):
            if model == 'XRD':
                if p1_list[i] <= s_angle and abs(j - pi_index) <= (2 * m / k_ln):
                    p_as_ji[j][i] = intensity_as(p1_list[i], x_list[j],asy_C,model)
            elif model == 'XPS':
                if check_in_intervals(p1_list[i],s_angle) and abs(j - pi_index) <= (2 * m / k_ln):
                    p_as_ji[j][i] = intensity_as(p1_list[i], x_list[j],asy_C,model)
    for j in range(m):
        for i in range(k_ln):
            i_l = i * 2
            i_n = i_l + 1
            numerator[i_l] = w_list[i_l] * lorenz_density(x_list[j], p1_list[i], p2_list[i_l]) * p_as_ji[j][i]
            numerator[i_n] = w_list[i_n] * normal_density(x_list[j], p1_list[i],p2_list[i_n]) * p_as_ji[j][i]
            numerator_l[i] = numerator[i_l] * lorenz_density(x_list[j],p1_list[i],p2_list[i_l])
        denominator = sum(numerator)
        for i in range(k):
            gamma_ji[j][i] = numerator[i] / denominator
        for i_l in range(k_ln):
            gamma_ji_l[j][i_l] = numerator_l[i_l] / denominator
    return gamma_ji, gamma_ji_l, p_as_ji

def check_in_intervals(x, intervals):
    if len(intervals) == 1 and isinstance(intervals[0], list):
        if intervals[0][0] <= x <= intervals[0][1]:
            return True
    elif len(intervals) == 2 and not isinstance(intervals[0], list):
        if intervals[0] <= x <= intervals[1]:
            return True
    else:
        for interval in intervals:
            if interval[0] <= x <= interval[1]:
                return True
    return False

# The diffraction peak positions obtained by EM algorithm are sorted in a certain order
def get_angle_sort(mui_cal_em_list, mui_abc_list):
    # Make a copy of the mui_abc_list so that we can sort it without modifying the original list
    mui_abc_copy = copy.deepcopy(mui_abc_list)
    # Sort the copy of mui_abc_list in ascending order
    mui_abc_copy.sort()
    # Create an empty list to store the matching values of mui_cal_em_list
    mui_cal_em_matching = []
    # Create an empty list to store the indices of the matched values of mui_abc_list
    j_index = []
    # Iterate through the values in mui_cal_em_list
    for i in range(len(mui_cal_em_list)):
        # Iterate through the values in mui_cal_em_list again
        for j in range(len(mui_cal_em_list)):
            # Check if the value of mui_abc_list at index i is equal to the value of mui_abc_copy at index j
            # and if j has not been matched before
            if mui_abc_list[i] == mui_abc_copy[j] and j not in j_index:
                # Append the corresponding value from mui_cal_em_list to the matching list
                mui_cal_em_matching.append(mui_cal_em_list[j])
                # Add the index of j to the list of matched indices
                j_index.append(j)
    # Return the list of matched values from mui_cal_em_list
    return mui_cal_em_matching

def solve_sigma2(x_list, normal_gamma_list, intensity, mu, denominator,limit):
    """
    Return the new list of γi or σi^2 which makes likelihood reaches maximum
    ref: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO formula (2.25 b,c)
    """
    x_list_i = (x_list - mu) ** 2
    numerator = np.multiply(np.multiply(normal_gamma_list.T, x_list_i), intensity).sum()
    new_sigma2 = (numerator / (denominator + 1e-12))
    if new_sigma2 <= limit:
        new_sigma2 = limit
    return new_sigma2

def Distribute_mu(new_p1_list, mui_abc, hkl_dat, lmd):
    em_mu_matching = []
    j_index = []
    em_mu = copy.deepcopy(new_p1_list)
    em_mu.sort()
    Bragg_mu = copy.deepcopy(mui_abc)
    Bragg_mu.sort()
    for i in range(len(Bragg_mu)):
        for j in range(len(Bragg_mu)):
            if mui_abc[i] == Bragg_mu[j] and j not in j_index:
                em_mu_matching.append(em_mu[j])
                j_index.append(j)

    sub_new_mu = []
    sub_new_mu_set = []
    for task in range(len(hkl_dat)):
        for j in range(lmd * len(hkl_dat[task][0])):
            sub_new_mu.append(em_mu_matching[0])
            del em_mu_matching[0]
        sub_new_mu_set.append(sub_new_mu)
        sub_new_mu = []

    if len(em_mu_matching) == 0:
        return sub_new_mu_set


