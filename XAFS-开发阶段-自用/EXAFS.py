# Extended X-ray absorption fine structure
# Author: Bin CAO <binjacobcao@gmail.com>
import os
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

class EXAFS(object):
    def __init__(self,XAFSdata,  power = 2, distance = 5, k_point = 10,de_bac = False):
        """
        XAFSdata : the document name of input data 
        k_point :  default k_point = 8, the cut off range of k points
        de_bac : default de_bac = False, 
        has been processed to remove the absorption background
        """
        self.XAFSdata = XAFSdata
        self.power = power 
        self.k_point = k_point
        self.distance = distance
        self.de_bac = de_bac

        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10

        os.makedirs('XAFS/EXAFS', exist_ok=True)

    def fit(self, Ezero = None, first_cutoff_energy=None,second_cutoff_energy=None):
        """
        Cutoff_energy: The data behind cutoff energy will be used to calculate the mean absorption.
        """
        warnings.filterwarnings("ignore")
        data = pd.read_csv(self.XAFSdata, header=None, names=['energy', 'absor'])
        energy = np.array(data.energy)
        absor = np.array(data.absor)
        if first_cutoff_energy == None:
            first_cutoff_energy = energy[0] + (energy[-1] - energy[0]) * 0.2
            print("For more accurate EXAFS results, it is preferable to input first_cutoff_energy into WPEM")
        else: pass 
        if second_cutoff_energy == None:
            second_cutoff_energy = energy[0] + (energy[-1] - energy[0]) * 0.6
            print("For more accurate EXAFS results, it is preferable to input second_cutoff_energy into WPEM")
        else: pass 

        first_cutoff_index = find_first_value_greater_than(energy, first_cutoff_energy)
        second_cutoff_index = find_first_value_greater_than(energy, second_cutoff_energy)
        if self.de_bac == True:  
            popt, _ = curve_fit(fitting_function, energy[0:first_cutoff_index] , absor[0:first_cutoff_index])
            k1_fit, k2_fit, c_fit = popt
            absor = absor - (k1_fit / energy ** 3 + k2_fit / energy ** 4 + c_fit)
            plt.plot(energy, absor, color='k', linewidth=2, )
            plt.xlabel('Energy(eV)' )
            plt.ylabel('\u03BC(E)', )
            plt.savefig('XAFS/EXAFS/NormalizedEnergy.png',dpi=800)
            plt.show()
            plt.clf()

        first_base = np.mean(absor[0:first_cutoff_index])
        mean_absor = np.mean(absor[second_cutoff_index:-1])

        delta =  first_base - mean_absor
        if  Ezero == None:
            Ezero = find_max_slope_x(energy, absor)
        else: pass
        print('E0 =',Ezero)
        zero_index = np.argmax(energy >= Ezero)

        # Spline_fun   
        spline = UnivariateSpline(energy[zero_index:-1], absor[zero_index:-1], s=2,k = 2) 
        energy_base = spline(energy[zero_index:-1])

        _energy = energy[zero_index:-1] 
        edge_frac = (absor[zero_index:-1] - energy_base) / delta
        add_index = find_index_in_range(edge_frac, -0.01, 0.01)
        if add_index == None:
            add_index = find_index_in_range(edge_frac, -0.02, 0.02)
        else:pass
        if add_index == None:
            add_index = find_index_in_range(edge_frac, -0.03, 0.03)
        else:pass
        _energy = _energy[add_index:-1]
        edge_frac = edge_frac[add_index:-1]
        energy_base = energy_base[add_index:-1]
     
        plt.plot(energy, absor, color='k', linewidth=2, )
        plt.plot(_energy, energy_base, '--',color='r', )
        plt.axvline(Ezero,linestyle='--',color='b',)
        plt.xlabel('Energy(eV)' )
        plt.ylabel('\u03BC(E)', )
        plt.savefig('XAFS/EXAFS/Splinefun.png',dpi=800)
        plt.show()
        plt.clf()

        plt.plot(_energy, edge_frac, color='k', linewidth=2, )
        plt.plot(_energy, np.zeros(len(_energy)), '--',color='r', )
        plt.xlabel('Energy(eV)' )
        plt.ylabel('\u03BC(E)', )
        plt.savefig('XAFS/EXAFS/SmoothEnergy.png',dpi=800)
        plt.show()
        plt.clf()
  
        # k = 2pi/h * np.aqrt(2 * m_e * (E - E0))
        K_space = np.sqrt(0.262449*(_energy - Ezero))

        # mask k point larger than self.k_point
        mask = K_space <= self.k_point
        K_space = K_space[mask]
        edge_frac = edge_frac[mask]

        plt.plot(K_space, edge_frac, color='k', linewidth=2, )
        plt.plot(K_space, np.zeros(len(K_space)), '--',color='r', )
        plt.xlabel('k(A\u207b\u00b9)', )
        plt.ylabel('\u03c7(k)', )
        plt.savefig('XAFS/EXAFS/Kspace.png',dpi=800)
        plt.show()
        plt.clf()

        _edge_frac = edge_frac * K_space ** self.power

        plt.plot(K_space, _edge_frac, color='k', linewidth=2, )
        plt.plot(K_space, np.zeros(len(K_space)), '--',color='r', )
        plt.xlabel('k(A\u207b\u00b9)' )
        plt.ylabel(f'k{self.power} \u03c7(k) (A\u00B0-{self.power})' )
        plt.savefig('XAFS/EXAFS/Kspace_enhanced_absorption.png',dpi=800)
        plt.show()
        plt.clf()

        # fourier transform
        r_dis, intensity = inverse_fourier_transform(K_space, _edge_frac,self.distance)
        plt.plot(r_dis, intensity, color='k', linewidth=2,)
        plt.xlabel('radial distance (A\u00B0)')
        plt.ylabel(f' |\u03c7(R)| (A\u00B0-{self.power+1})', )
        plt.savefig('XAFS/EXAFS/FFT_EXAFS.png',dpi=800)
        plt.show()
        plt.clf()

        return None
      









            





def find_first_value_greater_than(array, target):
    for index, num in enumerate(array):
        if num > target:
            return index
    return -1  


def fitting_function(x, k1, k2, c):
    # Victoreen equation
    return k1 / x**3 + k2 / x**4 + c


def inverse_fourier_transform(Kpoint, Intensity, real_point):
    # Calculate the frequency resolution
    df = Kpoint[1] - Kpoint[0]
    # Define the time range for the inverse Fourier transform
    num_points = len(Kpoint) * 10
    rel_dis = np.linspace(0, real_point, num_points)
    # Initialize the inverse Fourier transform result (time domain signal)
    y = np.zeros(num_points, dtype=np.complex128)
    window = np.hamming(num_points)
    # Perform the inverse Fourier transform
    for k in range(len(Kpoint)):
        y += df * Intensity[k] * np.exp(2j *  Kpoint[k] * rel_dis)*window
    return rel_dis, np.abs(y/np.sqrt(2*np.pi))

def find_max_slope_x(x, y, s=0.1):
    """
    Smooth the data and find the x value corresponding to the maximum slope.

    Parameters:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        s (float, optional): Smoothing factor. Controls the smoothness of the fitted curve.

    Returns:
        float: The x value corresponding to the maximum slope.

    """
    # Use UnivariateSpline to fit the data and obtain the smoothed curve
    spline = UnivariateSpline(x, y, s=s)
    # Generate a finer grid of x values for better resolution
    x_smooth = np.linspace(x.min(), x.max(), num=1000)
    # Evaluate the smoothed curve at the fine grid of x values
    y_smooth = spline(x_smooth)
    # Calculate the slope at each point on the smoothed curve
    slopes = np.gradient(y_smooth, x_smooth)
    # Find the index of the maximum slope
    max_slope_idx = np.argmax(slopes)
    # Get the corresponding x value for the maximum slope
    max_slope_x = x_smooth[max_slope_idx]

    return max_slope_x

def find_index_in_range(arr, lower_bound, upper_bound):
    for i in range(len(arr)):
        if lower_bound <= arr[i] <= upper_bound:
            return i
    return None 
