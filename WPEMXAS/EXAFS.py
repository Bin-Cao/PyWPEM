# Extended X-ray absorption fine structure
# Author: Bin CAO <binjacobcao@gmail.com>
import os
import warnings
import pywt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import plotly.graph_objs as go


class EXAFS(object):
    def __init__(self,XAFSdata,  power = 2, distance = 5, k_point = 8,k = 3,s= None,window_size=30,hop_size=None,Extend=50,name = 'unknown',transform ='wavelet', de_bac = False):
        """
        XAFSdata : the document name of input data 
        k_point :  default k_point = 8, the cut off range of k points
        de_bac : default de_bac = False, 
        has been processed to remove the absorption background

        k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        ``k = 3`` is a cubic spline. Default is 3.
        s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        If `s` is None, ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points. Default is None.

        window_size : int, Length of each segment. Defaults to 30.
        hop_size : int, Number of points to overlap between segments. If `None`,
        hop_size = window_size / 8. Defaults to `None`.
        """
        self.XAFSdata = XAFSdata
        self.power = power 
        self.k_point = k_point
        self.distance = distance
        self.de_bac = de_bac
        self.name = name
        self.k = k
        self.s = s
        self.window_size = window_size
        if hop_size == None:
            self.hop_size = int(window_size / 8)
        if type(hop_size) == int:
            self.hop_size = hop_size
        else:
            print('type error: %s' % type(hop_size))
        self.transform = transform
        self.Extend = Extend

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
            plt.savefig('XAFS/EXAFS/NormalizedEnergy_{}.png'.format(self.name),dpi=800)
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
        spline = UnivariateSpline(energy[zero_index:-1], absor[zero_index:-1], s=self.s,k = self.k) 
        energy_base = spline(energy[zero_index:-1])

        _energy = energy[zero_index:-1] 
        edge_frac = (absor[zero_index:-1] - energy_base) / delta
    
       
        # update the Extended edge 
        add_index = find_first_value_greater_than(_energy, _energy[0]+self.Extend)
        _energy = _energy[add_index:-1]
        edge_frac = edge_frac[add_index:-1]
        energy_base = energy_base[add_index:-1]
     
        plt.plot(energy, absor, color='k', linewidth=2, )
        plt.plot(_energy, energy_base, '--',color='r', )
        plt.axvline(Ezero,linestyle='--',color='b',)
        plt.xlabel('Energy(eV)' )
        plt.ylabel('\u03BC(E)', )
        plt.savefig('XAFS/EXAFS/Splinefun_{}.png'.format(self.name),dpi=800)
        plt.show()
        plt.clf()

        plt.plot(_energy, edge_frac, color='k', linewidth=2, )
        plt.plot(_energy, np.zeros(len(_energy)), '--',color='r', )
        plt.xlabel('Energy(eV)' )
        plt.ylabel('\u03BC(E)', )
        plt.savefig('XAFS/EXAFS/SmoothEnergy_{}.png'.format(self.name),dpi=800)
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
        plt.savefig('XAFS/EXAFS/Kspace_{}.png'.format(self.name),dpi=800)
        plt.show()
        plt.clf()

        _edge_frac = edge_frac * K_space ** self.power

        plt.plot(K_space, _edge_frac, color='k', linewidth=2, )
        plt.plot(K_space, np.zeros(len(K_space)), '--',color='r', )
        plt.xlabel('k(A\u207b\u00b9)' )
        plt.ylabel(f'k{self.power} \u03c7(k) (A\u00B0-{self.power})' )
        plt.savefig('XAFS/EXAFS/Kspace_enhanced_absorption_{}.png'.format(self.name),dpi=800)
        plt.show()
        plt.clf()

        # fourier transform
        if self.transform == 'fourier':
            r_dis, intensity = inverse_fourier_transform(K_space, _edge_frac,self.distance)
            plot_two_dim(K_space, _edge_frac,self.distance,self.window_size,self.hop_size,type='STFT')
        elif self.transform == 'wavelet':
            r_dis, intensity = inverse_wavelet_transform(K_space, _edge_frac,self.distance)
            plot_two_dim(K_space, _edge_frac,self.distance,self.window_size,self.hop_size,type='WT')
        else:
            print('Unknown transform, please input transform as a string of wavelet or fourier')

        r_dis += 0.5
        plt.plot(r_dis, intensity, color='k', linewidth=2,)
        plt.xlabel('radial distance (A\u00B0)')
        plt.ylabel(f' |\u03c7(R)| (A\u00B0-{self.power+1})', )
        plt.savefig('XAFS/EXAFS/FFT_EXAFS_{}.png'.format(self.name),dpi=800)
        plt.show()
        plt.clf()

        with open(os.path.join('XAFS/EXAFS/',
                                'FineStructure_{}.csv'.format(self.name)), 'w') as wfid:
                print('dist', end=',', file=wfid)
                print('I',  file=wfid)
                for j in range(len(r_dis)):
                    print(float(r_dis[j]), end=',', file=wfid)
                    print(float(intensity[j]), file=wfid)
                    
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
    window = np.blackman(num_points)
    # Perform the inverse Fourier transform
    for k in range(len(Kpoint)):
        y += df * Intensity[k] * np.exp(2j *  Kpoint[k] * rel_dis)*window
    return rel_dis, np.abs(y/np.sqrt(2*np.pi))

def FIFT(Kpoint, Intensity, rel_dis):
    # Calculate the frequency resolution
    df = Kpoint[1] - Kpoint[0]
    # Initialize the inverse Fourier transform result (time domain signal)
    y = np.zeros(len(rel_dis), dtype=np.complex128)
    window = np.blackman(len(rel_dis))
    # Perform the inverse Fourier transform
    for k in range(len(Kpoint)):
        y += df * Intensity[k] * np.exp(2j *  Kpoint[k] * rel_dis)*window
    return np.abs(y/np.sqrt(2*np.pi))

def inverse_wavelet_transform(Kpoint, Intensity, real_point, wavelet='cmor'):
    # Calculate the frequency resolution
    df = Kpoint[1] - Kpoint[0]
    
    # Define the time range for the inverse wavelet transform
    num_points = len(Kpoint) * 10
    rel_dis = np.linspace(0, real_point, num_points)
    
    # Initialize the inverse wavelet transform result (time domain signal)
    y = np.zeros(num_points, dtype=np.complex128)
    
    # Perform the inverse wavelet transform
    for k in range(len(Kpoint)):
        # Directly use pywt.waverec for inverse wavelet transform
        wavelet_coefficients = Intensity[k] * np.exp(2j * Kpoint[k] * rel_dis)
        inverse_transform = pywt.waverec([wavelet_coefficients], wavelet, mode='per')
        y += df * inverse_transform
    
    return rel_dis, y / np.sqrt(2 * np.pi)




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

def plot_two_dim(Kpoint, Intensity,real_point,window_size,hop_size,type='STFT'):
    num_points = len(Kpoint) * 10
    rel_dis = np.linspace(0, real_point, num_points)
    R, K = np.meshgrid(rel_dis,Kpoint) # along the real space axis
    Imatrix = np.zeros(R.shape)
    if type == 'STFT':
        segment_k = sliding_window(Kpoint, window_size, hop_size)
        segment_I = sliding_window(Intensity, window_size, hop_size)
        for k in range(len(segment_k)) :
            r_int = FIFT(segment_k[k], segment_I[k], rel_dis)
            for i in range(len(segment_k[k])) :
                kvalue = segment_k[k][i]
                index = np.where(Kpoint == kvalue)[0][0]
                if all(element == 0 for element in Imatrix[index,:]): # not overlapped k
                    Imatrix[index,:] = r_int 
                else:
                    Imatrix[index,:] = (Imatrix[index,:] + r_int ) / 2 # overlap at most once
    # Create a 3D scatter plot
    trace = go.Surface(x=R, y=K, z=Imatrix, colorscale='Viridis')

    # Create layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='radial distance (A\u00B0)'),
            yaxis=dict(title='k(A\u207b\u00b9)'),
            zaxis=dict(title='I / a.u')
        )
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    # Save as an interactive HTML file
    fig.write_html('XAFS/EXAFS/interactive_plot.html')
  
    return None


def sliding_window(arr, window_size, overlap):
    result = []
    for i in range(0, len(arr) - window_size + 1, overlap):
        window = arr[i:i+window_size]
        result.append(window)
    return result

