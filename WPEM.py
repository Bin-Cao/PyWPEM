"""
module WPEM: The main call interface of WPEM, including the following functions and subroutines.

Author: Bin CAO <binjacobcao@gmail.com>

GitHub : https://github.com/Bin-Cao/WPEM

FileTypeCovert:
    Convert "Free Format (2Theta, step, 2ThetaF)" to "X,Y Data".
    input:
        :param file_name (str): File name.
    output:
        "X,Y Data" in CSV format.

CIFpreprocess:
    CIF file processing module.
    input: 
        :param filepath (str): File path of the CIF file to be calculated.
    output:
        latt (list): Lattice constants [a, b, c, al1, al2, al3].
        structure_factor (list): Structure factors [['Cu2+', 0, 0, 0], ['O-2', 0.5, 1, 1], ...].

BackgroundFit:
    Strips the noise from the diffraction pattern.
    input:
        :param intensity_csv (str): Experimental measurement file.
    output:
        std (float/array): Standard deviation of the background distribution.

XRDfit:
    Whole pattern decomposition module through EM-Bragg solver.
    input:
        :param wavelength (list): List of wavelengths of diffraction waves.
        :param Var (float or array): Statistical variance of background.
        :param Lattice_constants (list): Initial values of lattice constants.
        :param no_bac_intensity_file (str): CSV document containing diffraction intensity without background intensity.
        :param original_file (str): CSV document containing diffraction intensity.
        :param background_file (str): CSV document containing the fitted background intensity.
    output:
        result_folder : the results.

Amorphous_fit:
    Amorphous qualitative description module.
    input:
        :param mix_component (int): Number of amorphous peaks.
    output:
        result_folder : the results.

AmorphousRDFun:
    Amorphous qualitative description module.
    input:
        :param wavelength (float): Wavelength of X-ray.
    return:
        circle_x (array): Circle X.

Plot_Components:
    Decomposition component drawing.
    input:
        :param low_boundary (float): The smallest diffraction angle studied.
        :param up_boundary (float): The largest diffraction angle studied.
        :param wavelength (list): List of X-ray wavelengths.
    output:
        result_folder : the results.

XRDSimulation:
    
"""


from .EMBraggOpt.EMBraggSolver import WPEMsolver
from .Background.BacDeduct import TwiceFilter
from .Amorphous.fitting.AmorphousFitting import Amorphous_fitting
from .Amorphous.QuantitativeCalculation.AmorphousRDF import RadialDistribution
from .DecomposePlot.plot import Decomposedpeaks
from .XRDSimulation.Simulation import XRD_profile
from .Extinction.XRDpre import profile
# from .Raman.Decompose.RamanFitting import fit
import datetime
from time import time


import datetime
now = datetime.datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')

print('A Diffraction Refinement Software : WPEM')
print('Bin Cao, MGI, Shanghai University, Shanghai, CHINA.')
print('URL : https://github.com/Bin-Cao/WPEM')
print('Executed on :',formatted_date_time, ' Have a great day.')
print('\n')


def XRDfit(wavelength, Var, Lattice_constants, no_bac_intensity_file, original_file, bacground_file, two_theta_range = None,structure_factor = None, 
        bta=0.8, bta_threshold = 0.5,limit=0.0005, iter_limit=0.05, w_limit=1e-17, iter_max=40, lock_num = 2, asy_C=0.5, s_angle=50, 
        subset_number=9, low_bound=65, up_bound=80, MODEL = 'REFINEMENT', Macromolecule =False, cpu = 4, num =1, EXACT = False,Cu_tao = None, Ave_Waves = False,):
    """
    PACKAGE: Whole Pattern fitting of powder X-ray diffraction by Expectation Maximum (WPEM) algorithm

    More information about WPEM algorithm, e.g., source code, user manual,  examples etc., are provided at the open-source, dynamic-update library of WPEM package: [github.com/Bin-Cao/WpenPattern]. Welcome to participate in the community building of WpenPattern package, and provide your code and suggestions. 

    :param wavelength: list type, The wavelength of diffraction waves

    :param Var: a constant or a array, Statistical variance of background 

    :param Lattice_constants: 2-dimensional list, initial value of Lattice_constants

    :param no_bac_intensity_file: csv document, Diffraction intensity file with out bacground intensity

    :param original_file: csv document, Diffraction intensity file

    :param bacground_file: csv document, The fitted background intensity 

    :param two_theta_range: The studied range of diffraction angels 

    :param structure_factor: list, if EXACT = True, the structure factor is used calculating the volume fraction of mixed components

    :param bta: float default = 0.8, the ratio of Lorentzian components in PV function

    :param bta_threshold: float default = 0.5, a preset lower boundary of bta

    :param limit: float default = 0.0005, a preset lower boundary of sigma2

    :param iter_limit: float default = 0.05, a preset threshold iteration promotio (likelihood) 

    :param w_limit: float default = 1e-17,  a preset lower boundary of peak weight

    :param iter_max: int default = 40, maximum number of iterations

    :param lock_num: int default = 3,  restriction  of loglikelihood iterations continously decrease    

    :param asy_C, s_angle: Peak Correction Parameters

    :param subset_number (default = 9), low_bound (default = 65), up_bound (default = 80): subset_number peaks between low_bound and up_bound are used to calculate the new lattice constants by bragg law

    :param MODEL: str default = 'REFINEMENT' for lattice constants REFINEMENT; 'ANALYSIS' for components ANALYSIS

    :param Macromolecule : Boolean default = False, for profile fitting of crystals. True for macromolecules

    :param cpu : int default = 4, the number of processors

    :param num : int default = 1, the number of the strongest peaks used in calculating volume fraction

    :param EXACT : Boolean default = False, True for exact calculation of volume fraction by diffraction intensity theory
    
    :return: An instantiated model
    """
    
    if Ave_Waves == 1:
        wavelength = [2/3 * wavelength[0]+ 1/3 * wavelength[1]]
    else:
        pass

    time0 = time()
    start_time = datetime.datetime.now()
    print('Started at', start_time.strftime('%c'),'\n')

    MultiTasks = len(Lattice_constants)

    # detect Lattice_constants and return a singal 
    singal = []
    for detect in range(MultiTasks):
        if len(Lattice_constants[detect]) == 6:
            singal.append(0)
            pass
        elif len(Lattice_constants[detect]) == 7:
            if Lattice_constants[detect][6] == 'fixed':
                singal.append(1)
                print('The lattice constants of input system {} are fixed'.format(detect))
                Lattice_constants[detect].remove('fixed')
                print('viz., fixed : {}'.format(Lattice_constants[detect]))
            else:
                print('Type Error - only \'fixed\' is allowed')
        else:
            print('Type Error - the form of input lattice constants are illegal')

    initial_peak_file = []
    for task in range(MultiTasks):
        initial_peak_file_task = "peak{task}.csv".format(task=task)
        initial_peak_file.append(initial_peak_file_task)
    # List of The file name of initial (2theta data)
    
    Inst_WPEM = WPEMsolver(wavelength,  Var,  asy_C,  s_angle,
        subset_number,  low_bound,  up_bound,
        Lattice_constants,singal,  no_bac_intensity_file,  original_file,
        bacground_file, two_theta_range, initial_peak_file,  bta,  bta_threshold,
        limit,  iter_limit, w_limit, iter_max,  lock_num,  structure_factor,  MODEL,
        Macromolecule, cpu,  num, EXACT, Cu_tao
    )
        
    Rp, Rwp, i_ter, flag = Inst_WPEM.cal_output_result()

    if flag == 1:
        print("%s-th iterations, convergence is achieved." % i_ter +
            '\n Rp: %.3f' % Rp + '\nRwp: %.3f ' % Rwp)
    elif flag == 2:
        print("%s-th iterations, reach the limit of ϵ." % i_ter +
            '\n Rp: %.3f' % Rp + '\nRwp: %.3f ' % Rwp)
    elif flag == 3:
        print("%s-th iterations, reach the maximum number of iteration steps." % i_ter +
            '\n Rp: %.3f' % Rp + '\nRwp: %.3f ' % Rwp)
    elif flag == 4:
        print("%s-th iterations, reach the limit of lock_num." % i_ter +
            '\n Rp: %.3f' % Rp + '\nRwp: %.3f ' % Rwp)
    elif flag == -1:
        print('The three input files do not match!')

    endtime = time()
    Durtime = "%.0f hours " % int((endtime - time0) / 3600) + "%.0f minute  " % int(
        ((endtime - time0) / 60) % 60) + "%.0f second  " % ((endtime - time0) % 60)
    print('\n')
    print('WPEM program running time : ', Durtime)
    return Durtime

def BackgroundFit(intensity_csv, LFctg = 0.5, lowAngleRange=None, bac_num=None, bac_split=5, window_length=17, 
                    polyorder=3, poly_n=6, mode='nearest', bac_var_type='constant', Model='XRD'):
    """
    :param intensity_csv: the experimental observation

    :param LFctg: low frequency filter Percentage, default  = 0.5

    :param lowAngleRange: low angle (2theta) with obvious background lift phenomenon

    :param bac_num: the number of background points in the background set

    :param bac_split: the background spectrum is divided into several segments
    
    :param window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `x`.
    
    :param polyorder: int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.

    :param poly_n: background mean function fitting polynomial degree

    :param mode:  str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.

    :param bac_var_type: 
        A pattern describing the background distribution
        one of constant, polynomial, multivariate gaussia

    :param  Model:
        Display the background curve of XRD diffraction spectrum (Model='XRD')
        and Raman spectrum (Model='Raman') according to the type

    :return:
        std of the background distribution
    """
    module = TwiceFilter(Model)
    module.FFTandSGFilter(intensity_csv, LFctg, lowAngleRange, bac_num, bac_split, window_length,polyorder,  poly_n, mode, bac_var_type)
    
def FileTypeCovert(file_name):
    module = TwiceFilter()
    module.convert_file(file_name)

def Amorphous_fit(mix_component, ang_range = None, sigma2_coef = 0.5, max_iter = 5000, peak_location = None,Wavelength = 1.54184):
    """
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
    Amorphous_fitting(mix_component, ang_range, sigma2_coef, max_iter, peak_location,Wavelength)

def AmorphousRDFun(wavelength, r_max = 5,density_zero=None,NAa=None,highlight= 4,value=0.6):
    module = RadialDistribution(wavelength, r_max)
    module.RDF(density_zero,NAa,highlight,value)

def Plot_Components(lowboundary, upboundary, wavelength,name = None, Macromolecule = False,phase = 1,Pic_Title = False):
    """
    :param lowboundary : float, the smallest diffraction angle studied

    :param upboundary : float, the largest diffraction angle studied 

    :param wavelength : list, the wavelength of the X rays

    :param name : list, assign the name of each crystal through this parameter

    :param Macromolecule: whether it contains amorphous, used in amorphous fitting

    :param phase: the number of compounds contained in diffraction signals

    :param Pic_Title: Whether to display the title of the pictures
    
    """
    module = Decomposedpeaks()
    module.decomposition_peak(lowboundary, upboundary, wavelength,name, Macromolecule ,phase,Pic_Title)

def XRDSimulation(structure_factor,mu_list,gamma_list, sigma2_list, Mult, HKL_list,  LatticCs, Wavelength=1.54184,two_theta_range=(0, 90,0.02)):
    """
    structure_factor ==> [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
    mu_list ==> calculated mui
    gamma_list ==> calculated gamma
    sigma2_list ==> calculated sigma2
    HKL_list ==> [H,K,L] in shape of n*3
    LatticCs ==> [a,b,c,alpha,beta,gamma]
    """
    XRD_profile(structure_factor,mu_list,gamma_list, sigma2_list, Mult, HKL_list,  LatticCs, Wavelength).Simulate(two_theta_range)
    

def CIFpreprocess(filepath, wavelength='CuKa',two_theta_range=(10, 90),latt = None, structure_factor = None):
    """
    for a single crystal
    Computes the XRD pattern and save to csv file
    Args:
        filepath (str): file path of the cif file to be calculated
        two_theta_range ([float of length 2]): Tuple for range of
            two_thetas to calculate in degrees. Defaults to (0, 90). Set to
            None if you want all diffracted beams within the limiting
            sphere of radius 2 / wavelength.
    return 
    latt: lattice constants : [a, b, c, al1, al2, al3]
    structure_factor : [['Cu2+',0,0,0,],['O-2',0.5,1,1,],.....]  
    """
    profile(wavelength,two_theta_range).generate(filepath,latt,structure_factor)