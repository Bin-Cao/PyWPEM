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
        AtomCoordinates (list): Structure factors [['Cu2+', 0, 0, 0], ['O-2', 0.5, 1, 1], ...].

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
    XRDSimulation
    input:
        :param filepath (str): file path of the cif file to be calculated
    output:
        files

... developing
"""

from .EMBraggOpt.EMBraggSolver import WPEMsolver
from .Background.BacDeduct import TwiceFilter, convert_file,read_xrdml
from .Amorphous.fitting.AmorphousFitting import Amorphous_fitting
from .Amorphous.QuantitativeCalculation.AmorphousRDF import RadialDistribution
from .DecomposePlot.plot import Decomposedpeaks
from .XRDSimulation.Simulation import XRD_profile
from .Extinction.XRDpre import profile
from .StructureOpt.SiteOpt import BgolearnOpt
from .WPEMXPS.XPSEM import XPSsolver
from .WPEMXAS.EXAFS import EXAFS
from .GraphStructure.graph import CrystalGraph
# from .Raman.Decompose.RamanFitting import fit
import datetime
import numpy as np
from time import time
import datetime


logo = '''
██╗    ██╗██████╗ ███████╗███╗   ███╗
██║    ██║██╔══██╗██╔════╝████╗ ████║
██║ █╗ ██║██████╔╝█████╗  ██╔████╔██║
██║███╗██║██╔═══╝ ██╔══╝  ██║╚██╔╝██║
╚███╔███╔╝██║     ███████╗██║ ╚═╝ ██║
 ╚══╝╚══╝ ╚═╝     ╚══════╝╚═╝     ╚═╝                                                  
'''

now = datetime.datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
print(logo)
print('A Diffraction Refinement Software : WPEM')
print('Bin Cao, Advanced Materials Thrust, Hong Kong University of Science and Technology (Guangzhou)')
print('URL : https://github.com/Bin-Cao/WPEM')
print('Executed on :',formatted_date_time, ' | Have a great day.')
print('='*100)

def XRDfit(wavelength, Var, Lattice_constants, no_bac_intensity_file, original_file, bacground_file, density_list=None, two_theta_range = None,structure_factor = None, 
        bta=0.8, bta_threshold = 0.5,limit=0.0005, iter_limit=0.05, w_limit=1e-17, iter_max=40, lock_num = 2, asy_C=0.5, s_angle=50, 
        subset_number=9, low_bound=65, up_bound=80, InitializationEpoch=2, MODEL = 'REFINEMENT', Macromolecule =False, cpu = 4, num =3, EXACT = False,
        Cu_tao = None, Ave_Waves = False,loadParams=False, ZeroShift=False,):
    """
    :param wavelength: list type, The wavelength of diffraction waves
    :param Var: a constant or a array, Statistical variance of background 
    :param Lattice_constants: 2-dimensional list, initial value of Lattice_constants
    :param no_bac_intensity_file: csv document, Diffraction intensity file with out bacground intensity
    :param original_file: csv document, Diffraction intensity file
    :param bacground_file: csv document, The fitted background intensity 
    :param density_list : list default is None, the densities of crytal, can be calculated by fun. WPEM.CIFpreprocess()
        e.g., 
        _,_,d1 = WPEM.CIFpreprocess()
        _,_,d2 = WPEM.CIFpreprocess()
        density_list = [d1,d2]
    :param two_theta_range: The studied range of diffraction angels 
    :param structure_factor: list, if EXACT = True, the structure factor is used calculating
            the mass fraction of mixed components
    :param bta: float default = 0.8, the ratio of Lorentzian components in PV function
    :param bta_threshold: float default = 0.5, a preset lower boundary of bta
    :param limit: float default = 0.0005, a preset lower boundary of sigma2
    :param iter_limit: float default = 0.05, a preset threshold iteration promotio (likelihood) 
    :param w_limit: float default = 1e-17,  a preset lower boundary of peak weight
    :param iter_max: int default = 40, maximum number of iterations
    :param lock_num: int default = 3,  restriction  of loglikelihood iterations continously decrease    
    :param asy_C, s_angle: Peak Correction Parameters
    :param subset_number (default = 9), low_bound (default = 65), up_bound (default = 80): subset_number peaks
            between low_bound and up_bound are used to calculate the new lattice constants by bragg law
    :param InitializationEpoch: int, default = 2, at initialization, frozen the peaks location for searching a satisified Model initial parameters.
    :param MODEL: str default = 'REFINEMENT' for lattice constants REFINEMENT; 'ANALYSIS' for components ANALYSIS
    :param Macromolecule : Boolean default = False, for profile fitting of crystals. True for macromolecules
    :param cpu : int default = 4, the number of processors
    :param num : int default = 3, the number of the strongest peaks used in calculating mass fraction
    :param EXACT : Boolean default = False, True for exact calculation of mass fraction by diffraction intensity theory
    :param Cu_tao: The restriction on the diffraction intensities of copper Kα1 and Kα2 rays.
    :param Ave_Waves: A boolean, default is False. Set to True to optimize using the average wavelength of Kα1 and Kα2.
    :param loadParams : Boolean default = False, for loading parameters
    :param ZeroShift : If ZeroShift == True and the standard sample is available, the instrument offset can be calibrated
    :return: An instantiated model
    """
    
    if Ave_Waves == 1 :
        wavelength = [2/3 * wavelength[0]+ 1/3 * wavelength[1]]
    else:
        pass
    if ZeroShift == True and len(wavelength) == 2:
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
        Lattice_constants, density_list, singal, no_bac_intensity_file,  original_file,
        bacground_file, two_theta_range, initial_peak_file,  bta,  bta_threshold,
        limit,  iter_limit, w_limit, iter_max,  lock_num,  structure_factor,  MODEL,
        InitializationEpoch,Macromolecule, cpu,  num, EXACT, Cu_tao,loadParams,ZeroShift
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
                    polyorder=3, poly_n=6, mode='nearest', bac_var_type='constant', Model='XRD',noise=None,segement=None):
    """
    :param intensity_csv: the dir of experimental XRD data
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
        or Raman spectrum (Model='Raman') or X-ray photoemission spectrography (Model='XPS') according to the type
    :param noise:
            float, default is None 
            the noise level applied to gaussian processes model
    :param segement:
            A list containing the background point range. It can be easily defined by the user to manually adjust the background domains.
    :return:
        std of the background distribution
    """
    module = TwiceFilter(Model,segement)
    return module.FFTandSGFilter(intensity_csv, LFctg, lowAngleRange, bac_num, bac_split, window_length,polyorder,  poly_n, mode, bac_var_type,noise)
    
def FileTypeCovert(file_name, file_type='dat'):
    """
    Convert XRD data from different file formats to the appropriate format.

    :param file_name: str
        The name of the original XRD data file (including file extension).
    
    :param file_type: str, optional, default='dat'
        The type of the original XRD data file. Can be 'dat' or 'xrdml'.
        - 'dat': Process the file as a .dat format.
        - 'xrdml': Process the file as a .xrdml format.

    :return: Processed data (depending on the file type)
    """
    if file_type == 'dat':
        # Convert .dat file
        print('The converted data are saved in the ConvertedDocuments folder.')
        return convert_file(file_name)
    elif file_type == 'xrdml':
        # Read .xrdml file
        print('The converted data are saved in the ConvertedDocuments folder.')
        return read_xrdml(file_name)
    else:
        # Handle unsupported file type
        print(f"Unsupported file type: {file_type}. Please use 'dat' or 'xrdml'.")
        return None


def Amorphous_fit(mix_component,amor_file = None, ang_range = None, sigma2_coef = 0.5, max_iter = 5000, peak_location = None,Wavelength = 1.54184):
    """
    :param mix_component : the number of amorphous peaks 
    :param amor_file : the amorphous file locationn
    :param ang_range : default is None
        two theta range of study, e.g., ang_range = (20,80)
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
    return Amorphous_fitting(mix_component, amor_file,ang_range, sigma2_coef, max_iter, peak_location,Wavelength)

def AmorphousRDFun(wavelength=1.54184, amor_file=None,r_max = 5,density_zero=None,Nf2=None,highlight= 4,value=0.6):
    """
    Function to compute the radial distribution function (RDF) for amorphous materials based on X-ray diffraction data.
    
    Reference: 
        J. Chem. Phys. 2, 551 (1934); https://doi.org/10.1063/1.1749528

    :param wavelength: float, optional, default=1.54184
        The wavelength of the X-ray used for diffraction. Typically set to 1.54184 Å for Cu Kα radiation.

    :param amor_file: str, optional, default=None
        The file path to the amorphous phase intensity data. If not provided, the function will search for the default file 
        located at '/DecomposedComponents/Amorphous.csv'.

    :param r_max: float, optional, default=5
        The maximum radius (in Å) from the center in the RDF plot. Defines the range of the radial distribution function.

    :param density_zero: float, optional, default=None
        The average density of the sample in atoms per cubic centimeter (atoms/cc). This is needed to calculate the 
        atomic distribution in the material.

    :param Nf2: float, optional, default=None
        The effective number of atoms in the sample (N) multiplied by the atom scattering intensity (Aa). This is used 
        to compute the intensity contributions of each atomic species.

    :param highlight: int, optional, default=4
        The number of peaks to highlight in the RDF plot. Peaks are marked in the graphical output for easier visualization.

    :param value: float, optional, default=0.6
        Assumes that scattering can be treated as independent at sin(θ/λ) = 0.6. This parameter influences how scattering 
        is modeled for the RDF calculation.

    :return: tuple
        Returns the RDF plot and specified peaks.
    
    """
    module = RadialDistribution(wavelength, r_max)
    return module.RDF(amor_file,density_zero,Nf2,highlight,value)

def Plot_Components(lowboundary, upboundary, wavelength, density_list=None, name = None, Macromolecule = False,phase = 1,Pic_Title = False,lifting=None):
    """
    :param lowboundary : float, the smallest diffraction angle studied
    :param upboundary : float, the largest diffraction angle studied 
    :param wavelength : list, the wavelength of the X-ray
    :param density_list : list default is None, the densities of crytal, can be calculated by fun. WPEM.CIFpreprocess()
        e.g., 
        _,_,d1 = WPEM.CIFpreprocess()
        _,_,d2 = WPEM.CIFpreprocess()
        density_list = [d1,d2]
    :param name : list, assign the name of each crystal through this parameter
    :param Macromolecule: whether it contains amorphous, used in amorphous fitting
    :param phase: the number of compounds contained in diffraction signals
    :param Pic_Title: Whether to display the title of the pictures, some title is very long
    :param lifting : list, whether to lift the base of each components
    """
    module = Decomposedpeaks()
    return module.decomposition_peak(lowboundary, upboundary, wavelength,density_list,name, Macromolecule ,phase,Pic_Title,lifting)

def XRDSimulation(filepath,wavelength='CuKa',two_theta_range=(10, 90, 0.01),SuperCell=False,PeriodicArr=[3,3,3],ReSolidSolution = None, RSSratio=0.1,
                   Vacancy=False, Vacancy_atom = None, Vacancy_ratio = None,GrainSize = None,LatticCs = None,PeakWidth=True, CSWPEMout = None,
                   orientation=None,thermo_vib=None,zero_shift = None, bacI=False,seed=42):
    """
    :param filepath (str): file path of the cif file to be calculated
    :param wavelength: The wavelength can be specified as either a
                float or a string. If it is a string, it must be one of the
                supported definitions in the dict of WAVELENGTHS.
                Defaults to "CuKa", i.e, Cu K_alpha radiation.
    :param two_theta_range ([float of length 2]): Tuple for range of
        two_thetas to calculate in degrees. Defaults to (0, 90). Set to
        None if you want all diffracted beams within the limiting
        sphere of radius 2 / wavelength.
    :param SuperCell : : bool, default False
        If True, a supercell will be established
    :param PeriodicArr : list, default [3, 3, 3]
        Periodic translation the lattice 3 times along x, y, z direction
    :param ReSolidSolution : list, default None
        If not None, should contain the original atom type and replace atom type
        e.g., ReSolidSolution = ['Ru4+', 'Na2+'], means 'Na2+' replaces the 'Ru4+' atom locations
    :param RSSratio :float, default 0.1
        In the supercell, the percentage of 'Ru4+' atoms to be replaced randomly by 'Na2+'
    :param Vacancy : bool, default False
        If True, consider the charge balance, otherwise do not consider
    :param Vacancy_atom : str, default None
        If Vacancy is True, the atom to be considered for charge balancing
        e.g., Vacancy_atom = 'O2+'
    :param Vacancy_ratio : int, default None
        If Vacancy is True, the ratio of the number of vacancy atoms to the number of replaced atoms
        e.g., 1 means for each 'Ru4+' atom replaced by 'Na2+'atom, a vacancy 'O2+' is created for balancing the charge
    :param GrainSize
        The default value is 'none,' or you can input a float representing 
        the grain size within a range of 5-30 nanometers.
    :param LatticCs: The lattice constants after WPEM refinement. The default is None. 
        If set to None, WPEM reads lattice constants from an input CIF file. Read parameters from CIF by using ..Extinction.XRDpre.
    :param PeakWidth
        PeakWidth=False, The peak width of the simulated peak is 0
        PeakWidth=True, The peak width of the simulated peak is set to the peak obtained by WPEM
    :param CSWPEMout : location of corresponding Crystal System WPEMout file
        if None, PEM simulates the peaks as the default Voigt function
        else WPEM simulates the peaks by the decomposed peak shapes
    :param orientation: The default value is 'none,' or you can input a list such as [-0.2, 0.3],
        adjusting intensity within the range of (1-20%)I to (1+30%)I.
    :param thermo_vib: The default is 'none,' or you can input a float, for example, thermo_vib=0.05, 
        representing the variability in the average atom position. It is recommended to use values between 0.05 and 0.5 angstrom.
    :param zero_shift: The default is 'none,' or you can input a float, like zero_shift=1.5,
        which represents the instrument zero shift. It is recommended to use values between 2θ = -3 and 3 degrees.
    :param bacI: The default is False. If bacI = True, a three-degree polynomial function is applied
        to simulate the background intensity.
    :param seed : default seed = 42
    return : Structure factors 
    """
    return XRD_profile(filepath,wavelength,two_theta_range,SuperCell,PeriodicArr,ReSolidSolution, RSSratio, GrainSize,LatticCs,PeakWidth, CSWPEMout).Simulate(Vacancy=Vacancy, Vacancy_atom = Vacancy_atom, Vacancy_ratio = Vacancy_ratio,orientation=orientation,thermo_vib=thermo_vib,zero_shift = zero_shift, bacI=bacI,seed=seed)
    
def CIFpreprocess(filepath, wavelength='CuKa',two_theta_range=(10, 90),latt = None, AtomCoordinates = None,show_unitcell=False,cal_extinction=True,relaxation=False):
    """
    For a single crystal:
    Computes the XRD pattern and saves it to a CSV file.

    :param filepath: str
        The file path to the CIF file for which the XRD pattern will be calculated.
    :param wavelength: float or str, optional, default="CuKa"
        The wavelength of the X-ray. If provided as a string, it must be one of the keys in the WAVELENGTHS dictionary.
        By default, this is set to "CuKa", corresponding to Cu K_alpha radiation.
    :param two_theta_range: list of float, length 2, optional, default=(0, 90)
        A tuple specifying the range of 2θ (in degrees) to calculate. Defaults to (0, 90). 
        Set to None to include all diffracted beams within the limiting sphere of radius 2 / wavelength.
    :param latt: list
        The lattice constants, formatted as [a, b, c, α, β, γ], where `a`, `b`, and `c` are the edge lengths, and 
        `α`, `β`, and `γ` are the angles between them (in degrees).
    :param AtomCoordinates: list of lists
        A list of atomic species and their coordinates in the unit cell, formatted as:
        [['Cu2+', 0, 0, 0], ['O-2', 0.5, 1, 1], ...].
        Note: '22' is the space group code.
        This input interface is designed to handle non-standard CIF files by allowing manual input for structure reading 
        and method definition.
    :param relaxation: bool, optional, default=False
        Whether to relax the structure using the M3Gnet relaxation potential field.

    :return: tuple
        A tuple containing:
        - `latt`: The lattice constants [a, b, c, α, β, γ].
        - `AtomCoordinates`: The atomic species and coordinates in the unit cell, e.g., [['Cu2+', 0, 0, 0], ['O-2', 0.5, 1, 1], ...].
        - `lattice_density`: The calculated lattice density (ρ).
    """

    return profile(wavelength,two_theta_range,show_unitcell,cal_extinction,relaxation).generate(filepath,latt,AtomCoordinates)



def SubstitutionalSearch(xrd_pattern, cif_file,random_num=8, wavelength='CuKa',search_cap=50,SolventAtom = None, SoluteAtom= None,max_iter = 100,cal_extinction=True):
    """
        :param xrd_pattern: str
        The path to the experimental XRD diffraction pattern of a single crystal, containing 2theta and intensity values. 
        :param cif_file: str
            The path to the CIF (Crystallographic Information File) associated with the crystal structure.
        :param random_num: int
            The number of times the structure will be randomly initialized to establish the training dataset for BGO.
        :param wavelength: float or str, optional, default="CuKa"
            The wavelength of the X-ray used. If provided as a string, it must be one of the keys in the WAVELENGTHS dictionary. 
            By default, this is set to "CuKa", which corresponds to Cu K_alpha radiation.
        :param search_cap: int
            This parameter limits the number of combinations considered during the search to avoid excessive memory usage. It helps to truncate the search process and prevent memory overflow.
        :param solvent_atoms: str, optional, default=None
            The solvent atoms in the system. If not provided, it defaults to None. For example, SolventAtom = 'Cu'.
        :param solute_atoms: str, optional, default=None
            The solute atoms in the system. If not provided, it defaults to None. For example, SoluteAtom = 'Ti'.
        :param max_iter: int
            The maximum number of iterations to run during the computation.
        :param cal_extinction: bool, optional, default=False
            Whether to consider the extinction effect in the calculation. Set to `True` if the extinction effect should be included, `False` otherwise.
        """
    return BgolearnOpt(xrd_pattern, cif_file, random_num,wavelength,search_cap,cal_extinction). Substitutional_SS(SolventAtom, SoluteAtom ,max_iter)


def XPSfit(Var, atomIdentifier, satellitePeaks,no_bac_df, original_df, bacground_df, energy_range = None, bta=0.8, bta_threshold = 0.5,limit=0.0005, 
           iter_limit=0.05, w_limit=1e-17, iter_max=40, lock_num = 2, asy_C=0., s_energy=[100,1000], tao=0.5, ratio=0.8,
       InitializationEpoch=2,loadParams=False,):
    """

    s_energy : asymmetric peak's range, a list [900,950]
    """
    time0 = time()
    start_time = datetime.datetime.now()
    print('Started at', start_time.strftime('%c'),'\n')

    
    XPS = XPSsolver(Var, asy_C, s_energy, atomIdentifier, satellitePeaks,no_bac_df,original_df,bacground_df, energy_range, bta, 
                    bta_threshold,limit, iter_limit,w_limit,iter_max,lock_num, InitializationEpoch, loadParams,tao,ratio
    )
        
    Rp, Rwp, i_ter, flag = XPS.cal_output_result()

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
    print('WPEM-XPS program running time : ', Durtime)
    return Durtime

def EXAFSfit(XAFSdata,  power = 2, distance = 5, k_point = 8,k = 3,s= None,window_size=30,hop_size=None,Extend=50,name = 'unknown',transform ='fourier',de_bac = False,Ezero = None, first_cutoff_energy=None,second_cutoff_energy=None):
    """
    :param XAFSdata : the document name of input data 
    :param k_point :  default k_point = 8, the cut off range of k points
    :param de_bac : default de_bac = False, 
        has been processed to remove the absorption background
    :param k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        ``k = 3`` is a cubic spline. Default is 3.
        s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        If `s` is None, ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points. Default is None.

    :param transform, default is 'wavelet', the inverse transform manner, 'wavelet', 'fourier'
    :paramCutoff_energy: The data behind cutoff energy will be used to calculate the mean absorption.

    """
    return EXAFS(XAFSdata,  power, distance , k_point ,k,s,window_size,hop_size,Extend,name,transform,de_bac).fit(Ezero , first_cutoff_energy,second_cutoff_energy)

def CryGraph(folder_path,BK_boundary_condition = False):
    """
    :param folder_path: str, the folder path to save CIF files.
    :param BK_boundary_condition: bool, default: False. If True, the Bon Kaman boundary condition is applied to construct graph edges. 
        This may result in longer calculation times.
    
    # read in data
    import numpy as np
    import pickle
    
    # define the node of graphs
    with open('./CifParserToGraph/Node.pkl', 'rb') as file:
        Node = pickle.load(file)
    
    # define the edge of graphs
    with open('./CifParserToGraph/Edge.pkl', 'rb') as file:
        Edge = pickle.load(file)
    
    # define the label of graphs
    y_sg = np.load('./CifParserToGraph/Y_group.npy')
    y_ls = np.load('./CifParserToGraph/Y_system.npy')
    """
    CrystalGraph(folder_path).generate_graph(BK_boundary_condition)




# ----------------------------------------------------------------
# ----------------------------------------------------------------
# there are several tools developed for facilitating the usage of WPEM packages
def ToMatrix(Node):
    """
    Convert a dictionary of values into a matrix.

    Parameters:
    - Node: Input dictionary in the form {key: (value1, value2, ...), ...}

    Returns:
    - matrix: Matrix representation of the values in the dictionary.
    """
    # Get the values from the dictionary
    values = list(Node.values())
    
    # Convert the values to a matrix
    matrix = np.array(values)
    
    return matrix


def ToAdj(Edge, size, sel=False):
    """
    Generate an adjacency matrix.

    Parameters:
    - Edge: Edge data, in the form [[0, 24], [1, 25], ...]
    - size: Size of the matrix
    - sel: Whether to set the diagonal to 1, default is False

    Returns:
    - adjacency_matrix: Generated adjacency matrix
    """
    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((size, size), dtype=int)

    # Fill the adjacency matrix with edge data
    for edge in Edge:
        adjacency_matrix[edge[0]][edge[1]] += 1

    # If sel is True, set the diagonal to 1
    if sel:
        adjacency_matrix = adjacency_matrix + np.eye(size, dtype=int)

    return adjacency_matrix


def split_datasets(Node, Edge, target, ratio =0.2):
    """
    Split three datasets (Node, Edge, target) into training and testing sets.

    Parameters:
    - Node: NumPy array representing the Node dataset
    - Edge: NumPy array representing the Edge dataset
    - target: NumPy array representing the target dataset

    Returns:
    - Node_train, Node_test: Training and testing sets for Node
    - Edge_train, Edge_test: Training and testing sets for Edge
    - target_train, target_test: Training and testing sets for target
    """
    # Create an index array representing the order of data
    data_indices = np.arange(len(target))

    # Randomize the index array
    np.random.shuffle(data_indices)

    # Determine the split point
    split_point = int((1-ratio) * len(data_indices))

    # Split the datasets
    train_indices = data_indices[:split_point]
    test_indices = data_indices[split_point:]

    # Convert datasets to NumPy arrays
    Node = np.array(Node)
    Edge = np.array(Edge)
    target = np.array(target)

    # Use index arrays to split Node, Edge, and target
    Node_train, Node_test = Node[train_indices], Node[test_indices]
    Edge_train, Edge_test = Edge[train_indices], Edge[test_indices]
    target_train, target_test = target[train_indices], target[test_indices]

    return Node_train, Node_test, Edge_train, Edge_test, target_train, target_test


def Laplacian(A):
    """
    Compute the normalized Laplacian matrix L = I - D^(-1/2) * A * D^(-1/2).

    Parameters:
    - A: Input matrix

    Returns:
    - L: Normalized Laplacian matrix
    - eigenvalues matrix of L
    - eigenvectors matrix of L 
    """
    # Degree matrix D
    D = np.diag(np.sum(A, axis=1))

    # Compute D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # Compute L = I - D^(-1/2) * A * D^(-1/2)
    L = np.identity(A.shape[0]) - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
    
    L_reg = L + 1e-5 * np.identity(A.shape[0])
    eigenvaluesMatrix, eigenvectorsMatrix = np.linalg.eig(L_reg)
    eigenvaluesMatrix = np.diag(eigenvaluesMatrix)

    return L_reg, eigenvaluesMatrix.real , eigenvectorsMatrix.real