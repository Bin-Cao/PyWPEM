from .WPEMProcess import WPEM
import datetime
from time import time


class Built(object):
    def fit(self, wavelength, Var, Lattice_constants, no_bac_intensity_file, original_file, bacground_file, two_theta_range = None,structure_factor = None, 
            LogPrint=False,bta=0.8, bta_threshold = 0.5,limit=0.0005, iter_limit=0.05, w_limit=1e-17, iter_max=40, lock_num = 2, asy_C=0.5, s_angle=50, 
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

        :param structure_factor: list, if EXACT = True, the structure factor is used calculating the volume fraction of mixed components

        :param LogPrint: Boolean default = False, if LogPrint = True, WPEM will print the log of fitting accuracy

        :param bta: float default = 0.8, the ratio of Lorentzian components in PV function

        :param bta_threshold: float default = 0.5, a preset lower boundary of bta

        :param limit: float default = 0.0005, a preset lower boundary of sigma2

        :param iter_limit: float default = 0.05, a preset threshold iteration promotion (likelihood) 

        :param w_limit: float default = 1e-17,  a preset lower boundary of peak weight

        :param iter_max: int default = 40, maximum number of iterations

        :param lock_num: int default = 3,  in case of loglikelihood iterations continously decrease    

        :param asy_C, s_angle: Peak Correction Parameters

        :param subset_number (default = 9), low_bound (default = 65), up_bound (default = 80): subset_number peaks between low_bound and up_bound are used to calculate the new lattice constants by bragg law

        :param MODEL: str default = 'REFINEMENT' for lattice constants REFINEMENT; 'ANALYSIS' for components ANALYSIS

        :param Macromolecule : Boolean default = False, for profile fitting of crystals. True for macromolecules

        :param cpu : int default = 4, Parallel computatis CPU core numerus

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
        print('Started at', start_time.strftime('%c'))

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

        Inst_WPEM = WPEM(wavelength,  Var,  asy_C,  s_angle,
            subset_number,  low_bound,  up_bound,
            Lattice_constants,singal,  no_bac_intensity_file,  original_file,
            bacground_file, two_theta_range, initial_peak_file,  bta,  bta_threshold,
            limit,  iter_limit, w_limit, iter_max,  lock_num,  structure_factor,  MODEL,
            Macromolecule,  cpu,  num, EXACT, Cu_tao
        )
        # whether or not print out the update log
        if  LogPrint == True:
            # Determine the latent parameters
            Rp, Rwp, i_ter, flag = Inst_WPEM.cal_output_result()
        elif  LogPrint == False:
            Rp, Rwp, i_ter, flag = Inst_WPEM.cal_output_result_reduce()
        else:
            print('The input value of LogPrint ust be True or False!')

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

        print('WPEM program running time : ', Durtime)
        return Durtime
 
        
