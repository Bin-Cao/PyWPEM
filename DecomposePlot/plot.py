# The drawing module 
# Author: Bin CAO <binjacobcao@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class Decomposedpeaks():
    # To draw the decomposition peak.
    def decomposition_peak(self, lowboundary, upboundary, wavelength, density_list=None, name = None, Macromolecule = False,phase = 1,Pic_Title = False,lifting = None):
        """
        :param lowboundary : float, the smallest diffraction angle studied

        :param upboundary : float, the largest diffraction angle studied 

        :param wavelength : list, the wavelength of the X rays

        :param density_list : list default is None, the densities of crytal, can be calculated by fun. WPEM.CIFpreprocess()
            e.g., 
            _,_,d1 = WPEM.CIFpreprocess()
            _,_,d2 = WPEM.CIFpreprocess()
            density_list = [d1,d2]

        :param name : list,  the name of each crystal through this parameter

        :param Macromolecule: whether it contains amorphous, used in amorphous fitting

        :param phase: the number of compounds contained in diffraction signals

        :param Pic_Title: Whether to display the title of the pictures

        :param lifting : whether to lift the base of each components
        
        """
        if density_list is not None:
            self.density_list= density_list # the densities of crytal, can be calculated by fun. WPEM.CIFpreprocess()
        else:
            self.density_list = np.ones(phase)

        if lifting == None:
            lifting = []
            for j in range(phase+1):
                lifting.append(0)
        elif type(lifting) == list:
            if len(lifting) != phase+1:
                print('User must assigned a lifting constant for each phase')
            else: pass
        else:
            print('type error: %s' % type(lifting), 'must be a list')
                      
        if name == None:
            name = []
            for j in range(phase):
                name.append('phase_%d' % (j+1))
        elif type(name) == list:
            print('Name assigned successfully')
        else:
            print('Type Error: name must be a list')

        # compare with no_bac intensity
        origianl_data = pd.read_csv(r'intensity.csv', header=None, names=['two_theta', 'intensity'])
        index1 = np.where((origianl_data.two_theta < lowboundary) | (origianl_data.two_theta > upboundary))
        origianl_data = origianl_data.drop(index1[0])
        o_x = np.array(origianl_data.two_theta)
        o_y = np.array(origianl_data.intensity)

        # Define the font of images
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12 

        if Macromolecule == False:
            # if Macromolecule == False, real and fitting profile are total intensity contains bac!
            if  phase == 1:

                fitting_data = pd.read_csv(r'./DecomposedComponents/fitting_profile.csv', header=None,  names=['two_theta', 'intensity'])
                index2 = np.where((fitting_data.two_theta < lowboundary) | (fitting_data.two_theta > upboundary))
                fitting_data = fitting_data.drop(index2[0])

                dec_peaks_data = pd.read_csv(r'./DecomposedComponents/sub_peaks.csv', header=0)
                index3 = np.where((dec_peaks_data.mu_i < lowboundary) | (dec_peaks_data.mu_i > upboundary))
                dec_peaks_data = dec_peaks_data.drop(index3[0])

                f_x = np.array(fitting_data.two_theta)
                f_y = np.array(fitting_data.intensity)
                cal_mu = np.array(dec_peaks_data.mu_i)
                cal_w = np.array(dec_peaks_data.wi)
                cal_A = np.array(dec_peaks_data.Ai)
                cal_gamma = np.array(dec_peaks_data.L_gamma_i)
                cal_sigma = np.array(dec_peaks_data.G_sigma2_i)

                peak_intens = []
                k = len(cal_mu)
                if k <= 1000:
                    for i in range(k):
                        w_l = cal_w[i] * cal_A[i]
                        w_n = cal_w[i] * (1 - cal_A[i])
                        peak_intens.append(np.array(self.draw_peak_density(o_x, w_l, w_n, cal_mu[i], cal_gamma[i], cal_sigma[i])))
                else:
                    print ('the input peaks are too many!')

                plt.xlabel('2\u03b8\u00B0')
                plt.ylabel('I (a.u.)')
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposition peak', size=20)
                plt.plot(o_x, o_y, label="experiment")
                plt.plot(f_x, f_y, label="WPEM fitting profile")
                for i in range(k):
                    plt.plot(o_x, peak_intens[i])
                plt.legend()
                plt.savefig('./DecomposedComponents/Decomposed_peaks.png', dpi=800)
                plt.savefig('./DecomposedComponents/Decomposed_peaks.svg', dpi=800)
                plt.show()

            elif type(phase) == int:
                DecomposepeaksIntensity = []
                for i in range(phase):
                    dec_peaks_data = pd.read_csv(r'./DecomposedComponents/System{Task}.csv'.format(Task=i), header=0)
                    index4 = np.where((dec_peaks_data.mu_i < lowboundary) | (dec_peaks_data.mu_i > upboundary))
                    dec_peaks_data = dec_peaks_data.drop(index4[0])

                    o_x = np.array(origianl_data.two_theta)
                    cal_mu = np.array(dec_peaks_data.mu_i)
                    cal_w = np.array(dec_peaks_data.wi)
                    cal_A = np.array(dec_peaks_data.Ai)
                    cal_gamma = np.array(dec_peaks_data.L_gamma_i)
                    cal_sigma = np.array(dec_peaks_data.G_sigma2_i)
                   
                    peak_intens = []

                    k = len(cal_mu)
                    if k <= 1000:

                        for p in range(k):
                            w_l = cal_w[p] * cal_A[p]
                            w_n = cal_w[p] * (1 - cal_A[p])
                            peak_intens.append(np.array(self.draw_peak_density(o_x, w_l, w_n, cal_mu[p], cal_gamma[p], cal_sigma[p])))
                        total_intens = np.zeros(len(o_x))
                        for j in range(k):
                            total_intens += np.array(peak_intens[j])
                        DecomposepeaksIntensity.append(total_intens)
                    else:
                        print('the input peaks are too many!')

                    maxmum_peak_index = np.argmax(total_intens)     # total_intens will updata at each iteration
                    MaxP_diffraction_angle = o_x[maxmum_peak_index]
                    MaxP_diffraction_intensity = total_intens[maxmum_peak_index]

                    value = np.sin(MaxP_diffraction_angle / 2 * np.pi / 180) / wavelength[0]
                   
                    MaxP_diffraction_angle = round(MaxP_diffraction_angle,3)
                    MaxP_diffraction_intensity = round(MaxP_diffraction_intensity,3)
                    value = round(value,3)

                    plt.xlabel('2\u03b8\u00B0')
                    plt.ylabel('I (a.u.)')
                    if Pic_Title == False:
                        pass
                    else:
                        plt.title('first peak diffraction angle = {angle}, diffraction intensity = {inten} \n System{Task} : [sin(theta)/wavelength] = {value}'.format(angle = MaxP_diffraction_angle, inten = MaxP_diffraction_intensity,Task = i, value = value) , size=12)
                  
                    plt.plot(o_x, total_intens, label="{}".format(name[i]))
                    for __peak in range(k):
                        plt.plot(o_x, peak_intens[__peak])
                    plt.legend()
                    plt.savefig('./DecomposedComponents/{Task}.png'.format(Task = name[i]), dpi=800)
                    plt.show()

                area = []   # defined for computing the mass fraction of components by the intensity area
                plt.xlabel('2\u03b8\u00B0')
                plt.ylabel('I (a.u.)')
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposited peaks - all components', size=15)
                plt.plot(o_x, o_y+lifting[-1], label="experiment")
                
                for l in range(phase):
                        plt.plot(o_x, DecomposepeaksIntensity[l]+ lifting[l],label="{System}".format(System = name[l]))
                        # calculate the integral area of each component
                        area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))      
                plt.legend()
                plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.svg', dpi=800)
                plt.show()

                # save the 2theta-intensity file of decomposed components
                for l in range(phase):
                    with open(os.path.join('DecomposedComponents','{System}.csv'.format(System = name[l])), 'w') as wfid:
                        for j in range(len(o_x)):
                            print(o_x[j], end=', ', file=wfid)
                            print(float(DecomposepeaksIntensity[l][j]), file=wfid)

                Sum = 0.0
                for system in range(len(area)):
                    Sum += float(area[system]* self.density_list[system])

                Fraction = []
                for system in range(len(area)):
                    Fraction.append(area[system] * self.density_list[system]/ Sum * 100)

                print('Mass fraction estimate in % (calculated by integral area):', str(Fraction), '\n Saved at the DecomposedComponents document')
                with open(os.path.join('WPEMFittingResults', 'MassFraction_estimate_integral_area.txt'), 'w') as wfid:
                    print('The estimated Mass fraction in % :', file=wfid)
                    print(str(Fraction), file=wfid)
            else:
                print('Input a error type of ', phase)
    
    
        elif Macromolecule == True:
            # if Macromolecule == True, real and fitting profile are total intensity contains bac! 

            # fitted crystalline inten
            fitting_data = pd.read_csv(r'./DecomposedComponents/fitting_profile.csv', header=None,  names=['two_theta', 'intensity'])
            index2 = np.where((fitting_data.two_theta < lowboundary) | (fitting_data.two_theta > upboundary))
            fitting_data = fitting_data.drop(index2[0])
         
            if  phase == 1:  
                # fitted crystalline peaks
                dec_peaks_data = pd.read_csv(r'./DecomposedComponents/sub_peaks.csv', header=0)
                index3 = np.where((dec_peaks_data.mu_i < lowboundary) | (dec_peaks_data.mu_i > upboundary))
                dec_peaks_data = dec_peaks_data.drop(index3[0])

                # fitted amorphous crystalline inten
                Amorphous_fitting_data = pd.read_csv(r'./DecomposedComponents/Amorphous.csv', header=None,  names=['two_theta', 'intensity'])
                index4 = np.where((Amorphous_fitting_data.two_theta < lowboundary) | (Amorphous_fitting_data.two_theta > upboundary))
                Amorphous_fitting_data = Amorphous_fitting_data.drop(index4[0])

                # fitted amorphous crystalline peaks
                Amorphous_dec_peaks_data = pd.read_csv(r'./DecomposedComponents/M_Amorphous_peaks.csv', header=0)
                index5 = np.where((Amorphous_dec_peaks_data.mu_i < lowboundary) | (Amorphous_dec_peaks_data.mu_i > upboundary))
                Amorphous_dec_peaks_data = Amorphous_dec_peaks_data.drop(index5[0])

                f_x = np.array(fitting_data.two_theta)
                f_y = np.array(fitting_data.intensity)
                cal_mu = np.array(dec_peaks_data.mu_i)
                cal_w = np.array(dec_peaks_data.wi)
                cal_A = np.array(dec_peaks_data.Ai)
                cal_gamma = np.array(dec_peaks_data.L_gamma_i)
                cal_sigma = np.array(dec_peaks_data.G_sigma2_i)
                
                Amorphous_f_x = np.array(Amorphous_fitting_data.two_theta)
                Amorphous_f_y = np.array(Amorphous_fitting_data.intensity)
                Amorphous_cal_mu = np.array(Amorphous_dec_peaks_data.mu_i)
                Amorphous_cal_w = np.array(Amorphous_dec_peaks_data.wi)
                Amorphous_cal_sigma = np.array(Amorphous_dec_peaks_data.sigma2_i)

                # peaks of crystalline part
                peak_intens = []
                k = len(cal_mu)
                if k <= 1000:
                    for i in range(k):
                        w_l = cal_w[i] * cal_A[i]
                        w_n = cal_w[i] * (1 - cal_A[i])
                        peak_intens.append(np.array(self.draw_peak_density(o_x, w_l, w_n,cal_mu[i], cal_gamma[i], cal_sigma[i])))
                    # peak_intens = [[array_peak1],[array_peak2]...]
                else:
                    print ('Crystalline: the input peaks are too many!')
                
                # peaks of amorphous part
                Amorphous_peak_intens = []
                _k = len(Amorphous_cal_mu)
                if _k <= 1000:
                    for i in range(_k):
                        Amorphous_peak_intens.append(Amorphous_cal_w[i] * np.array(self.normal_density(o_x, Amorphous_cal_mu[i], Amorphous_cal_sigma[i])))
                else:
                    print ('Amorphous: the input peaks are too many!')

                plt.xlabel('2\u03b8\u00B0')
                plt.ylabel('I (a.u.)')
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposition peak', size=20)
                plt.plot(o_x, o_y, label="experiment")
                plt.plot(f_x, f_y, label="WPEM fitting profile")
                for i in range(k):
                    plt.plot(o_x, peak_intens[i],linewidth=2)

                plt.plot(Amorphous_f_x,  Amorphous_f_y, linestyle='--',linewidth=2.5, c='k',label="Amorphous profile")
                for i in range(_k):
                    plt.plot(o_x, Amorphous_peak_intens[i],linestyle='--', c='b',linewidth=1.5,)
                plt.legend()
                plt.savefig('./DecomposedComponents/Decomposed_peaks.png', dpi=800)
                plt.savefig('./DecomposedComponents/Decomposed_peaks.svg', dpi=800)
                plt.show()
                
                # cal relative bulk crystallinity
                Amorphous_area = self.theta_intensity_area(Amorphous_f_x, Amorphous_f_y)
                _total_int = np.zeros(len(Amorphous_f_x))
                for peak in range(len(peak_intens)):
                    _total_int += peak_intens[peak]

                crystalline_area = self.theta_intensity_area(o_x, _total_int)
                RBC = crystalline_area / (crystalline_area + Amorphous_area) * 100
                print('Relative bulk crystallinity % (calculated by integral area):', str(RBC), '\n Saved at the WPEMFittingResults')
                with open(os.path.join('WPEMFittingResults', 'M_Macromolecule Relative bulk crystallinity.txt'), 'w') as wfid:
                    print('Relative bulk crystallinity % :', file=wfid)
                    print(str(RBC), file=wfid)
                

            elif type(phase) == int:
                 # fitted amorphous crystalline inten
                Amorphous_fitting_data = pd.read_csv(r'./DecomposedComponents/Amorphous.csv', header=None,  names=['two_theta', 'intensity'])
                index4 = np.where((Amorphous_fitting_data.two_theta < lowboundary) | (Amorphous_fitting_data.two_theta > upboundary))
                Amorphous_fitting_data = Amorphous_fitting_data.drop(index4[0])

                # fitted amorphous crystalline peaks
                Amorphous_dec_peaks_data = pd.read_csv(r'./DecomposedComponents/M_Amorphous_peaks.csv', header=0)
                index5 = np.where((Amorphous_dec_peaks_data.mu_i < lowboundary) | (Amorphous_dec_peaks_data.mu_i > upboundary))
                Amorphous_dec_peaks_data = Amorphous_dec_peaks_data.drop(index5[0])

                Amorphous_f_x = np.array(Amorphous_fitting_data.two_theta)
                Amorphous_f_y = np.array(Amorphous_fitting_data.intensity)
                Amorphous_cal_mu = np.array(Amorphous_dec_peaks_data.mu_i)
                Amorphous_cal_w = np.array(Amorphous_dec_peaks_data.wi)
                Amorphous_cal_sigma = np.array(Amorphous_dec_peaks_data.sigma2_i)

                 # peaks of amorphous part
                Amorphous_peak_intens = []
                _k = len(Amorphous_cal_mu)
                if _k <= 1000:
                    for i in range(_k):
                        Amorphous_peak_intens.append(Amorphous_cal_w[i] * np.array(self.normal_density(o_x, Amorphous_cal_mu[i], Amorphous_cal_sigma[i])))
                else:
                    print ('Amorphous: the input peaks are too many!')

                DecomposepeaksIntensity = []
                for i in range(phase):
                    dec_peaks_data = pd.read_csv(r'./DecomposedComponents/System{Task}.csv'.format(Task=i), header=0)
                    index = np.where((dec_peaks_data.mu_i < lowboundary) | (dec_peaks_data.mu_i > upboundary))
                    dec_peaks_data = dec_peaks_data.drop(index[0])

                    o_x = np.array(origianl_data.two_theta)
                    cal_mu = np.array(dec_peaks_data.mu_i)
                    cal_w = np.array(dec_peaks_data.wi)
                    cal_A = np.array(dec_peaks_data.Ai)
                    cal_gamma = np.array(dec_peaks_data.L_gamma_i)
                    cal_sigma = np.array(dec_peaks_data.G_sigma2_i)
                   
                    peak_intens = []
                    k = len(cal_mu)
                    if k <= 1000:

                        for p in range(k):
                            w_l = cal_w[p] * cal_A[p]
                            w_n = cal_w[p] * (1 - cal_A[p])
                            peak_intens.append(np.array(self.draw_peak_density(o_x, w_l, w_n,
                                                                            cal_mu[p], cal_gamma[p], cal_sigma[p])))
                        total_intens = np.zeros(len(o_x))
                        for j in range(k):
                            total_intens += np.array(peak_intens[j])
                        DecomposepeaksIntensity.append(total_intens)
                    else:
                        print('the input peaks are too many!')

                    maxmum_peak_index = np.argmax(total_intens)     # total_intens will updata at each iteration
                    MaxP_diffraction_angle = o_x[maxmum_peak_index]
                    MaxP_diffraction_intensity = total_intens[maxmum_peak_index]

                    value = np.sin(MaxP_diffraction_angle / 2 * np.pi / 180) / wavelength[0]
                    MaxP_diffraction_angle = round(MaxP_diffraction_angle,3)
                    MaxP_diffraction_intensity = round(MaxP_diffraction_intensity,3)
                    value = round(value,3)

                   
                    plt.xlabel('2\u03b8\u00B0')
                    plt.ylabel('I (a.u.)')
                    if Pic_Title == False:
                        pass
                    else:
                        plt.title(' First peak diffraction angle = {angle}, diffraction intensity = {inten} \n {Task} : [sin(theta)/wavelength] = {value}'.format(angle = MaxP_diffraction_angle, inten = MaxP_diffraction_intensity,Task = name[i], value = value) , size=12)
                    plt.plot(o_x, total_intens, label="{Task}".format(Task=name[i]))
                
                    for __peak in range(k):
                        plt.plot(o_x, peak_intens[__peak])
                    plt.legend()
                    plt.savefig('./DecomposedComponents/{Task}.png'.format(Task=name[i]), dpi=800)
                    plt.show()

                
                area = []   # defined for computing the mass fraction of components by the intensity area
                plt.xlabel('2\u03b8\u00B0')
                plt.ylabel('I (a.u.)')
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposited peaks - all components', size=15)
                plt.plot(o_x, o_y+lifting[-1], label="experiment")
                for l in range(phase):
                    plt.plot(o_x, DecomposepeaksIntensity[l]+ lifting[l],label=" {System}".format(System = name[l]))
                    # calculate the integral area of each component
                    area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))

                # save the 2theta-intensity file of decomposed components
                for l in range(phase):
                    with open(os.path.join('DecomposedComponents','{System}.csv'.format(System = name[l])), 'w') as wfid:
                        for j in range(len(o_x)):
                            print(o_x[j], end=', ', file=wfid)
                            print(float(DecomposepeaksIntensity[l][j]), file=wfid)
                        
                            
                plt.plot( Amorphous_f_x,  Amorphous_f_y, linestyle='--',linewidth=2.5, c='k',label="Amorphous profile")
                for i in range(_k):
                    plt.plot(Amorphous_f_x, Amorphous_peak_intens[i],linestyle='--', c='b',linewidth=1.5,)
                plt.legend()    
                plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.svg', dpi=800)
                plt.show()

                Sum = 0.0
                for system in range(len(area)):
                    Sum += float(area[system] * self.density_list[system])

                Fraction = []
                for system in range(len(area)):
                    Fraction.append(area[system] * self.density_list[system] / Sum * 100)

                print('Mass fraction estimate in % (calculated by integral area):', str(Fraction), '\n Saved at the WPEMFittingResults')
                with open(os.path.join('WPEMFittingResults', 'MassFraction_estimate_integral_area.txt'), 'w') as wfid:
                    print('The estimated Mass fraction in % :', file=wfid)
                    print(str(Fraction), file=wfid)
                
                # cal relative bulk crystallinity
                Amorphous_area = self.theta_intensity_area(Amorphous_f_x, Amorphous_f_y)
                Crystalline_area = self.theta_intensity_area(np.array(fitting_data.two_theta), np.array(fitting_data.intensity))
                RBC = Crystalline_area / (Crystalline_area + Amorphous_area) * 100
                print('Relative bulk crystallinity % (calculated by integral area):', str(RBC), '\n Saved at the WPEMFittingResults')
                with open(os.path.join('WPEMFittingResults', 'M_Macromolecule Relative bulk crystallinity.txt'), 'w') as wfid:
                    print('Relative bulk crystallinity % :', file=wfid)
                    print(str(RBC), file=wfid)

            else:
                print('Input a error type of ', phase)
    
    # Normal distribution
    def normal_density(self, x, mu, sigma2):
        density = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
        return density

    # Lorenz distribution
    def lorenz_density(self, x, mu, gamma):
        density = (1 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
        return density

    # To draw the decomposition peak.
    def draw_peak_density(self, x, w_l, w_g, mu, gamma, sigma2):
        peak_density = w_l * self.lorenz_density(x, mu, gamma) + w_g * self.normal_density(x, mu, sigma2)
        return peak_density

    def theta_intensity_area(self, theta_data, intensity):
        n = len(theta_data) - 1
        __area = 0
        for i in range(n):
            __h = (intensity[i] + intensity[i + 1]) / 2
            __l = theta_data[i + 1] - theta_data[i]
            __area += __h * __l
        return __area
        




