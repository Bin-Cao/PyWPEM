# The drawing module 
# Author: Bin CAO <binjacobcao@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class Decomposedpeaks():
    # To draw the decomposition peak.
    def decomposition_peak(self, lowboundary, upboundary, wavelength,name = None, Macromolecule = False,phase = 1,Pic_Title = False):
        self.lowboundary = lowboundary
        self.upboundary = upboundary
        self.wavelength = wavelength
        self.Macromolecule = Macromolecule
        self.phace = phase

        assign = 0
        if name == None:
            pass
        elif type(name) == list:
            print('Name assigned successfully')
            assign = True
        else:
            print('Type Error: name must be a list')

        # compare with no_bac intensity
        origianl_data = pd.read_csv(r'intensity.csv', header=None, names=['two_theta', 'intensity'])
        index1 = np.where((origianl_data.two_theta < self.lowboundary) | (origianl_data.two_theta > self.upboundary))
        origianl_data = origianl_data.drop(index1[0])
        o_x = np.array(origianl_data.two_theta)
        o_y = np.array(origianl_data.intensity)

        if self.Macromolecule == False:

            # if Macromolecule == False, real and fitting profile are total intensity contains bac!
            if  self.phace == 1:

                fitting_data = pd.read_csv(r'./DecomposedComponents/fitting_profile.csv', header=None,  names=['two_theta', 'intensity'])
                index2 = np.where((fitting_data.two_theta < self.lowboundary) | (fitting_data.two_theta > self.upboundary))
                fitting_data = fitting_data.drop(index2[0])

                dec_peaks_data = pd.read_csv(r'./DecomposedComponents/sub_peaks.csv', header=0)
                index3 = np.where((dec_peaks_data.mu_i < self.lowboundary) | (dec_peaks_data.mu_i > self.upboundary))
                dec_peaks_data = dec_peaks_data.drop(index3[0])


                f_x = np.array(fitting_data.two_theta)
                f_y = np.array(fitting_data.intensity)
                cal_mu = np.array(dec_peaks_data.mu_i)
                cal_w = np.array(dec_peaks_data.wi)
                cal_A = np.array(dec_peaks_data.Ai)
                cal_gamma = np.array(dec_peaks_data.gamma_i)
                cal_sigma = np.array(dec_peaks_data.sigma_i)

                peak_intens = []
                k = len(cal_mu)
                if k <= 1000:
                    for i in range(k):
                        w_l = cal_w[i] * cal_A[i]
                        w_n = cal_w[i] * (1 - cal_A[i])
                        peak_intens.append(np.array(self.draw_peak_density(o_x, w_l, w_n, cal_mu[i], cal_gamma[i], cal_sigma[i])))
                else:
                    print ('the input peaks are too many!')

                plt.xlabel('2Theta/(°)', size=10)
                plt.ylabel('Intensity/(a.u.)', size=10)
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposition peak', size=20)
                plt.plot(o_x, o_y, label="Real intensity")
                plt.plot(f_x, f_y, label="WPEM fitting profile")
                for i in range(k):
                    plt.plot(o_x, peak_intens[i])
                plt.legend()
                plt.savefig('./DecomposedComponents/Decomposed_peaks.png', dpi=800)
                plt.show()


            elif type(self.phace) == int:
                DecomposepeaksIntensity = []
                for i in range(self.phace):
                    dec_peaks_data = pd.read_csv(r'./DecomposedComponents/System{Task}.csv'.format(Task=i), header=0)
                    index4 = np.where((dec_peaks_data.mu_i < self.lowboundary) | (dec_peaks_data.mu_i > self.upboundary))
                    dec_peaks_data = dec_peaks_data.drop(index4[0])

                    o_x = np.array(origianl_data.two_theta)
                    cal_mu = np.array(dec_peaks_data.mu_i)
                    cal_w = np.array(dec_peaks_data.wi)
                    cal_A = np.array(dec_peaks_data.Ai)
                    cal_gamma = np.array(dec_peaks_data.gamma_i)
                    cal_sigma = np.array(dec_peaks_data.sigma_i)
                   

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

                    value = np.sin(MaxP_diffraction_angle / 2 * np.pi / 180) / self.wavelength[0]
                   
                    MaxP_diffraction_angle = round(MaxP_diffraction_angle,3)
                    MaxP_diffraction_intensity = round(MaxP_diffraction_intensity,3)
                    value = round(value,3)

                    
                    plt.xlabel('2Theta/(°)', size=10)
                    plt.ylabel('Intensity/(a.u.)', size=10)
                    if Pic_Title == False:
                        pass
                    else:
                        plt.title(' First peak diffraction angle = {angle}, diffraction intensity = {inten} \n System{Task} : [sin(theta)/wavelength] = {value}'.format(angle = MaxP_diffraction_angle, inten = MaxP_diffraction_intensity,Task = i, value = value) , size=12)
                    
                    if assign == False:
                        plt.plot(o_x, total_intens, label="{}".format(name[i]))
                        for __peak in range(k):
                            plt.plot(o_x, peak_intens[__peak])
                        plt.legend()
                        plt.savefig('./DecomposedComponents/Decomposed_peaks{Task}.png'.format(Task=i), dpi=800)
                        plt.show()
                    else: 
                        plt.plot(o_x, total_intens, label="{}".format(name[i]))
                        for __peak in range(k):
                            plt.plot(o_x, peak_intens[__peak])
                        plt.legend()
                        plt.savefig('./DecomposedComponents/{Task}.png'.format(Task = name[i]), dpi=800)
                        plt.show()

                area = []   # defined for computing the volume fraction of components by the intensity area
                plt.xlabel('2Theta/(°)', size=10)
                plt.ylabel('Intensity/(a.u.)', size=10)
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposited peaks - all components', size=15)
                plt.plot(o_x, o_y, label="Real intensity")
                if assign == False:
                    for l in range(self.phace):
                        plt.plot(o_x, DecomposepeaksIntensity[l],label="Decomposed profile of system  {System}".format(System = l))
                        
                        # calculate the integral area of each component
                        area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))      
                    plt.legend()
                    plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                    plt.show()
                else:
                    for l in range(self.phace):
                            plt.plot(o_x, DecomposepeaksIntensity[l],label="{System}".format(System = name[l]))
                            
                            # calculate the integral area of each component
                            area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))      
                    plt.legend()
                    plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                    plt.show()

              
                if assign == False:
                    # save the 2theta-intensity file of decomposed components
                    for l in range(self.phace):
                        with open(os.path.join('DecomposedComponents','Profile_Decomposed_system{System}.csv'.format(System = l)), 'w') as wfid:
                            for j in range(len(o_x)):
                                print(o_x[j], end=', ', file=wfid)
                                print(float(DecomposepeaksIntensity[l][j]), file=wfid)
                elif assign == True:
                    # save the 2theta-intensity file of decomposed components
                    for l in range(self.phace):
                        with open(os.path.join('DecomposedComponents','{System}.csv'.format(System = name[l])), 'w') as wfid:
                            for j in range(len(o_x)):
                                print(o_x[j], end=', ', file=wfid)
                                print(float(DecomposepeaksIntensity[l][j]), file=wfid)


                Sum = 0.0
                for system in range(len(area)):
                    Sum += float(area[system])

                Fraction = []
                for system in range(len(area)):
                    Fraction.append(area[system] / Sum * 100)

                print('volume fraction estimate in % (calculated by integral area):', str(Fraction), '\n Saved at the DecomposedComponents document')
                with open(os.path.join('WPEMFittingResults', 'VolumeFraction_estimate_integral_area.txt'), 'w') as wfid:
                    print('The estimated volume fraction in % :', file=wfid)
                    print(str(Fraction), file=wfid)

            else:
                print('Input a error type of ', self.phace)
    
    
        elif self.Macromolecule == True:
            # if Macromolecule == True, real and fitting profile are total intensity contains bac! 

            # fitted crystalline inten
            fitting_data = pd.read_csv(r'./DecomposedComponents/fitting_profile.csv', header=None,  names=['two_theta', 'intensity'])
            index2 = np.where((fitting_data.two_theta < self.lowboundary) | (fitting_data.two_theta > self.upboundary))
            fitting_data = fitting_data.drop(index2[0])
         
            if  self.phace == 1:  

                # fitted crystalline peaks
                dec_peaks_data = pd.read_csv(r'./DecomposedComponents/sub_peaks.csv', header=0)
                index3 = np.where((dec_peaks_data.mu_i < self.lowboundary) | (dec_peaks_data.mu_i > self.upboundary))
                dec_peaks_data = dec_peaks_data.drop(index3[0])

                # fitted amorphous crystalline inten
                Amorphous_fitting_data = pd.read_csv(r'./DecomposedComponents/M_Amorphous_whole_profile.csv', header=None,  names=['two_theta', 'intensity'])
                index4 = np.where((Amorphous_fitting_data.two_theta < self.lowboundary) | (Amorphous_fitting_data.two_theta > self.upboundary))
                Amorphous_fitting_data = Amorphous_fitting_data.drop(index4[0])

                # fitted amorphous crystalline peaks
                Amorphous_dec_peaks_data = pd.read_csv(r'./DecomposedComponents/M_Amorphous_peaks.csv', header=0)
                index5 = np.where((Amorphous_dec_peaks_data.mu_i < self.lowboundary) | (Amorphous_dec_peaks_data.mu_i > self.upboundary))
                Amorphous_dec_peaks_data = Amorphous_dec_peaks_data.drop(index5[0])

                f_x = np.array(fitting_data.two_theta)
                f_y = np.array(fitting_data.intensity)
                cal_mu = np.array(dec_peaks_data.mu_i)
                cal_w = np.array(dec_peaks_data.wi)
                cal_A = np.array(dec_peaks_data.Ai)
                cal_gamma = np.array(dec_peaks_data.gamma_i)
                cal_sigma = np.array(dec_peaks_data.sigma_i)

                
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



                plt.xlabel('2Theta/(°)', size=10)
                plt.ylabel('Intensity/(a.u.)', size=10)
                if Pic_Title == False:
                    pass
                else:
                    plt.title('Decomposition peak', size=20)
                plt.plot(o_x, o_y, label="Real intensity")
                plt.plot(f_x, f_y, label="WPEM fitting profile")
                for i in range(k):
                    plt.plot(o_x, peak_intens[i],linewidth=2)

                plt.plot( Amorphous_f_x,  Amorphous_f_y, linestyle='--',linewidth=2.5, c='k',label=" Amorphous profile")
                for i in range(_k):
                    plt.plot(o_x, Amorphous_peak_intens[i],linestyle='--', c='b',linewidth=1.5,)
                plt.legend()
                plt.savefig('./DecomposedComponents/Decomposed_peaks.png', dpi=800)
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
                


            elif type(self.phace) == int:

                 # fitted amorphous crystalline inten
                Amorphous_fitting_data = pd.read_csv(r'./DecomposedComponents/Amorphous.csv', header=None,  names=['two_theta', 'intensity'])
                index4 = np.where((Amorphous_fitting_data.two_theta < self.lowboundary) | (Amorphous_fitting_data.two_theta > self.upboundary))
                Amorphous_fitting_data = Amorphous_fitting_data.drop(index4[0])

                # fitted amorphous crystalline peaks
                Amorphous_dec_peaks_data = pd.read_csv(r'./DecomposedComponents/M_Amorphous_peaks.csv', header=0)
                index5 = np.where((Amorphous_dec_peaks_data.mu_i < self.lowboundary) | (Amorphous_dec_peaks_data.mu_i > self.upboundary))
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
                for i in range(self.phace):
                    dec_peaks_data = pd.read_csv(r'./DecomposedComponents/System{Task}.csv'.format(Task=i), header=0)
                    index = np.where((dec_peaks_data.mu_i < self.lowboundary) | (dec_peaks_data.mu_i > self.upboundary))
                    dec_peaks_data = dec_peaks_data.drop(index[0])

                    o_x = np.array(origianl_data.two_theta)
                    cal_mu = np.array(dec_peaks_data.mu_i)
                    cal_w = np.array(dec_peaks_data.wi)
                    cal_A = np.array(dec_peaks_data.Ai)
                    cal_gamma = np.array(dec_peaks_data.gamma_i)
                    cal_sigma = np.array(dec_peaks_data.sigma_i)
                   


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


                    value = np.sin(MaxP_diffraction_angle / 2 * np.pi / 180) / self.wavelength[0]
                    MaxP_diffraction_angle = round(MaxP_diffraction_angle,3)
                    MaxP_diffraction_intensity = round(MaxP_diffraction_intensity,3)
                    value = round(value,3)

                    if assign == False:
                        plt.xlabel('2Theta/(°)', size=10)
                        plt.ylabel('Intensity/(a.u.)', size=10)
                        if Pic_Title == False:
                            pass
                        else:
                            plt.title(' First peak diffraction angle = {angle}, diffraction intensity = {inten} \n System{Task} : [sin(theta)/wavelength] = {value}'.format(angle = MaxP_diffraction_angle, inten = MaxP_diffraction_intensity,Task = i, value = value) , size=12)
                        plt.plot(o_x, total_intens, label="Fitted profile of System{Task}".format(Task=i))

                        for __peak in range(k):
                            plt.plot(o_x, peak_intens[__peak])
                        plt.legend()
                        plt.savefig('./DecomposedComponents/Decomposed_peaks{Task}.png'.format(Task=i), dpi=800)
                        plt.show()
                    elif assign == True:
                        plt.xlabel('2Theta/(°)', size=10)
                        plt.ylabel('Intensity/(a.u.)', size=10)
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

                if assign == False:
                    area = []   # defined for computing the volume fraction of components by the intensity area
                    plt.xlabel('2Theta/(°)', size=10)
                    plt.ylabel('Intensity/(a.u.)', size=10)
                    if Pic_Title == False:
                        pass
                    else:
                        plt.title('Decomposited peaks - all components', size=15)
                    plt.plot(o_x, o_y, label="Real intensity")
                    for l in range(self.phace):
                        plt.plot(o_x, DecomposepeaksIntensity[l],label="Decomposed profile of system  {System}".format(System = l))
                        # calculate the integral area of each component
                        area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))

                    # save the 2theta-intensity file of decomposed components
                    for l in range(self.phace):
                        with open(os.path.join('DecomposedComponents','Profile_Decomposed_system{System}.csv'.format(System = l)), 'w') as wfid:
                            for j in range(len(o_x)):
                                print(o_x[j], end=', ', file=wfid)
                                print(float(DecomposepeaksIntensity[l][j]), file=wfid)
                            
                                
                    plt.plot( Amorphous_f_x,  Amorphous_f_y, linestyle='--',linewidth=2.5, c='k',label=" Amorphous profile")
                    for i in range(_k):
                        plt.plot(Amorphous_f_x, Amorphous_peak_intens[i],linestyle='--', c='b',linewidth=1.5,)
                    plt.legend()    
                    plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                    plt.show()
                elif assign == True:
                    area = []   # defined for computing the volume fraction of components by the intensity area
                    plt.xlabel('2Theta/(°)', size=10)
                    plt.ylabel('Intensity/(a.u.)', size=10)
                    if Pic_Title == False:
                        pass
                    else:
                        plt.title('Decomposited peaks - all components', size=15)
                    plt.plot(o_x, o_y, label="Real intensity")
                    for l in range(self.phace):
                        plt.plot(o_x, DecomposepeaksIntensity[l],label=" {System}".format(System = name[l]))
                        # calculate the integral area of each component
                        area.append(self.theta_intensity_area(o_x, DecomposepeaksIntensity[l]))

                    # save the 2theta-intensity file of decomposed components
                    for l in range(self.phace):
                        with open(os.path.join('DecomposedComponents','{System}.csv'.format(System = name[l])), 'w') as wfid:
                            for j in range(len(o_x)):
                                print(o_x[j], end=', ', file=wfid)
                                print(float(DecomposepeaksIntensity[l][j]), file=wfid)
                            
                                
                    plt.plot( Amorphous_f_x,  Amorphous_f_y, linestyle='--',linewidth=2.5, c='k',label=" Amorphous profile")
                    for i in range(_k):
                        plt.plot(Amorphous_f_x, Amorphous_peak_intens[i],linestyle='--', c='b',linewidth=1.5,)
                    plt.legend()    
                    plt.savefig('./DecomposedComponents/Decomposed_peaks_totalview.png', dpi=800)
                    plt.show()

                Sum = 0.0
                for system in range(len(area)):
                    Sum += float(area[system])

                Fraction = []
                for system in range(len(area)):
                    Fraction.append(area[system] / Sum * 100)

                print('volume fraction estimate in % (calculated by integral area):', str(Fraction), '\n Saved at the WPEMFittingResults')
                with open(os.path.join('WPEMFittingResults', 'VolumeFraction_estimate_integral_area.txt'), 'w') as wfid:
                    print('The estimated volume fraction in % :', file=wfid)
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
                print('Input a error type of ', self.phace)
     # Normal distribution
    def normal_density(self, x, mu, sigma2):
        """
        :param x: sample data (2theta)
        :param mu: mean (μi)
        :param sigma2: variance (σi^2)
        :return: Return the probability density of Normal distribution x~N(μi,σi^2)
        """
        self.x = x;
        self.mu = mu;
        self.sigma2 = sigma2
        density = (1 / np.sqrt(2 * np.pi * self.sigma2)) * np.exp(-((self.x - self.mu) ** 2) / (2 * self.sigma2))
        return density

    # Lorenz distribution
    def lorenz_density(self, x, mu=0, gamma=1):
        """
        :param x: sample data (2theta)
        :param mu: mean (μi)
        :param gamma: FWHM of Lorenz distribution
        :return: Return the probability density of Lorenz distribution
        """
        self.x = x;
        self.mu = mu;
        self.gamma = gamma

        density = (1 / np.pi) * (self.gamma / ((self.x - self.mu) ** 2 + self.gamma ** 2))
        return density


    # To draw the decomposition peak.
    def draw_peak_density(self, x, w_l, w_g, mu, gamma, sigma2):
        self.x = x
        self.w_l = w_l
        self.w_g = w_g
        self.mu = mu
        self.gamma = gamma
        self.sigma2 = sigma2
        peak_density = self.w_l * self.lorenz_density(self.x, self.mu, self.gamma) + self.w_g * self.normal_density(self.x, self.mu, self.sigma2)
        return peak_density

    def theta_intensity_area(self, theta_data, intensity):
        n = len(theta_data) - 1
        __area = 0
        for i in range(n):
            __h = (intensity[i] + intensity[i + 1]) / 2
            __l = theta_data[i + 1] - theta_data[i]
            __area += __h * __l
        return __area
        




