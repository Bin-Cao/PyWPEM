# The background intensity distribution module of WPEM
# Author: Bin CAO <binjacobcao@gmail.com>

import re
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as Gpr
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class TwiceFilter:
    """
    background intensity module
    Smoothing technique Savitzky-Golay filter is applied to improve the signal/noise ratio 
    after inverse Fourier transform and give a set of slowly varying background points
    """
    def __init__(self, Model='XRD'):
        """
        Display the background curve of XRD diffraction spectrum (Model='XRD')
        and Raman spectrum (Model='Raman') according to the type
        """

        # Model = 'XRD' or 'Raman'
        self.Model = Model
        os.makedirs('ConvertedDocuments', exist_ok=True)

    def convert_file(self, file_name):
        """
        Convert "Free Format(2Theta, step, 2ThetaF)" to "X,Y Data"

        This function is defined to convert the XRD diffractometer output file format
        into a WPEM acceptable input file

        file_name : The file name of original XRD data (Free Format(2Theta, step, 2ThetaF))
        """
        digital_pattern = re.compile(r'[0-9.]+')
        intensity = []
        with open(file_name, 'r') as xrdfid:
            first_line = xrdfid.readline()
            tmp_list = digital_pattern.findall(first_line)
            start = float(tmp_list[0])
            step = float(tmp_list[1])
            end = float(tmp_list[2])
            while tmp_list:
                tmp_list = digital_pattern.findall(xrdfid.readline())
                for i in tmp_list:
                    intensity.append(float(i))

        two_theta = np.arange(start, end + step, step)
        with open(os.path.join('ConvertedDocuments', 'intensity.csv'), 'w') as wfid:
            for i in range(len(intensity)):
                print(two_theta[i], end=',', file=wfid)
                print(intensity[i], file=wfid)

    def FFTandSGFilter(self, intensity_csv, LFctg = 0.5, lowAngleRange=None, bac_num=None, bac_split=5, window_length=17, 
                       polyorder=3,  poly_n=6, mode='nearest', bac_var_type='constant'):
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

        :return:
        """
        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 15

        import heapq
        import numpy.fft as nf
        from scipy.signal import savgol_filter

        angle = intensity_csv.iloc[:, 0]
        signal = intensity_csv.iloc[:, 1]
        # The complex-valued FFT computes the Fourier transform of signal, 
        # which represents the decomposition of the signal into its component frequencies.
        complex_array = nf.fft(signal)
        pows = np.abs(complex_array)

        DC = []
        # tuple_type = np.where(complex_array.imag == 0)[0]
        # for i in tuple_type:
        # DC.append(tuple_type[i])
        hDCnum = int(LFctg * len(angle))
        for hDC in range(hDCnum):
            DC.append(heapq.nsmallest(hDCnum, enumerate(pows), key=lambda x: x[1])[hDC][0])

        filter_complex_array = copy.deepcopy(complex_array)
        filter_complex_array[DC] = 0
        FFT_filter_intensity = nf.ifft(filter_complex_array).real

        # Savitzky-Golay Filter
        intensity_up = copy.deepcopy(FFT_filter_intensity)
        if bac_num == None:
            bac_num = int(1 / 4 * len(angle))
        else:
            bac_num = bac_num
        SG_filter_intensity = savgol_filter(intensity_up, window_length, polyorder, mode=mode)

        PairIndex = []
        Twotheta = []
        BacSelected = []
        BacOrigPoint = []

        if bac_split == None:
            print('You must input the parameter bac_split')

        elif type(bac_split) == int:
            if lowAngleRange == None:
                choiselg_num = int(bac_num / bac_split)
                Split_filter_intensity = self.chunks(SG_filter_intensity, bac_split)
                for part in range(bac_split):
                    for num in range(choiselg_num):
                        PairIndex.append((heapq.nsmallest(choiselg_num, enumerate(Split_filter_intensity[part]),
                                                          key=lambda x: x[1])[num][0]) + part * len(Split_filter_intensity[0]))

            elif type(lowAngleRange) == int or type(lowAngleRange) == float:
                lowAngleRange_index = np.where(angle >= lowAngleRange)[0][0]
                lowangle_num = np.max((int(bac_num * (lowAngleRange_index / len(angle))), int(0.2*(lowAngleRange_index+1))))
                choiselg_num = np.max((int(lowangle_num / 5), 1))
                other_num = bac_num - lowangle_num
                Split_filter_intensity = self.chunks(SG_filter_intensity[0:lowAngleRange_index], 5)
                for part in range(len(Split_filter_intensity)):
                    for num in range(choiselg_num):
                        PairIndex.append((heapq.nsmallest(choiselg_num, enumerate(Split_filter_intensity[part]),
                                                          key=lambda x: x[1])[num][0]) + part * len(Split_filter_intensity[0]))

                __choiselg_num = int(other_num / bac_split)
                __Split_filter_intensity = self.chunks(SG_filter_intensity[lowAngleRange_index:-1], bac_split)

                for __part in range(bac_split):
                    for __num in range(__choiselg_num):
                        PairIndex.append((heapq.nsmallest(__choiselg_num, enumerate(__Split_filter_intensity[__part]),
                                                          key=lambda x: x[1])[__num][0]) + __part * len(__Split_filter_intensity[0]) + lowAngleRange_index)
            else:
                print('Type Error \'lowAngleRange\'')

        else:
            print('Type Error \'segmentation strategy\'')

        PairIndex.sort()
        for pair in range(len(PairIndex)):
            Twotheta.append(angle[PairIndex[pair]])
            # intensity of experimental measurement
            BacOrigPoint.append(signal[PairIndex[pair]])
            # intensity after fft and SG filter
            BacSelected.append(SG_filter_intensity[PairIndex[pair]])

        # calculate the variance of background function
        if bac_var_type == 'constant' :
            Sum = 0
            for bac in range(len(BacSelected)):
                Sum += (BacOrigPoint[bac] - BacSelected[bac]) ** 2
            Var_2 = Sum / len(BacOrigPoint)    # 1/n * sum( (x-E(x))^2 )
            standard_deviation = float(np.sqrt(Var_2))

            bac = np.polyfit(Twotheta, BacSelected, poly_n)
            poly_fit = np.poly1d(bac)
            Intensity_mean = poly_fit(angle)
            inten_no_bac = list(map(lambda x: abs(x[0] - x[1]), zip(signal, Intensity_mean)))
            # Save non-background intensity
            with open(os.path.join('ConvertedDocuments', "no_bac_intensity.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(inten_no_bac[j]), file=wfid)
            # Save background intensity
            with open(os.path.join('ConvertedDocuments', "bac.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(Intensity_mean[j]), file=wfid)


            plt.plot(angle, Intensity_mean, color='k', label='background means')
            lowbound = Intensity_mean - standard_deviation
            upbound =  Intensity_mean + standard_deviation

            plt.plot(angle, lowbound, 'g--', lw=1)
            plt.plot(angle, upbound, 'g--', lw=1)
            plt.fill_between(angle, lowbound, upbound,
                             color='lightblue', label='one sigma \n confidence interval')
            plt.scatter(Twotheta, BacSelected, color='r', s=5,zorder=2, label='background points')
            if self.Model == 'Raman':
                plt.xlabel('Raman shift (cm\u207B\u00B9)')
            else:
                plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            plt.legend()
            plt.savefig('./ConvertedDocuments/background function distribution_constant.png', dpi=800)
            plt.show()
            plt.clf()

        # Be careful! if choose 'polynomial', WPEM models the variance as a three-times polynomial function of diffraction angles
        # the corresponding background function is fitted by a polynomial function latter
        elif bac_var_type == 'polynomial' :
            sta_dev_array = list(map(lambda x: abs(x[0]-x[1]), zip(BacOrigPoint,BacSelected)))
            bac_var = np.polyfit(Twotheta, sta_dev_array, 3)     # here default ploy = 3
            poly_var_fit = np.poly1d(bac_var)
            standard_deviation = poly_var_fit(angle)             # here standard_deviation is a N (2theta) array


            bac = np.polyfit(Twotheta, BacSelected, poly_n)
            poly_fit = np.poly1d(bac)
            Intensity_mean = poly_fit(angle)
            inten_no_bac = list(map(lambda x: abs(x[0] - x[1]), zip(signal, Intensity_mean)))
            # Save non-background intensity
            with open(os.path.join('ConvertedDocuments', "no_bac_intensity.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(inten_no_bac[j]), file=wfid)
            # Save background intensity
            with open(os.path.join('ConvertedDocuments', "bac.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(Intensity_mean[j]), file=wfid)


            plt.plot(angle, Intensity_mean, color='k', label='background means')
            lowbound = list(map(lambda x: abs(x[0] - x[1]), zip(Intensity_mean, standard_deviation)))
            upbound = list(map(lambda x: abs(x[0] + x[1]), zip(Intensity_mean, standard_deviation)))

            plt.plot(angle, lowbound, 'g--', lw=1)
            plt.plot(angle, upbound, 'g--', lw=1)
            plt.fill_between(angle, lowbound, upbound,
                             color='lightblue', label='one sigma \n confidence interval')
            plt.scatter(Twotheta, BacSelected, color='r', s=5, zorder=2,label='background points')
            if self.Model == 'Raman':
                plt.xlabel('Raman shift (cm\u207B\u00B9)')
            else:
                plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            plt.legend()
            plt.savefig('./ConvertedDocuments/background function distribution_polynomial.png', dpi=800)
            plt.show()
            plt.clf()


        #  if choose 'multivariate gaussian', WPEM models the background function as a multivariate gaussian of diffraction angles with variance
        elif bac_var_type == 'multivariate gaussian' :

            kernel = 1 * RBF() + WhiteKernel()
            model = Gpr(kernel = kernel, n_restarts_optimizer = 10, alpha = 0, normalize_y = True, random_state = 0).fit(np.array(Twotheta).reshape(-1,1), BacOrigPoint)
            background_meanfunction, standard_deviation = model.predict(np.array(angle).reshape(-1,1), return_std=True)



            Intensity_mean, Intensity_dev = model.predict(angle[:, np.newaxis], return_std=True)

            inten_no_bac = list(map(lambda x: abs(x[0] - x[1]), zip(signal, background_meanfunction)))

            # Save non-background intensity
            with open(os.path.join('ConvertedDocuments', "no_bac_intensity.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(inten_no_bac[j]), file=wfid)
            # Save background intensity
            with open(os.path.join('ConvertedDocuments', "bac.csv"), 'w') as wfid:
                for j in range(len(angle)):
                    print(angle[j], end=', ', file=wfid)
                    print(float(background_meanfunction[j]), file=wfid)


            plt.plot(angle, background_meanfunction,  color='k', label='background means')
            plt.plot(angle, Intensity_mean - Intensity_dev,  'g--',lw=1)
            plt.plot(angle, Intensity_mean + Intensity_dev, 'g--',lw=1)
            plt.fill_between(angle, Intensity_mean - Intensity_dev, Intensity_mean + Intensity_dev, color='lightblue', label='one sigma \n confidence interval')
            plt.scatter(Twotheta, BacSelected, color='r', s=5, zorder=2,label='background points')
            if self.Model == 'Raman':
                plt.xlabel('Raman shift/(cm-1)')
            else:
                plt.xlabel('2\u03b8\u00B0')
            plt.ylabel('I (a.u.)')
            plt.legend()
            plt.savefig('./ConvertedDocuments/background function distribution _ multivariate gaussian.png', dpi=800)
            plt.show()
            plt.clf()



        # Save intensity document after FFT with DC
        with open(os.path.join('ConvertedDocuments', "intensity_fft.csv"), 'w') as wfid:
            for j in range(len(angle)):
                print(angle[j], end=', ', file=wfid)
                print(float(FFT_filter_intensity[j]), file=wfid)

        # Save background points intensity  after FFT and SG filter
        with open(os.path.join('ConvertedDocuments', "bac_points.csv"), 'w') as wfid:
            for j in range(len(Twotheta)):
                print(Twotheta[j], end=', ', file=wfid)
                print(float(BacSelected[j]), file=wfid)

        plt.plot(angle, signal, color='cyan', label='original intensity')
        plt.plot(angle, SG_filter_intensity, color='k', label='FFT-SG intensity')
        plt.scatter(Twotheta, BacSelected, s=5, c='r', zorder=2,label='selected background points')
        if self.Model == 'Raman':
            plt.xlabel('Raman shift (cm\u207B\u00B9)')
        else:
            plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig('./ConvertedDocuments/background points.png',dpi=800)
        plt.show()
        plt.clf()

        plt.plot(angle, signal, color='cyan', label='original intensity')
        plt.plot(angle, Intensity_mean, color='k', label='background function')
        plt.scatter(Twotheta, BacSelected, c='r',zorder=2, label='background points')
        if self.Model == 'Raman':
            plt.xlabel('Raman shift (cm\u207B\u00B9)')
        else:
            plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig('./ConvertedDocuments/backgroundfittingcurve.png',dpi=800)
        plt.show()
        plt.clf()
        
        plt.plot(angle, signal, color='cyan', label='original intensity')
        plt.plot(angle, inten_no_bac, color='k', label='de_background intensity')
        if self.Model == 'Raman':
            plt.xlabel('Raman shift (cm\u207B\u00B9)')
        else:
            plt.xlabel('2\u03b8\u00B0')
        plt.ylabel('I (a.u.)')
        plt.legend()
        plt.savefig('./ConvertedDocuments/de_backgroundfittingcurve.png',dpi=800)
        plt.show()
        plt.clf()

        return standard_deviation

    def chunks(self, arr, m):
        """
        Auxiliary function for splitting the array into m segments 
        """
        import math
        arr = arr
        m = m
        n = int(math.ceil(len(arr) / float(m)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]

