# Amorphous Quantitative Description Module
# Author: Bin CAO <binjacobcao@gmail.com>

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class RadialDistribution(object):
    # radial distribution function
    def __init__(self,wavelength, r_max = 5,work_dir =None):
        self.wavelength = wavelength
        # the maximum of radius from a center
        self.r_max = r_max
        self.work_dir = work_dir

        # Define the font of the image
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12 

    def RDF(self,amor_file=None,density_zero=None,Nf2=None,highlight= 4,value=0.6):

        if self.work_dir is None:
            DCfolder = 'DecomposedComponents'
        else:
            DCfolder = os.path.join(work_dir, 'DecomposedComponents')

        os.makedirs(DCfolder, exist_ok=True)

        # density_zero : the average density of the sample in atoms per cc.
        # Nf2 : N is the effective number of atoms in the sample ; Aa is the atom scatter intensity
        # value : assuming that the scattering can be taken as independent at sin(theta/lamda) =0.6,
        if amor_file == None:
            data = pd.read_csv(os.path.join(DCfolder, "Amorphous.csv"), header=None, names=['ang','int'])
            b_data = pd.read_csv(os.path.join(DCfolder,"M_background_amorphous_stripped.csv"), header=None, names=['ang','int'])
        else : 
            data = pd.read_csv(amor_file, header=None, names=['ang','int'])
        
        if self.wavelength*value > 1:
            print('The input value must be smaller than % f' % 1/self.wavelength*value)

        # Define the high angle and intercept the K value
        angle = np.arcsin(self.wavelength*value) * 180 / np.pi
        
        # N is the effective number of atoms in the sample
        # Aa is the atom scatter intensity
        # at large angles of scattering I(k) approaches Nf2
        if Nf2 == None:
            # ref :
            if amor_file == None:
                # read from the amorphous background 
                for idx, ang_value in enumerate(b_data.ang):
                    if ang_value >= angle:
                        mean_int = b_data.int[idx:].mean()
                        print('Nf2 = {Nf2}, is evaluated at angel {angle}'.format(Nf2= mean_int, angle = angle))
                        break

            else:
                for idx, ang_value in enumerate(data.ang):
                    if ang_value >= angle:
                        mean_int = data.int[idx:].mean()
                        print('Nf2 = {Nf2}, is evaluated at angel {angle}'.format(Nf2= mean_int, angle = angle))
                        break
        else:
            Nf2 = Nf2

        plt.xlabel('2\u03b8\u00B0', )
        plt.ylabel('I (a.u.)')
        plt.plot(data.ang, data.int,label="Amorphous diffraction pattern")
        plt.legend()  # fontsize = 15
        plt.savefig(os.path.join(DCfolder,'Amorphous_DP.png'), dpi=800)
        plt.show()
        plt.clf()

        k = 4 * np.pi / self.wavelength * np.sin(data.ang/2)
        # 𝑖(𝑘)𝑘 ； k(I(k)-1)
        # reduced interference function
        int_k = (data.int / Nf2 -1) * k

        r = np.arange(0,self.r_max,0.01)
        RDF_r_list = []
        for i in r:
            RDF_r = cal_RDF(k, int_k, i)
            RDF_r_list.append(RDF_r)
        
        circle_x, circle_y, dis= peak_detect(r,RDF_r_list,highlight)
        plt.xlabel('r/A\u00b0', )
        plt.ylabel('RDF(r)', )
        plt.plot(r, RDF_r_list,color='k',label="4Pir\u00b2Pu(r)-4Pir\u00b2Pu\u2080(r)")
        plt.axhline(0.0, color='b', linestyle='--',)
        plt.scatter(circle_x,circle_y, color='white', marker='o', edgecolors='g', s=200)
        plt.legend()
        plt.savefig(os.path.join(DCfolder,'RDF.png'), dpi=800)
        plt.savefig(os.path.join(DCfolder,'RDF.svg'), dpi=800)
        plt.show()
        plt.clf()
        if density_zero == None: density_zero =40,
        elif type(density_zero) == float or type(density_zero) == int: pass
        plt.xlabel('r/A\u00b0',)
        plt.ylabel('RDF(r)', )
        base = 4 * np.pi * r**2 * density_zero
        circle_x, circle_y, dis= peak_detect_based(r,RDF_r_list,base,highlight)
        plt.plot(r, base, color='b',linestyle='--',label="4Pir\u00b2Ru\u2080(r)")
        plt.plot(r, RDF_r_list+ base,color='k',label="4Pir\u00b2Ru(r)")
        plt.scatter(circle_x,circle_y, color='white', marker='o', edgecolors='g', s=200)
        plt.legend()
        plt.savefig(os.path.join(DCfolder,'RDF_hasbase.png'), dpi=800)
        plt.savefig(os.path.join(DCfolder,'RDF_hasbase.svg'), dpi=800)
        plt.show()
        plt.clf()

        print('interatomic distances is %f A\u00b0' % np.round(dis,3))

        return circle_x

def cal_RDF(k, int_k, r):
    """
    Calculate RDF
    :param k:
    :param int_k:
    :param r:
    :return:
    """
    # 𝑖(𝑘)𝑘𝑠𝑖𝑛𝑘𝑟
    x = np.array(k)
    y = np.array(int_k) * np.sin(x *r )
    n = len(x) - 1
    area = 0
    for i in range(n):
        h = (y[i] + y[i + 1]) / 2
        l = x[i + 1] -x[i]
        area += h * l
    return area*2*r/np.pi

# calculate the peak location and intervals
def peak_detect(r,RDF_r_list,highlight):
    """
    Peak detection / 1d array
    :param r: x
    :param RDF_r_list: y
    :return:
    """
    b_index = [0]
    for i in range(len(RDF_r_list)-1):
        if RDF_r_list[i] *  RDF_r_list[i+1]  < 0:
            b_index.append(i)
    circle_x = []
    circle_y = []
    index = 0
    for hl in range(highlight):
        left = b_index[hl]
        right = b_index[hl+1]
        index = int(np.flatnonzero(np.abs(np.array(RDF_r_list)) ==np.abs(np.array(RDF_r_list[left:right])).max()))
        circle_y.append(RDF_r_list[index])
        circle_x.append(r[index])
    return circle_x, circle_y, circle_x[2]-circle_x[0]

def peak_detect_based(r,RDF_r_list,base, highlight):
    """
    Peak detection / 1d array
    :param r: x
    :param RDF_r_list: y
    :return:
    """ 
    b_index = [0]
    for i in range(len(RDF_r_list)-1):
        if RDF_r_list[i] *  RDF_r_list[i+1]  < 0:
            b_index.append(i)
    circle_x = []
    circle_y = []
    index = 0
    for hl in range(highlight):
        left = b_index[hl]
        right = b_index[hl+1]
        index = int(np.flatnonzero(np.abs(np.array(RDF_r_list)) ==np.abs(np.array(RDF_r_list[left:right])).max()))
        circle_y.append(RDF_r_list[index] + base[index])
        circle_x.append(r[index])

    return circle_x, circle_y,circle_x[2]-circle_x[0]



