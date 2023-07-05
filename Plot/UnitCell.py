import re
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
from IPython.display import display
import ipywidgets as widgets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..XRDSimulation.DiffractionGrometry.atom import atomics

class plotUnitCell(object):
    def __init__(self,atom_coordinatses,lattice_param,):
        self.atom_coordinates = atom_coordinatses
        self.lattice_param = lattice_param


    def plot(self,):
        elevation_slider = widgets.IntSlider(value=0, min=-90, max=90, description='elevation')
        azimuth_slider = widgets.IntSlider(value=0, min=-180, max=180, description='azimuth')
        interact_func = interactive(self.sub_plot,elevation=elevation_slider, azimuth=azimuth_slider)
        interact_func.children[0].value = 30  
        interact_func.children[1].value = 60  
        display(interact_func)
        plt.show()

    def sub_plot(self,elevation, azimuth):
        a = self.lattice_param[0]
        b = self.lattice_param[1]
        c = self.lattice_param[2]
        alpha = self.lattice_param[3]
        beta = self.lattice_param[4]
        gamma = self.lattice_param[5]

        # Calculate vertex coordinates
        vertices = np.array([[0, 0, 0],
                            [a, 0, 0],
                            [a, b, 0],
                            [0, b, 0],
                            [0, 0, c],
                            [a, 0, c],
                            [a, b, c],
                            [0, b, c]])

        # Compute the rotation matrix
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        rotation_matrix = np.array([[np.cos(alpha_rad) * np.cos(beta_rad), -np.sin(alpha_rad), np.cos(alpha_rad) * np.sin(beta_rad)],
                                    [np.sin(alpha_rad) * np.cos(beta_rad) * np.sin(gamma_rad) + np.cos(alpha_rad) * np.sin(gamma_rad), np.cos(alpha_rad) * np.cos(gamma_rad), -np.sin(alpha_rad) * np.sin(beta_rad) * np.sin(gamma_rad) + np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)],
                                    [-np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad) + np.sin(alpha_rad) * np.sin(gamma_rad), np.cos(alpha_rad) * np.sin(gamma_rad), np.sin(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad) + np.cos(alpha_rad) * np.sin(beta_rad) * np.sin(gamma_rad)]])

        # translation vector
        translation_vector = np.zeros(3)

        transformed_vertices = (rotation_matrix @ (vertices.T - translation_vector[:, np.newaxis])).T

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot unit cell
        cube_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

        for cube_face in cube_faces:
            polygon = Poly3DCollection([transformed_vertices[cube_face]], linewidths=1, edgecolor='black')
            polygon.set_facecolor((0.5, 0.5, 0.5, 0.01))
            ax.add_collection3d(polygon)

        # plot atoms
        atom_list = []
        for atom in range(len(self.atom_coordinates)):
            ion = self.atom_coordinates[atom][0]
            x = self.atom_coordinates[atom][1]
            y = self.atom_coordinates[atom][2]
            z = self.atom_coordinates[atom][3]

            dict =  atomics()
            # in case errors 
            try:
                # read in the form factor
                dict[ion]
            except:
                # atomic unionized forms
                # Plan to replaces with Thomas-Fermi method
                ion = getHeavyatom(ion)

            if ion not in atom_list: atom_list.append(ion)
            size = dict[ion]['0']

            point = np.array([x*a, y*b, z*c])
            transformed_point = (rotation_matrix @ (point - translation_vector)).T
            ax.scatter(transformed_point[0], transformed_point[1], transformed_point[2], c=colors[atom_list.index(ion)], s=(5*size),marker='o', edgecolor='black',alpha=1.0)

        ax.view_init(elev=elevation, azim=azimuth)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel('X',labelpad=-20)
        ax.set_ylabel('Y',labelpad=-20)
        ax.set_zlabel('Z',labelpad=-20)

        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.grid(False)
        

def getHeavyatom(s):
    """
    Some atomic ionization forms not defined in the table are replaced by their unionized forms
    """
    # Define a function called getHeavyatom that takes one parameter: s, a string that contains letters and/or non-letter characters.
    return re.sub(r'[^A-Za-z]+', "", s)
    # Use the re.sub() function to replace all non-letter characters in s with an empty string. Return the modified string.

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080', '#808000',
              '#800080', '#008080', '#FFA500', '#FFC0CB', '#FFD700', '#008B8B', '#00FF7F', '#7B68EE', '#FF4500', '#FF1493']