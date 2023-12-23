# Calculate the graph structure of crystals
# Author: Bin CAO <binjacobcao@gmail.com>

from ..Extinction.XRDpre import UnitCellAtom,det_system
from ..Extinction.CifReader import CifFile
from ..XRDSimulation.Simulation import  group_elements_by_first_element,getHeavyatom,lattice_parameters_to_matrix,generate_super_cell
import numpy as np
import pickle
import os
import re 
import copy

class CrystalGraph(object):
    def __init__(self, folder_path):
        os.makedirs('CifParserToGraph', exist_ok=True)
        lattice = []
        atom_coordinates = []
        file = []
        lattice_system = []
        space_groups = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".cif"):
                # constructe the file path
                file_path = os.path.join(folder_path, filename)
                latt, space_g, Asymmetric_atomic_coordinates,_, symmetric_operation = read_cif(file_path)
                if space_g== None or latt==None or Asymmetric_atomic_coordinates == None:
                    print('CIF parse error')
                    print('='*60,'\n')
                    pass
                else:
                    try:
                        AtomCoordinates= UnitCellAtom(copy.deepcopy(Asymmetric_atomic_coordinates),symmetric_operation)
                        system = det_system(latt)
                        
                        lattice_system.append(system)
                        space_groups.append(Asymmetric_atomic_coordinates[0])
                        lattice.append(latt)
                        atom_coordinates.append(group_elements_by_first_element(AtomCoordinates))
                        file.append(file_path)
                        print('file : %s' % file_path, 'was Parsered by WPEM') 
                        print('='*60,'\n')
                    except:
                        print('file : %s' % file_path, 'was omitted by WPEM') 
                        print('='*60,'\n')

        self.lattice = lattice
        self.atom_coordinates = atom_coordinates
        self.lattice_system = lattice_system
        self.space_groups = space_groups
        self.file = file
       

    
    def generate_graph(self, BK_boundary_condition=False):
        node_list = []
        edge_list = []
        pass_cif = []
        if BK_boundary_condition==True:
            print('Periodic boundary conditions are considered in establishing edges')
        for i in range(len(self.lattice)):
            try:
                node,edge = graph_exp(self.atom_coordinates[i],self.lattice[i],BK_boundary_condition)
                node_list.append(node)
                edge_list.append(edge)
            except:
                pass_cif.append(i)
                pass
        
        file = np.delete(self.file, pass_cif)
        with open('./CifParserToGraph/CifPath.path', 'w') as savfile:
            for item in file:
                savfile.write("%s\n" % str(item))

        

        with open('./CifParserToGraph/Node.pkl', 'wb') as file:
            pickle.dump(node_list, file)
        with open('./CifParserToGraph/Edge.pkl', 'wb') as file:
            pickle.dump(edge_list, file)

        ya = np.delete(self.space_groups, pass_cif)
        yb = np.delete(self.lattice_system, pass_cif)

        np.save('./CifParserToGraph/Y_group.npy', ya)
        np.save('./CifParserToGraph/Y_system.npy', yb)

        print('Parsing failed for {} CIF files.'.format(len(pass_cif)))
        print('ALL {} CIFs were configured as graph data.'.format(len(ya)))
        

def read_cif(cif_dir):
    cif = CifFile.from_file(cif_dir)
    for k in cif.data:
        v = cif.data[k].data
        try:
            a = getFloat(v['_cell_length_a'])
            b = getFloat(v['_cell_length_b'])
            c = getFloat(v['_cell_length_c']) 
            alpha = getFloat(v['_cell_angle_alpha'])
            beta = getFloat(v['_cell_angle_beta'])
            gamma = getFloat(v['_cell_angle_gamma'])
            latt = [a, b, c, alpha, beta, gamma]
        except: latt = None
        try:
            space_group_code = int(v['_symmetry_Int_Tables_number'])
        except KeyError:
            try:
                space_group_code = int(v['_space_group_IT_number'])
            except:
                space_group_code = -1

        sites =[space_group_code]
        try:
            for i, name in enumerate(v['_atom_site_type_symbol']):
                sites.append([name, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
        except KeyError:
            try:
                for i, name in enumerate(v['_atom_site_label']):
                    sites.append([name, getFloat(v['_atom_site_fract_x'][i]), getFloat(v['_atom_site_fract_y'][i]), getFloat(v['_atom_site_fract_z'][i])])
            except:
                sites = None
       
        try:
            symmetric_operation = list(v['_space_group_symop_operation_xyz'])
        except KeyError:
            try:
                symmetric_operation = list(v['_symmetry_equiv_pos_as_xyz'])
            except KeyError:
                symmetric_operation = None

        if '_symmetry_space_group_name_H-M' in v:
            symbol = v['_symmetry_space_group_name_H-M'][0]
            spaceG = v['_symmetry_space_group_name_H-M']
        elif '_symmetry_space_group_name_Hall' in v:
            symbol = v['_symmetry_space_group_name_Hall'][0]
            spaceG = v['_symmetry_space_group_name_Hall']
        else:
            symbol = None
            spaceG = None
        
    return latt, spaceG, sites, symbol,symmetric_operation

def getFloat(s):
    return float(re.sub(u"\\(.*?\\)", "", s))


def graph_exp(atom_coordinates,lattice,BK_boundary_condition = False):
    if BK_boundary_condition == False:
        encoded_node = encode_atoms(atom_coordinates)
        encoded_edg = find_nearest_neighbors(atom_coordinates,lattice,)
    elif BK_boundary_condition == True:
        encoded_node = encode_atoms(atom_coordinates)
        supercell = generate_super_cell(atom_coordinates, 3, 3, 3)
        centralized_cell = movecell(supercell)
        encoded_edg = find_nearest_neighbors(atom_coordinates,lattice,centralized_cell)

    return encoded_node,encoded_edg

def encode_atoms(input_coordinates):
    atomic_data = {
        'H': (1, 1.008, 1),
        'He': (2, 4.0026, 2),
        'Li': (3, 6.94, 1),
        'Be': (4, 9.0122, 2),
        'B': (5, 10.81, 3),
        'C': (6, 12.011, 4),
        'N': (7, 14.007, 5),
        'O': (8, 15.999, 6),
        'F': (9, 18.998, 7),
        'Ne': (10, 20.180, 8),
        'Na': (11, 22.990, 1),
        'Mg': (12, 24.305, 2),
        'Al': (13, 26.982, 3),
        'Si': (14, 28.085, 4),
        'P': (15, 30.974, 5),
        'S': (16, 32.06, 6),
        'Cl': (17, 35.45, 7),
        'K': (19, 39.098, 1),
        'Ar': (18, 39.948, 8),
        'Ca': (20, 40.078, 2),
        'Sc': (21, 44.956, 3),
        'Ti': (22, 47.867, 4),
        'V': (23, 50.942, 5),
        'Cr': (24, 51.996, 6),
        'Mn': (25, 54.938, 7),
        'Fe': (26, 55.845, 8),
        'Ni': (28, 58.693, 10),
        'Co': (27, 58.933, 9),
        'Cu': (29, 63.546, 11),
        'Zn': (30, 65.38, 12),
        'Ga': (31, 69.723, 13),
        'Ge': (32, 72.630, 14),
        'As': (33, 74.922, 15),
        'Se': (34, 78.971, 16),
        'Br': (35, 79.904, 17),
        'Kr': (36, 83.798, 18),
        'Rb': (37, 85.468, 1),
        'Sr': (38, 87.62, 2),
        'Y': (39, 88.906, 3),
        'Zr': (40, 91.224, 4),
        'Nb': (41, 92.906, 5),
        'Mo': (42, 95.95, 6),
        'Tc': (43, 98.0, 7),
        'Ru': (44, 101.07, 8),
        'Rh': (45, 102.91, 9),
        'Pd': (46, 106.42, 10),
        'Ag': (47, 107.87, 11),
        'Cd': (48, 112.41, 12),
        'In': (49, 114.82, 13),
        'Sn': (50, 118.71, 14),
        'Sb': (51, 121.76, 15),
        'Te': (52, 127.60, 16),
        'I': (53, 126.90, 17),
        'Xe': (54, 131.29, 18),
        'Cs': (55, 132.91, 1),
        'Ba': (56, 137.33, 2),
        'La': (57, 138.91, 3),
        'Ce': (58, 140.12, 4),
        'Pr': (59, 140.91, 5),
        'Nd': (60, 144.24, 6),
        'Pm': (61, 145.0, 7),
        'Sm': (62, 150.36, 8),
        'Eu': (63, 151.96, 9),
        'Gd': (64, 157.25, 10),
        'Tb': (65, 158.93, 11),
        'Dy': (66, 162.50, 12),
        'Ho': (67, 164.93, 13),
        'Er': (68, 167.26, 14),
        'Tm': (69, 168.93, 15),
        'Yb': (70, 173.04, 16),
        'Lu': (71, 174.97, 17),
        'Hf': (72, 178.49, 4),
        'Ta': (73, 180.95, 5),
        'W': (74, 183.84, 6),
        'Re': (75, 186.21, 7),
        'Os': (76, 190.23, 8),
        'Ir': (77, 192.22, 9),
        'Pt': (78, 195.08, 10),
        'Au': (79, 196.97, 11),
        'Hg': (80, 200.59, 12),
        'Tl': (81, 204.38, 13),
        'Pb': (82, 207.2, 14),
        'Bi': (83, 208.98, 15),
        'Th': (90, 232.04, 6),
        'Pa': (91, 231.04, 7),
        'U': (92, 238.03, 6),
        'Np': (93, 237.0, 7),
        'Pu': (94, 244.0, 8),
        'Am': (95, 243.0, 9),
        'Cm': (96, 247.0, 10),
        'Bk': (97, 247.0, 11),
        'Cf': (98, 251.0, 12),
        'Es': (99, 252.0, 13),
        'Fm': (100, 257.0, 14),
        'Md': (101, 258.0, 15),
        'No': (102, 259.0, 16),
        'Lr': (103, 262.0, 17),
    }

    encoded_atoms = {}

    for i in range(len(input_coordinates)):
        atom = getHeavyatom(input_coordinates[i][0])
        if atom in atomic_data:
            atomic_number, atomic_mass, valence_electrons = atomic_data[atom]
        elif atom[:2] in atomic_data:
            atomic_number, atomic_mass, valence_electrons = atomic_data[atom[:2]]
        elif atom[:1] in atomic_data:
            atomic_number, atomic_mass, valence_electrons = atomic_data[atom[:1]]
        else:
            print('The atom format is not supported, -1,-1,-1 is assigned to the node')
            atomic_number, atomic_mass, valence_electrons = -1,-1,-1
        encoded_atoms[i] = (atomic_number, atomic_mass, valence_electrons)

    return encoded_atoms

def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def find_nearest_neighbors(input_coordinates,lattice,supercell=None):
    if supercell == None:
        # lattice matrix in cartesian coords
        lattice_vectors = lattice_parameters_to_matrix(lattice[0], lattice[1], lattice[2], lattice[3], lattice[4], lattice[5])
        fractional_coords = np.array([atom[1:] for atom in input_coordinates])
        cartesian_coords = fractional_to_cartesian(fractional_coords, lattice_vectors)
        # search the nearest Neighbor inner a lattice cell
        num_atoms = len(cartesian_coords)
        nearest_neighbors = []
        for i in range(num_atoms):
            current_atom = cartesian_coords[i]
            distances = [calculate_distance(current_atom, cartesian_coords[j]) for j in range(num_atoms)]
            # Exclude the current atom itself
            distances[i] = np.inf
            nearest_neighbor_index = np.where(distances == np.min(distances))[0]
            for k in nearest_neighbor_index:
                nearest_neighbors.append([i,k]) 
    else:
        # for supercell defined in WPEM , the cartesian coordinates can be computed by fraction coordinates (we does not reduce to 1) multiplies the lattice matrix
        lattice_vectors = lattice_parameters_to_matrix(lattice[0], lattice[1], lattice[2], lattice[3], lattice[4], lattice[5])
        original_cell_fract = np.array([atom[1:] for atom in input_coordinates])
        original_cell_cart = fractional_to_cartesian(original_cell_fract, lattice_vectors)
        super_cell_fract = np.array([atom[1:] for atom in supercell])
        super_cell_cart = fractional_to_cartesian(super_cell_fract, lattice_vectors)
        # consider the Born-Kaman boundary conditions
        num_atoms = len(original_cell_cart) # centralized grid
        nearest_neighbors = []
        for i in range(num_atoms):
            current_atom = original_cell_cart[i]
            distances = [calculate_distance(current_atom, super_cell_cart[j]) for j in range(len(super_cell_cart))]
            # serach the index of current_atom on the super_cell_cart list
            indices = [index for index, element in enumerate(super_cell_cart) if are_arrays_equal(element, current_atom)]
            # Exclude the current atom itself
            distances[indices[0]] = np.inf
            nearest_neighbor_index = np.where(distances == np.min(distances))[0] # this index is calculated at super cell 
            for k in nearest_neighbor_index:
                # find the cooresponding atom at the original lattice cell
                nearest_atom = super_cell_cart[k]
                # check whether the atom located at the original lattice
                # Convert the point to fractional coordinates
                fractional_point = fractional_to_cartesian(nearest_atom,np.linalg.inv(lattice_vectors))
                if np.round(fractional_point[0],5) <0 :fractional_point[0] += 1
                elif np.round(fractional_point[0],5) >=1 :fractional_point[0] -= 1
                elif np.round(fractional_point[1],5) <0 : fractional_point[1] += 1
                elif np.round(fractional_point[1],5) >=1 : fractional_point[1] -= 1
                elif np.round(fractional_point[2],5) <0 : fractional_point[2] += 1
                elif np.round(fractional_point[2],5) >=1 : fractional_point[2] -= 1
                # Convert back to Cartesian coordinates
                _nearest_atom = fractional_to_cartesian(fractional_point,lattice_vectors)
                # find nearest neighbor qual atom at the original lattice cell 
                nearest_indices = [index for index, element in enumerate(original_cell_cart) if are_arrays_equal(element, _nearest_atom)]
                if nearest_indices:
                    nearest_neighbors.append([i,nearest_indices]) 
                else : 
                    _distances = [calculate_distance(_nearest_atom, original_cell_cart[j]) for j in range(len(original_cell_cart))]
                    nearest_neighbors.append([i,np.argmin(_distances)])  
                  

    return nearest_neighbors

def fractional_to_cartesian(fractional_coordinates, lattice_vectors):
    return np.dot(fractional_coordinates, lattice_vectors)

def movecell(supercell):
    for i in range(len(supercell)):
        supercell[i][1] -= 1
        supercell[i][2] -= 1
        supercell[i][3] -= 1
    return  supercell

def are_arrays_equal(arr1, arr2, decimal_places=3):
    return np.allclose(arr1, arr2, atol=10**(-decimal_places))


