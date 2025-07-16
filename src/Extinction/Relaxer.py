import warnings
from .m3gnet.models import Relaxer
from pymatgen.core import Lattice, Structure
import numpy as np

def _Relaxer(cry_type,lattice_constants,atom_coords):
    for category in (UserWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=category, module="tensorflow")

    atom = Parse(cry_type,lattice_constants,atom_coords) 

    relaxer = Relaxer() 

    relax_results = relaxer.relax(atom, verbose=True)

    final_structure = relax_results['final_structure']
    final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(atom))
  

    lattices = [final_structure.lattice.abc[0], final_structure.lattice.abc[1], final_structure.lattice.abc[2],
                 final_structure.lattice.alpha, final_structure.lattice.beta, final_structure.lattice.gamma]
    
    # print('Relaxed lattice parameters by M3GNet',lattices)
    print(f"Final energy is {final_energy_per_atom:.3f} eV/atom by M3GNet")

    return lattices , final_energy_per_atom


def Parse(cry_type,lattice_constants,atom_coords,):
    name_list,frac_coords = process_data(atom_coords)
    if cry_type == 1:
        atom = Structure(Lattice.cubic(lattice_constants[0]),name_list,frac_coords)
    elif cry_type == 2: 
        atom = Structure(Lattice.hexagonal(lattice_constants[0],lattice_constants[2]),name_list,frac_coords)
    elif cry_type == 3:
        atom = Structure(Lattice.tetragonal(lattice_constants[0],lattice_constants[2]),name_list,frac_coords)
    elif cry_type == 4:
        atom = Structure(Lattice.orthorhombic(lattice_constants[0],lattice_constants[1],lattice_constants[2]),name_list,frac_coords)
    elif cry_type == 5: 
        atom = Structure(Lattice.rhombohedral(lattice_constants[0],lattice_constants[3]),name_list,frac_coords)
    elif cry_type == 6:
        atom = Structure(Lattice.monoclinic(lattice_constants[0],lattice_constants[1],lattice_constants[2],lattice_constants[4]),name_list,frac_coords)
    else : print('triclinic crystal is not supported in M3GNet') 

    return atom

def process_data(input_data):
    elements = []
    coordinates = []

    for item in input_data:
        # parse the atoms 
        element = ''.join(filter(str.isalpha, item[0]))
        elements.append(element)

        # parse the fractional coordinates
        coordinates.append(item[1:])

    return elements, coordinates

# def for convert lattice matrix to lattice constants
def calculate_lattice_constants(cell_matrix):
    a = np.linalg.norm(cell_matrix[0])
    b = np.linalg.norm(cell_matrix[1])
    c = np.linalg.norm(cell_matrix[2])

    alpha = np.degrees(np.arccos(np.dot(cell_matrix[1], cell_matrix[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(cell_matrix[0], cell_matrix[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(cell_matrix[0], cell_matrix[1]) / (a * b)))

    return [a, b, c, alpha, beta, gamma]