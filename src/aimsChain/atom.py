"""
This module defines the basic Atom object.
Atoms is the basic geometry object, simply a list of Atom
"""

import numpy as np


class Atom(object):
    """
    Class for a single atom
    Parameters:
    position: 3 floats for xyz
    force: 3 floats for xyz forces
    extra: list of string to store extra info from geometry files
    constraint: mask for constraint, 0 values will be masked
    """
    
    def __init__(self, positions=(0,0,0),
                 forces=(0,0,0),
                 symbol='',
                 constraint=[1,1,1]):
        self.__positions = np.array(positions)
        self.__forces = np.array(forces)
        self.__extra = []
        self.__symbol = symbol
        self.__constraint=np.array(constraint)

    @property
    def positions(self):
        """get positions, numpy array with 3 float"""
        return self.__positions
    @positions.setter
    def positions(self, positions):
        """set positions"""
        self.__positions = self.__positions * np.abs(np.array(self.constraint - 1))
        self.__positions = self.__positions +  np.array(positions) * self.constraint 
        
    @property
    def forces(self):
        """
        get forces, numpy array with 3 float
        map constraint over the forces
        """
        return self.__forces*self.__constraint
    @forces.setter
    def forces(self, forces):
        """set forces"""
        self.__forces = np.array(forces)

    @property
    def extra(self):
        """get extra, list of strings"""
        return self.__extra
    @extra.setter
    def extra(self, extra):
        """set the extra"""
        self.__extra = extra
    def add_extra(self,extra):
        """add a new line to the end of existing extra"""
        self.__extra.append(extra)

    @property
    def symbol(self):
        """set the symbol for the atom"""
        return self.__symbol
    @symbol.setter
    def symbol(self,symbol):
        """get symbol for the atom"""
        self.__symbol=symbol

    @property
    def constraint(self):
        """
        get the constraint of the geometry
        """
        return self.__constraint
    @constraint.setter
    def constraint(self, constraint):
        """
        set the constraint for geometry
        constraint can either be combination of x,y,z, or all
        or a array of 3 item for x,y,z axes, either 1 or 0
        1 is free, 0 is fixed
        """
        if isinstance(constraint,str):
            if 'x' in constraint:
                self.__constraint = self.constraint&np.array([0,1,1])
            if 'y' in constraint:
                self.__constraint = self.constraint&np.array([1,0,1])
            if 'z' in constraint:
                self.__constraint = self.constraint&np.array([1,1,0])
            if constraint == 'all':
                self.__constraint = np.array([0,0,0])
        else:
            self.__constraint = np.array(constraint)


class Atoms(object):
    """
    Class for a single geometry
    Parameters:
    
    atoms: list of Atom object, basis of Atoms
    lattice: the lattice vectors, for periodic system
    ener: energy of the geometry
    """
    
    def __init__(self, atoms=[], 
                 lattice=None,
                 ener = 0):
        import copy
        self.__atoms=[]
        for atom in atoms:
            if isinstance(atom, Atom):
                self.__atoms.append(atom)
        self.__lattice = None
        if np.shape(lattice) == (3,3):
           self.__lattice = copy.deepcopy(lattice) 
        self.__ener = float(ener)
        # Lattice relaxation support
        self.__lattice_forces = np.zeros((3,3))
        self.__lattice_constraint = np.ones((3,3))  # all free by default

    @property
    def atoms(self):
        """
        get a list of all atoms in the geometry
        """
        return self.__atoms
    @atoms.setter
    def atoms(self, atom):
        """
        add a new atom to the end of the current list if atom is a single atom
        or else set atoms to the list of atom
        """
        if isinstance(atom, Atom):
            self.__atoms.append(atom)
        elif isinstance(atom, list):
            self.__atoms = atom
    
    @property
    def forces(self):
        """
        get a list of forces, 
        index correspond to position of atom in the list
        """
        forces = []
        for atom in self.__atoms:
            forces.append(atom.forces)
        return np.array(forces)
    @forces.setter
    def forces(self, forces):
        """
        set forces of all atoms
        no.of forces must match no. of atoms
        """
        for i, triplet in enumerate(forces):
            self.__atoms[i].forces = triplet

    @property
    def positions(self):
        """
        get a list of positions,
        index correspond to position of atom in the list
        """
        positions = []
        for atom in self.__atoms:
            positions.append(atom.positions)
        return np.array(positions)
    @positions.setter
    def positions(self, positions):
        """
        set positions of all atoms
        no. of positions must match no. of atoms
        """
        for i, triplet in enumerate(positions):
            self.__atoms[i].positions = triplet

    @property
    def lattice(self):
        """
        get the lattice constant stored in the geometry
        """
        return self.__lattice
    @lattice.setter
    def lattice(self,lattice):
        """
        set the lattice vector for the geometry
        """
        if len(lattice) == 3 and len(lattice[0]) == 3:
            self.__lattice = np.array(lattice)
    @property
    def constraints(self):
        """
        get a list of constraints,
        index correspond to position of atom in the list
        """
        constraints = []
        for atom in self.__atoms:
            constraints.append(atom.constraint)
        return np.array(constraints)
    @constraints.setter
    def constraints(self, constraints):
        """
        set the constraints for all atom at once
        no. of constraints must match no. of atoms
        """
        for i, single_constraint in enumerate(constraints):
            self.__atoms[i].constraint = single_constraint
    @property
    def ener(self):
        """
        get the energy of this geometry
        default value is 0, if not set otherwise
        """
        return self.__ener
    @ener.setter
    def ener(self, value):
        """
        Set the energy of this geometry
        """
        self.__ener = float(value)

    @property
    def lattice_forces(self):
        """
        Get the lattice forces (stress tensor, 3x3)
        Constraint mask applied: fixed components return zero
        """
        return self.__lattice_forces * self.__lattice_constraint
    @lattice_forces.setter
    def lattice_forces(self, value):
        """
        Set the lattice forces (stress tensor)
        """
        self.__lattice_forces = np.array(value)

    @property
    def lattice_constraint(self):
        """
        Get the lattice constraint mask (3x3).
        1 = free to relax, 0 = fixed.
        """
        return self.__lattice_constraint
    @lattice_constraint.setter
    def lattice_constraint(self, value):
        """
        Set the lattice constraint mask.
        Accepts a 3x3 array, or a string like 'abc' for diagonal only,
        or 'all' to fix everything.
        """
        if isinstance(value, str):
            if value == 'all':
                self.__lattice_constraint = np.zeros((3,3))
            elif value == 'none':
                self.__lattice_constraint = np.ones((3,3))
            elif value == 'diagonal':
                mask = np.zeros((3,3))
                mask[0,0] = mask[1,1] = mask[2,2] = 1
                self.__lattice_constraint = mask
            elif value == 'volume':
                # isotropic: all diagonal elements move together
                self.__lattice_constraint = np.eye(3)
            else:
                # parse individual chars like 'abc' for diagonal
                mask = np.zeros((3,3))
                for c in value:
                    idx = ord(c) - ord('a')
                    if 0 <= idx < 3:
                        mask[idx, idx] = 1
                self.__lattice_constraint = mask
        else:
            self.__lattice_constraint = np.array(value)

    @property
    def all_positions(self):
        """
        Get combined positions: atom positions (natoms, 3) followed by
        lattice vectors (3, 3). Returns a (natoms+3, 3) array.
        If no lattice, returns just atom positions.
        """
        atom_pos = self.positions
        if self.__lattice is not None:
            return np.vstack([atom_pos, self.__lattice])
        return atom_pos

    @all_positions.setter
    def all_positions(self, combined):
        """
        Set positions from a combined (natoms+3, 3) array.
        First natoms rows -> atom positions, last 3 rows -> lattice vectors.
        If the array has only natoms rows, sets only atom positions.
        """
        combined = np.array(combined)
        natoms = len(self.__atoms)
        if combined.shape[0] == natoms + 3:
            self.positions = combined[:natoms]
            self.lattice = combined[natoms:]
        elif combined.shape[0] == natoms:
            self.positions = combined
        else:
            raise ValueError("Combined positions shape mismatch: "
                             f"expected ({natoms},3) or ({natoms+3},3), "
                             f"got {combined.shape}")

    @property
    def all_forces(self):
        """
        Get combined forces: atom forces (natoms, 3) followed by
        lattice forces (3, 3). Returns a (natoms+3, 3) array.
        If no lattice, returns just atom forces.
        """
        atom_forces = self.forces
        if self.__lattice is not None:
            lat_forces = self.lattice_forces
            return np.vstack([atom_forces, lat_forces])
        return atom_forces

    @all_forces.setter
    def all_forces(self, combined):
        """
        Set forces from a combined (natoms+3, 3) array.
        First natoms rows -> atom forces, last 3 rows -> lattice forces.
        If the array has only natoms rows, sets only atom forces.
        """
        combined = np.array(combined)
        natoms = len(self.__atoms)
        if combined.shape[0] == natoms + 3:
            self.forces = combined[:natoms]
            self.__lattice_forces = combined[natoms:]
        elif combined.shape[0] == natoms:
            self.forces = combined
        else:
            raise ValueError("Combined forces shape mismatch: "
                             f"expected ({natoms},3) or ({natoms+3},3), "
                             f"got {combined.shape}")

    @property
    def all_constraints(self):
        """
        Get combined constraints: atom constraints (natoms, 3) followed by
        lattice constraints (3, 3). Returns a (natoms+3, 3) array.
        If no lattice, returns just atom constraints.
        """
        atom_con = self.constraints
        if self.__lattice is not None:
            return np.vstack([atom_con, self.__lattice_constraint])
        return atom_con
