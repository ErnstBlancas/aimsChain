"""
This module defines basic io functions for aims
Read/Write aims geometry
Read aims output
"""
def read_aims(filename):
    """
    Import aims geometry file into the standard atoms type
    Store all info other than comments, this includes constraints, spins, and etc.
    Lattice vector is stored separately.
    """
    from aimsChain.atom import Atoms
    from aimsChain.atom import Atom
    import numpy as np

    atoms = Atoms()
    geo = open(filename,'r')
    lines = geo.readlines()
    geo.close()
    current_atom = None
    lattice=[]
    for line in lines:
        inp = line.split()
        if inp == []: 
            continue
        if inp[0][0] == '#':
            continue
        elif inp[0] == "atom":
            if current_atom != None:
                atoms.atoms = current_atom
            current_atom = Atom()
            current_atom.positions=[float(inp[1]), float(inp[2]), float(inp[3])]
            current_atom.symbol=inp[4]
        elif inp[0] == "lattice_vector":
            lattice.append([float(inp[1]), float(inp[2]), float(inp[3])])
        elif inp[0] == "constrain_relaxation":
            if inp[1] in 'xyz':
                current_atom.constraint=inp[1]
            else:
                current_atom.constraint="all"
        else:
            current_atom.add_extra(line.replace("\n"," "))
    if current_atom != None:
        atoms.atoms = current_atom
    if len(lattice) == 3:
        atoms.lattice = lattice
    return atoms


def write_aims(filename, atoms):
    """
    Wrtie atoms into a geometry file with "filename"
    """
    import numpy as np

    geo = open(filename, 'w')
    geo.write('#=======================================#\n')
    geo.write('# '+filename+'\n')
    geo.write('#=======================================#\n')
    if atoms.lattice != None:
        for vector in atoms.lattice:
            geo.write('lattice_vector ')
            for i in range(3):
                geo.write('%16.16f ' % vector[i])
            geo.write('\n')
    
    for atom in atoms.atoms:
        geo.write('atom ')
        for coord in atom.positions:
            geo.write('%16.16f ' % coord)
        geo.write(atom.symbol + '\n')
        constraint = ( atom.constraint == [0,0,0] )
        if constraint[0] and constraint[1] and constraint[2]:
            geo.write('constrain_relaxation .true.\n')
        elif constraint[0]:
            geo.write('constrain_relaxation x\n')
        elif constraint[1]:
            geo.write('constrain_relaxation y\n')
        elif constraint[2]:
            geo.write('constrain_relaxation z\n')
        for line in atom.extra:
            geo.write(line + '\n')
    geo.close()

def write_mapped_aims(filename, atoms):
    """
    Wrtie atoms into a geometry file with "filename"
    """
    import numpy as np
    
    if atoms.lattice == None:
        write_aims(filename, atoms)
        return
    else:
        lattice = atoms.lattice

    geo = open(filename, 'w')
    geo.write('#=======================================#\n')
    geo.write('# '+filename+'\n')
    geo.write('# mapped to central unit cell \n')
    geo.write('#=======================================#\n')
    for vector in atoms.lattice:
        geo.write('lattice_vector ')
        for i in range(3):
            geo.write('%16.16f ' % vector[i])
        geo.write('\n')
    
    for atom in atoms.atoms:
        geo.write('atom ')
        positions = np.linalg.solve(lattice.transpose(), atom.positions.transpose()).transpose()
        positions %= 1.0
        positions %= 1.0
        positions = np.dot(positions, lattice)
        for coord in positions:
            geo.write('%16.16f ' % coord)
        geo.write(atom.symbol + '\n')
        constraint = ( atom.constraint == [0,0,0] )
        if constraint[0] and constraint[1] and constraint[2]:
            geo.write('constrain_relaxation .true.\n')
        elif constraint[0]:
            geo.write('constrain_relaxation x\n')
        elif constraint[1]:
            geo.write('constrain_relaxation y\n')
        elif constraint[2]:
            geo.write('constrain_relaxation z\n')
        for line in atom.extra:
            geo.write(line + '\n')
    geo.close()

def write_xyz(filename, path, repeat=[2,2,1]):
    """
    Write the path into a multi-image xyz file
    """
    import numpy as np
    lattice = path.lattice_vector
        
    geo = open(filename, 'w')
    if not path.periodic:
        for node in path.nodes:
            geometry = node.geometry
            geo.write('%d \n' % len(geometry.atoms))
            geo.write("Energy: %.16e \n" % node.ener)
            for atom in geometry.atoms:
                geo.write(atom.symbol +'\t')
                for coord in atom.positions:
                    geo.write('%16.12f \t' % coord)
                geo.write('\n') 
    else:
        for node in path.nodes:
            geometry = node.geometry
            geo.write('%d \n' % (len(geometry.atoms)*np.prod(repeat)))
            geo.write("Energy: %.16e \n" % node.ener)
            for atom in geometry.atoms:
                pos = atom.positions
                for a in range(repeat[0]):
                    for b in range(repeat[1]):
                        for c in range(repeat[2]):
                            geo.write(atom.symbol +'\t')
                            for coord in (pos+a*lattice[0]+b*lattice[1]+c*lattice[2]):
                                geo.write('%16.12f \t' % coord)
                            geo.write('\n') 

    geo.close()



def read_aims_output(filename):
    """
    Read the aims output file
    Will return the total energy corrected, and the forces
    Only suitable for single calculations (i.e. no relaxations etc)
    May read more stuff with little modification, if desired
    """
    import numpy as np
    output = open(filename, 'r')
    n_atoms = 0
    ener = 0
    forces = []
    while True:
        line = output.readline()
        if not line:
            break
        if "| Number of atoms                   :" in line:
            inp = line.split()
            n_atoms = int(inp[5])
        if "| Total energy corrected        :" in line:
            ener = float(line.split()[5])
        if "Total atomic forces (unitary forces cleaned)" in line:
            for i in range(n_atoms):
                inp = output.readline().split()
                forces.append([float(inp[2]), float(inp[3]), float(inp[4])])
    output.close()
    return ener, np.array(forces)

