"""
This module defines basic io functions for aims
Read/Write aims geometry
Read aims output
"""

# 1 GPa = 1e9 J/m^3 ; 1 eV/A^3 = 160.21766208 GPa
_GPA_TO_EV_PER_ANG3 = 1.0 / 160.21766208


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
            elif inp[1] == 'all':
                current_atom.constraint="all"
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
    #if atoms.lattice != None:
    try:
        is_lat = len(atoms.lattice)
        lat=True
    except TypeError: 
        lat = False
    if lat:
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


def stress_to_lattice_forces(stress, lattice, scale=1.0):
    """
    Convert a stress tensor (eV/A^3) into forces on the lattice vectors (eV/A).

    With the cell matrix h whose *rows* are the lattice vectors and the
    strain convention h -> (1 + eps) h, the energy change is

        dE = V * sigma : deps ,            deps = dh . h^{-1}

    so dE/dh = V * sigma * h^{-T} and the force conjugate to the lattice
    vectors is

        F_lat = -V * sigma * h^{-T} ,      V = |det(h)|

    Rows of the returned 3x3 array correspond to the lattice vectors
    a, b, c.  `scale` is an optional damping factor (lattice_force_scale).
    """
    import numpy as np
    stress = np.asarray(stress, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    volume = abs(np.linalg.det(lattice))
    return -scale * volume * (stress @ np.linalg.inv(lattice).T)


def _parse_stress_block(header, lines):
    """
    Parse the numeric stress block that follows a "stress tensor" header
    line in an FHI-aims output stream.  Handles the two formats found in
    real outputs:

      * 3x3 block  : three consecutive lines with three floats each
      * Voigt list : a single line with six floats (xx yy zz yz xz xy)

    Returns a 3x3 numpy array in eV/A^3 (converted from GPa when the
    header indicates GPa), or None when no stress numbers are found.
    """
    import numpy as np
    unit = _GPA_TO_EV_PER_ANG3 if "gpa" in header.lower() else 1.0
    rows = []
    for raw in lines:
        vals = []
        for tok in raw.split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        if len(vals) >= 3:
            rows.append(vals)
        if len(rows) >= 3:
            break
    if not rows:
        return None
    if len(rows) == 1 and len(rows[0]) >= 6:
        xx, yy, zz, yz, xz, xy = rows[0][:6]
        return unit * np.array([[xx, xy, xz],
                                [xy, yy, yz],
                                [xz, yz, zz]])
    if len(rows) >= 3:
        return unit * np.array([rows[0][:3], rows[1][:3], rows[2][:3]],
                               dtype=float)
    return None


def read_aims_output(filename):
    """
    Read the aims output file
    Will return the total energy corrected, the forces, and the stress tensor.
    Only suitable for single calculations (i.e. no relaxations etc)
    May read more stuff with little modification, if desired
    
    Returns:
        (energy, forces, stress) where stress is a 3x3 array (eV/A^3)
        or None if not found
    """
    import numpy as np
    output = open(filename, 'r')
    n_atoms = 0
    ener = 0
    forces = []
    stress = None
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
                vals = []
                for tok in inp:
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        continue
                # drop a leading atom index if present: "1  fx fy fz [extra]"
                if len(vals) >= 4 and vals[0].is_integer():
                    vals = vals[1:]
                if len(vals) >= 3:
                    forces.append(vals[:3])
        if "stress tensor" in line.lower():
            # The symmetrized block puts its three numeric rows 5-7 lines
            # after the header, so the window must be >= 7; it must also stay
            # well below ~20, since these lines are consumed and the total
            # energy and forces follow shortly after the block.
            follow = []
            for _ in range(10):
                nxt = output.readline()
                if not nxt:
                    break
                follow.append(nxt)
            parsed = _parse_stress_block(line, follow)
            if parsed is not None:
                stress = parsed
    output.close()
    return ener, np.array(forces), stress
