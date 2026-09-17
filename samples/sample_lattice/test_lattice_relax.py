#!/usr/bin/env python
"""
Test script for lattice relaxation in aimsChain.

Creates a path between two MgO rocksalt structures with different
lattice constants and verifies that:
1. Positions include lattice vectors when relax_lattice is enabled
2. Forces/stress are correctly parsed and combined
3. Interpolation produces intermediate lattice vectors
4. The optimizer step correctly handles combined arrays
"""

import sys
import os
import numpy as np

# Add aimsChain to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimsChain.aimsio import read_aims
from aimsChain.node import Node
from aimsChain.string_path import StringPath
from aimsChain.config import Control
from aimsChain.interpolate import spline_pos, get_t

def test_read_periodic_geometry():
    """Test reading a geometry file with lattice vectors"""
    geo = read_aims('samples/sample_lattice/ini.in')
    
    print("=== Test 1: Read periodic geometry ===")
    print(f"Number of atoms: {len(geo.atoms)}")
    print(f"Lattice vectors shape: {np.shape(geo.lattice)}")
    assert np.shape(geo.lattice) == (3,3), "Lattice should be 3x3"
    print(f"Lattice constant a: {geo.lattice[0,0]:.4f} Å")
    
    # Check all_positions
    all_pos = geo.all_positions
    natoms = len(geo.atoms)
    assert all_pos.shape == (natoms + 3, 3), \
        f"all_positions shape should be ({natoms+3}, 3), got {all_pos.shape}"
    print(f"Combined positions shape: {all_pos.shape}")
    
    # Check lattice constraints
    assert np.all(geo.lattice_constraint == np.ones((3,3))), \
        "Default lattice constraint should be all free"
    
    # Test setting constraints via string
    geo.lattice_constraint = "diagonal"
    expected_mask = np.zeros((3,3))
    expected_mask[0,0] = expected_mask[1,1] = expected_mask[2,2] = 1
    assert np.all(geo.lattice_constraint == expected_mask), \
        "Diagonal constraint should give diag(1,1,1)"
    print("Lattice constraint 'diagonal': OK")
    
    geo.lattice_constraint = "none"
    assert np.all(geo.lattice_constraint == np.ones((3,3))), \
        "'none' constraint should give all ones"
    
    geo.lattice_constraint = "all"
    assert np.all(geo.lattice_constraint == np.zeros((3,3))), \
        "'all' constraint should give all zeros"
    print("All constraint modes: OK")
    
    print("PASSED\n")
    return geo


def test_combined_positions_forces():
    """Test setting and getting combined positions and forces"""
    print("=== Test 2: Combined positions and forces ===")
    
    # Create a simple geometry
    from aimsChain.atom import Atoms, Atom
    
    atoms_list = [
        Atom(positions=(0,0,0), symbol='Mg'),
        Atom(positions=(2,2,2), symbol='O'),
    ]
    lattice = np.array([[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]])
    geo = Atoms(atoms=atoms_list, lattice=lattice)
    
    # Set positions via combined setter
    new_all_pos = np.array([
        [0.1, 0.0, 0.0],  # Mg
        [2.0, 2.1, 2.0],  # O
        [4.3, 0.0, 0.0],  # lattice a
        [0.0, 4.3, 0.0],  # lattice b
        [0.0, 0.0, 4.3],  # lattice c
    ])
    geo.all_positions = new_all_pos
    
    # Verify
    np.testing.assert_almost_equal(geo.positions[0], [0.1, 0.0, 0.0])
    np.testing.assert_almost_equal(geo.positions[1], [2.0, 2.1, 2.0])
    np.testing.assert_almost_equal(geo.lattice[0,0], 4.3)
    print("Combined positions setter/getter: OK")
    
    # Set forces via combined setter
    new_all_forces = np.array([
        [0.1, 0.0, 0.0],   # Mg force
        [-0.1, 0.0, 0.0],  # O force
        [0.01, 0, 0],      # lattice a stress
        [0, 0.01, 0],      # lattice b stress
        [0, 0, 0.01],      # lattice c stress
    ])
    geo.all_forces = new_all_forces
    
    # Verify
    np.testing.assert_almost_equal(geo.forces[0], [0.1, 0.0, 0.0])
    np.testing.assert_almost_equal(geo.forces[1], [-0.1, 0.0, 0.0])
    np.testing.assert_almost_equal(geo.lattice_forces, np.diag([0.01, 0.01, 0.01]))
    print("Combined forces setter/getter: OK")
    
    # Test constraint masking on forces
    geo.lattice_constraint = "diagonal"
    constrained_lat_forces = geo.lattice_forces
    expected = np.diag([0.01, 0.01, 0.01])
    np.testing.assert_almost_equal(constrained_lat_forces, expected)
    print("Lattice force constraint masking: OK")
    
    print("PASSED\n")
    return geo


def test_interpolation_with_lattice():
    """Test that spline interpolation handles combined positions"""
    print("=== Test 3: Interpolation with lattice ===")
    
    from aimsChain.atom import Atoms, Atom
    
    # Create two geometries with different lattice constants
    atoms_list1 = [
        Atom(positions=(0,0,0), symbol='Mg'),
        Atom(positions=(2.1,2.1,2.1), symbol='O'),
    ]
    atoms_list2 = [
        Atom(positions=(0,0,0), symbol='Mg'),
        Atom(positions=(2.125,2.125,2.125), symbol='O'),
    ]
    
    lat1 = np.array([[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]])
    lat2 = np.array([[4.25, 0, 0], [0, 4.25, 0], [0, 0, 4.25]])
    
    geo1 = Atoms(atoms=atoms_list1, lattice=lat1)
    geo2 = Atoms(atoms=atoms_list2, lattice=lat2)
    
    # Get combined positions
    pos1 = geo1.all_positions  # (2+3, 3)
    pos2 = geo2.all_positions
    
    # Interpolate
    positions = np.array([pos1, pos2])
    new_t = np.array([0.0, 0.5, 1.0])
    interpolated = spline_pos(positions, new_t)
    
    print(f"Interpolated shape: {interpolated.shape}")  # (3, 5, 3)
    assert interpolated.shape == (3, 5, 3), \
        f"Expected (3, 5, 3), got {interpolated.shape}"
    
    # Check midpoint lattice
    mid_lattice = interpolated[1, 2:, :]  # last 3 rows
    expected_mid = (lat1 + lat2) / 2  # linear interpolation
    np.testing.assert_almost_equal(mid_lattice, expected_mid, decimal=4)
    print(f"Midpoint lattice a: {mid_lattice[0,0]:.4f} (expected {expected_mid[0,0]:.4f})")
    print("Spline interpolation with lattice: OK")
    
    print("PASSED\n")


def test_node_with_relax_lattice():
    """Test Node positions/forces with relax_lattice enabled"""
    print("=== Test 4: Node with relax_lattice ===")
    
    from aimsChain.atom import Atoms, Atom
    
    # Create a mock control with relax_lattice=True
    control = Control()
    control.relax_lattice = True
    
    atoms_list = [
        Atom(positions=(0,0,0), symbol='Mg'),
        Atom(positions=(2.1,2.1,2.1), symbol='O'),
    ]
    lattice = np.array([[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]])
    geo = Atoms(atoms=atoms_list, lattice=lattice)
    
    # Create a mock path with the control
    class MockPath:
        pass
    mock_path = MockPath()
    mock_path.control = control
    
    node = Node(param=0.0, geometry=geo, path=mock_path)
    
    # Check _use_lattice_relax
    assert node._use_lattice_relax(), "Should use lattice relax"
    print("_use_lattice_relax(): True")
    
    # Check positions include lattice
    pos = node.positions
    assert pos.shape == (2 + 3, 3), f"Expected (5,3), got {pos.shape}"
    print(f"Node positions shape: {pos.shape}")
    
    # Check forces shape (even with zero forces)
    node.forces = np.array([[0,0,0], [0,0,0]])
    node.geometry.lattice_forces = np.zeros((3,3))
    forces = node.forces
    assert forces.shape == (5, 3), f"Expected (5,3), got {forces.shape}"
    print(f"Node forces shape: {forces.shape}")
    
    # Test setting positions via combined array
    new_pos = np.array([
        [0.1, 0, 0],
        [2.0, 2.0, 2.0],
        [4.3, 0, 0],
        [0, 4.3, 0],
        [0, 0, 4.3],
    ])
    node.positions = new_pos
    np.testing.assert_almost_equal(node.geometry.lattice[0,0], 4.3)
    print("Node combined positions setter: OK")
    
    # Test fixed node returns zeros
    node.fixed = True
    forces_fixed = node.forces
    assert np.all(forces_fixed == 0), "Fixed node forces should be zero"
    assert forces_fixed.shape == (5, 3)
    print("Fixed node forces: all zero, correct shape")
    
    print("PASSED\n")


def test_path_with_lattice():
    """Test creating a full path with lattice relaxation"""
    print("=== Test 5: Full path with lattice relaxation ===")
    
    # Read geometries
    ini = read_aims('samples/sample_lattice/ini.in')
    fin = read_aims('samples/sample_lattice/fin.in')
    
    # Create config with lattice relaxation
    control = Control()
    control.ini = 'samples/sample_lattice/ini.in'
    control.fin = 'samples/sample_lattice/fin.in'
    control.nimage = 5
    control.relax_lattice = True
    control.lattice_constraint = "diagonal"
    control.periodic_interp = True
    
    # Create path
    path = StringPath(control=control)
    
    # Create initial and final nodes
    ininode = Node(param=0.0, geometry=ini)
    finnode = Node(param=1.0, geometry=fin)
    ininode.fixed = True
    finnode.fixed = True
    
    path.nodes = [ininode, finnode]
    
    # Interpolate
    path.interpolate(control.nimage)
    
    print(f"Number of nodes: {path.n_nodes()}")
    assert path.n_nodes() == control.nimage + 2, \
        f"Expected {control.nimage + 2} nodes, got {path.n_nodes()}"
    
    # Check each node has combined positions
    for i, node in enumerate(path.nodes):
        pos = node.positions
        natoms = len(ini.atoms)
        assert pos.shape == (natoms + 3, 3), \
            f"Node {i}: expected ({natoms+3}, 3), got {pos.shape}"
        
        # Check lattice is interpolated between ini and fin
        lat = pos[natoms:]
        # For diagonal constraint and linear interp, the diagonal should
        # be between 4.21 and 4.25
        param = node.param
        expected_a = 4.21 + param * (4.25 - 4.21)
        np.testing.assert_almost_equal(lat[0,0], expected_a, decimal=3)
    
    print("All nodes have correct combined positions")
    print("Lattice vectors correctly interpolated between endpoints")
    
    print("PASSED\n")


def test_tangent_with_lattice():
    """Test that tangent computation works with combined vectors"""
    print("=== Test 6: Tangent with lattice ===")
    
    from aimsChain.atom import Atoms, Atom
    
    control = Control()
    control.relax_lattice = True
    
    class MockPath:
        pass
    mock_path = MockPath()
    mock_path.control = control
    mock_path.nodes = []
    
    def make_geo(lat_a, atom_shift=0):
        lattice = np.array([[lat_a, 0, 0], [0, 4.2, 0], [0, 0, 4.2]])
        atoms_list = [
            Atom(positions=(0+atom_shift, 0, 0), symbol='Mg'),
            Atom(positions=(lat_a/2+atom_shift, 2.1, 2.1), symbol='O'),
        ]
        return Atoms(atoms=atoms_list, lattice=lattice)
    
    # Create three nodes
    geo0 = make_geo(4.20, 0)
    geo1 = make_geo(4.22, 0.01)
    geo2 = make_geo(4.25, 0.02)
    
    n0 = Node(param=0.0, geometry=geo0, path=mock_path, fixed=True)
    n1 = Node(param=0.5, geometry=geo1, path=mock_path)
    n2 = Node(param=1.0, geometry=geo2, path=mock_path, fixed=True)
    
    mock_path.nodes = [n0, n1, n2]
    
    # Get tangent for the middle node
    tangent = n1.get_tangent()
    
    # Tangent should have shape (5, 3) = (natoms + 3, 3)
    assert tangent.shape == (5, 3), f"Expected (5,3), got {tangent.shape}"
    
    # Tangent should be approximately a unit vector
    tangent_flat = tangent.reshape(-1)
    magnitude = np.sqrt(np.dot(tangent_flat, tangent_flat))
    assert abs(magnitude - 1.0) < 1e-10, f"Tangent should be unit, got {magnitude}"
    print(f"Tangent magnitude: {magnitude:.10f}")
    print(f"Tangent shape: {tangent.shape}")
    print("Tangent with lattice vectors: OK")
    
    print("PASSED\n")


def test_normal_forces_with_lattice():
    """Test that normal forces projection works with lattice"""
    print("=== Test 7: Normal forces with lattice ===")
    
    from aimsChain.atom import Atoms, Atom
    
    control = Control()
    control.relax_lattice = True
    control.method = "string"
    
    class MockPath:
        pass
    mock_path = MockPath()
    mock_path.control = control
    mock_path.nodes = []
    
    lattice = np.array([[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]])
    atoms_list = [
        Atom(positions=(0, 0, 0), symbol='Mg'),
        Atom(positions=(2.105, 2.105, 2.105), symbol='O'),
    ]
    geo = Atoms(atoms=atoms_list, lattice=lattice)
    
    n0 = Node(param=0.0, geometry=geo, path=mock_path, fixed=True)
    n1 = Node(param=0.5, geometry=geo, path=mock_path)
    n2 = Node(param=1.0, geometry=geo, path=mock_path, fixed=True)
    
    mock_path.nodes = [n0, n1, n2]
    
    # Set forces on the middle node (atom forces + lattice stress)
    all_forces = np.array([
        [0.5, 0.0, 0.0],   # Mg force
        [-0.5, 0.0, 0.0],  # O force
        [0.01, 0, 0],      # lattice a stress
        [0, 0.01, 0],      # lattice b stress
        [0, 0, 0.01],      # lattice c stress
    ])
    n1.geometry.all_forces = all_forces
    
    # Get normal forces
    normal_f = n1.normal_forces
    
    assert normal_f.shape == (5, 3), f"Expected (5,3), got {normal_f.shape}"
    
    # Normal forces should be perpendicular to tangent
    tangent = n1.get_tangent()
    normal_flat = normal_f.reshape(-1)
    tangent_flat = tangent.reshape(-1)
    projection = np.dot(normal_flat, tangent_flat)
    assert abs(projection) < 1e-10, \
        f"Normal forces should be perpendicular to tangent, dot={projection}"
    print(f"Normal forces orthogonal to tangent: dot = {projection:.2e}")
    print("Normal forces with lattice: OK")
    
    print("PASSED\n")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("=" * 60)
    print("aimsChain Lattice Relaxation Tests")
    print("=" * 60)
    print()
    
    try:
        test_read_periodic_geometry()
        test_combined_positions_forces()
        test_interpolation_with_lattice()
        test_node_with_relax_lattice()
        test_path_with_lattice()
        test_tangent_with_lattice()
        test_normal_forces_with_lattice()
        
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
