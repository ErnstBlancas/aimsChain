#!/usr/bin/env python3
"""
test_aims_output.py — unit tests for the aimsChain FHI-AIMS output parsing
and the stress -> lattice-force conversion used by the molecule_chain
example.

Run:
    cd examples/molecule_chain
    .venv/bin/python test_aims_output.py
"""

import os
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))

from aimsChain.aimsio import read_aims_output, stress_to_lattice_forces

PASS = 0


def check(name, cond):
    global PASS
    if cond:
        PASS += 1
        print("  [ok] %s" % name)
    else:
        print("  [FAIL] %s" % name)
        raise SystemExit(1)


def write_tmp(content):
    fd, path = tempfile.mkstemp(suffix=".out")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_forces_with_index():
    print("== forces: '1 fx fy fz' lines (FHI-AIMS format) ==")
    content = """\
  Invoking FHI-aims ...
  | Number of atoms                   :  3
  | Total energy corrected        :   -1.23456789012345E+03 eV
  Total atomic forces (unitary forces cleaned)
    1   0.1000000000E+00   0.2000000000E+00  -0.3000000000E+00
    2  -0.4000000000E+00   0.5000000000E+00   0.6000000000E+00
    3   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
"""
    p = write_tmp(content)
    ener, forces, stress = read_aims_output(p)
    os.unlink(p)
    check("energy parsed", abs(ener - (-1234.56789012345)) < 1e-6)
    check("3 forces read", forces.shape == (3, 3))
    check("force[0] correct", np.allclose(forces[0], [0.1, 0.2, -0.3]))
    check("force[1] correct", np.allclose(forces[1], [-0.4, 0.5, 0.6]))
    check("stress None when absent", stress is None)


def test_forces_with_extra_columns():
    print("== forces: lines with trailing columns ==")
    content = """\
  | Number of atoms                   :  2
  | Total energy corrected        :   -5.00000000000000E+00 eV
  Total atomic forces (unitary forces cleaned)
    1   0.1000000000E+00   0.2000000000E+00  -0.3000000000E+00   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
    2  -0.1000000000E+00  -0.2000000000E+00   0.3000000000E+00   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
"""
    p = write_tmp(content)
    ener, forces, stress = read_aims_output(p)
    os.unlink(p)
    check("energy parsed", abs(ener - (-5.0)) < 1e-12)
    check("forces shape", forces.shape == (2, 3))
    check("force[1] correct", np.allclose(forces[1], [-0.1, -0.2, 0.3]))


def test_stress_3x3_ev_per_ang3():
    print("== stress: 3x3 block in eV/A^3 ==")
    content = """\
  | Number of atoms                   :  1
  | Total energy corrected        :   -1.00000000000000E+00 eV
  Total atomic forces (unitary forces cleaned)
    1   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
  |  Stress tensor (eV/A^3):
  |  -------------------------------------------------------------
  |   0.1000000000E-01   0.0000000000E+00   0.0000000000E+00
  |   0.0000000000E+00   0.2000000000E-01   0.0000000000E+00
  |   0.0000000000E+00   0.0000000000E+00   0.3000000000E-01
"""
    p = write_tmp(content)
    ener, forces, stress = read_aims_output(p)
    os.unlink(p)
    check("stress is 3x3", stress is not None and stress.shape == (3, 3))
    check("stress values", np.allclose(stress, np.diag([0.01, 0.02, 0.03])))


def test_stress_3x3_gpa():
    print("== stress: 3x3 block in GPa (converted to eV/A^3) ==")
    content = """\
  | Number of atoms                   :  1
  | Total energy corrected        :   -1.00000000000000E+00 eV
  Total atomic forces (unitary forces cleaned)
    1   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
  Stress tensor (GPa):
      0.1000000000E+02   0.0000000000E+00   0.0000000000E+00
      0.0000000000E+00   0.2000000000E+02   0.0000000000E+00
      0.0000000000E+00   0.0000000000E+00   0.3000000000E+02
"""
    p = write_tmp(content)
    ener, forces, stress = read_aims_output(p)
    os.unlink(p)
    gpa2ev = 1.0 / 160.21766208
    check("stress shape", stress is not None and stress.shape == (3, 3))
    check("GPa converted",
          np.allclose(stress, np.diag([10.0, 20.0, 30.0]) * gpa2ev))


def test_stress_voigt_six():
    print("== stress: single Voigt line (xx yy zz yz xz xy) ==")
    content = """\
  | Number of atoms                   :  1
  | Total energy corrected        :   -1.00000000000000E+00 eV
  Total atomic forces (unitary forces cleaned)
    1   0.0000000000E+00   0.0000000000E+00   0.0000000000E+00
  Stress tensor (eV/A^3):
      0.1000000000E-01   0.2000000000E-01   0.3000000000E-01   0.4000000000E-02   0.5000000000E-02   0.6000000000E-02
"""
    p = write_tmp(content)
    ener, forces, stress = read_aims_output(p)
    os.unlink(p)
    expect = np.array([[0.01, 0.006, 0.005],
                       [0.006, 0.02, 0.004],
                       [0.005, 0.004, 0.03]])
    check("Voigt -> 3x3", stress is not None and np.allclose(stress, expect))


def test_lattice_forces_cubic():
    print("== stress -> lattice forces, cubic cell ==")
    # h = 10 I  ->  V = 1000, h^-T = 0.1 I
    # F = -V * sigma * h^-T = -1000 * 0.01 * 0.1 = -1 per diagonal
    lattice = np.diag([10.0, 10.0, 10.0])
    stress = np.diag([0.01, 0.02, 0.03])
    F = stress_to_lattice_forces(stress, lattice)
    check("F_lat diag", np.allclose(F, np.diag([-1.0, -2.0, -3.0])))


def test_lattice_forces_scale():
    print("== lattice_force_scale damping ==")
    lattice = np.diag([10.0, 10.0, 10.0])
    stress = np.diag([0.01, 0.02, 0.03])
    F = stress_to_lattice_forces(stress, lattice, scale=0.5)
    check("scale applied", np.allclose(F, np.diag([-0.5, -1.0, -1.5])))


def test_lattice_forces_nonorthogonal():
    print("== stress -> lattice forces, non-orthogonal cell (energy check) ==")
    # Verify the conversion is the negative gradient of E(h) = V * sigma : eps
    # (thermodynamics convention: dE = V*sigma:deps, sigma tensile-positive).
    # Numerically: F_lat = -dE/dh.
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 3))
    h[0] += 8.0; h[1] += 5.0; h[2] += 4.0   # make it invertible-ish
    sigma = rng.normal(size=(3, 3))
    sigma = (sigma + sigma.T) / 2.0
    V = abs(np.linalg.det(h))
    F = stress_to_lattice_forces(sigma, h)

    def E(hh):
        eps = (hh - h) @ np.linalg.inv(h)
        return V * np.sum(sigma * eps)  # dE/dh = V*sigma*h^-T

    eps_step = 1e-6
    num = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = np.zeros((3, 3)); d[i, j] = eps_step
            num[i, j] = (E(h + d) - E(h - d)) / (2 * eps_step)
    check("F_lat matches -dE/dh", np.allclose(F, -num, atol=1e-4))


if __name__ == "__main__":
    print("aimsChain FHI-AIMS output parsing tests\n")
    test_forces_with_index()
    test_forces_with_extra_columns()
    test_stress_3x3_ev_per_ang3()
    test_stress_3x3_gpa()
    test_stress_voigt_six()
    test_lattice_forces_cubic()
    test_lattice_forces_scale()
    test_lattice_forces_nonorthogonal()
    print("\n%d checks passed." % PASS)
