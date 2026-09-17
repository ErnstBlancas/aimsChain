#!/usr/bin/env python3
"""
run_mgo_chain.py — Minimum-energy path between compressed and expanded MgO
using aimsChain with a cheap custom ASE double-well calculator.

Every node in the chain is free to relax BOTH its lattice parameter *a*
and its atomic positions.  The calculator has two minima at a=3.98 Å and
a=4.25 Å, separated by a 5 eV barrier.

Lattice relaxation is enabled — the string method operates on the full
(3N + 9)-dimensional configuration space.

Usage:
    cd examples/mgo_chain
    python run_mgo_chain.py
"""

import os
import sys
import shutil
import numpy as np

# ---------------------------------------------------------------------------
# aimsChain from repo source
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))

from aimsChain.string_path import StringPath
from aimsChain.node      import Node
from aimsChain.aimsio    import read_aims, write_aims, write_xyz
from aimsChain.config    import Control

# ---------------------------------------------------------------------------
# ASE
# ---------------------------------------------------------------------------
from ase import Atoms as AseAtoms
from ase.calculators.calculator import Calculator


# ===================================================================
# Double-well calculator  (cell + atom-position terms)
# ===================================================================

class DoubleWellMgO(Calculator):
    """
    E = E_cell(a) + k_pos * sum |pos_i - ideal(a, i)|^2

    E_cell(a) = A * (a - a1)^2 * (a - a2)^2
      → minima at a1 and a2, barrier at the midpoint.

    The harmonic term anchors each atom to its ideal rock-salt site
    at the *current* lattice constant, so both the cell and the atoms
    contribute to the energy landscape.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, a1=3.98, a2=4.25, barrier_height=5.0, k_pos=1.0):
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        half = (a1 + a2) / 2
        raw = (half - a1) ** 2 * (half - a2) ** 2
        self.A = barrier_height / raw
        self.k_pos = k_pos

    def calculate(self, atoms, properties, system_changes):
        cell = atoms.get_cell().array
        a = cell[0, 0]                     # assume cubic
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # --- ideal rock-salt positions at lattice constant *a* ---
        ideal = self._ideal_rocksalt(a, symbols)

        # --- cell (double-well) term ---
        da1 = a - self.a1
        da2 = a - self.a2
        E_cell = self.A * da1 * da1 * da2 * da2

        # --- atom-position harmonic term ---
        diff = positions - ideal
        E_pos = self.k_pos * np.sum(diff * diff)

        energy = E_cell + E_pos

        # --- atom forces ---
        forces = -2.0 * self.k_pos * diff

        # --- stress from the cell term only ---
        dE_da = self.A * 2.0 * da1 * da2 * (2.0 * a - self.a1 - self.a2)
        V = atoms.get_volume()
        stress = np.zeros((3, 3))
        stress[0, 0] = a * dE_da / V

        self.results = {"energy": energy, "forces": forces, "stress": stress}

    @staticmethod
    def _ideal_rocksalt(a, symbols):
        half = a / 2.0
        mg = np.array([[0, 0, 0], [half, half, 0],
                       [half, 0, half], [0, half, half]])
        ox = np.array([[half, 0, 0], [0, half, 0],
                       [0, 0, half], [half, half, half]])
        pos = []
        for sym in symbols:
            if sym == "Mg":
                pos.append(mg[len(pos) % 4])
            else:
                pos.append(ox[len(pos) % 4])
        return np.array(pos)


# ===================================================================
# Helpers
# ===================================================================

def eval_node(node: Node, calc: DoubleWellMgO, control: Control) -> None:
    """Run the calculator; store energy, atom forces, and lattice forces."""
    geo = node.geometry
    symbols = [a.symbol for a in geo.atoms]
    positions = np.array([a.positions for a in geo.atoms])

    atoms = AseAtoms(symbols=symbols, positions=positions, pbc=True)
    if geo.lattice is not None:
        atoms.set_cell(geo.lattice)
    atoms.calc = calc

    node.ener = atoms.get_potential_energy()
    node.forces = atoms.get_forces()

    if geo.lattice is not None and control.relax_lattice:
        stress = atoms.get_stress(voigt=False)
        geo.lattice_forces = stress * control.lattice_force_scale


def atom_rms(node: Node) -> float:
    """RMS displacement of atoms from their ideal rock-salt positions."""
    geo = node.geometry
    if geo.lattice is None:
        return 0.0
    a = geo.lattice[0, 0]
    symbols = [at.symbol for at in geo.atoms]
    ideal = DoubleWellMgO._ideal_rocksalt(a, symbols)
    pos = np.array([at.positions for at in geo.atoms])
    return float(np.sqrt(np.mean((pos - ideal) ** 2)))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    control = Control()
    print(f"ini  = {control.ini}")
    print(f"fin  = {control.fin}")
    print(f"nimg = {control.nimage}")
    print(f"thres= {control.thres}")
    if control.relax_lattice:
        print(f"lattice  relax={control.relax_lattice}"
              f"  constraint={control.lattice_constraint}"
              f"  scale={control.lattice_force_scale}")

    # ---- Read geometries  (ALL nodes free — no fixed endpoints) ----
    ini_node = Node(param=0.0, geometry=read_aims(control.ini), fixed=False)
    fin_node = Node(param=1.0, geometry=read_aims(control.fin), fixed=False)

    if control.periodic_interp and fin_node.geometry.lattice is not None:
        lattice = fin_node.geometry.lattice
        init_pos = ini_node.positions
        curr_pos = fin_node.positions
        fin_pos = []
        for i, p in enumerate(curr_pos):
            best_d, best = 1e10, None
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        trial = p + a*lattice[0] + b*lattice[1] + c*lattice[2]
                        d = np.linalg.norm(trial - init_pos[i])
                        if d < best_d:
                            best_d, best = d, trial
            fin_pos.append(best)
        fin_node.positions = np.array(fin_pos)

    # ---- Calculator ----
    calc = DoubleWellMgO(a1=3.98, a2=4.25, barrier_height=5.0, k_pos=1.0)
    print(f"Calculator: DoubleWell  a1={calc.a1}  a2={calc.a2}"
          f"  barrier=5.0 eV  k_pos={calc.k_pos}")

    # ---- Build path ----
    path = StringPath(nodes=[ini_node, fin_node], control=control)
    path.interpolate(control.nimage)
    print(f"Path: {path.n_nodes()} nodes  (all free to relax)\n")

    # ---- Output dirs ----
    for d in ("paths", "iterations", "optimized"):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.mkdir(d)
    path.runs = 0
    path.write_all_node()
    path.write_path("iterations/path.dat")

    # ---- String-method loop ----
    with open("forces.log", "w") as flog:
        flog.write("# Iter  residual_force  a_values  atom_rms\n")
        flog.flush()

        force = 10.0
        max_iter = 30

        while force > control.thres and path.runs < max_iter:
            run = path.runs
            a_vals = []
            energies = []
            rms_vals = []
            for node in path.nodes:
                eval_node(node, calc, control)
                a_vals.append(node.geometry.lattice[0, 0])
                energies.append(node.ener)
                rms_vals.append(atom_rms(node))

            energies = np.array(energies)
            rel = energies - energies[0]

            print(f"--- Iteration {run:04d} ---")
            print("  a (Å):     " + " ".join(f"{v:7.4f}" for v in a_vals))
            print("  E_rel (eV):" + " ".join(f"{v:7.4f}" for v in rel))
            print("  atom RMS:  " + " ".join(f"{v:7.4f}" for v in rms_vals))

            force = path.move_nodes()
            print(f"  residual   = {force:.6f} eV/Å\n")

            flog.write(f"{run:04d}  {force:12.6f}  "
                       + " ".join(f"{v:.4f}" for v in a_vals)
                       + "  |  "
                       + " ".join(f"{v:.4f}" for v in rms_vals) + "\n")
            flog.flush()

            if force > control.thres:
                path.add_runs()
                path.write_node()
            path.write_path("iterations/path.dat")
            _write_current(path, control)

        msg = "converged\n" if force <= control.thres \
              else f"stopped after {max_iter} iters\n"
        flog.write("# " + msg)

    # ---- Final output ----
    opt_dir = "optimized"
    if os.path.isdir(opt_dir):
        shutil.rmtree(opt_dir)
    os.mkdir(opt_dir)
    for i, d in enumerate(path.get_paths(), start=1):
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(opt_dir, f"image{i:03d}"),
                            dirs_exist_ok=True)
    _write_current(path, control, final=True)

    print("=" * 60)
    print("Final path  (all nodes free):")
    print(f"{'Node':>5} {'param':>8} {'a (Å)':>8} {'E_rel (eV)':>12} {'atom RMS':>10}")
    for i, node in enumerate(path.nodes):
        a_val = node.geometry.lattice[0, 0]
        rms = atom_rms(node)
        print(f"{i:5d} {node.param:8.4f} {a_val:8.4f}"
              f" {node.ener - path.nodes[0].ener:12.6f} {rms:10.6f}")
    print("=" * 60)


def _write_current(path, control, final=False):
    dname = "optimized" if final else f"paths/iteration{path.runs:04d}"
    os.makedirs(dname, exist_ok=True)
    write_xyz(os.path.join(dname, "path.xyz"), path, control.xyz_lattice)
    for i, node in enumerate(path.nodes, start=1):
        write_aims(os.path.join(dname, f"image{i:03d}.in"), node.geometry)
    energies = np.array([n.ener for n in path.nodes])
    energies -= energies[0]
    with open(os.path.join(dname, "ener.lst"), "w") as f:
        f.write("# image  energy(eV)  status\n")
        for i, (node, e) in enumerate(zip(path.nodes, energies), start=1):
            st = "FIXED" if node.fixed else "FREE"
            f.write(f"image{i:03d}  {e:16.10f}  {st}\n")


if __name__ == "__main__":
    main()
