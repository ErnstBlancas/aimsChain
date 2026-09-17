#!/usr/bin/env python3
"""
run_mgo_chain.py — Minimum-energy path between compressed and expanded
molecular crystal using aimsChain with a cheap custom ASE double-well
calculator.

Every node in the chain is free to relax BOTH its lattice and its atomic
positions.  The calculator has two minima at cells C1 and C2, separated by
a 5 eV barrier.  Atomic positions are harmonically anchored to an
interpolation between the initial and final geometries.

Compatible with organic molecular crystals (S, C, H, etc.) — no element
hardcoding.

Lattice relaxation is enabled — the string method operates on the full
(3N + 9)-dimensional configuration space.  All three lattice vectors
relax freely.

Usage:
    cd examples/molecule_chain
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
# Molecular-crystal calculator  (double-well cell + harmonic atoms)
# ===================================================================

class MolecularCrystal(Calculator):
    """
    E = E_cell(cell) + k_pos * sum_i |pos_i - ref_i(cell)|^2

    E_cell(cell) = A * ||cell - ini_cell||_F^2 * ||cell - fin_cell||_F^2
      → minima at ini_cell and fin_cell, barrier at the midpoint.

    The reference position for each atom is a linear interpolation
    between the initial (ini_pos) and final (fin_pos) geometries,
    controlled by the full 3x3 cell matrix:

        t = ||cell - ini_cell||_F / ||fin_cell - ini_cell||_F
        ref_i = (1 - t) * ini_pos_i + t * fin_pos_i

    At cell = ini_cell the atoms relax toward the initial geometry; at
    cell = fin_cell they relax toward the final geometry.  All three
    lattice vectors contribute to the energy and stress.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, ini_positions, fin_positions,
                 ini_cell, fin_cell, barrier_height=5.0, k_pos=1.0):
        super().__init__()
        self.ini_pos = np.asarray(ini_positions)
        self.fin_pos = np.asarray(fin_positions)
        self.ini_cell = np.asarray(ini_cell, dtype=float)
        self.fin_cell = np.asarray(fin_cell, dtype=float)
        self.k_pos = k_pos
        self.delta_pos = self.fin_pos - self.ini_pos
        delta_cell = self.fin_cell - self.ini_cell
        self.delta_norm = np.sqrt(np.sum(delta_cell * delta_cell))
        # A so E_cell(midpoint) = barrier_height
        self.A = 16.0 * barrier_height / (self.delta_norm ** 4)

    def _reference(self, cell):
        """Linearly interpolate reference positions for cell matrix."""
        diff = cell - self.ini_cell
        t = np.sqrt(np.sum(diff * diff)) / self.delta_norm
        t = np.clip(t, 0.0, 1.0)
        return self.ini_pos + t * self.delta_pos

    def calculate(self, atoms, properties, system_changes):
        cell = atoms.get_cell().array   # 3x3
        positions = atoms.get_positions()

        # --- cell term: double-well in 3x3 cell space ---
        diff_i = cell - self.ini_cell
        diff_f = cell - self.fin_cell
        ni2 = np.sum(diff_i * diff_i)     # ||·||_F^2
        nf2 = np.sum(diff_f * diff_f)
        E_cell = self.A * ni2 * nf2

        # --- harmonic atom-position term ---
        ref = self._reference(cell)
        diff_p = positions - ref
        E_pos = self.k_pos * np.sum(diff_p * diff_p)

        energy = E_cell + E_pos

        forces = -2.0 * self.k_pos * diff_p

        # --- full stress tensor ---
        V = atoms.get_volume()
        dE_dcell = 2.0 * self.A * (nf2 * diff_i + ni2 * diff_f)
        stress = dE_dcell @ cell.T / V

        self.results = {"energy": energy, "forces": forces, "stress": stress}


# ===================================================================
# Helpers
# ===================================================================

def eval_node(node: Node, calc: MolecularCrystal, control: Control) -> None:
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


def atom_rms(node: Node, calc: MolecularCrystal) -> float:
    """RMS displacement of atoms from their interpolated reference positions."""
    geo = node.geometry
    if geo.lattice is None:
        return 0.0
    ref = calc._reference(geo.lattice)
    pos = np.array([at.positions for at in geo.atoms])
    return float(np.sqrt(np.mean((pos - ref) ** 2)))


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

    # Extract reference positions and lattice cells
    ini_positions = np.array([a.positions for a in ini_node.geometry.atoms])
    fin_positions = np.array([a.positions for a in fin_node.geometry.atoms])
    ini_cell = ini_node.geometry.lattice
    fin_cell = fin_node.geometry.lattice

    if control.periodic_interp and fin_node.geometry.lattice is not None:
        lattice = fin_node.geometry.lattice
        init_pos = ini_positions
        curr_pos = fin_positions
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
        fin_positions = np.array(fin_pos)
        # Write unwrapped positions back to geometry atoms
        for i, atom in enumerate(fin_node.geometry.atoms):
            atom.positions = fin_positions[i].tolist()

    # ---- Calculator ----
    calc = MolecularCrystal(
        ini_positions=ini_positions,
        fin_positions=fin_positions,
        ini_cell=ini_cell,
        fin_cell=fin_cell,
        barrier_height=5.0, k_pos=1.0,
    )
    norm_ini = np.sqrt(np.sum(ini_cell * ini_cell))
    norm_fin = np.sqrt(np.sum(fin_cell * fin_cell))
    print(f"Calculator: MolecularCrystal"
          f"  ||ini_cell||={norm_ini:.4f}  ||fin_cell||={norm_fin:.4f}"
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
        flog.write("# Iter  residual_force  cell_norms  atom_rms\n")
        flog.flush()

        force = 10.0
        max_iter = 30

        while force > control.thres and path.runs < max_iter:
            run = path.runs
            cell_norms = []
            energies = []
            rms_vals = []
            for node in path.nodes:
                eval_node(node, calc, control)
                cell = node.geometry.lattice
                cell_norms.append(np.sqrt(np.sum(cell * cell)))
                energies.append(node.ener)
                rms_vals.append(atom_rms(node, calc))

            energies = np.array(energies)
            rel = energies - energies[0]

            print(f"--- Iteration {run:04d} ---")
            print("  ||cell||:  " + " ".join(f"{v:7.4f}" for v in cell_norms))
            print("  E_rel (eV):" + " ".join(f"{v:7.4f}" for v in rel))
            print("  atom RMS:  " + " ".join(f"{v:7.4f}" for v in rms_vals))

            force = path.move_nodes()
            print(f"  residual   = {force:.6f} eV/Å\n")

            flog.write(f"{run:04d}  {force:12.6f}  "
                       + " ".join(f"{v:.4f}" for v in cell_norms)
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
    print("Final path  (all nodes free, full cell relaxation):")
    print(f"{'Node':>5} {'param':>8} {'||cell||':>10} {'E_rel (eV)':>12} {'atom RMS':>10}")
    for i, node in enumerate(path.nodes):
        cell = node.geometry.lattice
        cell_norm = np.sqrt(np.sum(cell * cell))
        rms = atom_rms(node, calc)
        print(f"{i:5d} {node.param:8.4f} {cell_norm:10.4f}"
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
