#!/usr/bin/env python3
"""
run_chain.py — FHI-AIMS string-method driver for the molecule_chain example.

Finds the minimum-energy path between ini.in and fin.in (96-atom molecular
crystal, S/C/H) with full relaxation of the atomic positions AND the lattice
cell.  The endpoints stay fixed at the initial/final geometries; every
intermediate node relaxes atoms + cell, sampling the PES to find the optimal
path between them (--free-endpoints also relaxes the endpoints).
FHI-AIMS is used only as a single-point provider of energies, forces
and stresses; every step of the relaxation is driven by aimsChain (string
method + dampedBFGS).

Two evaluation engines:

  engine = "aims"  (default)
      Writes geometry.in + control.in + aims.sub into every chain-node
      directory and submits one FHI-AIMS job per node through the SLURM
      submitter/waiter implemented in tools/runchain.py
      (submitter -> sbatch, waiter -> squeue).  Requires a SLURM cluster
      and the FHI-AIMS binary (see aims.sub).

  engine = "ase"   (test only, no FHI-AIMS)
      Evaluates every node in-process with an ASE calculator (the
      double-well MolecularCrystal used by the original run_mgo_chain.py)
      and writes synthetic FHI-AIMS-style output files, so the identical
      load/optimize code path is exercised end-to-end.

Usage:
    cd examples/molecule_chain
    python run_chain.py                            # FHI-AIMS via SLURM
    python run_chain.py --engine ase --max-iter 60 # ASE double-well test
    python run_chain.py --engine aims --dry-run    # prepare node dirs +
                                                   # print sbatch commands
"""

import argparse
import importlib.util
import os
import shutil
import sys

import numpy as np

# ---------------------------------------------------------------------------
# aimsChain from repo source
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

from aimsChain.string_path import StringPath
from aimsChain.node import Node
from aimsChain.aimsio import read_aims, write_aims, write_xyz
from aimsChain.config import Control

from ase.calculators.calculator import Calculator


# ===================================================================
# Molecular-crystal calculator (double-well cell + harmonic atoms)
# -------------------------------------------------------------------
# Used only by engine="ase".  E = E_cell(cell) + k_pos * sum |pos-ref|^2
# with minima at the ini/fin cells; the atoms are anchored to a linear
# interpolation between the initial and final geometries.
# ===================================================================

class MolecularCrystal(Calculator):
    """ASE calculator with energy / forces / stress (3x3, eV/A^3)."""

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
        diff = cell - self.ini_cell
        t = np.sqrt(np.sum(diff * diff)) / self.delta_norm
        t = np.clip(t, 0.0, 1.0)
        return self.ini_pos + t * self.delta_pos

    def calculate(self, atoms=None, properties=None, system_changes=()):
        cell = atoms.get_cell().array          # 3x3
        positions = atoms.get_positions()

        diff_i = cell - self.ini_cell
        diff_f = cell - self.fin_cell
        ni2 = np.sum(diff_i * diff_i)          # ||.||_F^2
        nf2 = np.sum(diff_f * diff_f)
        E_cell = self.A * ni2 * nf2

        ref = self._reference(cell)
        diff_p = positions - ref
        E_pos = self.k_pos * np.sum(diff_p * diff_p)

        energy = E_cell + E_pos
        forces = -2.0 * self.k_pos * diff_p

        V = atoms.get_volume()
        dE_dcell = 2.0 * self.A * (nf2 * diff_i + ni2 * diff_f)
        stress = dE_dcell @ cell.T / V         # eV/A^3, tensile positive

        self.results = {"energy": energy, "forces": forces, "stress": stress}


# ===================================================================
# Engine: FHI-AIMS through the SLURM submitter/waiter
# ===================================================================

def _load_runchain():
    """Import tools/runchain.py (submitter / waiter) without a package."""
    spec = importlib.util.spec_from_file_location(
        "runchain_tools", os.path.join(_REPO, "tools", "runchain.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_aims_nodes(paths, control, dry_run=False):
    """
    Run one FHI-AIMS single-point per chain-node directory using the
    submitter and waiter implemented in tools/runchain.py.

    For every node dir: copy the external job template (control.external_jobname,
    e.g. aims.sub), substitute the three placeholders it expects
    ("aims.out" -> <node>.out, "title" -> job label, "name" -> <node>/aims),
    then submit all jobs with submitter() and wait with waiter().
    """
    rch = _load_runchain()
    if not os.path.isfile(control.external_jobname):
        raise FileNotFoundError(
            "Missing external job template '%s' — expected in the example "
            "directory (see aims.sub)." % control.external_jobname)

    scripts = []
    for path in paths:
        if path.endswith("/"):
            path = path[:-1]
        filename = path[path.rfind("/") + 1:] + ".out"
        dst = os.path.join(path, control.external_jobname)
        shutil.copy(control.external_jobname, dst)
        with open(dst) as f:
            content = f.read()
        content = content.replace("aims.out", filename)
        content = content.replace("title", filename[:-4])
        content = content.replace("name", path + "/aims")
        with open(dst, "w") as f:
            f.write(content)
        scripts.append(dst)

    if dry_run:
        print("[dry-run] prepared %d FHI-AIMS job script(s):" % len(scripts))
        for s in scripts:
            print("   sbatch", s)
        return
    rch.submit_and_wait(scripts)


# ===================================================================
# Engine: ASE calculator (test only) — writes synthetic FHI-AIMS output
# ===================================================================

def _write_fake_aims_output(filename, n_atoms, energy, forces, stress):
    """Write a minimal FHI-AIMS-style output readable by read_aims_output()."""
    lines = ["  Invoking FHI-aims ...\n",
             "  | Number of atoms                   :  %d\n" % n_atoms,
             "  ...scf iterations...\n",
             "  | Total energy corrected        :   %.12E eV\n" % energy,
             "  ...\n",
             "  Total atomic forces (unitary forces cleaned)\n"]
    for i, f in enumerate(forces, start=1):
        lines.append("  %4d  %16.10E  %16.10E  %16.10E\n"
                     % (i, f[0], f[1], f[2]))
    lines.append("  Stress tensor (eV/A^3):\n")
    for row in stress:
        lines.append("  %16.10E  %16.10E  %16.10E\n"
                     % (row[0], row[1], row[2]))
    with open(filename, "w") as f:
        f.writelines(lines)


def run_ase_nodes(paths, calc, control):
    """Evaluate each node dir in-process with an ASE calculator (built-in
    or custom) and write a synthetic FHI-AIMS output file per node, so the
    standard path.load_nodes() -> move_nodes() pipeline runs unchanged."""
    from ase import Atoms as AseAtoms

    for path in paths:
        if path.endswith("/"):
            path = path[:-1]
        geo = read_aims(os.path.join(path, "geometry.in"))
        symbols = [a.symbol for a in geo.atoms]
        positions = np.array([a.positions for a in geo.atoms])
        atoms = AseAtoms(symbols=symbols, positions=positions, pbc=True)
        if geo.lattice is not None:
            atoms.set_cell(geo.lattice)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        _write_fake_aims_output(
            os.path.join(path, os.path.basename(path) + ".out"),
            len(geo.atoms), energy, forces, stress)


# ===================================================================
# Helpers
# ===================================================================

def _write_current(path, control, final=False):
    dname = "optimized" if final else "paths/iteration%04d" % path.runs
    os.makedirs(dname, exist_ok=True)
    write_xyz(os.path.join(dname, "path.xyz"), path, control.xyz_lattice)
    for i, node in enumerate(path.nodes, start=1):
        write_aims(os.path.join(dname, "image%03d.in" % i), node.geometry)
    energies = np.array([n.ener for n in path.nodes])
    energies -= energies[0]
    with open(os.path.join(dname, "ener.lst"), "w") as f:
        f.write("# image  energy(eV)  status\n")
        for i, (node, e) in enumerate(zip(path.nodes, energies), start=1):
            st = "FIXED" if node.fixed else "FREE"
            f.write("image%03d  %16.10f  %s\n" % (i, e, st))


def _unwrap_periodic(ini_node, fin_node):
    """Map the final geometry onto the periodic images closest to the initial
    one (atom-by-atom), so the interpolation follows the shortest path."""
    lattice = fin_node.geometry.lattice
    init_pos = np.array([a.positions for a in ini_node.geometry.atoms])
    curr_pos = np.array([a.positions for a in fin_node.geometry.atoms])
    fin_pos = []
    for i, p in enumerate(curr_pos):
        best_d, best = 1e10, None
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    trial = p + a * lattice[0] + b * lattice[1] + c * lattice[2]
                    d = np.linalg.norm(trial - init_pos[i])
                    if d < best_d:
                        best_d, best = d, trial
        fin_pos.append(best)
    for i, atom in enumerate(fin_node.geometry.atoms):
        atom.positions = fin_pos[i].tolist()


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["aims", "ase"], default="aims",
                    help="evaluation backend: 'aims' = FHI-AIMS via SLURM "
                         "submitter/waiter (default), 'ase' = in-process ASE "
                         "calculator (test only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="aims engine: prepare node dirs and job scripts, "
                         "print the sbatch commands, do not submit")
    ap.add_argument("--max-iter", type=int, default=100,
                    help="maximum string-method iterations (default 100)")
    ap.add_argument("--n-image", type=int, default=None,
                    help="override n_images from chain.in")
    ap.add_argument("--free-endpoints", action="store_true",
                    help="also relax the initial/final nodes.  Default: keep "
                         "them fixed at ini.in / fin.in, relax every "
                         "intermediate node (atoms + lattice)")
    ap.add_argument("--ase-calc", choices=["doublewell", "lj"], default="doublewell",
                    help="ASE test calculator for engine=ase: 'doublewell' "
                         "(custom molecular-crystal double well) or 'lj' "
                         "(ASE built-in LennardJones)")
    args = ap.parse_args()

    control = Control()
    if args.n_image is not None:
        control.nimage = args.n_image

    print("== molecule_chain: FHI-AIMS string method ==")
    print("  ini    = %s" % control.ini)
    print("  fin    = %s" % control.fin)
    print("  nimage = %d   thres = %g   engine = %s"
          % (control.nimage, control.thres, args.engine))
    if control.relax_lattice:
        print("  lattice relax = %s  constraint = %s  force scale = %g"
              % (control.relax_lattice, control.lattice_constraint,
                 control.lattice_force_scale))
    if args.engine == "aims":
        print("  external job template = %s" % control.external_jobname)

    # ---- Read endpoints ----
    # Default (standard string method): the endpoints stay fixed at
    # ini.in / fin.in; every intermediate node relaxes its atomic
    # positions AND its lattice cell, sampling the PES to find the
    # optimal path between the two geometries.  --free-endpoints also
    # relaxes the endpoints.
    fixed_end = not args.free_endpoints
    ini_node = Node(param=0.0, geometry=read_aims(control.ini), fixed=fixed_end)
    fin_node = Node(param=1.0, geometry=read_aims(control.fin), fixed=fixed_end)

    if control.periodic_interp and fin_node.geometry.lattice is not None:
        _unwrap_periodic(ini_node, fin_node)

    # ---- Build path ----
    path = StringPath(nodes=[ini_node, fin_node], control=control)
    path.interpolate(control.nimage)
    print("  path nodes = %d   (intermediates relax atoms + lattice%s)"
          % (path.n_nodes(),
             "" if fixed_end else "; endpoints also free"))

    # ---- ASE test calculator ----
    calc = None
    if args.engine == "ase":
        if args.ase_calc == "lj":
            from ase.calculators.lj import LennardJones
            calc = LennardJones()
            print("  ASE test calculator: built-in LennardJones (epsilon=%.4f eV,"
                  " sigma=%.3f A)" % (calc.parameters["epsilon"],
                                      calc.parameters["sigma"]))
        else:
            ini_positions = np.array([a.positions for a in ini_node.geometry.atoms])
            fin_positions = np.array([a.positions for a in fin_node.geometry.atoms])
            calc = MolecularCrystal(
                ini_positions=ini_positions,
                fin_positions=fin_positions,
                ini_cell=ini_node.geometry.lattice,
                fin_cell=fin_node.geometry.lattice,
                barrier_height=5.0, k_pos=1.0)
            print("  ASE test calculator: MolecularCrystal double-well")

    # ---- Output dirs ----
    for d in ("paths", "iterations", "optimized"):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.mkdir(d)
    path.runs = 0
    path.write_all_node()
    path.write_path("iterations/path.dat")

    # ---- String-method loop ----
    path_to_run = path.get_paths()
    with open("forces.log", "w") as flog:
        flog.write("# Iter  residual_force  cell_norms\n")
        flog.flush()

        force = 10.0
        while force > control.thres and path.runs < args.max_iter:
            run = path.runs

            if args.engine == "aims":
                run_aims_nodes(path_to_run, control, dry_run=args.dry_run)
                if args.dry_run:
                    return
            else:
                run_ase_nodes(path_to_run, calc, control)

            path.load_nodes()
            force = path.move_nodes()

            cell_norms = [np.sqrt(np.sum(n.geometry.lattice * n.geometry.lattice))
                          for n in path.nodes]
            energies = np.array([n.ener for n in path.nodes])
            rel = energies - energies[0]

            print("--- Iteration %04d ---" % run)
            print("  ||cell||:  " + " ".join("%7.4f" % v for v in cell_norms))
            print("  E_rel (eV):" + " ".join("%7.4f" % v for v in rel))
            print("  residual   = %.6f eV/A\n" % force)

            flog.write("%04d  %12.6f  "
                       % (run, force)
                       + " ".join("%.4f" % v for v in cell_norms) + "\n")
            flog.flush()

            if force > control.thres:
                path.add_runs()
                path_to_run = path.write_node()
            path.write_path("iterations/path.dat")
            _write_current(path, control)

        msg = ("converged\n" if force <= control.thres
               else "stopped after %d iters\n" % args.max_iter)
        flog.write("# " + msg)

    # ---- Final output ----
    opt_dir = "optimized"
    if os.path.isdir(opt_dir):
        shutil.rmtree(opt_dir)
    os.mkdir(opt_dir)
    for i, d in enumerate(path.get_paths(), start=1):
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(opt_dir, "image%03d" % i),
                            dirs_exist_ok=True)
    _write_current(path, control, final=True)

    print("=" * 64)
    print("Final path  (endpoints fixed, intermediates fully relaxed):")
    print("%5s %8s %10s %12s" % ("Node", "param", "||cell||", "E_rel (eV)"))
    for i, node in enumerate(path.nodes):
        cell = node.geometry.lattice
        cell_norm = np.sqrt(np.sum(cell * cell))
        print("%5d %8.4f %10.4f %12.6f"
              % (i, node.param, cell_norm, node.ener - path.nodes[0].ener))
    print("=" * 64)


if __name__ == "__main__":
    main()
