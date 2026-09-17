# molecule_chain — string-method MEP with FHI-AIMS + lattice relaxation

Minimum-energy path between `ini.in` and `fin.in` (96-atom molecular crystal,
S/C/H, ~920 A^3 cell) computed with the aimsChain string method.  The
endpoints stay fixed at the initial/final geometries; every intermediate
chain node relaxes **both its atomic positions and its lattice cell**,
sampling the potential energy surface to find the optimal path between them
(`--free-endpoints` also relaxes the endpoints).
FHI-AIMS is used only as a single-point provider of energies, forces and
stresses — the optimization is driven entirely by aimsChain.

## Layout

    run_chain.py       driver (string method + dampedBFGS, all nodes free)
    chain.in           aimsChain settings (method, n_images, thresholds, ...)
    control.in         FHI-AIMS single-point input (no relax_geometry,
                       compute_analytical_stress .true.)
    aims.sub           SLURM job template for one FHI-AIMS run per node
    ini.in / fin.in    endpoint geometries
    test_aims_output.py    unit tests for output parsing / lattice forces
    requirements.txt   python dependencies
    setup_env.sh       create the venv on the target machine
    transfer.sh        package everything into a transferable tarball
    .venv/             local python environment (numpy, scipy, ase)

The driver imports the aimsChain library from the repo (`../../..`), the
SLURM submitter/waiter from `tools/runchain.py`, and the two engines:

* `engine=aims` (default): one FHI-AIMS job per node via `sbatch`/`squeue`
  (submitter/waiter already implemented in `tools/runchain.py`).
* `engine=ase` (test only): evaluates every node in-process with an ASE
  calculator and writes synthetic FHI-AIMS-style output files so the
  identical load/optimize code path is exercised without FHI-AIMS.
  Calculator: `--ase-calc doublewell` (default, custom molecular-crystal
  double well) or `--ase-calc lj` (ASE built-in LennardJones).

## Run on the local machine (ASE test, no FHI-AIMS)

    cd aimsChain/examples/molecule_chain
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python test_aims_output.py          # parser unit tests
    .venv/bin/python run_chain.py --engine ase    # full string-method test
    .venv/bin/python run_chain.py --engine ase --ase-calc lj  # with ASE's
                                                  # built-in LennardJones

## Run with FHI-AIMS (SLURM cluster)

1. Edit `aims.sub`:
   - `AIMS_BINARY=/path/to/your/aims.x`
   - `--partition`, `--account`, `--ntasks` to match your cluster.
2. Optional: `python run_chain.py --dry-run` to prepare the node
   directories and print the `sbatch` commands without submitting.
3. Launch:

       .venv/bin/python run_chain.py

   Per iteration the driver submits one FHI-AIMS single point per chain
   node (12 nodes with the current `chain.in`), waits for all of them,
   parses energy/forces/stress from each `<node>.out`, relaxes atoms and
   cell, and repeats until the residual force is below `force_thres`.

## Transfer to a remote machine

    ./transfer.sh            # -> molecule_chain_fhi_aims.tar.gz

The tarball keeps the `aimsChain/examples/molecule_chain` layout and
includes `src/aimsChain` + `tools/runchain.py`, minus run artifacts and
the venv.  On the remote:

    tar xzf molecule_chain_fhi_aims.tar.gz
    cd aimsChain/examples/molecule_chain
    ./setup_env.sh           # recreates .venv + installs requirements

`./transfer.sh --with-venv` also ships the local `.venv`; this only works
when the remote has the same python at the same path, so prefer
`setup_env.sh`.

## Notes

* Lattice forces are the physical cell forces `F_lat = -V * sigma * h^-T`
  (eV/A), converted from the FHI-AIMS stress tensor; `lattice_force_scale`
  damps them (1.0 = undamped).  If the cell diverges on your system, check
  the sign convention of your FHI-AIMS stress output.
* The output parser (`aimsChain.aimsio.read_aims_output`) accepts the
  stress tensor as a 3x3 block or a 6-component Voigt line, and converts
  GPa to eV/A^3 when the header says so.
