#!/usr/bin/env python
import subprocess, os, shutil, time, glob, sys
from pathlib import Path
from typing import List
import pickle as cp
import numpy as np

from aimsChain.string_path import StringPath
from aimsChain.neb_path import NebPath
from aimsChain.gs_path import GrowingStringPath
from aimsChain.node import Node
from aimsChain.aimsio import read_aims
from aimsChain.config import Control
from aimsChain.interpolate import get_t
from aimsChain.aimsio import write_mapped_aims, write_xyz, write_aims
"""
do a single aims run for the path given
faster now
"""

def submitter(sbatch_script: str) -> str | None:
    try:
        home=Path.cwd()
        path="/".join(sbatch_script.split("/")[0:-1])+'/'
        jobname = sbatch_script.split("/")[-1]
        os.chdir(path)
        result = subprocess.run(
            ["sbatch", jobname],
            capture_output=True,
            text=True,
            check=True
        )

        job_id = result.stdout.strip().split()[-1]
        os.chdir(home)
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {sbatch_script}: {e.stderr.strip()}")
        return None

def waitter(job_ids: List[str], poll_interval: int = 1):
    if not job_ids:
        print("No jobs to wait for.")
        return
    job_list = ",".join(job_ids)

    while True:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_list],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            print("All jobs have left the queue.")
            break
        time.sleep(poll_interval)

    print("All jobs finished.")

def submit_and_wait(scripts):
    job_ids = []
    for script in scripts:
        jid = submitter(script)
        if jid:
            job_ids.append(jid)
    if not job_ids:
        print("No jobs were submitted successfully. Exiting.")
        sys.exit(1)
    waitter(job_ids)
    return True

def run_aims(paths):
    global control
    ## fixme devolver lo que tenian ellos...
    if control.external_job:
        if not os.path.isfile(control.external_jobname):
            print("Missing ",control.external_jobname, " Fix it")
            exit()
        ## prepare 
        scripts = []
        for path in paths:
            if path[-1] == "/":
                path = path[:-1]
            filename = path[path.rfind('/')+1:]+'.out'
            os.system(f'cp {control.external_jobname} {path}/')
            os.system(f'sed -i -e "s|aims.out|{filename}|g" {path}/{control.external_jobname}')
            os.system(f'sed -i -e "s|title|{filename[:-4]}|g" {path}/{control.external_jobname}')
            os.system(f'sed -i -e "s|name|{path}/aims|g" {path}/{control.external_jobname}')
            scripts.append(path+'/'+control.external_jobname)
        submit_and_wait(scripts)

        ##if control.restart:
        ##     save_restart(paths)
    else:
        print("fixme")

"""
Ugly works for initial interpolation of path
"""
def initial_interpolation():
    global control
    global path
    #set the initial and final image
    ininode = Node(param = 0.0, 
                   geometry = read_aims(control.ini))
    finnode = Node(param = 1.0,
                   geometry = read_aims(control.fin))

    #does periodic transformation between initial and final node
    #minimize distance between initial and final image atom-wise
    #maybe this should be moved to another file...runchain is getting too messy
    #leave it like this for now
    #print("D0:", control.periodic_interp ) # these are here for debugging #
    #print("D1:", finnode.geometry.lattice)
    #if control.periodic_interp and finnode.geometry.lattice != None: # original part from python2 has to changed #
    if control.periodic_interp and not np.all(np.array(finnode.geometry.lattice) == 0. ) :
        print("D2:", "inside periodic interpolation")
        lattice = finnode.geometry.lattice
        initial_pos = ininode.positions
        curr_pos = finnode.positions
        fin_pos = []
        for i,atom_pos in enumerate(curr_pos):
            #large separation to start with
            dis = 1000000
            pos = None
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    for c in [-1,0,1]:
                        pos_tmp = (atom_pos +
                                   a*lattice[0] + b*lattice[1] + c*lattice[2])
                        dis_tmp = np.sum(np.array(pos_tmp - initial_pos[i])**2)**0.5
                        if dis_tmp <= dis:
                            pos = pos_tmp
                            dis = dis_tmp
            fin_pos.append(pos)

        finnode.positions = fin_pos                    


    #fix the initial and final node    
    ininode.fixed = True
    finnode.fixed = True

    if control.use_gs:
        path.nodes = [ininode, finnode]
        path.add_lower()
        path.add_upper()
        return

    #parse the externl geometry
    try:
        nodes = [ininode]
        if control.ext_geo and os.path.isfile(str(control.ext_geo)):
            geo = open(control.ext_geo)
            lines = geo.readlines()
            geo.close()
            for line in lines:
                inp = line.split()
                if inp == []:
                    continue
                elif inp[0][0] == '#':
                    continue
                else:
                    if os.path.isfile(inp[0]):
                        nodes.append(Node(param = 0.5, 
                                          geometry = read_aims(inp[0]), 
                                          fixed = False))
            nodes.append(finnode)
            geos = []
            for node in nodes:
                geos.append(node.positions)
            t = get_t(geos)
            for i in range(len(t)):
                nodes[i].param = t[i]
            path.nodes = nodes
            #if resample is turned on and n_image does not equal to 
            #number of images from external source,
            #then resample the path
            if control.resample and control.nimage != (len(nodes)-2):
                path.interpolate(control.nimage)

    except:

        print('!Error interprating the external geometries\n')
        print('!Using standard interpolation method for initial geometries\n')
        nodes = [ininode, finnode]


    #if there were no external geometry, linear interpolate the image
    if len(nodes) <= 2:
        path.nodes = [ininode, finnode]
        #interpolate the images
        path.interpolate(control.nimage)

def save_restart(path):
    global restart_stage
    global force
    restart = open("iterations/restart_file", 'wb')
    cp.dump((path, restart_stage, force), restart)
    restart.close()

def read_restart():
    global path_to_run
    global force
    global restart_stage
    global path
    file_exist = os.path.isfile("iterations/restart_file")
    if file_exist:
        restart = open("iterations/restart_file", 'rb')
        path_to_run, restart_stage, force = cp.load(restart)
        path.read_path("iterations/path.dat")
        path.load_nodes()
        restart.close()
    return file_exist

def write_current(final = False):
    global path
    if not final:
        dir_name = "iteration%04d" % path.runs
        dir_name = os.path.join("paths", dir_name)
    else:
        dir_name = "optimized"
        
    try:
        os.mkdir(dir_name)
    except OSError:
        pass
    ener = []
    climb = []
    fixed = []
    write_xyz(os.path.join(dir_name, "path.xyz"), path, control.xyz_lattice)
    for i,node in enumerate(path.nodes):
        ener.append(node.ener)
        climb.append(node.climb)
        fixed.append(node.fixed)
        i += 1
        file_name = os.path.join(dir_name, "image%03d.in" % i)
        if not final:
            if control.map_unit_cell:
                write_mapped_aims(file_name, node.geometry)
            else:
                write_aims(file_name, node.geometry)
    ener = np.array(ener) - ener[0]
    enerfile = open(os.path.join(dir_name, "ener.lst"), 'w')
    enerfile.write("#image num.\tenergy(eV)\t status\n")
    for i,energy in enumerate(ener):
        enerfile.write("image%03d\t%.10f" % (i+1, energy))
        if climb[i]:
            enerfile.write("\t CLIMB")
        elif fixed[i]:
            enerfile.write("\t FIXED")
        else:
            enerfile.write("\t NORMAL")
        enerfile.write("\n")
    enerfile.close()
    pathfile = open(os.path.join(dir_name, "path.lst") ,'w')
    pathfile.write("#image num.\tpath\t\t\t\t\t\t\t status\n")
    for i,item in enumerate(path.get_paths()):
        pathfile.write("image%03d\t%s" % (i+1,item))
        if climb[i]:
            pathfile.write("\t CLIMB")
        elif fixed[i]:
            pathfile.write("\t FIXED")
        else:
            pathfile.write("\t NORMAL")

        pathfile.write("\n")
    pathfile.close()
        
    
def main():
    global control
    global path_to_run
    global force
    global restart_stage
    global path

    force = 10.0
    control = Control()

    #print(control)

    if control.method == "neb":
        path = NebPath(control=control)
    else:
        path = StringPath(control=control)
    restart_stage = "mep"


    print(control.restart)
    print(control.use_gs)


    is_restart = control.restart and read_restart()

    
    # no lo estamos usando
    if control.use_gs:
        if not is_restart or restart_stage == "growing":
            path = GrowingStringPath(control=control)
            read_restart()
            restart_stage = "growing"
            growing = True
            gsforcelog = open("growing_forces.log", 'a')


    #check if the system is a restart
    ## lo queremos
    if not is_restart:
        for directory in ["paths", "iterations", "optimized"]:
            if os.path.isdir(directory):
                shutil.rmtree(directory)
        os.mkdir("paths")


        initial_interpolation()

        #write directory for images
        path_to_run = path.write_all_node() ## write all the geometries in for iter0
        path.write_path("iterations/path.dat")

        if control.use_gs: 
            gsforcelog.write("Iteration\tResidual force\t\tLower end force\t\tUpper end force \n")
            gsforcelog.flush()

    print(restart_stage)
    if restart_stage == "growing":
        while growing:

            run_aims(path_to_run)
            path.load_nodes()
            force,low_force,high_force = path.move_nodes()
            write_current()
            curr_runs = path.runs

            if low_force < control.gs_thres:
                growing = path.add_lower() and growing
            if high_force < control.gs_thres:
                growing = path.add_upper() and growing

            path.add_runs()
            path_to_run = path.write_node()

            gsforcelog.write('iteration%04d\t%16.8f\t%16.8f\t%16.8f \n' %
                           (curr_runs,force,low_force, high_force))
            gsforcelog.flush()
            path.write_path("iterations/path.dat")
        run_aims(path_to_run)
        write_current()
        path.write_path("iterations/path.dat")


        force = 10.0
        gsforcelog.write("Path is grown.\n")

        gsforcelog.close()
        path.write_path("iterations/path.dat")

        grown_path = Path('grownstring')
        grown_path.mkdir(parents=True, exist_ok=True)

        for i, directory in enumerate(path.get_paths(), start=1):
            shutil.copytree(directory, grown_path / f"image{i:03d}", dirs_exist_ok=True)
        write_xyz("grownstring/path.xyz", path, control.xyz_lattice)

        restart_stage = "grown"

    if restart_stage == "grown":

        if control.method == "neb":
            path = NebPath(control=control)
        else:
            path = StringPath(control=control)
        path.read_path("iterations/path.dat")
        path.load_nodes()

        #resample the path
        path.interpolate(control.nimage)
        path.add_runs()
        path_to_run = path.write_all_node()

        restart_stage = "mep"

    forcelog = open("forces.log", 'a')
    if not is_restart:
        forcelog.write("#Residual Forces in the system:\n")
        forcelog.flush()


    if restart_stage == "mep":
        while force > control.thres:
            run_aims(path_to_run)
            path.load_nodes()
            force = path.move_nodes()
            write_current()
            curr_runs = path.runs
            if force > control.thres:
                path.add_runs()
                path_to_run = path.write_node()
            forcelog.write('iteration%04d\t%16.10f\n' % (curr_runs,force))
            forcelog.flush()
            path.write_path("iterations/path.dat")
        force = 10.0
        forcelog.write("System has converged.\n")

    forcelog.close()

    if control.use_climb:
        forcelog = open("climbing_forces.log", 'a')
        if restart_stage != "CI":
            path.find_climb()
            path.add_runs()
            if control.climb_control != "control.in":
                path_to_run = path.write_all_node(control.climb_control)
            else:
                path_to_run = path.write_node()

            forcelog.write("#Residual Forces in the Climbing image:\n")
            forcelog.flush()
            restart_stage = "CI"


        while force > control.climb_thres:
            run_aims(path_to_run)
            path.load_nodes()
            force = path.move_climb()
            write_current()
            curr_runs = path.runs
            if force > control.climb_thres:
                path.add_runs()
                path_to_run = path.write_node(control.climb_control)
            forcelog.write('iteration%04d\t%16.16f \n' % (curr_runs, force))
            forcelog.flush()
            path.write_path("iterations/path.dat")
        forcelog.write('Climbing image has converged.\n')
        forcelog.close()

    opt_path = Path("optimized")
    opt_path.mkdir(parents=True, exist_ok=True)

    for i, directory in enumerate(path.get_paths(), start=1):
        shutil.copytree(directory, opt_path / f"image{i:03d}", dirs_exist_ok=True)

    write_current(True)

if __name__ == "__main__":
    os.system("rm -rf iterations paths optimized forces.log")
    main()

