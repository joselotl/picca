#!/usr/bin/env python

import scipy as sp
from scipy import random
import fitsio
import argparse
import sys
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, utils

def calc_dmat(p):
    xcf.fill_neighs(p)
    tmp = xcf.dmat(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--drq', type = str, default = None, required=True,
                        help = 'drq')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--rp-max', type = float, default = 200., required=False,
                        help = 'max rp [h^-1 Mpc]')

    parser.add_argument('--rp-min', type = float, default = -200., required=False,
                        help = 'min rp [h^-1 Mpc]')

    parser.add_argument('--rt-max', type = float, default = 200., required=False,
                        help = 'max rt [h^-1 Mpc]')

    parser.add_argument('--np', type = int, default = 100, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--lambda-abs', type = str, default = "LYA", required=False,
                        help = 'name of the absorption in picca.constants')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--rej', type = float, default = 1., required=False,
                    help = 'fraction rejected: -1=no rejection, 1=all rejection')

    parser.add_argument('--z-evol-del', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the object field')

    parser.add_argument('--z-min-obj', type = float, default = None, required=False,
                    help = 'min redshift for object field')

    parser.add_argument('--z-max-obj', type = float, default = None, required=False,
                    help = 'max redshift for object field')

    parser.add_argument('--z-cut-min', type = float, default = 0., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift higher than z-cut-min')

    parser.add_argument('--z-cut-max', type = float, default = 10., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift smaller than z-cut-min')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--mpi', action="store_true", required=False,
                    help = 'use mpi parallelization')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    print("nproc",args.nproc)

    xcf.rp_max = args.rp_max
    xcf.rp_min = args.rp_min
    xcf.rt_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.nside = args.nside
    xcf.zref = args.z_ref
    xcf.alpha = args.z_evol_del
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    xcf.rej = args.rej

    cosmo = constants.cosmo(args.fid_Om)

    comm = None
    rank = 0 
    size = 1
    
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

    if rank == 0:
        ### Read deltas
        dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs,
            args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec)

        sys.stderr.write("\n")
        print("done, npix = {}\n".format(len(dels)))

        ### Find the redshift range
        if (args.z_min_obj is None):
            dmin_pix = cosmo.r_comoving(zmin_pix)
            dmin_obj = max(0.,dmin_pix+xcf.rp_min)
            args.z_min_obj = cosmo.r_2_z(dmin_obj)
            sys.stderr.write("\r z_min_obj = {}\r".format(args.z_min_obj))
        if (args.z_max_obj is None):
            dmax_pix = cosmo.r_comoving(zmax_pix)
            dmax_obj = max(0.,dmax_pix+xcf.rp_max)
            args.z_max_obj = cosmo.r_2_z(dmax_obj)
            sys.stderr.write("\r z_max_obj = {}\r".format(args.z_max_obj))

        ### Read objects
        objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
        sys.stderr.write("\n")

    if comm is not None:
        dels = comm.bcast(dels, root=0)
        ndels = comm.bcast(ndels, root=0)
        objs = comm.bcast(objs, root=0)
        z_min_pix = comm.bcast(z_min_pix, root=0)
        z_min_obj = comm.bcast(z_min_obj, root=0)

    xcf.objs = objs
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels

    ###
    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)

    xcf.counter = Value('i',0)

    xcf.lock = Lock()


    cpu_data = list(dels.keys())[rank::size]
    cpu_data = sorted(cpu_data)
    cpu_data = [cpu_data[i::args.nproc] for i in range(args.nproc)]

    random.seed(0)
    pool = Pool(processes=args.nproc)
    dm = pool.map(calc_dmat,cpu_data)
    pool.close()

    dm = sp.array(dm)
    wdm =dm[:,0].sum(axis=0)
    npairs=dm[:,2].sum(axis=0)
    npairs_used=dm[:,3].sum(axis=0)
    dm=dm[:,1].sum(axis=0)

    if comm is not None:
        dm = comm.gather(dm)
        wdm = comm.gather(wdm)
        npairs = comm.gather(npairs)
        npairs_used = comm.gather(npairs_used)

        dm = sp.array(dm).sum(axis=0)
        wdm = sp.array(wdm).sum(axis=0)
        npairs = sp.array(npairs).sum(axis=0)
        npairs_used = sp.array(npairs_used).sum(axis=0)

    if rank == 0:
        w = wdm>0
        dm[w,:] /= wdm[w,None]

        out = fitsio.FITS(args.out,'rw',clobber=True)
        head = {}
        head['REJ']=args.rej
        head['RPMAX']=xcf.rp_max
        head['RPMIN']=xcf.rp_min
        head['RTMAX']=xcf.rt_max
        head['Z_CUT_MAX']=xcf.z_cut_max
        head['Z_CUT_MIN']=xcf.z_cut_min
        head['NT']=xcf.nt
        head['NP']=xcf.np
        head['NPROR']=npairs
        head['NPUSED']=npairs_used

        out.write([wdm,dm],names=['WDM','DM'],header=head)
        out.close()
