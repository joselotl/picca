#!/usr/bin/env python
from __future__ import print_function
import sys
import fitsio
import argparse
import scipy as sp
from scipy.interpolate import interp1d
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, utils, io
from picca.utils import print

def calc_wickT(p):
    if cf.x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    sp.random.seed(p[0])
    tmp = cf.wickT(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the wick covariance for the auto-correlation of forests')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--in-dir2', type=str, default=None, required=False,
        help='Directory to 2nd delta files')

    parser.add_argument('--rp-min', type=float, default=0., required=False,
        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max', type=float, default=200., required=False,
        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max', type=float, default=200., required=False,
        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np', type=int, default=50, required=False,
        help='Number of r-parallel bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of r-transverse bins')

    parser.add_argument('--z-cut-min', type=float, default=0., required=False,
        help='Use only pairs of forest x object with the mean of the last absorber \
        redshift and the object redshift larger than z-cut-min')

    parser.add_argument('--z-cut-max', type=float, default=10., required=False,
        help='Use only pairs of forest x object with the mean of the last absorber \
        redshift and the object redshift smaller than z-cut-max')

    parser.add_argument('--lambda-abs', type=str, default='LYA', required=False,
        help='Name of the absorption in picca.constants defining the redshift of the delta')

    parser.add_argument('--lambda-abs2', type=str, default=None, required=False,
        help='Name of the absorption in picca.constants defining the redshift of the 2nd delta')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--max-diagram', type=int, default=3, required=False,
        help='Maximum diagram to compute')

    parser.add_argument('--cf1d', type=str, required=True,
        help='1D auto-correlation of pixels from the same forest file: do_cf1d.py')

    parser.add_argument('--cf1d2', type=str, default=None, required=False,
        help='1D auto-correlation of pixels from the same forest file of the 2nd delta field: do_cf1d.py')

    parser.add_argument('--cf', type=str, default=None, required=False,
        help='3D auto-correlation of pixels from different forests: picca_cf.py')

    parser.add_argument('--remove-same-half-plate-close-pairs', action='store_true', required=False,
        help='Reject pairs in the first bin in r-parallel from same half plate')

    parser.add_argument('--rej', type=float, default=1., required=False,
        help='Fraction of rejected pairs: -1=no rejection, 1=all rejection')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    print("nproc",args.nproc)

    cf.rp_max = args.rp_max
    cf.rt_max = args.rt_max
    cf.rp_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.np = args.np
    cf.nt = args.nt
    cf.nside = args.nside
    cf.zref = args.z_ref
    cf.alpha = args.z_evol
    cf.alpha2 = args.z_evol
    cf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    cf.rej = args.rej
    cf.remove_same_half_plate_close_pairs = args.remove_same_half_plate_close_pairs
    cf.max_diagram = args.max_diagram

    cosmo = constants.cosmo(args.fid_Om)

    ### Read data
    data, ndata, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, cf.nside, cf.lambda_abs, cf.alpha, cf.zref, cosmo, nspec=args.nspec)
    for p,datap in data.items():
        for d in datap:
            for k in ['co','de','order','iv','diff','m_SNR','m_reso','m_z','dll']:
                setattr(d,k,None)
    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata
    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,zmin_pix)
    sys.stderr.write("\n")
    print("done, npix = {}".format(cf.npix))

    ### Load cf1d for data
    h = fitsio.FITS(args.cf1d)
    head = h[1].read_header()
    llmin = head['LLMIN']
    llmax = head['LLMAX']
    dll = head['DLL']
    nv1d   = h[1]['nv1d'][:]
    cf.v1d = h[1]['v1d'][:]
    ll = llmin + dll*sp.arange(len(cf.v1d))
    cf.v1d = interp1d(ll[nv1d>0],cf.v1d[nv1d>0],kind='nearest',fill_value='extrapolate')

    nb1d   = h[1]['nb1d'][:]
    cf.c1d = h[1]['c1d'][:]
    cf.c1d = interp1d((ll-llmin)[nb1d>0],cf.c1d[nb1d>0],kind='nearest',fill_value='extrapolate')
    cf.v1d2 = cf.v1d
    cf.c1d2 = cf.c1d
    h.close()

    ### Load cf
    if not args.cf is None:
        h = fitsio.FITS(args.cf)
        head = h[1].read_header()
        cf.cf_np = head['NP']
        cf.cf_nt = head['NT']
        cf.cf_rp_min = head['RPMIN']
        cf.cf_rp_max = head['RPMAX']
        cf.cf_rt_max = head['RTMAX']
        cf.cf_angmax = utils.compute_ang_max(cosmo,cf.cf_rt_max,zmin_pix)
        da = h[2]['DA'][:]
        we = h[2]['WE'][:]
        da = (da*we).sum(axis=0)
        we = we.sum(axis=0)
        w = we>0.
        da[w] /= we[w]
        cf.cf = da.copy()
        h.close()

    ### Read data 2
    if args.in_dir2 or args.lambda_abs2:
        cf.x_correlation = True
        cf.alpha2 = args.z_evol2
        if args.in_dir2 is None:
            args.in_dir2 = args.in_dir
        if args.lambda_abs2:
            cf.lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
        else:
            cf.lambda_abs2 = cf.lambda_abs

        data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir2, cf.nside, cf.lambda_abs2, cf.alpha2, cf.zref, cosmo, nspec=args.nspec)
        for p,datap in data2.items():
            for d in datap:
                for k in ['co','de','order','iv','diff','m_SNR','m_reso','m_z','dll']:
                    setattr(d,k,None)
        cf.data2 = data2
        cf.ndata2 = ndata2
        cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,zmin_pix,zmin_pix2)
        print("")
        print("done, npix = {}".format(len(data2)))

        ### Load cf1d for data2
        h = fitsio.FITS(args.cf1d2)
        head = h[1].read_header()
        llmin = head['LLMIN']
        llmax = head['LLMAX']
        dll = head['DLL']
        nv1d   = h[1]['nv1d'][:]
        cf.v1d2 = h[1]['v1d'][:]
        ll = llmin + dll*sp.arange(len(cf.v1d2))
        cf.v1d2 = interp1d(ll[nv1d>0],cf.v1d2[nv1d>0],kind='nearest',fill_value='extrapolate')

        nb1d   = h[1]['nb1d'][:]
        cf.c1d2 = h[1]['c1d'][:]
        cf.c1d2 = interp1d((ll-llmin)[nb1d>0],cf.c1d2[nb1d>0],kind='nearest',fill_value='extrapolate')
        h.close()


    cf.counter = Value('i',0)
    cf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(list(data.keys()))):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    pool = Pool(processes=args.nproc)
    print(" \nStarting\n")
    wickT = pool.map(calc_wickT,sorted(list(cpu_data.values())))
    print(" \nFinished\n")
    pool.close()

    wickT = sp.array(wickT)
    wAll = wickT[:,0].sum(axis=0)
    nb = wickT[:,1].sum(axis=0)
    npairs = wickT[:,2].sum(axis=0)
    npairs_used = wickT[:,3].sum(axis=0)
    T1 = wickT[:,4].sum(axis=0)
    T2 = wickT[:,5].sum(axis=0)
    T3 = wickT[:,6].sum(axis=0)
    T4 = wickT[:,7].sum(axis=0)
    T5 = wickT[:,8].sum(axis=0)
    T6 = wickT[:,9].sum(axis=0)
    we = wAll*wAll[:,None]
    w = we>0.
    T1[w] /= we[w]
    T2[w] /= we[w]
    T3[w] /= we[w]
    T4[w] /= we[w]
    T5[w] /= we[w]
    T6[w] /= we[w]
    T1 *= 1.*npairs_used/npairs
    T2 *= 1.*npairs_used/npairs
    T3 *= 1.*npairs_used/npairs
    T4 *= 1.*npairs_used/npairs
    T5 *= 1.*npairs_used/npairs
    T6 *= 1.*npairs_used/npairs
    Ttot = T1+T2+T3+T4+T5+T6

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [
        {'name':'RPMIN','value':cf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':cf.rej,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    comment = ['Sum of weight','Covariance','Nomber of pairs','T1','T2','T3','T4','T5','T6']
    out.write([Ttot,wAll,nb,T1,T2,T3,T4,T5,T6],names=['CO','WALL','NB','T1','T2','T3','T4','T5','T6'],comment=comment,header=head,extname='COV')
    out.close()
