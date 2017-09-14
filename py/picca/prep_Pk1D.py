import fitsio
import scipy as sp



def exp_diff(file,ll) :

    nexp_per_col = file[0].read_header()['NEXP']/2
    fltotodd  = sp.zeros(ll.size)
    ivtotodd  = sp.zeros(ll.size)
    fltoteven = sp.zeros(ll.size)
    ivtoteven = sp.zeros(ll.size)
        
    for iexp in range (nexp_per_col) :
        for icol in range (2):
            llexp = file[4+iexp+icol*nexp_per_col]["loglam"][:]
            flexp = file[4+iexp+icol*nexp_per_col]["flux"][:]
            ivexp = file[4+iexp+icol*nexp_per_col]["ivar"][:]
            bins = sp.searchsorted(ll,llexp)

            if iexp%2 == 1 :
                civodd=sp.bincount(bins,weights=ivexp)
                cflodd=sp.bincount(bins,weights=ivexp*flexp)
                fltotodd[:civodd.size-1] += cflodd[:-1]
                ivtotodd[:civodd.size-1] += civodd[:-1]
            else :
                civeven=sp.bincount(bins,weights=ivexp)
                cfleven=sp.bincount(bins,weights=ivexp*flexp)
                fltoteven[:civeven.size-1] += cfleven[:-1]
                ivtoteven[:civeven.size-1] += civeven[:-1]

    w=ivtotodd>0
    fltotodd[w]/=ivtotodd[w]
    w=ivtoteven>0
    fltoteven[w]/=ivtoteven[w]

    alpha = 1
    if (nexp_per_col%2 == 1) :
        n_even = (nexp_per_col-1)/2
        alpha = sp.sqrt(4.*n_even*(n_even+1))/nexp_per_col
    diff = 0.5 * (fltoteven-fltotodd) * alpha ### CHECK THE * alpha (Nathalie)
    
    return diff
