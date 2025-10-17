#python disc_halo_test.py --fitsout disc_halo_test.fits --o disc_halo_test.npy -n 10 --xsize_fin 64
#!/home/xingwei/.3ml/bin/python
# %env OMP_NUM_THREADS=1
# %env MKL_NUM_THREADS=1
# %env NUMEXPR_NUM_THREADS=1
import argparse
parser = argparse.ArgumentParser(
    description="Random souce with WCDA response")
parser.add_argument('--fitsout', type=str, required=True,  help='filename of fits output file')
parser.add_argument('--o', type=str, required=True,  help='filename of npy output file')
parser.add_argument('-n', type=int, required=True, help='number of map to generate')
parser.add_argument('--pw', type=float, default=0.1, help='pixel width (in deg)')
parser.add_argument('--xsize', type=int, default=100, help='imgsize_input')
parser.add_argument('--xsize_fin', type=int, default=64, help='imgsize_output')
args = parser.parse_args()


from threeML import silence_logs
silence_logs()
from response import *
from scipy.special import k0
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import tqdm

def halo(cx, cy, r1, width, pw, xsize):
    '''
    cx, cy: center, float, 0-1
    width: \sim 0.x
    '''
    src = np.zeros((xsize, xsize))
    x = np.arange(xsize)*pw
    ix = np.arange(xsize)
    xx,yy=np.meshgrid(x,x)
    cpx = np.interp(cx, ix/xsize, x)+ np.pi* 1e-2
    cpy = np.interp(cy, ix/xsize, x)+ np.pi* 1e-2
    rr = np.sqrt((xx-cpx)**2 + (yy-cpy)**2)
    src = np.exp( -(rr-r1)**2/2/width**2)
    srcsum = src.sum()
    assert np.isfinite(srcsum), "Sum must be finite"
    
    return src/srcsum * (pw/180 * np.pi)**2

def disc(cx, cy, r1, theta, e, pw, xsize):
    
    src = np.zeros((xsize, xsize))
    x = np.arange(xsize)*pw
    ix = np.arange(xsize)
    xx,yy=np.meshgrid(x,x)
    cpx = np.interp(cx, ix/xsize, x)+ np.pi* 1e-2
    cpy = np.interp(cy, ix/xsize, x)+ np.pi* 1e-2
    rr = np.sqrt(((xx-cpx) * np.cos(theta) + (yy-cpy) * np.sin(theta))**2 +
                 e* (-(xx-cpx) * np.sin(theta) + (yy-cpy) * np.cos(theta))**2)
    src = np.where(rr < r1, 1,0)
    srcsum = src.sum()
    assert np.isfinite(srcsum), "Sum must be finite"
    
    return src/srcsum * (pw/180 * np.pi)**2


def Proc(filename):
    global xsize, pw, xsize_fin, map_tree, det_res, flux_min
    np.random.seed()
    fitsfile = filename
    nsrc = np.random.randint(MINNSRC,MAXNSRC+1)

    # print(nsrc)
    srcmap = np.zeros((xsize, xsize))
    cxmin = (xsize-xsize_fin)/2/xsize
    cxmax = 1-cxmin
    
    for n in range(nsrc):
        srctype = np.random.randint(0,2)
        if srctype ==0:
            # halo
            srcmap += halo(
                cx=np.random.uniform(cxmin, cxmax),
                cy=np.random.uniform(cxmin, cxmax),
                r1=np.random.uniform(0.2,2),
                width=np.random.uniform(0.05,0.2),
                pw=pw,
                xsize=xsize
                               ) * np.random.uniform(flux_min,1)
        elif srctype ==1:
            # disc  
            srcmap += disc(
                cx=np.random.uniform(cxmin, cxmax),
                cy=np.random.uniform(cxmin, cxmax),
                r1=np.random.uniform(0.5,2),
                theta=np.random.uniform(0,2*np.pi),
                e=np.random.uniform(1,3),
                pw=pw,
                xsize=xsize
                               ) * np.random.uniform(flux_min,1)
            
    cp = int(xsize/2)
    hw = int(xsize_fin/2)
    orig = srcmap[cp-hw:cp+hw,cp-hw:cp+hw]

    ra = np.random.uniform(0,360)
    dec=  np.random.uniform(0,40)
    data2fits(srcmap, ra, dec, fitsfile, xsize_fin, pixelw = pw)

    plindex = np.random.uniform(2,4)
    piv = 1e12
    specK = 10**np.random.uniform(-20,-19)
    
    blur = respconv(
        fitsfile, map_tree, det_res, './disc_halo_test.hd5',  
        xsize_fin, srcname='src',
        plindex=plindex, piv=1e12, specK=specK,
        bins='0', PLOT=False
            )
    return orig, blur


NIMG = args.n
MAXNSRC = 5
MINNSRC = 1
NTHREAD = 1
pw = args.pw
xsize = args.xsize
xsize_fin = args.xsize_fin
map_tree = './map_20240731_WCDA.root'
det_res = './res_20240731_WCDA.root'
flux_min = 0.2 # minimum flux ratio of source
fitsname = args.fitsout
savename = args.o

iteration = np.arange(NIMG).tolist()
orig = []
blur = []
for i in tqdm.tqdm(iteration):
    origmap, blurmap = Proc(fitsname)
    orig.append(origmap)
    blur.append(blurmap)
    
orig = np.array(orig)
blur = np.array(blur)
result = np.array([orig, blur])
np.save(savename, result)
print('finish ' + savename)