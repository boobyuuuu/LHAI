#python halo_test_width.py --fitsout halo_test_width.fits --o halo_test_width.npy -n 100 --xsize_fin 64
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
parser.add_argument('--idx_list', type=str, default=None, help='comma separated indices to reprocess')
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
def Proc(filename, width_val):
    global xsize, pw, xsize_fin, map_tree, det_res, r1_fixed, flux_fixed
    np.random.seed()
    fitsfile = filename
    
    srcmap = np.zeros((xsize, xsize))
    srcmap = halo(
        cx=0.5,
        cy=0.5,
        r1=r1_fixed,
        width=width_val,
        pw=pw,
        xsize=xsize
    ) * flux_fixed
            
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
        fitsfile, map_tree, det_res, './halo_test_width.hd5',  
        xsize_fin, srcname='src',
        plindex=plindex, piv=1e12, specK=specK,
        bins='0', PLOT=False
    )
    return orig, blur
NIMG = args.n
pw = args.pw
xsize = args.xsize
xsize_fin = args.xsize_fin
map_tree = './map_20240731_WCDA.root'
det_res = './res_20240731_WCDA.root'
fitsname = args.fitsout
savename = args.o
r1_fixed = 1.0
flux_fixed = 1.0
width_continuous = np.linspace(0.4, 0.7, NIMG)

if args.idx_list is None:
    orig = []
    blur = []
    for i in tqdm.tqdm(range(NIMG)):
        current_width = width_continuous[i]
        origmap, blurmap = Proc(fitsname, width_val=current_width)
        orig.append(origmap)
        blur.append(blurmap)
    orig = np.array(orig)
    blur = np.array(blur)
else:
    result_reshaped = np.load(savename)
    orig = result_reshaped[:,0]
    blur = result_reshaped[:,1]

    idx_to_process = [int(x) for x in args.idx_list.split(',')]
    for i in tqdm.tqdm(idx_to_process):
        current_width = width_continuous[i]
        orig_new, blur_new = Proc(fitsname, width_val=current_width)
        orig[i] = orig_new
        blur[i] = blur_new

result = np.array([orig, blur])
result_reshaped = np.transpose(result, (1, 0, 2, 3))
np.save(savename, result_reshaped)
print(f'finish {savename}, with shape {result_reshaped.shape}')
