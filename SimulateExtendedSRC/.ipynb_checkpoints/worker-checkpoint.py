#!/home/innov2021/miniconda3/envs/3ml/bin/python3.11
import argparse

parser = argparse.ArgumentParser(
    description="Random souce with WCDA response")
parser.add_argument('-o', type=str, required=True,  help='filename')
parser.add_argument('-n', type=int, required=True, help='number of map to generate')
parser.add_argument('--pw', type=float, default=0.1, help='pixel width (in deg)')
parser.add_argument('--xsize', type=int, default=100, help='imgsize_input')
parser.add_argument('--xsize_fin', type=int, default=64, help='imgsize_output')
args = parser.parse_args()

from threeML import silence_logs
silence_logs()
from response import *
from srcgen import *




map_tree = '../res_file/map_20240731_WCDA.root'
det_res = '../res_file/res_20240731_WCDA.root'

pw = args.pw
xsize = args.xsize_fin
xsize_large = args.xsize
NSAMPLE = args.n


ORIG = []
BLUR = []

for i in range(NSAMPLE):
    fitsfile = './fits/'+args.o+'_%d.fits'%i
    mapname = './hd5file/'+args.o+'_%d.hd5'%i
    nsrc = np.random.randint(1,5)
    ra = np.random.uniform(0,360)
    dec=  np.random.uniform(0,40)
    
    data = srcgen(nsrc, xsize=xsize_large)
    data = zeroes_like(data)
    x = np.arange(xsize_large)
    xx,yy = np.meshgrid((x,y))
    rr = (xx**2 + yy**2)**0.5
    data = np.where(rr < 30, 1, 0)
    
    
    orig = data2fits(data, ra, dec, fitsfile, xsize, pixelw = pw)
    
    blur = respconv(
        fitsfile, map_tree, det_res, mapname,
        xsize, srcname='src',
        plindex=2.63, piv=1e9, specK=2e-25,
        bins='0'
            )
    ORIG.append(orig)
    BLUR.append(blur)
    print(i)
result = np.array([np.array(ORIG),np.array(BLUR)])
print(result.shape)
np.save('./data/'+args.o+'.npy',result)
