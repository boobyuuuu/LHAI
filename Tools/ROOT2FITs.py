import argparse
from ROOT import TFile, TH2D
import numpy as np
from astropy.io import fits

Fmap = TFile.Open('test.root')
Hmap = Fmap.Get("hSig")
NbinsX = int(Hmap.GetNbinsX())
NbinsY = int(Hmap.GetNbinsY())
WbinX = Hmap.GetXaxis().GetBinWidth(1)
WbinY = Hmap.GetYaxis().GetBinWidth(1)
X0 = Hmap.GetXaxis().GetBinLowEdge(1)
Y0 = Hmap.GetYaxis().GetBinLowEdge(1)
print("NbinsX=%d, NbinsY=%d"%(NbinsX, NbinsY))
print("WbinX=%.2lf, WbinY=%.2lf"%(WbinX, WbinY))
print("X0=%.2lf, Y0=%.2lf"%(X0, Y0))

Data = np.zeros((NbinsY, NbinsX))
for ix in range(0, NbinsX):
    for iy in range(0, NbinsY):
        Data[iy][ix] = Hmap.GetBinContent(NbinsX-ix, iy+1)

# 创建一个PrimaryHDU对象，并将数据存储在其中
hdu = fits.PrimaryHDU(Data)
hdr = hdu.header
hdr.set('CRVAL1', X0+NbinsX*WbinX)
hdr.set('CDELT1', -WbinX)
hdr.set('CRPIX1', 0)
hdr.set('CTYPE1', 'RA---CAR')
hdr.set('CUNIT1', '')
hdr.set('CRVAL2', Y0-WbinY/2)
hdr.set('CDELT2', WbinY)
hdr.set('CRPIX2', 0)
hdr.set('CTYPE2', 'DEC--CAR')
hdr.set('CUNIT2', '')
hdr.set('EQUINOX', '2000.0')
hdr.set('RADESYS', 'FK5     ')
hdr.set('Content', 'Source significance map')
# 写入头信息
print(hdu.header)
# 将HDU对象保存为FITS文件
hdu.writeto('test.fits')
