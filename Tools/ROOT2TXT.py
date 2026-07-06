from ROOT import TFile, TH2D
import numpy as np
from astropy.io import fits

Fmap = TFile.Open("test.root")
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

with open('test.txt', 'w') as f:
    for ix in range(0, NbinsX):
        xx = X0+(ix+0.5)*WbinX
        for iy in range(0, NbinsY):
            yy = Y0+(iy+0.5)*WbinY
            print("%.5lf  %.5lf  %.5lf"%(xx, yy, Hmap.GetBinContent(iy+1, ix+1)), file=f)
