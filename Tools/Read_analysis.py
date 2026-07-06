from astropy import coordinates
import astropy.units as u
from astropy.io import ascii
import numpy as np
import scipy as sp
import matplotlib as mpb
import astropy.io.fits as fits
from astropy.visualization import MinMaxInterval as mmi
from astropy.visualization import SqrtStretch as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import imshow_norm
import astropy.wcs as wcs
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import Circle
import scipy.ndimage as ndimage
from scipy.integrate import simps
import argparse

def Column_density(T):
    sub = np.where(np.isnan(T))
    T[sub] = 0
    sub = np.where(T<0.05*3) ## Here 0.05 is typcal rmd noise of the CO survey used.
    T[sub]=0
    wco = T*1.3
    xco = 1.5e20 #units cm-2/( K km/s)
    NH = 2*wco*xco 
    return NH

def angle(ra1, dec1, ra2, dec2):
    zen1 = np.deg2rad(90-dec1)
    azi1 = np.deg2rad(ra1)
    zen2 = np.deg2rad(90-dec2)
    azi2 = np.deg2rad(ra2)
    z1 = np.cos(zen1)
    x1 = np.sin(zen1)*np.cos(azi1)
    y1 = np.sin(zen1)*np.sin(azi1)
    z2 = np.cos(zen2)
    x2 = np.sin(zen2)*np.cos(azi2)
    y2 = np.sin(zen2)*np.sin(azi2)
    
    space = np.rad2deg(np.arccos(x1*x2+y1*y2+z1*z2))
    return space

def main(COFile, Mode): # OutDir, OutName, Velocity, Postion, Radius):
    COPHDU = fits.open(COFile)
    # Pixel width
    CDELT1 = COPHDU[0].header['CDELT1']   # l
    CDELT2 = COPHDU[0].header['CDELT2']   # b
    CDELT3 = COPHDU[0].header['CDELT3']   # v
    # Pixid of refer point
    CRPIX1 = COPHDU[0].header['CRPIX1']
    CRPIX2 = COPHDU[0].header['CRPIX2']
    CRPIX3 = COPHDU[0].header['CRPIX3']
    # value of refer point
    CRVAL1 = COPHDU[0].header['CRVAL1']
    CRVAL2 = COPHDU[0].header['CRVAL2']
    CRVAL3 = COPHDU[0].header['CRVAL3']
    # Npixels
    NAXIS1 = COPHDU[0].header['NAXIS1']
    NAXIS2 = COPHDU[0].header['NAXIS2']
    NAXIS3 = COPHDU[0].header['NAXIS3']

    COData = COPHDU[0].data

    if Mode == '0':
        # M1 figure, Column density
        V1, V2 = -150, 150  # Velocity slice
        Vpix1 = int((V1-CRVAL3/1000)/(CDELT3/1000)+CRPIX3-1)
        Vpix2 = int((V2-CRVAL3/1000)/(CDELT3/1000)+CRPIX3-1)
        print(Vpix1, Vpix2)
        COMap = COData[0][Vpix1:Vpix2].sum(axis=0)*CDELT3/1000
        COMap = Column_density(COMap)
    
        myHDU = fits.PrimaryHDU()
        myHDU.header['BITPIX'] = -30
        myHDU.header['NAXIS'] = 2 
        myHDU.header['NAXIS1']=COPHDU[0].header["NAXIS1"]
        myHDU.header['NAXIS2']=COPHDU[0].header["NAXIS2"]
        myHDU.header['CTYPE1']=COPHDU[0].header['CTYPE1']
        myHDU.header['CTYPE2']=COPHDU[0].header['CTYPE2']    
        
        myHDU.header['CRVAL1']= COPHDU[0].header['CRVAL1']
        myHDU.header['CRPIX1']= COPHDU[0].header['CRPIX1']
        myHDU.header['CDELT1']= COPHDU[0].header['CDELT1']
        myHDU.header['CUNIT1']=('deg', 'Physical unit of X-axis')
        
        myHDU.header['CRVAL2']= COPHDU[0].header['CRVAL2']
        myHDU.header['CRPIX2']= COPHDU[0].header['CRPIX2']
        myHDU.header['CDELT2']= COPHDU[0].header['CDELT2']
        myHDU.header['CUNIT2']=('deg', 'Physical unit of y-axis')
        # print(myHDU.header)
        wcsobj=wcs.WCS(myHDU)
        
        del_ra=np.abs(COPHDU[0].header['CDELT1'])
        del_dec=np.abs(COPHDU[0].header['CDELT2'])    
        left,right,bottom,top = 0,COPHDU[0].header['NAXIS1'],0,COPHDU[0].header['NAXIS2']
        coor_x_ref,coor_y_ref = wcsobj.wcs_pix2world(0.5*right+0.5,0.5*top+0.5,1)
        figure_x2y = float(right*del_ra*np.cos(np.radians(coor_y_ref)))/float(top*del_dec)
        # wcsobj_plot=wcs.WCS(COPHDU[0].header)
        subplot_kw = {'projection':wcsobj}
        fig,ax=plt.subplots(1,1,subplot_kw=subplot_kw,figsize=(figure_x2y*10,8),dpi=500)
        ax.coords[0].set_major_formatter('d.d')
        ax.coords[1].set_major_formatter('d.d')
        mycolor = ['black','midnightblue','mediumblue','b','purple','r','orange','yellow','white']
        cmap_color_def = colors.LinearSegmentedColormap.from_list('my_list',mycolor)
        color_bar_set = plt.cm.gnuplot2
        color_bar_set = cmap_color_def
        im1=plt.imshow(COMap,aspect='auto',cmap=color_bar_set,extent=[left,right,bottom,top],vmin=np.min(COMap),vmax=np.max(COMap))
        #cd=plt.contour(COMap,levels=[1.e22,3.e22,6.e22,7.e22],colors='g',linewidths=0.8,linestyles='dashed',alpha=0.8)
        fig.colorbar(im1,orientation='vertical',label=r'$N_H/cm^{-2}$')  
        plt.xlabel('l [ degree ]')
        plt.ylabel('b [ degree ]')
        plt.title('Velocity: 46km/s to 66km/s')
        plt.savefig("J1908_46_66.png")
    

if __name__ == "__main__":

    argv = argparse.ArgumentParser()
    argv.add_argument("-f",  "--infile",   dest="COFile")
    argv.add_argument("-m",  "--Mode",   dest="Mode")
    #argv.add_argument("-vel",  "--velocity",   dest="Velocity", type=float)
    #argv.add_argument("-posi", "--postion",  dest="Position", type=float)
    #argv.add_argument("-rd", "--radius",  dest="Radius", type=float)
    args = argv.parse_args()

    COFile = args.COFile
    Mode = args.Mode
    #Velocity = args.Velocity
    #Postion = args.Position
    #Radius = args.Radius

    main(COFile, Mode) #OutDir, OutName, Velocity, Postion, Radius)
