#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "TNtuple.h"
#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TNtupleD.h"
#include "TF1.h"
#include "TRandom.h"
#include "slalib.h"
#include "slamac.h"
#include "Sun.h"
#include "Moon.h"

#define hNDEG 10
#define hNRA 1400
#define hNDEC 1000
#define hPI 3.14159265358979312
#define NP 12 
#define WCDALA 0.51238798666678265 
#define WCDALO 1.7477516044432506
#define WCDA_phi0 29.45
//#define DEC_BIN  1000  // -20 - 80
#define DEC_BIN  1200  // -30 - 90
#define RA_BIN   1800  // -90 - 90
#define ALL_BIN   3600  // 0 - 360
#define BIN_WIDTH 0.1  //
#define NUM_E      1 

#define deg2rad 0.017453293
#define rad2deg 57.29577951


double Li_Ma(double aa,double ddon, double ddoff)
{
     double off,on,ss,all,aa1;
         all = ddon + ddoff;
         aa1 = aa + 1.;
	 if(ddon>0.5&&aa<0.2&&aa>0){
           if(fabs(ddon-ddoff*aa)>0.4){
            on  = ddon*log(aa1/aa*ddon/all);
            off = ddoff*log(aa1*ddoff/all);
            ss  = sqrt(2.)*sqrt(on+off);
	    if(ddon<aa*ddoff)ss=-ss;
           }
           else ss=(ddon-ddoff*aa)/sqrt(ddoff*aa+aa*ddon);
	 }
	 else ss=0;
         return ss;
}
int src_position(char *name, double mjd,double *ra1,double *dec1)
{
   double ra,dec;	
   if(name=="MOON")
           moon_orbit(mjd, &ra, &dec);
   if(name=="CRAB"){
	   ra=83.63;
           dec=22.02;
   }
   if(name=="SUN")
           sun_orbit(mjd, &ra, &dec);
   if(name=="MARK421"){
         ra  = 166.11;
	 dec = 38.21;  //got from web??      
   }
   if(name=="MARK501"){
          ra  = 253.47;
          dec = 39.76;  //got from web??
    }
   if(name=="CYGNUS"){
          ra  = 304.83;
	  dec = 36.83; // got by milagro
   }
   if(name=="MR"){
          ra  = 112.5;
          dec = 14.; // got by MAKET-ANI
   }	
   if(name=="GC40"){
          ra  = 286.19;
          dec = 6.43; // got by Zhangjl PHD
   }
   if(name=="GC41"){
          ra  = 286.28;
          dec = 7.05; // got by Zhangjl PHD
    }		 
    if(name=="MILAGRO1"){
          ra  = 305.023;
          dec = 36.719; // got by Milagro result ApJ 664:L91-L94, 2007
    }  	   
    if(name=="MILAGRO2"){
          ra  = 286.976;
          dec = 6.268; // got by HESS result
    }
   *ra1  = ra;
   *dec1 = dec;
   return 0;
}
//mjd to sidereal time in arc
double slaLst(double mjd, double LONGITUDE, double *lst )
{
  double t;
  t = 0.671262 + 1.002737909*(mjd - 40000)+LONGITUDE/D2PI;
  t = t - (int)t;
  *lst = D2PI * t;
}

//Horizon to hour angle:  Az,El to HA,Dec
int h2eh_wcda(double mjd, double zen, double az, double *hra, double *ddec)
{
    double z,a,tra,tdec;
    a=360-(az+270.+WCDA_phi0);
    if(a>=360) a = a-360;
    if(a<0)    a = a+360;
    z=90-zen;
    a=a/DR2D;
    z=z/DR2D;  
    slaDh2e(a, z, WCDALA,&tra, &tdec);
    *hra=-tra*DR2D;
    *ddec=tdec*DR2D;
    return 0;
}
//Horizon to equatorial coordinates:  Az,El to RA,Dec
int h2e_wcda(double mjd,double zen, double az, double *hra, double *ddec)
{
    double z,a,tra,tdec,tside;
     h2eh_wcda(mjd,zen,az,&tra,&tdec); 
     slaLst(mjd,WCDALO, &tside );
     tra=tside*DR2D+tra;
     if(tra<0)tra+=360.;
     if(tra>=360)tra-=360;  
    *hra=tra;
    *ddec=tdec;
    return 0;
}

//Horizon to equatorial coordinates in solar time:  Az,El to HA,Dec
int h2es_wcda(double mjd,double az, double zen, double *hra, double *ddec)
{
    double z,a,tra,tdec,tsolar;
     h2eh_wcda(mjd,az,zen,&tra,&tdec);
     tsolar=(mjd-(int)mjd)*360;
     tra=tsolar+tra;
     if(tra<0)tra+=360.;
     if(tra>=360)tra-=360;
    *hra=tra;
    *ddec=tdec;
    return 0;
}
// equatorial to Horizon  coordinates:  Az,El to HA,Dec
int e2h_wcda(double mjd,double hra, double ddec,double *zen, double *az)
{
    double z,a,tra,tdec,tside;
     slaLst(mjd,WCDALO,&tside);
     tra=tside*DR2D - hra;
     tra=tra/DR2D; tdec=ddec/DR2D;
     slaDe2h(tra,tdec,WCDALA,&a, &z);
     a=a*DR2D;
      a = 360. - a;
      a = a - (270.+WCDA_phi0);
      if(a<0) a = a+360.;
     *az=a;
     *zen=90-z*DR2D;
    return 0;
}

int e2g(double tra, double tdec, double *tlong, double *tlat)
{
  double ra,dec,l,b; 
     ra=tra/DR2D;
     dec=tdec/DR2D;
     slaEqgal(ra,dec,&l,&b);
     *tlong=l*DR2D;
     *tlat=b*DR2D;
     return 1;
}
int g2e(double tlong, double tlat, double *tra, double *tdec)
{
  double ra,dec,l,b;
     l=tlong/DR2D;
     b=tlat/DR2D;
     slaGaleq(l, b,&ra,&dec);
     *tra=ra*DR2D;
     *tdec=dec*DR2D;
     return 1;
}
//for panel draw
void slaAitoff(double ra,double dec,double *rra ,double *ddec)
{
 ra=ra*DPI/180.;
 dec=dec*DPI/180.;
 double x, y;
 double alpha2 = (ra-DPI)/2.;
 double delta = dec;
 double r2 = sqrt(2.);
 double f = 2*r2/DPI;
 double cdec = cos(delta);
 double denom = sqrt(1+cdec*cos(alpha2));
 x = cdec*sin(alpha2)*2*r2/denom;
 y = sin(delta)*r2/denom;
 x /= f;
 y /= f;
 x += DPI;
 *rra = x*180./DPI;
 *ddec = y*180./DPI;
}

Double_t Gaus(Double_t *x, Double_t *par)
{
         Double_t arg = 0;
         if (par[2] != 0) arg = (x[0] - par[1])/par[2];
         //Double_t fitval = par[0]*TMath::Exp(-0.5*arg*arg);
         Double_t fitval = par[0]*exp(-0.5*arg*arg);
	 return fitval;
}

//calculate the space angle
double GetSpace(double decc1, double raa1, double decc2, double raa2)
{
    double dr,dl0,dm0,dz0,dl1,dm1,dz1,ra,dec; 
  
    ra=raa1*hPI/180.;
    dec=(90.-decc1)*hPI/180.;              
       dl0= sin(dec)*cos(ra);
       dm0= sin(dec)*sin(ra);
       dz0= cos(dec);
     ra=raa2*hPI/180.;
     dec=(90.-decc2)*hPI/180.;
       dl1 = sin(dec)*cos(ra);
       dm1 = sin(dec)*sin(ra);
       dz1 = cos(dec);
     dr = acos(dl0*dl1+dm0*dm1+dz0*dz1)*180./hPI;
    return dr;
}

/********** caluculation of position angle **********/

double position( double RAS, double DEC ,double ras, double dec ){

  double sd, sD, sf, cd, cD, cf;
  double stheta, ctheta;

  sd = sin( dec * deg_rad );
  sD = sin( DEC * deg_rad );
  sf = sin( ( ras - RAS ) * deg_rad );
  cd = cos( dec * deg_rad );
  cD = cos( DEC * deg_rad );
  cf = cos( ( ras - RAS ) * deg_rad );

  stheta = cd * sf;
  ctheta = cD * sd - sD * cd * cf;
  return ( atan2( stheta, ctheta ) * rad_deg );

}

/********** caluculation of distance by (zenith & azimuth) *********/

double distance( double zen, double azi, double ZEN, double AZI ){

  double e1, e2, e3;

  e1 = sin(zen*deg2rad)*cos(azi*deg2rad) * sin(ZEN*deg2rad)*cos(AZI*deg2rad);
  e2 = sin(zen*deg2rad)*sin(azi*deg2rad) * sin(ZEN*deg2rad)*sin(AZI*deg2rad);
  e3 = cos(zen*deg2rad) * cos(ZEN*deg2rad);

  return ( acos( e1 + e2 + e3 ) * rad2deg );

}

/********** caluculation of distance by (ra & dec) *********/

double distance_radec( double ra, double dec, double RA, double DEC ){

  double e1, e2, e3;

  e1 = sin(dec*deg2rad)*cos(ra*deg2rad) * sin(DEC*deg2rad)*cos(RA*deg2rad);
  e2 = sin(dec*deg2rad)*sin(ra*deg2rad) * sin(DEC*deg2rad)*sin(RA*deg2rad);
  e3 = cos(dec*deg2rad) * cos(DEC*deg2rad);

  return ( acos( e1 + e2 + e3 ) * rad2deg );

}
