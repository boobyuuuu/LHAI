#ifndef papi_h
#define papi_h
//Made by Zhiguo Yao <zhiguo.yao@ihep.ac.cn>, 2019/07/20. Some functions are
//based on my own fortran code written 16 years ago.

#include <slamac.h>
#include <slalib.h>
#include <TRandom.h>
#include "lhaasosite.h"

class papi {
public:
  //We put these constants as the member variables
  static const double pi, twopi, halfpi, degrad, raddeg, c_light;

  //normalize the angle to range [angle0,angle0+twopi)
  template<typename T>
  static T angfix(T x, T angle0 = 0) {
    x -= angle0;
    return x - (twopi*std::floor(x/twopi)) + angle0;
  }

  template<typename T>
  static int nint(T x) {
    return (x>0)?int(x+0.5):-int(-x+0.5);
  }

  template<typename T>
  static long long nlong(T x) {
    return (x>0)?(long long)(x+0.5):-(long long)(-x+0.5);
  }

  //For multi-thread usage: please do NOT call papi constructor
  //directly, use papi::Instance() instead
  papi *Instance() {
    if (!fInstance) fInstance = new papi();
    return fInstance;
  }

  //Function for converting azimuth from wcda to HCS and from HCS to wcda.
  //The function is same for the two conversions.
  //When calling any functions in this package:
  //if azim_in is the input value in WCDA, please use
  //    papi::somefunction(...,papi::azimxwcda(azim_in),....)
  //If azim_out is the output, please use
  //    azimwcda = papi::azimxwcda(azim_out)
  //to get the azimuth in the WCDA coordinate system.
  inline static double azimxwcda(double azim,
      double wcdaphi = wcdaphimydet) {
    return angfix(halfpi - wcdaphi*degrad - azim);
  }

  //from azimuth to lhaaso phi or from lhaaso phi to azimuth
  inline static double azimxlhaaso(double azim,
      double lhaasophi = lhaasophimydet) {
    return angfix(halfpi - lhaasophi*degrad - azim);
  }

  static double getdut(double mjd);
  static double getlast(double mjd, double tsec);

  static void eqm2hcs(double mjd, double tsec, double ram, double dnm,
      double &zeni, double &azim);

  static void hcs2eqm(double mjd, double tsec, double zeni, double azim,
      double &ram, double &dnm);

  static void hcs2eql(double cz, double sz, double ca, double sa,
      double &ha, double &dn);

  static void eql2hcs(double ch, double sh, double cd, double sd,
      double &ze, double &az);

  inline static void hcs2eql(double zeni, double azim,
      double &ha, double &dn) {
    hcs2eql(cos(zeni),sin(zeni),cos(azim),sin(azim),ha,dn);
  }
  
  static void hcs2eql2(double mjd, double tsec, double zeni, double azim, double &ha, double &dn);

  inline static void eql2hcs(double ha, double dn,
      double &zeni, double &azim) {
    eql2hcs(cos(ha),sin(ha),cos(dn),sin(dn),zeni,azim);
  }

  //conversion with precession corrected
  inline static void hcs2eqlm(double mjd, double tsec,
      double zeni, double azim, double &ha, double &dn) {
    double ra;
    hcs2eqm(mjd,tsec,zeni,azim,ra,dn);
    ha = angfix(getlast(mjd,tsec) - ra);
  }

  //From the local equatorial system with precession corrected to zeni, azim
  inline static void eqlm2hcs(double mjd, double tsec,
      double ha, double dn, double &zeni, double &azim) {
    double ra = getlast(mjd,tsec) - ha;
    eqm2hcs(mjd,tsec,ra,dn,zeni,azim);
  }

  inline static void eql2eqm(double mjd, double tsec,
      double ha, double dn, double &ram, double &dnm) {
    ram = angfix(getlast(mjd,tsec)-ha);
    dnm = dn;
  }

  inline static void eqm2eql(double mjd, double tsec,
      double ram, double dnm, double &ha, double &dn) {
    ha = angfix(getlast(mjd,tsec)-ram);
    dn = dnm;
  }

  static void idplanet2hcs(double mjd, double tsec, int id,
      double &zeni, double &azim, double &radobs);

  inline static void moonhcs(double mjd, double tsec,
      double &zeni, double &azim, double &radobs) {
    idplanet2hcs(mjd,tsec,3,zeni,azim,radobs);
  }

  inline static void sunhcs(double mjd, double tsec,
      double &zeni, double &azim, double &radobs) {
    idplanet2hcs(mjd,tsec,0,zeni,azim,radobs);
  }

  static void idplanet2eqm(double mjd, double tsec, int id,
      double &ram, double &dnm, double &radobs);

  inline static void mooneqm(double mjd, double tsec,
      double &ram, double &dnm, double &radobs) {
    idplanet2eqm(mjd,tsec,3,ram,dnm,radobs);
  }

  inline static void suneqm(double mjd, double tsec,
      double &ram, double &dnm, double &radobs) {
    idplanet2eqm(mjd,tsec,0,ram,dnm,radobs);
  }

  static void ppp2dir(double px, double py, double pz,
      double &theta, double &phi);

  static void lmn2dir(double px, double py, double pz,
      double &theta, double &phi);

  static void dir2ppp(double theta, double phi,
      double &px, double &py, double &pz);

  inline static void dir2lmn(double theta, double phi,
      double &_l, double &_m, double &_n) {
    dir2ppp(theta,phi,_l,_m,_n);
  }

  static double ppp2ang(double x1, double y1, double z1,
      double x2, double y2, double z2);

  inline static void ppp2ang(double x1, double y1, double z1,
      double x2, double y2, double z2, double &ang) {
    ang = ppp2ang(x1,y1,z1,x2,y2,z2);
  }

  static double lmn2ang(double _l1, double _m1, double _n1,
      double _l2, double _m2, double _n2);

  inline static void lmn2ang(double _l1, double _m1, double _n1,
      double _l2, double _m2, double _n2, double &ang) {
    ang = lmn2ang(_l1,_m1,_n1,_l2,_m2,_n2);
  }

  static double dir2ang(double theta1, double phi1, double theta2, double phi2);

  inline static void dir2ang(double theta1, double phi1,
      double theta2, double phi2, double &ang) {
     ang = dir2ang(theta1,phi1,theta2,phi2);
  }

  inline static double eql2ang(double ra1, double dec1,
      double ra2, double dec2) {
    return dir2ang(halfpi-dec1,ra1,halfpi-dec2,ra2);
  }

  inline static void eql2ang(double ra1, double dec1, double ra2, double dec2,
      double &ang) {
     ang = eql2ang(ra1,dec1,ra2,dec2);
  }

  inline static double eqm2ang(double ra1, double dec1,
      double ra2, double dec2) {
    return dir2ang(halfpi-dec1,ra1,halfpi-dec2,ra2);
  }

  inline static void eqm2ang(double ra1, double dec1, double ra2, double dec2,
      double &ang) {
     ang = eqm2ang(ra1,dec1,ra2,dec2);
  }

  static void euler_trans(double *v0, double *vin, double *vout,
      bool isreverse = false);

  static void euler_trans(double the, double phi, double *vin, double *vout,
      bool isreverse = false);

  static void euler_trans(double  the0, double  phi0,
      double  the1, double  phi1,
      double &the2, double &phi2,
      bool isreverse = false);

  static void zz_trans(double *v0, double *vin, double *vout,
      bool isreverse = false);

  static void zz_trans(double the, double phi, double *vin, double *vout,
      bool isreverse = false);

  static void zz_trans(double  the0, double  phi0,
      double  the1, double  phi1,
      double &the2, double &phi2,
      bool isreverse = false);

  static void en_trans(double alpha0, double delta0,
      double alpha, double delta,
      double &yita, double &ksi);
 
  static void smearangle(double theta0, double phi0, double sigma,
      double &theta, double &phi);

  static void planeconv(double theta, double phi,
      double *vin, double *vout, bool isreverse = false);

  static void planeconv(double theta, double phi,
      double x, double y, double z, double t,
      double &xpp, double &ypp, double &zpp, double &tpp,
      bool isreverse = false);

private:
  papi() {}
 ~papi() {}

  static papi *fInstance;
  //Note: only integral values can be intialized in the header (and only once)!
  //https://stackoverflow.com/questions/2605520/
  //                                c-where-to-initialize-static-const
  //"""Only integral values (e.g., static const int ARRAYSIZE) are initialized
  //in header file because they are usually used in class header to define
  //something such as the size of an array. Non-integral values are
  //initialized in implementation file."""
  static const int eopdutver = 20220624;
  static const int neopdut = 1643;

  static double eopdut[neopdut][2];
};
#endif /* papi_h */


#ifdef papi_cc
const double papi::pi = 3.141592653589793;
const double papi::twopi = papi::pi*2;
const double papi::halfpi = papi::pi/2;
const double papi::degrad = papi::pi/180.;
const double papi::raddeg = 180./papi::pi;
const double papi::c_light = 2.99792458e8; //m/s

papi *papi::fInstance = 0;
#include "eopdut.h"

double papi::getdut(double mjd) {
  int n1,n2,n;

  n1 = 0;
  n2 = neopdut - 1;

  if (mjd<eopdut[n1][0] || mjd>eopdut[n2][0]) return 0.;

  if (n1==n2) {
    if (int(mjd)==int(eopdut[n1][0])) return eopdut[n1][0];
    else return 0.;
  }

  while (true) {
    if (mjd>=eopdut[n1][0] && mjd<=eopdut[n1+1][0]) {
      n = n1 + 1;
      break;
    }

    if (mjd>=eopdut[n2-1][0] && mjd<=eopdut[n2][0]) {
      n = n2;
      break;
    }

    n = (n1+n2)/2;

    if (mjd<=eopdut[n][0]) {
      n1++;
      n2 = n;
    }
    else {
      n1 = n;
      n2--;
    }
  }

  double dt = eopdut[n][0] - eopdut[n-1][0];
  if (dt==0.) return eopdut[n][1];
  else return (mjd-eopdut[n-1][0])/dt*(eopdut[n][1]-eopdut[n-1][1]) +
      eopdut[n-1][1];
}


void papi::eqm2hcs(double mjd, double tsec, double ram, double dnm,
    double &zeni, double &azim) {
  //For astronomical only positions (e.g., r.a., declination), we
  //use the sequence e.g., r.a., declination; For Zenith and Azimuth,
  //we use sequence zenith, azimuth.
  double mjdnow,tdb,tt,raa,dna,dut;
  double aob,zob,hob,dob,rob;

  mjdnow = mjd + tsec/86400.;
  tt = mjdnow + slaDtt(mjdnow)/86400.;
  tdb = tt; //not necessary to correct it

  //Mean to geocentric apparent.
  slaMap(ram,dnm,0.,0.,0.,0.,2000.,tdb,&raa,&dna);

  //Get UT1-UTC
  dut = getdut(mjdnow);

  //Geocentric apparent to observed
  //We set the air pressure to 0, since refraction will not be considered
  slaAopCR(raa,dna,mjdnow,dut,
      lonmydet*degrad,latmydet*degrad,altmydet,
      0.,0.,273.15,0.,0.,1.e-20,0.,&aob,&zob,&hob,&dob,&rob);

  zeni = zob;
  azim = angfix(aob);
}


double papi::getlast(double mjd, double tsec) {
  //get the local apparent sidereal time
  double mjdnow,tt,tdb;
  double dut;
  //last range: 0,twopi

  mjdnow = mjd + tsec/86400.;
  tt = mjdnow + slaDtt(mjdnow)/86400.;
  tdb = tt; //not necessary to correct it

  dut = getdut(mjdnow);
  //tdb to local apparent sidereal time
  return angfix(slaGmst(mjdnow+dut/86400.) + lonmydet*degrad + slaEqeqx(tdb));
}


void papi::hcs2eql(double cz, double sz, double ca, double sa,
    double &ha, double &dn) {
  //cz,sz: cos,sin of zenith; ca,sa: cos,sin of azimuth
  double x,y,z,r;
  static double sinlat,coslat;
  static bool first = true;
  //Inline version of slaDh2e
  //ha range: -pi,pi; dn range: -halfpi,halfpi
  //No polar motion included, so no time variations.

  if (first) {
    sinlat = sin(latmydet*degrad);
    coslat = cos(latmydet*degrad);
    first = false;
  }

  //change ce to sz, se to cz.
  //a: azimuth; e: elevation; p: latitude
  //  -ca*ce*sp     + se*cp
  x = -ca*sz*sinlat + cz*coslat;
  //  -sa*ce
  y = -sa*sz;
  //   ca*ce*cp     + se*sp
  z =  ca*sz*coslat + cz*sinlat;

  r = sqrt(x*x+y*y);
  if (r==0.) ha = 0.;
  else ha = atan2(y,x);
  if (ha<0.) ha = ha + twopi;
  dn = atan2(z,r);
}

void papi::hcs2eql2(double mjd, double tsec, double zeni, double azim, double &ha, double &dn){

    double azi, zen;
    /*azi = 360-(azim+270.+WCDA_phi0);
    if(azi>=360) azi = azi-360;
    if(azi<0)    azi = azi+360;*/
    azi = azimxwcda(azim/DR2D);
    zen = zeni/DR2D;  

    double ram, dnm;
    hcs2eqm(mjd, tsec, zen, azi, ram, dnm);
    double last = getlast(mjd, tsec);
    dn = dnm*DR2D;
    ha = -last + ram;
    if (ha <-3.141592654) ha += 2*3.141592654;
    if (ha >3.141592654) ha -= 2*3.141592654;
    //if (ha<0) ha = ha + 2*3.141592654;
    ha = ha*DR2D;//-(mjd-51544.5)/365.5*360/26000.;
}

void papi::eql2hcs(double ch, double sh, double cd, double sd,
    double &ze, double &az) {
  //ch,sh: cos,sin of hour angle; cd,sd: cos,sin of declination
  //ze: zenith, az: azimuth
  double x,y,z,r;
  static double sinlat,coslat;
  static bool first = true;
  //Inline version of slaDe2h
  //ze range: 0,pi; az range: 0,twopi
  //No polar motion included, so no time variations.

  if (first) {
    sinlat = sin(latmydet*degrad);
    coslat = cos(latmydet*degrad);
    first = false;
  }

  //h: hour angle; d: declination; p: latitude
  //  -ch*cd*sp     + sd*cp
  x = -ch*cd*sinlat + sd*coslat;
  //  -sh*cd
  y = -sh*cd;
  //   ch*cd*cp     + sd*sp
  z =  ch*cd*coslat + sd*sinlat;

  r = sqrt(x*x+y*y);
  if (r==0.) az = 0.;
  else az = atan2(y,x);

  if (az<0.) az = az + twopi;

  ze = halfpi - atan2(z,r);
}


void papi::hcs2eqm(double mjd, double tsec, double zeni, double azim,
    double &ram, double &dnm) {
  double mjdnow,dut;
  double rap,dnp;
  double tt,tdb;
  //ram range: 0,twopi;  dnm range: -halfpi,halfpi

  mjdnow = mjd + tsec/86400.;
  dut = getdut(mjdnow);

  //Observed to geocentric apparent
  static char copt[2] = "A";
  slaOapCR(copt,azim,zeni,mjdnow,dut,
      lonmydet*degrad,latmydet*degrad,altmydet,
      0.,0.,273.15,0.,0.,0.,0.,
      &rap,&dnp);

  tt = mjdnow + slaDtt(mjdnow)/86400.;
  tdb = tt; //not necessary to correct it

  //Geocentric apparent to mean place.
  slaAmp(rap,dnp,tdb,2000.,&ram,&dnm);
}


void papi::idplanet2hcs(double mjd, double tsec, int id,
    double &zeni, double &azim, double &radobs) {
  double mjdnow;
  double raatop,dnatop,diam;
  double last,hatop;
  double az,el;

  mjdnow = mjd + tsec/86400.;

  //Topocentric apparent and angular size of a planet
  //(Topocentric: geocentric parallax is corrected)
  slaRdplan(mjdnow,id,lonmydet*degrad,latmydet*degrad,
      &raatop,&dnatop,&diam);

  last = getlast(mjd,tsec);
  //Aparent to hour angle
  hatop = last - raatop;

  //Local equator to horizon
  slaDe2h(hatop,dnatop,latmydet*degrad,&az,&el);
  zeni = halfpi - el;
  azim = az;
  radobs = diam/2.;
}


void papi::idplanet2eqm(double mjd, double tsec, int id,
    double &ram, double &dnm, double &radobs) {
  double mjdnow;
  double raatop,dnatop,diam;
  double tt, tdb;

  mjdnow = mjd + tsec/86400.;

  //Topocentric apparent and angular size of a planet
  //(Topocentric: geocentric parallax is corrected)
  slaRdplan(mjdnow,id,lonmydet*degrad,latmydet*degrad,
      &raatop,&dnatop,&diam);

  tt = mjdnow + slaDtt(mjdnow)/86400.;
  tdb = tt; //not necessary to correct it

  //Geocentric apparent to mean place.
  slaAmp(raatop,dnatop,tdb,2000.,&ram,&dnm);
  radobs = diam/2.;
}


void papi::dir2ppp(double theta, double phi,
    double &px, double &py, double &pz) {
  double costheta,sintheta,cosphi,sinphi;

  costheta = cos(theta);
  sintheta = sin(theta);

  cosphi = cos(phi);
  sinphi = sin(phi);

  px = sintheta*cosphi;
  py = sintheta*sinphi;
  pz = costheta;
  return;
}


void papi::ppp2dir(double px, double py, double pz,
    double &theta, double &phi) {
  //returns theta: [0.,pi]; phi: [0.,2*pi)
  double p,costheta;

  p = sqrt(px*px+py*py+pz*pz);
  if (p==0) {
    theta = 0.;
    phi = 0.;
    return;
  }

  costheta = pz/p;
  if (costheta>1.) costheta = 1.;
  if (costheta<-1.) costheta = -1.;
  theta = acos(costheta);
  phi = atan2(py,px);

  if (phi<0.) phi += twopi;
  return;
}


double papi::ppp2ang(double x1, double y1, double z1,
    double x2, double y2, double z2) {
  double cosa;
  cosa = (x1*x2+y1*y2+z1*z2)/sqrt((x1*x1+y1*y1+z1*z1)*(x2*x2+y2*y2+z2*z2));
  if (cosa<-1.) cosa = -1.;
  else if (cosa>1.) cosa = 1.;
  return acos(cosa);
}


double papi::lmn2ang(double x1, double y1, double z1,
    double x2, double y2, double z2) {
  double cosa;
  cosa = x1*x2 + y1*y2 + z1*z2;
  if (cosa<-1.) cosa = -1.;
  else if (cosa>1.) cosa = 1.;
  return acos(cosa);
}


double papi::dir2ang(double t1, double p1, double t2, double p2) {
  double x1,y1,z1,x2,y2,z2;
  double cosa;
  double st1,st2;

  st1 = sin(t1);
  x1 = st1*cos(p1);
  y1 = st1*sin(p1);
  z1 = cos(t1);

  st2 = sin(t2);
  x2 = st2*cos(p2);
  y2 = st2*sin(p2);
  z2 = cos(t2);

  cosa = x1*x2+y1*y2+z1*z2;
  if (cosa<-1.) cosa = -1.;
  else if (cosa>1.) cosa = 1.;

  return acos(cosa);
}


void papi::euler_trans(double *v0, double *vin, double *vout,
    bool isreverse) {

  //isreverse = true: inverse conversion (x'',y'',z'' back to x,y,z)
  //v0[0-2]: l, m, n, of course it is not necesarrily normalized
  double x, y, z, xp, yp, zp, xpp, ypp, zpp;
  double ct, st, cp, sp;

  //ZYZ convention (or phi-theta-psi convention):
  //phi: azimuth, theta: elevation, psi: tilt.
  //i.    (x   -> xp ,  y   -> yp  ) around z,   phi;   zp   = z
  //ii.   (zp  -> zpp,  xp  -> xpp ) around yp,  theta; ypp  = yp
  //iii:  (xpp -> xppp, ypp -> yppp) around zpp, psi;   zppp = zpp
  //The step iii is ignored in this subroutine.
  //Of course there are some other conventions.

  double rr = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
  double rv = 1.; //inverse of the normalized factor
  if (fabs(rr-1)>1.e-20) rv = 1./sqrt(rr);

  ct = v0[2]*rv;
  st = sqrt(1.0-ct*ct);

  if (st==0) {
    cp = 1.;
    sp = 0.;
  }
  else {
    cp = v0[0]*rv/st;
    sp = v0[1]*rv/st;
  }

  if (isreverse) {
    //from x"-y"-z" to x-y-z, inverse convertion
    //In x"-y"-z" plane
    xpp = vin[0]; ypp = vin[1]; zpp = vin[2];
    //In x'-y'-z' plane
    yp = ypp; zp = zpp*ct - xpp*st; xp = xpp*ct + zpp*st;
    //In x-y-z plane
    x = xp*cp - yp*sp; y = yp*cp + xp*sp; z = zp;

    vout[0] = x; vout[1] = y; vout[2] = z;
  } else {
    //from x-y-z to x"-y"-z"
    //In x-y-z plane
    x = vin[0]; y = vin[1]; z = vin[2];
    //In x'-y'-z' plane
    xp = x*cp + y*sp; yp = y*cp - x*sp; zp = z;
    //In x"-y"-z" plane
    ypp = yp; zpp = zp*ct + xp*st; xpp = xp*ct - zp*st;

    vout[0] = xpp; vout[1] = ypp; vout[2] = zpp;
  }
  return;
}


void papi::euler_trans(double the, double phi, double *vin, double *vout,
    bool isreverse) {
  //Get Euler angles from the,phi
  //x = sin(the)*cos(phi)
  //y = sin(the)*sin(phi)
  //z = cos(the)

  //If the convertion is ZXZ convention:
  //the 'rotation phi' is of 90 degree difference from 'azimuth phi'.
  //if (phi<=1.5*pi) phi += 0.5*pi;
  //else phi -= 1.5*pi;

  //the, phi are in radian
  double ct = cos(the);
  double st = sin(the);
  double cp = cos(phi);
  double sp = sin(phi);

  double v0[3];
  v0[0] = st*cp;
  v0[1] = st*sp;
  v0[2] = ct;

  euler_trans(v0,vin,vout,isreverse);

  return;
}


void papi::euler_trans(double  the0, double  phi0,
  double  the1, double  phi1, double &the2, double &phi2,
  bool isreverse) {

  //all thes, phis are in radian
  double v0[3];
  double vin[3];
  double vout[3];

  //the, phi are in radian
  double st = sin(the0);

  v0[0] = st*cos(phi0);
  v0[1] = st*sin(phi0);
  v0[2] = cos(the0);

  st = sin(the1);
  vin[0] = st*cos(phi1);
  vin[1] = st*sin(phi1);
  vin[2] = cos(the1);

  euler_trans(v0,vin,vout,isreverse);

  the2 = acos(vout[2]);
  phi2 = atan2(vout[1],vout[0]);
  if (phi2<0.) phi2 += 2*pi;
  return;
}


void papi::zz_trans(double *v0, double *vin, double *vout,
    bool isreverse) {
  //This is a transformation invented by Zhiguo Yao.
  //Rotation x-, y-, z-axis around the axis perpendicular to
  //z"-z (v0: direction of z") for theta (angle from z to z"), and
  //then return the vin in x-y-z frame to vout in xn-yn-zn frame.
  //isreverse = true: inverse conversion (vin in xn-yn-zn frame back to
  //vout in x-y-z frame)

  double ct, st;
  double vx[3], vy[3], vz[3];

  double rr = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
  double rv; //inverse of the normalized factor
  if (fabs(rr-1)>1.e-20) rv = 1./sqrt(rr);
  else rv = 1;

  double x = v0[0]*rv;
  double y = v0[1]*rv;
  double z = v0[2]*rv;

  ct = z;
  st = sqrt(1.0-ct*ct);
  double rxy2 = x*x + y*y;
  double rxy = sqrt(rxy2);

  /*       ^      ^
           |z/z' / z"
           |    /
           |   /
           |  /
           | /
           |/
           o---------> y'
           x'
    (set r = sqrt(x*x+y*y), (x): cross-product)
  x'-axis (set y'-axis in z"z plane): z" (x) z
    = (y*i + -x*j + 0*k)/r
  y'-axis: z (x) x'
    = (x*i + y*j + 0*k)/r
  z'-axis:
    = (0*i + 0*j + 1*k)

  x-axis (1,0,0) in x'y'z' plane:
      (i*(y*i-x*j)/r)*i' + (i*(x*i+y*j)/r)*j' + i*(1*k)*k')
    = (y/r*i' + x/r*j' + 0*k')
  y-axis (0,1,0) in x'y'z' plane:
      (j*(y*i-x*j)/r*i' + j*(x*i+y*j)/r*j' + j*(1*k)*k')
    = (-x/r*i' + y/r*j' + 0*k')
  z-axis (0,0,1) in x'y'z' plane:
      (k*(y*i-x*j)/r*i' + k*(x*i+y*j)/r*j' + k*(1*k)*k')
    = (0, 0, k')

  Rotate axises with angle theta (phi0 -> phi0+theta) around x'-axis.
  (as now the frame is y'-z'-x', phi0 is the angle to the y'-axis)
  In x'y'z' plane, the tranformation relation for any vector (xa,ya,za) is:
    xb=xa, yb=ya*cos(theta)-za*sin(theta), zb=za*cos(theta)+ya*sin(theta)

  These axises change to
  xn-axis:
      (y/r*i' + x/r*cos(theta)*j' + x/r*sin(theta)*k')
    =  y/r*(y*i-x*j)/r + (x/r*cos(theta)*(x*i+y*j)/r + x/r*sin(theta)*k
    = (y*y+x*x*cos(theta))/rr*i + (-x*y+x*y*cos(theta))/rr*j + x/r*sin(theta)*k

  yn-axis:
      (-x/r*i' + y/r*cos(theta)*j' +  y/r*sin(theta)*k')
    = -x/r*(y*i-x*j)/r + y/r*cos(theta)*(x*i+y*j)/r +  y/r*sin(theta)*k
    = (-x*y+x*y*cos(theta))/rr*i + (x*x+y*y*cos(theta))/rr*j + y/r*sin(theta)*k

  zn-axis:
      (0*i' + -sin(theta)*j' + cos(theta)*k')
    = 0     + -sin(theta)*(x*i+y*j)/r + cos(theta)*k
    = -sin(theta)*x/r*i - sin(theta)*y/r*j + cos(theta)*k
  */

  if (fabs(ct)<1-1.e-10) {
    //xn-axis in the original frame
    vx[0] = (y*y+x*x*ct)/rxy2;
    vx[1] = (-x*y+x*y*ct)/rxy2;
    vx[2] = x/rxy*st;

    //yn-axis in the original frame
    vy[0] = (-x*y+x*y*ct)/rxy2;
    vy[1] = (x*x+y*y*ct)/rxy2;
    vy[2] = y/rxy*st;

    //zn-axis in the original frame
    vz[0] = -x*st/rxy;
    vz[1] = -y*st/rxy;
    vz[2] = ct;
  }
  else {
    if (ct>0) {
      vx[0] = 1;
      vx[1] = 0;
      vx[2] = 0;

      //yn-axis in the original frame
      vy[0] = 0;
      vy[1] = 1;
      vy[2] = 0;

      //zn-axis in the original frame
      vz[0] = 0;
      vz[1] = 0;
      vz[2] = 1;
    }
    else {
      vx[0] = -1;
      vx[1] = 0;
      vx[2] = 0;

      //yn-axis in the original frame
      vy[0] = 0;
      vy[1] = 1;
      vy[2] = 0;

      //zn-axis in the original frame
      vz[0] = 0;
      vz[1] = 0;
      vz[2] = -1;
    }
  }

  if (!isreverse) {
    //convert vin in frame x-y-z to vout in frame xn-yn-zn 
    //vout[0] = vin (.) i_n
    //vout[1] = vin (.) j_n
    //vout[2] = vin (.) k_n
    vout[0] = vin[0]*vx[0] + vin[1]*vx[1] + vin[2]*vx[2];
    vout[1] = vin[0]*vy[0] + vin[1]*vy[1] + vin[2]*vy[2];
    vout[2] = vin[0]*vz[0] + vin[1]*vz[1] + vin[2]*vz[2];
  }
  else {
    //now vin is in the xn-yn-zn frame
    //vout = vin[0]*i_n + vin[1]*j_n + vin[2]*k_n
    vout[0] = vin[0]*vx[0] + vin[1]*vy[0] + vin[2]*vz[0];
    vout[1] = vin[0]*vx[1] + vin[1]*vy[1] + vin[2]*vz[1];
    vout[2] = vin[0]*vx[2] + vin[1]*vy[2] + vin[2]*vz[2];
  }

  return;
}


void papi::zz_trans(double the, double phi, double *vin, double *vout,
    bool isreverse) {

  //the, phi are in radian
  double ct = cos(the);
  double st = sin(the);
  double cp = cos(phi);
  double sp = sin(phi);

  double v0[3];
  v0[0] = st*cp;
  v0[1] = st*sp;
  v0[2] = ct;

  zz_trans(v0,vin,vout,isreverse);

  return;
}


void papi::zz_trans(double  the0, double  phi0,
  double  the1, double  phi1, double &the2, double &phi2,
  bool isreverse) {

  //all thes, phis are in radian
  double v0[3];
  double vin[3];
  double vout[3];

  //the, phi are in radian
  double st = sin(the0);
  v0[0] = st*cos(phi0);
  v0[1] = st*sin(phi0);
  v0[2] = cos(the0);

  st = sin(the1);
  vin[0] = st*cos(phi1);
  vin[1] = st*sin(phi1);
  vin[2] = cos(the1);

  zz_trans(v0,vin,vout,isreverse);

  the2 = acos(vout[2]);
  phi2 = atan2(vout[1],vout[0]);
  if (phi2<0.) phi2 += 2*pi;

  return;
}


void papi::en_trans(double alpha0, double delta0,
  double alpha, double delta,
  double &yita, double &ksi) {
 /*
  * Adopted from my "shadow" program of 1997
  * Transform a direction to to the plane perpendicular to the sight.
  * "en" means East-North.
  * alaph0, delta0: right ascension, declination of the star/moon
  * alpha, delta: the incoming direction of an event
  * yita is 'theta' (angular distance), ksi is phi (phase angle)
  * ksi starts from east, rotating toward north.
  * So a map can be drawn with x = yita*cos(ksi), y = yita*sin(ksi)
  * It can be proven that dN/dOmega = const * dN/dx/dy (when theta is
  * very small, i.e., theta = sin(theta), using the Jacobian Determinant).
  * All the angles are in radian.
  */
  double xx,xy,xz,yx,yy,yz,zx,zy,zz;
  double x0,y0,z0,x,y,z;
  double ca,sa;
  double cd,sd;

  double ca0 = cos(alpha0);
  double sa0 = sin(alpha0);

  double cd0 = cos(delta0);
  double sd0 = sin(delta0);

  ca = cos(alpha);
  sa = sin(alpha);
  cd = cos(delta);
  sd = sin(delta);

 /*
  * first rotate xy plane -(90-alpha0) to x',y',z'
  * then rotate z'y' plane 90-delta0 to x'',y'',z''
  * then change x''' = -x'', y''' = -y''
  * After that, x is point the direction where alpha increases
  * So, x positive is EAST, y positive is NORTH.
  * New reference axises in the original coordinate
 */

  xx = -sa0;
  xy = ca0;
  xz = 0.;

  yx = -sd0*ca0;
  yy = -sd0*sa0;
  yz = cd0;

  zx = cd0*ca0;
  zy = cd0*sa0;
  zz = sd0;

  //Source position in origin coordinate
  x0 = cd*ca;
  y0 = cd*sa;
  z0 = sd;

  //Source position in circle reference
  x = x0*xx + y0*xy + z0*xz;
  y = x0*yx + y0*yy + z0*yz;
  z = x0*zx + y0*zy + z0*zz;

  yita = acos(z);
  ksi = atan2(y,x);
  if (ksi<0.) ksi = ksi + twopi;

  return;
}


void papi::smearangle(double theta0, double phi0, double sigma,
    double &theta, double &phi) {

  double thetap, phip;
  double a1 = 1.;
  double a2 = exp(-0.5*(pi/sigma)*(pi/sigma));

  //dN = A*sintheta/theta * d(exp(-0.5*(theta/sigma)*(theta/sigma)))

  double y;
  while (true) {
    thetap = a1 + gRandom->Rndm()*(a2-a1);
    if (thetap<1.e-40) thetap = 1.e-40;
    else if (thetap>1-1.e-10) thetap = 1-1.e-10;
    thetap = sqrt(-2*log(thetap))*sigma;
    y = sin(thetap)/thetap;
    if (gRandom->Rndm()<y) break;
  }

  phip = gRandom->Rndm()*twopi;
  euler_trans(theta0,phi0,thetap,phip,theta,phi,1);
}


void papi::planeconv(double theta, double phi,
    double *vin, double *vout, bool isreverse) {
  //isreverse = false: detector plane to shower plane
  //          = true : shower plane to detector plane
  // x towards y, angle phi:
  // |x'|   | cos  sin|  |x|
  // |  | = |         |  | |
  // |y'|   |-sin  cos|  |y|

  double xp,yp,zp;

  double costheta = cos(theta);
  double sintheta = sin(theta);

  double cosphi = cos(phi);
  double sinphi = sin(phi);

  double l = sintheta*cosphi;
  double m = sintheta*sinphi;
  double n = costheta;
  double k;

  if (!isreverse) {
    // x -> y, phi
    xp = vin[0]*cosphi + vin[1]*sinphi;
    yp = vin[1]*cosphi - vin[0]*sinphi;
    zp = vin[2];
    // z -> x', theta
    vout[2] = zp*costheta + xp*sintheta;
    vout[0] = xp*costheta - zp*sintheta;
    vout[1] = yp;
    k = -(l*vin[0]+m*vin[1]+n*vin[2]);
    vout[3] = vin[3] + k/c_light;
  }
  else {
    // x" -> z", theta
    xp = vin[0]*costheta + vin[2]*sintheta;
    zp = vin[2]*costheta - vin[0]*sintheta;
    yp = vin[1];
    //y' -> x', phi
    vout[1] = yp*cosphi + xp*sinphi;
    vout[0] = xp*cosphi - yp*sinphi;
    vout[2] = zp;

    k = -(l*vout[0]+m*vout[1]+n*vout[2]);
    vout[3] = vin[3] - k/c_light;
  }

  return;
}


void papi::planeconv(double theta, double phi,
    double x, double y, double z, double t,
    double &xpp, double &ypp, double &zpp, double &tpp,
    bool isreverse) {

  double xp,yp,zp;

  double costheta = cos(theta);
  double sintheta = sin(theta);

  double cosphi = cos(phi);
  double sinphi = sin(phi);

  double l = sintheta*cosphi;
  double m = sintheta*sinphi;
  double n = costheta;
  double k;

  if (!isreverse) {
    // x -> y, phi
    xp = x*cosphi + y*sinphi;
    yp = y*cosphi - x*sinphi;
    zp = z;
    // z -> x', theta
    zpp = zp*costheta + xp*sintheta;
    xpp = xp*costheta - zp*sintheta;
    ypp = yp;

    k = -(l*x+m*y+n*z);
    tpp = t + k/c_light;
  }
  else {
    // x" -> z", theta
    xp = x*costheta + z*sintheta;
    zp = z*costheta - x*sintheta;
    yp = y;
    //y' -> x', phi
    ypp = yp*cosphi + xp*sinphi;
    xpp = xp*cosphi - yp*sinphi;
    zpp = zp;

    k = -(l*xpp+m*ypp+n*zpp);
    tpp = t - k/c_light;
  }

  return;
}
#endif /* papi_cc */
