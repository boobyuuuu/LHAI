#ifndef lhaasosite_h
#define lhaasosite_h
//Made by Zhiguo Yao <zhiguo.yao@ihep.ac.cn>, 2018/11/01

/*
LHAASO site:
29d21m27.6106s N, 100d8m19.6522s E
29.35766961 N, 100.13879228 E
UTM: 610538.101, 3248152.001
Geomagnetic field:
  Input:
   2019-12-31 29.35766961 100.13879228 4410
  Output:
   -1.17 46.33 34562.5 34555.3 -708.7 36206.3 50054.6
    the declination (the direction of the horizontal component of
         the magnetic field measured clockwise from north) in degrees,
    the inclination (the direction of the magnetic field measured
         down from the horizontal) in degrees,
    the horizontal field in nanotesla (nT),
    the north component of the field in nT,
    the east component of the field in nT,
    the vertical component of the field in nT (down is positive),
    the total field in nT.
*/

//Geodetic parameters of the detector.
//The following values are in degree
//Altitude (note: in meter!) is useful for slalib only.
const double lonmydet = 100.138794639;
const double latmydet = 29.357656306;
//WCDA water level
const double altmydet = 4372.07+4.3;
const double angmagmydet = -1.17;
//X axis of WCDA from the true East, in degree
const double wcdaphimydet = 29.442;
//X axis of LHAASO from the true East, in degree
const double lhaasophimydet = -0.558;
#endif /* lhaasosite_h */

