#ifndef hpatimer_h
#define hpatimer_h
//hpatimer:
//  High Precision Absolute Time Representation (up to at least pico second)
//  Designed & Made by Zhiguo Yao <zhiguo.yao@ihep.ac.cn>, 2019/01/28

//Note:
//  Once new leap second occurs, please modify the getdtai(double), and
//  run Tdate_t::test(), the other 2 getdtai() functions will be printed
//  explicitly in the output - use them to replace the old code in this file.

//Time zone:
//  There is no timezone concept in this code. But, as there are some
//  parameters dependent on the time zone (such as the leap second), you
//  may regard all the time as UTC!

//Date and unix second can be converted to each other by "ctime" utilities.
//Define it if you want to use ctime; otherwise a solution based on slalib
//will be adopted.
//Caveat: Better not use ctime, as ctime (such as gmtime() and localtime())
//has the data racing problem for concurrent calls, which would bring
//troubles in multi-thread or parallel processing mode.
#undef HPTR_USE_CTIME

#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef HPTR_USE_CTIME
#include <ctime>
#endif

//Check whether c++11 is supported
#if __cplusplus >= 201103L
#include <chrono>
#else
#include <sys/time.h>
#endif

#ifndef _Nint_func
#define _Nint_func
template<typename T>
int _Nint(T x) { return (x>0)?int(x+0.5):-int(-x+0.5); }
//#define _Nint(x) (((x)>0)?int((x)+0.5):-int(-(x)+0.5))
#endif

//nearest integer in type long long
#ifndef _Nlong_func
#define _Nlong_func
template<typename T>
long long _Nlong(T x) { return (x>0)?(long long)(x+0.5):-(long long)(-x+0.5); }
//#define _Nlong(x) (((x)>0)?(long long)((x)+0.5):-(long long)(-(x)+0.5))
#endif

//nearest integer in type float/double: use std::round(x) in <cmath>
//modulus for double: use std::fmod(x,y) in <cmath>

//normalize the angle to range [angle0,angle0+twopi)
#ifndef _Angfix_func
#define _Angfix_func
#define V_PI 3.1415926535897932
#define V_TWOPI 6.2831853071795864
template<typename T>
T _Angfix(T x, T _angle0 = 0) {
  x -= _angle0;
  return x - (V_TWOPI*std::floor(x/V_TWOPI)) + _angle0;
}
#endif

//around 1000 years after unix zero
#define HPTR_SEC_MAX  31536000000LL
#define HPTR_DAY_MAX  405587
#define HPTR_DATE_MAX 29700615

//around 1000 year before the unix zero
//Time less than the MIN is not valid
#define HPTR_SEC_MIN -31536000000LL
#define HPTR_DAY_MIN -324413
#define HPTR_DATE_MIN 9700615

//around 100 year before the MIN
#define HPTR_SEC_NULL  HPTR_SEC_MIN-3153600000LL
#define HPTR_DAY_NULL  HPTR_DAY_MIN-36500
#define HPTR_DATE_NULL HPTR_DATE_MIN-1000000

class Tunix_t;
class Ttai_t;
class Tdate_t;
class Tmjd_t;
class Ttt_t;


class Tunix_t {
public:
  long long sec;
  double subsec;

  Tunix_t() : sec(HPTR_SEC_NULL), subsec(0) { }
  Tunix_t(long long _sec, double _subsec) : sec(_sec), subsec(_subsec) { }

  inline double span(const Tunix_t &t0) const {
    return (sec-t0.sec) + (subsec-t0.subsec);
  }

  //<0: early to t0; =0: same as t0; >0: later than t0
  inline int compare(const Tunix_t &t0) const {
    if (sec==t0.sec) {
      if (subsec>t0.subsec) return 1;
      else if (subsec<t0.subsec) return -1;
      else return 0;
    }
    else if (sec>t0.sec) return 1;
    else return -1;
  }

  //Replaced with the global domain function
  //inline bool operator< (const Tunix_t& t0) const { return (compare(t0)<0); }
  //inline bool operator> (const Tunix_t& t0) const { return (compare(t0)>0); }
  //inline bool operator<=(const Tunix_t& t0) const { return (compare(t0)<=0); }
  //inline bool operator>=(const Tunix_t& t0) const { return (compare(t0)>=0); }
  //inline bool operator==(const Tunix_t& t0) const { return (compare(t0)==0); }
  //inline bool operator!=(const Tunix_t& t0) const { return (compare(t0)!=0); }
  inline static bool lessthan(Tunix_t a, Tunix_t b) { return a.compare(b)<0; }

  inline void add(long long dsec, double dsubsec) {
    long long secp = (long long)((sec+dsec)+(subsec+dsubsec));
    subsec += dsubsec - (secp-sec-dsec);
    sec = secp;
  }

  //in order to keep precision, please use small dsec
  inline void add(double dsec) { add(0,dsec); }
  inline double unpack() const { return sec + subsec; }

  inline void setmin() { sec = HPTR_SEC_MIN; subsec = 0.; }
  inline void setmax() { sec = HPTR_SEC_MAX; subsec = 0.; }
  inline void setnull() { sec = HPTR_SEC_NULL; subsec = 0.; }
  inline bool ismin() { return sec==HPTR_SEC_MIN; }
  inline bool ismax() { return sec==HPTR_SEC_MAX; }
  inline bool isnull() { return sec<HPTR_SEC_MIN; }

  void setnow();

#ifdef HPTR_USE_CTIME
  struct tm gettm();
#endif /* HPTR_USE_CTIME */
  int getdtai() const;

  inline void assign(const Tunix_t &tu) {
    sec = tu.sec;
    subsec = tu.subsec;
  }

  inline void assign(long long _sec, double _subsec) {
    sec = _sec; subsec = 0.;
    add(0,_subsec);
  }

  void assign(const Tmjd_t &mjd);
  void assign(const Tdate_t &td);
  void assign(const Ttai_t &tai);
  void assign(const Ttt_t &tt);

  void print(const char *padding = 0) const;
};


class Ttai_t {
public:
  long long sec;
  double subsec;

  Ttai_t() : sec(HPTR_SEC_NULL), subsec(0) { }
  Ttai_t(long long _sec, double _subsec) : sec(_sec), subsec(_subsec) { }

  inline double span(const Ttai_t &t0) const {
    return (sec-t0.sec) + (subsec-t0.subsec);
  }

  //<0: early to t0; =0: same as t0; >0: later than t0
  inline int compare(const Ttai_t &t0) const {
    if (sec==t0.sec) {
      if (subsec>t0.subsec) return 1;
      else if (subsec<t0.subsec) return -1;
      else return 0;
    }
    else if (sec>t0.sec) return 1;
    else return -1;
  }

  inline static bool lessthan(Ttai_t a, Ttai_t b) { return a.compare(b)<0; }

  inline void add(long long dsec, double dsubsec) {
    long long secp = (long long)((sec+dsec)+(subsec+dsubsec));
    subsec += dsubsec - (secp-sec-dsec);
    sec = secp;
  }

  //in order to keep precision, please use small dsec
  inline void add(double dsec) { add(0,dsec); }

  inline double unpack() const { return sec + subsec; }

  inline void setmin() { sec = HPTR_SEC_MIN; subsec = 0.; }
  inline void setmax() { sec = HPTR_SEC_MAX; subsec = 0.; }
  inline void setnull() { sec = HPTR_SEC_NULL; subsec = 0.; }
  inline bool ismin() { return sec==HPTR_SEC_MIN; }
  inline bool ismax() { return sec==HPTR_SEC_MAX; }
  inline bool isnull() { return sec<HPTR_SEC_MIN; }

  void setnow();

  int getdtai() const;

  inline void assign(const Ttai_t &tai) {
    sec = tai.sec;
    subsec = tai.subsec;
  }

  inline void assign(long long _sec, double _subsec) {
    sec = _sec; subsec = 0.;
    add(0,_subsec);
  }

  void assign(const Tunix_t &tu);
  void assign(const Tdate_t &td);
  void assign(const Tmjd_t &mjd);
  void assign(const Ttt_t &tt);

  void print(const char *padding = 0) const;
};


class Tmjd_t {
public:
  int day;
  int sec;
  double subsec;

  Tmjd_t() : day(HPTR_DAY_NULL), sec(0), subsec(0) { }
  Tmjd_t(int _day, int _sec, double _subsec)
    : day(_day), sec(_sec), subsec(_subsec) { }

  inline double span(const Tmjd_t &t0) const {
    return (day-t0.day)*86400. + (sec-t0.sec) + (subsec-t0.subsec);
  }

  //<0: early to t0; =0: same as t0; >0: later than t0
  inline int compare(const Tmjd_t &t0) const {
    if (day==t0.day) {
      if (sec==t0.sec) {
        if (subsec>t0.subsec) return 1;
        else if (subsec<t0.subsec) return -1;
        else return 0;
      }
      else if (sec>t0.sec) return 1;
      else return -1;
    }
    else if (day>t0.day) return 1;
    else return -1;
  }

  inline static bool lessthan(Tmjd_t a, Tmjd_t b) { return a.compare(b)<0; }

  inline void add(int dday, int dsec, double dsubsec) {
    long long isec = day*86400LL + sec;
    long long isecp = (long long)((day+dday)*86400LL+
        (sec+dsec)+(subsec+dsubsec));
    subsec += dsubsec - (isecp-dday*86400LL-isec-dsec);
    day = isecp/86400LL;
    sec = isecp%86400LL;
  }

  //in order to keep precision, please use small dsec
  inline void add(double dsec) { add(0,0,dsec); }

  inline double unpack() const { return day + (sec+subsec)/86400.; }

  inline void setmin() { day = HPTR_DAY_MIN; sec = 0, subsec = 0.; }
  inline void setmax() { day = HPTR_DAY_MAX; sec = 0, subsec = 0.; }
  inline void setnull() { day = HPTR_DAY_NULL; sec = 0, subsec = 0.; }
  inline bool ismin() { return day==HPTR_DAY_MIN; }
  inline bool ismax() { return day==HPTR_DAY_MAX; }
  inline bool isnull() { return day<HPTR_DAY_MIN; }

  void setnow();

  static long long getmjd(int iy, int im, int id);
  static double getdtai(double mjd);

  double getdtai() const {
    return getdtai(day+(sec+subsec)/86400.);
  }

  inline void assign(const Tmjd_t &mjd) {
    day = mjd.day;
    sec = mjd.sec;
    subsec = mjd.subsec;
  }

  inline void assign(int _day, int _sec, double _subsec) {
    day = _day; sec = _sec; subsec = 0.;
    add(0,0,_subsec);
  }

  inline void assign(double mjd) {
    day = int(mjd+1.e-15);
    sec = int((mjd-day)*86400.);
    subsec = (mjd-day)*86400. - sec;
  }

  void assign(const Tunix_t &tu);
  void assign (const Tdate_t &td);
  void assign(const Ttai_t &tai);
  void assign(const Ttt_t &tt);

  void print(const char *padding = 0) const;
};


//TT expressed in MJD
class Ttt_t {
public:
  int day;
  int sec;
  double subsec;

  Ttt_t() : day(HPTR_DAY_NULL), sec(0), subsec(0) { }
  Ttt_t(int _day, int _sec, double _subsec)
    : day(_day), sec(_sec), subsec(_subsec) { }

  inline double span(const Ttt_t &t0) const {
    return (day-t0.day)*86400. + (sec-t0.sec) + (subsec-t0.subsec);
  }

  //<0: early to t0; =0: same as t0; >0: later than t0
  inline int compare(const Ttt_t &t0) const {
    if (day==t0.day) {
      if (sec==t0.sec) {
        if (subsec>t0.subsec) return 1;
        else if (subsec<t0.subsec) return -1;
        else return 0;
      }
      else if (sec>t0.sec) return 1;
      else return -1;
    }
    else if (day>t0.day) return 1;
    else return -1;
  }

  inline static bool lessthan(Ttt_t a, Ttt_t b) { return a.compare(b)<0; }

  inline void add(int dday, int dsec, double dsubsec) {
    long long isec = day*86400LL + sec;
    long long isecp = (long long)((day+dday)*86400LL+
        (sec+dsec)+(subsec+dsubsec));
    subsec += dsubsec - (isecp-dday*86400LL-isec-dsec);
    day = isecp/86400LL;
    sec = isecp%86400LL;
  }

  //in order to keep precision, please use small dsec
  inline void add(double dsec) { add(0,0,dsec); }

  inline void setmin() { day = HPTR_DAY_MIN; sec = 0, subsec = 0.; }
  inline void setmax() { day = HPTR_DAY_MAX; sec = 0, subsec = 0.; }
  inline void setnull() { day = HPTR_DAY_NULL; sec = 0, subsec = 0.; }
  inline bool ismin() { return day==HPTR_DAY_MIN; }
  inline bool ismax() { return day==HPTR_DAY_MAX; }
  inline bool isnull() { return day<HPTR_DAY_MIN; }

  void setnow();

  inline void assign(const Ttt_t &tt) {
    day = tt.day;
    sec = tt.sec;
    subsec = tt.subsec;
  }

  inline void assign(double tt) {
    day = int(tt+1.e-15);
    sec = int((tt-day)*86400.);
    subsec = (tt-day)*86400. - sec;
  }

  inline void assign(int _day, int _sec, double _subsec) {
    day = _day; sec = _sec; subsec = 0.;
    add(0,0,_subsec);
  }

  void assign(const Tmjd_t &mjd);
  void assign(const Tunix_t &tu);
  void assign(const Tdate_t &td);
  void assign(const Ttai_t &tai);

  void print(const char *padding = 0) const;
};


class Tdate_t {
public:
  int date;
  int sec;
  double subsec;

  Tdate_t() : date(HPTR_DATE_NULL), sec(0), subsec(0) { }
  Tdate_t(int _date, int _sec, double _subsec)
    : date(_date), sec(_sec), subsec(_subsec) { }

  double span(const Tdate_t &t0) const;
  //<0: early to t0; =0: same as t0; >0: later than t0
  inline int compare(const Tdate_t &t0) const {
    if (date==t0.date) {
      if (sec==t0.sec) {
        if (subsec>t0.subsec) return 1;
        else if (subsec<t0.subsec) return -1;
        else return 0;
      }
      else if (sec>t0.sec) return 1;
      else return -1;
    }
    else if (date>t0.date) return 1;
    else return -1;
  }

  inline static bool lessthan(Tdate_t a, Tdate_t b) { return a.compare(b)<0; }

  void add(int dday, int dsec, double dsubsec);
  //in order to keep precision, please use small dsec
  inline void add(double dsec) { add(0,0,dsec); }

  inline void unpack(int &iyear, int &imon, int &iday) const {
    iyear = date/10000;
    imon = date/100%100;
    iday = date%100;
  }

  inline int pack(int iyear, int imon, int iday) const {
    if (iyear>=0 && iyear<=49) iyear += 2000;
    else if (iyear>=50 && iyear<=99) iyear += 1900;
    return iyear*10000 + imon*100 + iday;
  }

  inline int packtime() const {
    return sec/3600*10000 + sec%3600/60*100 + sec%60;
  }

  inline void setmin() { date = HPTR_DATE_MIN; sec = 0, subsec = 0.; }
  inline void setmax() { date = HPTR_DATE_MAX; sec = 0, subsec = 0.; }
  inline void setnull() { date = HPTR_DATE_NULL; sec = 0, subsec = 0.; }
  inline bool ismin() { return date==HPTR_DATE_MIN; }
  inline bool ismax() { return date==HPTR_DATE_MAX; }
  inline bool isnull() { return date<HPTR_DATE_MIN; }

  void setnow();

  inline void assign(const Tdate_t &td) {
    date = td.date;
    sec = td.sec;
    subsec = td.subsec;
  }

  inline void assign(int iyear, int imon, int iday,
      int ihour, int imin, int isec,
      double fsec = 0.) {
    date = pack(iyear,imon,iday);
    sec = ihour*3600 + imin*60 + isec;
    subsec = fsec;
  }

  inline void assign(int _date, int _sec, double _subsec) {
    date = _date; sec = _sec; subsec = 0.;
    add(0,0,_subsec);
  }

  inline void assign(long long datetime, double fsec) { 
    date = pack(datetime/10000000000LL,datetime/100000000LL%100LL,
        datetime/1000000LL%100LL);
    sec = datetime/10000LL%100LL*3600LL + datetime/100LL%100LL*60LL +
      datetime%100LL;
    subsec = fsec;
  }

  inline void assignh(int yyyymmdd, int hhmmss, double fsec) { 
    date = pack(yyyymmdd/10000,yyyymmdd/100%100,yyyymmdd%100);
    sec = hhmmss/10000*3600 + hhmmss/100%100*60 + hhmmss%100;
    subsec = fsec;
  }

  void assign(const Tmjd_t &mjd);
  void assign(const Tunix_t &tu);
  void assign(const Ttai_t &tai);
  void assign(const Ttt_t &tt);

  inline void print(const char *padding = 0) const {
    if (padding) std::cout << padding;
    std::cout << "Date time: "
              << std::setw(4) << std::setfill('0') << date/10000 << "/"
              << std::setw(2) << std::setfill('0') << date%10000/100 << "/"
              << std::setw(2) << std::setfill('0') << date%100 << " "
              << std::setw(2) << std::setfill('0') << sec/3600 << ":"
              << std::setw(2) << std::setfill('0') << sec%3600/60 << ":"
              << std::setw(2) << std::setfill('0') << sec%60 << " + "
              << std::setprecision(12) << subsec
              << " second" << std::setfill(' ') << std::endl;
  }

  static void test();
};


//******************************************************************
//Global domain functions:

inline std::ostream& operator<<(std::ostream& out, const Tunix_t &t) {
  return out << t.sec << " " << std::setprecision(12) << t.subsec;
}

inline std::ostream& operator<<(std::ostream& out, const Ttai_t &t) {
  return out << t.sec << " " << std::setprecision(12) << t.subsec;
}

inline std::ostream& operator<<(std::ostream& out, const Tmjd_t &t) {
  return out << t.day << " " << t.sec << " " << std::setprecision(12) << t.subsec;
}

inline std::ostream& operator<<(std::ostream& out, const Ttt_t &t) {
  return out << t.day << " " << t.sec << " " << std::setprecision(12) << t.subsec;
}

inline std::ostream& operator<<(std::ostream& out, const Tdate_t &t) {
  return out << t.date << " " << t.sec << " " << std::setprecision(12) << t.subsec;
}



//These for the requirement of the standard library such std::sort
inline bool operator==(const Tunix_t &l, const Tunix_t &r) { return l.compare(r)==0; }
inline bool operator!=(const Tunix_t &l, const Tunix_t &r) { return l.compare(r)!=0; }
inline bool operator< (const Tunix_t &l, const Tunix_t &r) { return l.compare(r) <0; }
inline bool operator> (const Tunix_t &l, const Tunix_t &r) { return l.compare(r) >0; }
inline bool operator<=(const Tunix_t &l, const Tunix_t &r) { return l.compare(r)<=0; }
inline bool operator>=(const Tunix_t &l, const Tunix_t &r) { return l.compare(r)>=0; }


inline bool operator==(const Ttai_t &l, const Ttai_t &r) { return l.compare(r)==0; }
inline bool operator!=(const Ttai_t &l, const Ttai_t &r) { return l.compare(r)!=0; }
inline bool operator< (const Ttai_t &l, const Ttai_t &r) { return l.compare(r) <0; }
inline bool operator> (const Ttai_t &l, const Ttai_t &r) { return l.compare(r) >0; }
inline bool operator<=(const Ttai_t &l, const Ttai_t &r) { return l.compare(r)<=0; }
inline bool operator>=(const Ttai_t &l, const Ttai_t &r) { return l.compare(r)>=0; }

inline bool operator==(const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r)==0; }
inline bool operator!=(const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r)!=0; }
inline bool operator< (const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r) <0; }
inline bool operator> (const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r) >0; }
inline bool operator<=(const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r)<=0; }
inline bool operator>=(const Tmjd_t &l, const Tmjd_t &r) { return l.compare(r)>=0; }


inline bool operator==(const Ttt_t &l, const Ttt_t &r) { return l.compare(r)==0; }
inline bool operator!=(const Ttt_t &l, const Ttt_t &r) { return l.compare(r)!=0; }
inline bool operator< (const Ttt_t &l, const Ttt_t &r) { return l.compare(r) <0; }
inline bool operator> (const Ttt_t &l, const Ttt_t &r) { return l.compare(r) >0; }
inline bool operator<=(const Ttt_t &l, const Ttt_t &r) { return l.compare(r)<=0; }
inline bool operator>=(const Ttt_t &l, const Ttt_t &r) { return l.compare(r)>=0; }

inline bool operator==(const Tdate_t &l, const Tdate_t &r) { return l.compare(r)==0; }
inline bool operator!=(const Tdate_t &l, const Tdate_t &r) { return l.compare(r)!=0; }
inline bool operator< (const Tdate_t &l, const Tdate_t &r) { return l.compare(r) <0; }
inline bool operator> (const Tdate_t &l, const Tdate_t &r) { return l.compare(r) >0; }
inline bool operator<=(const Tdate_t &l, const Tdate_t &r) { return l.compare(r)<=0; }
inline bool operator>=(const Tdate_t &l, const Tdate_t &r) { return l.compare(r)>=0; }

//Time subtraction is uselful; Time addition is nonsense
inline double operator-(const Tunix_t &l, const Tunix_t &r) { return l.span(r); }
inline double operator-(const Ttai_t &l, const Ttai_t &r) { return l.span(r); }

inline double operator-(const Tmjd_t &l, const Tmjd_t &r) { return l.span(r); }
inline double operator-(const Ttt_t &l, const Ttt_t &r) { return l.span(r); }
inline double operator-(const Tdate_t &l, const Tdate_t &r) { return l.span(r); }


#ifdef hpatimer_cc
//******************************************************************

void Tunix_t::assign(const Tmjd_t &mjd) {
  sec = (mjd.day-40587LL)*86400LL + (long long)(mjd.sec);
  subsec = mjd.subsec;
}

#ifdef HPTR_USE_CTIME
struct tm Tunix_t::gettm() {
  struct tm atm;
  memset(&atm,0,sizeof(atm));

  time_t epoch = 0;
  epoch += sec;
  //forget about the time zone, so use localtime rather than gmtime!
  struct tm *ptm = localtime(&epoch);
  memcpy(&atm,ptm,sizeof(tm));
  return atm;
}

void Tunix_t::assign(const Tdate_t &td) {
  time_t epoch = 0;
  struct tm atm;
  memset(&atm,0,sizeof(atm));

  td.unpack(atm.tm_year,atm.tm_mon,atm.tm_mday);
  atm.tm_year -= 1900;
  atm.tm_mon -= 1;

  //mktime will add the timezone!!!!
  //timezone must be removed! There is no timezone in this toolset
  sec = difftime(mktime(&atm)-timezone,epoch) + td.sec;
  subsec = td.subsec;
}
#else /* HPTR_USE_CTIME */
void Tunix_t::assign(const Tdate_t &td) {
  Tmjd_t mjd;
  mjd.assign(td);
  assign(mjd);
}
#endif /* HPTR_USE_CTIME */


//Must return integer (actually it is true), otherwise make things complex
int Tunix_t::getdtai() const {
  if (sec >= 1483228800LL) return 37;
  if (sec >= 1435708800LL) return 36;
  if (sec >= 1341100800LL) return 35;
  if (sec >= 1230768000LL) return 34;
  if (sec >= 1136073600LL) return 33;
  if (sec >= 915148800LL) return 32;
  if (sec >= 867715200LL) return 31;
  if (sec >= 820454400LL) return 30;
  if (sec >= 773020800LL) return 29;
  if (sec >= 741484800LL) return 28;
  if (sec >= 709948800LL) return 27;
  if (sec >= 662688000LL) return 26;
  if (sec >= 631152000LL) return 25;
  if (sec >= 567993600LL) return 24;
  if (sec >= 489024000LL) return 23;
  if (sec >= 425865600LL) return 22;
  if (sec >= 394329600LL) return 21;
  if (sec >= 362793600LL) return 20;
  if (sec >= 315532800LL) return 19;
  if (sec >= 283996800LL) return 18;
  if (sec >= 252460800LL) return 17;
  if (sec >= 220924800LL) return 16;
  if (sec >= 189302400LL) return 15;
  if (sec >= 157766400LL) return 14;
  if (sec >= 126230400LL) return 13;
  if (sec >= 94694400LL) return 12;
  if (sec >= 78796800LL) return 11;
  if (sec >= 63072000LL) return 10;
  return 0;
}


void Tunix_t::assign(const Ttai_t &tai) {
  sec = tai.sec - tai.getdtai();
  subsec = tai.subsec;
}


void Tunix_t::assign(const Ttt_t &tt) {
  Tmjd_t mjd;
  mjd.assign(tt);
  assign(mjd);
}


//Must return integer (actually it is true), otherwise make things complex
int Ttai_t::getdtai() const {
  if (sec >= 1483228837LL) return 37;
  if (sec >= 1435708836LL) return 36;
  if (sec >= 1341100835LL) return 35;
  if (sec >= 1230768034LL) return 34;
  if (sec >= 1136073633LL) return 33;
  if (sec >= 915148832LL) return 32;
  if (sec >= 867715231LL) return 31;
  if (sec >= 820454430LL) return 30;
  if (sec >= 773020829LL) return 29;
  if (sec >= 741484828LL) return 28;
  if (sec >= 709948827LL) return 27;
  if (sec >= 662688026LL) return 26;
  if (sec >= 631152025LL) return 25;
  if (sec >= 567993624LL) return 24;
  if (sec >= 489024023LL) return 23;
  if (sec >= 425865622LL) return 22;
  if (sec >= 394329621LL) return 21;
  if (sec >= 362793620LL) return 20;
  if (sec >= 315532819LL) return 19;
  if (sec >= 283996818LL) return 18;
  if (sec >= 252460817LL) return 17;
  if (sec >= 220924816LL) return 16;
  if (sec >= 189302415LL) return 15;
  if (sec >= 157766414LL) return 14;
  if (sec >= 126230413LL) return 13;
  if (sec >= 94694412LL) return 12;
  if (sec >= 78796811LL) return 11;
  if (sec >= 63072010LL) return 10;
  return 0;
}


void Ttai_t::assign(const Tunix_t &tu) {
  sec = tu.sec + tu.getdtai();
  subsec = tu.subsec;
}


void Ttai_t::assign(const Tdate_t &td) {
  Tunix_t tu;
  tu.assign(td);
  assign(tu);
}


void Ttai_t::assign(const Tmjd_t &mjd) {
  Tunix_t tu;
  tu.assign(mjd);
  assign(tu);
}


void Ttai_t::assign(const Ttt_t &tt) {
  Tmjd_t mjd;
  mjd.assign(tt);
  assign(mjd);
}


long long Tmjd_t::getmjd(int iy, int im, int id) {
  //This is made based on slalib
  //Gregorian calendar to Modified Julian Date
  if (iy>=0 && iy<=49) iy += 2000;
  else if (iy>=50 && iy<=99) iy += 1900;

  long long iyL, imL;

  //Month lengths in days
  static int mtab[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

  //Validate year
  if (iy<-4699) {
    std::cout << "Warning! Tmjd_t::getmjd: year number = " << iy
              << " is not supported!" << std::endl;
    return -1;
  }

  //Validate month
  if (im<1 || im>12) {
    std::cout << "Warning! Tmjd_t::getmjd: month number = " << im
              << " is not valid!" << std::endl;
    return -1;
  }

  //Allow for leap year
  mtab[1] = (iy%4==0 && (iy%100!=0 || iy%400==0)) ?  29 : 28;

  //Validate day
  if (id < 1 || id > mtab[im-1]) {
    std::cout << "Warning! Tmjd_t::getmjd: day number = " << id
              << " is not valid!" << std::endl;
    return -1;
  }

  //Lengthen year and month numbers to avoid overflow
  iyL = (long long)iy;
  imL = (long long)im;

  //Perform the conversion
  long long mjd = 
      (( 1461LL * ( iyL - ( 12LL - imL ) / 10LL + 4712LL ) ) / 4LL +
       ( 306LL * ( ( imL + 9LL ) % 12LL ) + 5LL ) / 10LL -
       ( 3LL * ( ( iyL - ( 12LL - imL ) / 10LL + 4900LL ) / 100LL ) ) / 4LL +
       (long long)id - 2399904LL );

  return mjd;
}


double Tmjd_t::getdtai(double mjd) {
  //This is made based on slalib
  //return TAI-UTC in seconds
  //2017 Jan 1
  if ( mjd >= 57754.0 ) return 37.0;
  //2015 July 1
  if ( mjd >= 57204.0 ) return 36.0;
  //2012 July 1
  if ( mjd >= 56109.0 ) return 35.0;
  //2009 January 1
  if ( mjd >= 54832.0 ) return 34.0;
  //2006 January 1
  if ( mjd >= 53736.0 ) return 33.0;
  //1999 January 1
  if ( mjd >= 51179.0 ) return 32.0;
  //1997 July 1
  if ( mjd >= 50630.0 ) return 31.0;
  //1996 January 1
  if ( mjd >= 50083.0 ) return 30.0;
  //1994 July 1
  if ( mjd >= 49534.0 ) return 29.0;
  //1993 July 1
  if ( mjd >= 49169.0 ) return 28.0;
  //1992 July 1
  if ( mjd >= 48804.0 ) return 27.0;
  //1991 January 1
  if ( mjd >= 48257.0 ) return 26.0;
  //1990 January 1
  if ( mjd >= 47892.0 ) return 25.0;
  //1988 January 1
  if ( mjd >= 47161.0 ) return 24.0;
  //1985 July 1
  if ( mjd >= 46247.0 ) return 23.0;
  //1983 July 1
  if ( mjd >= 45516.0 ) return 22.0;
  //1982 July 1
  if ( mjd >= 45151.0 ) return 21.0;
  //1981 July 1
  if ( mjd >= 44786.0 ) return 20.0;
  //1980 January 1
  if ( mjd >= 44239.0 ) return 19.0;
  //1979 January 1
  if ( mjd >= 43874.0 ) return 18.0;
  //1978 January 1
  if ( mjd >= 43509.0 ) return 17.0;
  //1977 January 1
  if ( mjd >= 43144.0 ) return 16.0;
  //1976 January 1
  if ( mjd >= 42778.0 ) return 15.0;
  //1975 January 1
  if ( mjd >= 42413.0 ) return 14.0;
  //1974 January 1
  if ( mjd >= 42048.0 ) return 13.0;
  //1973 January 1
  if ( mjd >= 41683.0 ) return 12.0;
  //1972 July 1
  if ( mjd >= 41499.0 ) return 11.0;
  //1972 January 1
  if ( mjd >= 41317.0 ) return 10.0;
  //1968 February 1
  if ( mjd >= 39887.0 ) return 4.2131700 + ( mjd - 39126.0 ) * 0.002592;
  //1966 January 1
  if ( mjd >= 39126.0 ) return 4.3131700 + ( mjd - 39126.0 ) * 0.002592;
  //1965 September 1
  if ( mjd >= 39004.0 ) return 3.8401300 + ( mjd - 38761.0 ) * 0.001296;
  //1965 July 1
  if ( mjd >= 38942.0 ) return 3.7401300 + ( mjd - 38761.0 ) * 0.001296;
  //1965 March 1
  if ( mjd >= 38820.0 ) return 3.6401300 + ( mjd - 38761.0 ) * 0.001296;
  //1965 January 1
  if ( mjd >= 38761.0 ) return 3.5401300 + ( mjd - 38761.0 ) * 0.001296;
  //1964 September 1
  if ( mjd >= 38639.0 ) return 3.4401300 + ( mjd - 38761.0 ) * 0.001296;
  //1964 April 1
  if ( mjd >= 38486.0 ) return 3.3401300 + ( mjd - 38761.0 ) * 0.001296;
  //1964 January 1
  if ( mjd >= 38395.0 ) return 3.2401300 + ( mjd - 38761.0 ) * 0.001296;
  //1963 November 1
  if ( mjd >= 38334.0 ) return 1.9458580 + ( mjd - 37665.0 ) * 0.0011232;
  //1962 January 1
  if ( mjd >= 37665.0 ) return 1.8458580 + ( mjd - 37665.0 ) * 0.0011232;
  //1961 August 1
  if ( mjd >= 37512.0 ) return 1.3728180 + ( mjd - 37300.0 ) * 0.001296;
  //1961 January 1
  if ( mjd >= 37300.0 ) return 1.4228180 + ( mjd - 37300.0 ) * 0.001296;
  //Before that
                        return 1.4178180 + ( mjd - 37300.0 ) * 0.001296;
}


void Tmjd_t::assign(const Tunix_t &tu) {
  //epoch 0: 40587
  day = 40587 + tu.sec/86400;
  sec = tu.sec%86400;
  subsec = tu.subsec;
}


void Tmjd_t::assign (const Tdate_t &td) {
  int iy, im, id;
  td.unpack(iy,im,id);
  day = getmjd(iy,im,id);
  sec = td.sec;
  subsec = td.subsec;
}


void Tmjd_t::assign(const Ttai_t &tai) {
  Tunix_t tu;
  tu.assign(tai);
  assign(tu);
}


void Tmjd_t::assign(const Ttt_t &tt) {
  day = tt.day;
  sec = tt.sec;
  subsec = tt.subsec;
  add(-32.184);
}


void Ttt_t::assign(const Tmjd_t &mjd) {
  day = mjd.day;
  sec = mjd.sec;
  subsec = mjd.subsec;
  add(32.184);
}


void Ttt_t::assign(const Tdate_t &td) {
  Tmjd_t mjd;
  mjd.assign(td);
  assign(mjd);
}


void Ttt_t::assign(const Tunix_t &tu) {
  Tmjd_t mjd;
  mjd.assign(tu);
  assign(mjd);
}


void Ttt_t::assign(const Ttai_t &tai) {
  Tmjd_t mjd;
  mjd.assign(tai);
  assign(mjd);
}


double Tdate_t::span(const Tdate_t &t0) const {
  if (date/100==t0.date/100) {
    return (date-t0.date)*86400. + (sec-t0.sec) + (subsec-t0.subsec);
  }
  else {
    Tmjd_t mjd;
    mjd.assign(*this);
    Tmjd_t mjd0;
    mjd0.assign(t0);
    return mjd.span(mjd0);
  }
}


void Tdate_t::add(int dday, int dsec, double dsubsec) {
  Tmjd_t mjd;
  mjd.assign(*this);
  mjd.add(dday,dsec,dsubsec);
  assign(mjd);
}


void Tdate_t::assign(const Tmjd_t &mjd) {
  //This is made based on slalib
  //Check if date is acceptable
  if ((mjd.day <= -2395520) || (mjd.day >= 1e9)) return;

  //Express day in Gregorian calendar
  long long jd = (long long)(mjd.day) + 2400001LL;
  long long n4 = 4LL*
    (jd+((6LL*((4LL*jd-17918LL)/146097LL))/4LL+1LL)/2LL-37LL);
  long long nd10 = 10LL*(((n4-237LL)%1461LL)/4LL)+5LL;

  date = pack(n4/1461LL-4712LL,
              ((nd10/306LL+2LL)%12LL)+1LL,
              (nd10%306LL)/10LL+1LL);
  sec = mjd.sec;
  subsec = mjd.subsec;
}


#ifdef HPTR_USE_CTIME
void Tdate_t::assign(const Tunix_t &tu) {
  struct tm atm = tu.gettm();
  date = (atm.tm_year+1900)*10000 + (atm.tm_mon+1)*100 + atm.tm_mday;
  sec = atm.tm_hour*3600 + atm.tm_min*60 + atm.tm_sec;
  subsec = tu.subsec;
}
#else /* HPTR_USE_CTIME */
void Tdate_t::assign(const Tunix_t &tu) {
  Tmjd_t mjd;
  mjd.assign(tu);
  assign(mjd);
}
#endif /* HPTR_USE_CTIME */


void Tdate_t::assign(const Ttai_t &tai) {
  Tunix_t tu;
  tu.assign(tai);
  assign(tu);
}


void Tdate_t::assign(const Ttt_t &tt) {
  Tmjd_t mjd;
  mjd.assign(tt);
  assign(mjd);
}


void Tunix_t::print(const char *padding) const {
  if (padding) std::cout << padding;
  std::cout << "Unix time: " << sec << " + " << std::setprecision(12)
            << subsec << " second" << std::endl;
  Tdate_t td;
  td.assign(*this);
  td.print(padding);
}


void Ttai_t::print(const char *padding) const {
  if (padding) std::cout << padding;
  std::cout << "TAI time: " << sec << " + " << std::setprecision(12)
            << subsec << " second" << std::endl;
  Tdate_t td;
  td.assign(*this);
  td.print(padding);
}


void Tmjd_t::print(const char *padding) const {
  if (padding) std::cout << padding;
  std::cout << "MJD: " << day << " day + (" << sec << " + "
            << std::setprecision(12) << subsec << ") second" << std::endl;
  Tdate_t td;
  td.assign(*this);
  td.print(padding);
}


void Ttt_t::print(const char *padding) const {
  if (padding) std::cout << padding;
  std::cout << "Terrestrial Time: " << day << " day + (" << sec << " + "
            << std::setprecision(12) << subsec << ") second" << std::endl;
  Tdate_t td;
  td.assign(*this);
  td.print(padding);
}


void Tunix_t::setnow() {
#if __cplusplus >= 201103L
  unsigned long long _now =
      std::chrono::duration_cast<std::chrono::microseconds>
      (std::chrono::system_clock::now().time_since_epoch()).count();
  sec = _now/1000000ULL;
  subsec = _now%1000000ULL*1.e-6;
#else
  struct timeval _tp;
  gettimeofday(&_tp, NULL);
  sec = _tp.tv_sec;
  subsec = _tp.tv_usec*1.e-6;
#endif
}

void Ttai_t::setnow() {
  Tunix_t ut;
  ut.setnow();
  assign(ut);
}

void Tmjd_t::setnow() {
  Tunix_t ut;
  ut.setnow();
  assign(ut);
}


void Ttt_t::setnow() {
  Tunix_t ut;
  ut.setnow();
  assign(ut);
}


void Tdate_t::setnow() {
  Tunix_t ut;
  ut.setnow();
  assign(ut);
}


void Tdate_t::test() {
  Tunix_t t0(0,0.);
  Tmjd_t mjd0;
  mjd0.assign(t0);

  std::cout << "MJD of unix epoch 0 = " << mjd0.day << std::endl;

  //Everytime when you update the leap second in getdtai(double),
  //please run this test to update the rest getdtai(...)
  double vmjd[28] = { 57754.0, 57204.0, 56109.0, 54832.0, 53736.0,
                      51179.0, 50630.0, 50083.0, 49534.0, 49169.0,
                      48804.0, 48257.0, 47892.0, 47161.0, 46247.0,
                      45516.0, 45151.0, 44786.0, 44239.0, 43874.0,
                      43509.0, 43144.0, 42778.0, 42413.0, 42048.0,
                      41683.0, 41499.0, 41317.0 };

  for (int j=0; j<3; ++j) {
    if (j==1) 
      std::cout << "  double Ttai_t::getdtai() {" << std::endl;
    else if (j==2)
      std::cout << "  double Tunix_t::getdtai() {" << std::endl;
    else if (j==3)
      std::cout << "  double Tdate_t::getdtai() {" << std::endl;

    Tmjd_t mjd;
    Tunix_t tu;
    Tdate_t td;
    Tunix_t tup;
    Ttai_t tai;

    for (int i=0; i<28; ++i) {
      mjd.assign(vmjd[i]);
      tu.assign(mjd);
      td.assign(mjd);
      tup.assign(td);

      tai.sec = tu.sec;
      tai.subsec = tu.subsec;
      double dtai = mjd.getdtai();
      tai.sec += _Nlong(dtai);

      int iyear, imon, iday;
      td.unpack(iyear,imon,iday);

      if (j==0) std::cout << "i = " << i << " mjd = " << vmjd[i]
                          << " sec = " << tu.sec << " secp = " << tup.sec
                          << " date = " << iyear << "/" << imon << "/"
                          << iday << "+" << td.sec+td.subsec << std::endl;
      else if (j==1) std::cout << "    if (sec >= " << tai.sec
        << "LL) return " << dtai << ";" << std::endl;
      else if (j==2) std::cout << "    if (sec >= " << tu.sec
        << "LL) return " << dtai << ";" << std::endl;
      else if (j==3) std::cout << "    if (date >= " << td.date
        << ") return " << dtai << ";" << std::endl;
    }

    if (j>0) {
      std::cout << "    return 0;" << std::endl;
      std::cout << "  }" << std::endl;
    }
  }
}
#endif /* hpatimer_cc */
#endif /* hpatimer_h */

