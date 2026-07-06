# include <iostream>
# include "papi.h"

using namespace std;

int main(){

    double RA = 83.63, DEC = 22.02;
    double MJD = 59000;
    double ZEN, AZI;
    papi::eqm2hcs(MJD, 0, RA*papi::degrad, DEC*papi::degrad, ZEN, AZI);
    cout<<ZEN*papi::raddeg<<" "<<AZI*papi::raddeg<<endl;
    return 0;

}
