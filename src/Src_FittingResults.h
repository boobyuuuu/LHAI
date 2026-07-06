# ifndef Src_FittingResults_h
# define Src_FittingResults_h

# include <iostream>
# include <string>
# include <vector>

using namespace std;

class Src_FittingResults {

    public :

        Src_FittingResults();
        ~Src_FittingResults();
        void Init();

        // Component parameters
        Src_Template *Template;
        void SetTemplate(Src_Template *temp);

        // SED points
        double **WEmedian;
        double **KEmedian;
        double **KEmiddle;

        double **WFlux;
        double **WFluxErr;
        double **KFlux;
        double **KFluxErr;
        double **KFluxMd;
        double **KFluxMdErr;
        void DrawSED();

        // TS_src
        double *TS_comp;
        double **WTS_comp_bin;
        double **KTS_comp_bin;

        // TSMap
        void DrawTSMap();

};

Src_FittingResults::Src_FittingResults(){}

Src_FittingResults::~Src_FittingResults(){

    for (int ii=0;ii<Template->NComp;ii++){
        delete[] WEmedian[ii];
        delete[] KEmedian[ii];
        delete[] KEmiddle[ii];
        delete[] WFlux[ii];
        delete[] WFluxErr[ii];
        delete[] KFlux[ii];
        delete[] KFluxErr[ii];
        delete[] KFluxMd[ii];
        delete[] KFluxMdErr[ii];
        delete[] WTS_comp_bin[ii];
        delete[] KTS_comp_bin[ii];
    }

    delete[] WEmedian;
    delete[] KEmedian;
    delete[] KEmiddle;
    delete[] WFlux;
    delete[] WFluxErr;
    delete[] KFlux;
    delete[] KFluxErr;
    delete[] KFluxMd;
    delete[] KFluxMdErr;
    delete[] WTS_comp_bin;
    delete[] KTS_comp_bin;
    delete[] TS_comp;

}

void Src_FittingResults::SetTemplate(Src_Template *temp){ Template = temp; }

void Src_FittingResults::Init(){

    WEmedian = new double*[Template->NComp];
    KEmedian = new double*[Template->NComp];
    KEmiddle = new double*[Template->NComp];
    WFlux    = new double*[Template->NComp];
    WFluxErr = new double*[Template->NComp];
    KFlux    = new double*[Template->NComp];
    KFluxErr = new double*[Template->NComp];
    KFluxMd  = new double*[Template->NComp];
    KFluxMdErr   = new double*[Template->NComp];
    WTS_comp_bin = new double*[Template->NComp];
    KTS_comp_bin = new double*[Template->NComp];
    TS_comp      = new double[Template->NComp];

    for (int ii=0;ii<Template->NComp;ii++){
        WEmedian[ii] = new double[cf.NnhitUsed];
        WFlux[ii]    = new double[cf.NnhitUsed];
        WFluxErr[ii] = new double[cf.NnhitUsed];
        WTS_comp_bin[ii] = new double[cf.NnhitUsed];
        for (int jj=0;jj<cf.NnhitUsed;jj++){
            WEmedian[ii][jj] = 0;     
            WFlux[ii][jj]    = 0;        
            WFluxErr[ii][jj] = 0;     
            WTS_comp_bin[ii][jj] = 0;
        }

        KEmedian[ii] = new double[cf.KNEbinUsed];
        KEmiddle[ii] = new double[cf.KNEbinUsed];
        KFlux[ii]    = new double[cf.KNEbinUsed];
        KFluxErr[ii] = new double[cf.KNEbinUsed];
        KFluxMd[ii]  = new double[cf.KNEbinUsed];
        KFluxMdErr[ii]   = new double[cf.KNEbinUsed];
        KTS_comp_bin[ii] = new double[cf.KNEbinUsed];
        for (int jj=0;jj<cf.KNEbinUsed;jj++){
            KEmedian[ii][jj] = 0;     
            KEmiddle[ii][jj] = 0;     
            KFlux[ii][jj]    = 0;        
            KFluxErr[ii][jj] = 0;     
            KFluxMd[ii][jj]  = 0;      
            KFluxMdErr[ii][jj]   = 0;   
            KTS_comp_bin[ii][jj] = 0; 
        }

        TS_comp[ii] = 0;
    }

}

void Src_FittingResults::DrawSED(){

    return;

}

void Src_FittingResults::DrawTSMap(){

    return;

}

# endif
