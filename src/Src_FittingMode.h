# ifndef Src_FittingMode_h
# define Src_FittingMode_h

# include <iostream>
# include <string>
# include <vector>
# include <random>
# include <unistd.h>

# include "TRandom.h"
# include "TMarker.h"
# include "TEllipse.h"

using namespace std;

class Src_FittingMode {

    public :

        Src_FittingMode();
        ~Src_FittingMode();
        void Init(Src_Template* Template);
        void SetParStatus(int ipmode, int ismode, int icomp, Src_Template* Template, int binloop);
        void SetNPar_total(int ipmode, int ismode, int icomp, Src_Template* Template);

        int nPmode;
        int *NSmode;
        int *NSmode_int;
        int nSmode;
        string *ModeTag;
        string *MinAlgo;
        double *ConfLevel;
        int *NPar_total;
        bool *ErrOpt;
        int **NBinUsed;
        int **ParStatus;

        // Results
        void InitRes(Src_Template* Template);
        double TS_WCDA;
        double TS_KM2A;
        double TS_Total;
        double logL_total;
        int FITSTATUS;
        double *ParVal;
        double *ParErr;
        double **Emedian;
        double **FNorm;
        double **FNormErr;
        double **Flux;
        double **FluxErr;
        double *SEDChi2;
        double **TS_Src;
        int **FitStatus_Src;
        double TS_UL[3];
        double **TS_Bin;
        int **FitStatus_Bin;
        double **FNormUL;
        double **FluxUL;
        int *FitStatus_UL;
        void CalFlux(int ipmode, Src_Template* Template);
        void CalTS(int ipmode, Src_Template* Template);
        void CalSEDChi2(Src_Template* Template);
        void DrawSED(Src_Template* Template);
        void PrintRes(Src_Template* Template, int neffbins);
        void MkOutdir();
        void OutPara(Src_Template* Template, int DGEsFlag, int SrcsFlag, double epiv_global, int parstatus_global[4]);
        void DrawSigMap(Src_Template* Template, vector<long int> cellid, double **Wnon, double **Wnbkg, double **Knon, double **Knbkg, double **Wnmodel_convo, double **Knmodel_convo, double **Wpsf, double **Kpsf);
        void OutConvExcess(Src_Template* Template, vector<long int> cellid, double **Wnmodel_convo, double **Knmodel_convo);
        void OutConvExcess_ai(Src_Template* Template, vector<long int> cellid, double **Wnmodel_convo, double **Knmodel_convo, double **WNbkg, double **KNbkg, int poisson_flag, UInt_t poisson_seed);
        void GeneTSJOB(char* fityaml);

};

Src_FittingMode::Src_FittingMode(){}

Src_FittingMode::~Src_FittingMode(){

    for (int ii=0;ii<nSmode;ii++){
        delete[] NBinUsed[ii];
        delete[] ParStatus[ii];
    }
    delete[] NBinUsed;
    delete[] ParStatus;

    delete[] NSmode;
    delete[] NSmode_int;
    delete[] ModeTag;
    delete[] MinAlgo;
    delete[] ConfLevel;
    delete[] ErrOpt;

    delete[] ParVal;
    delete[] ParErr;

}

void Src_FittingMode::Init(Src_Template* Template){ 

    nPmode = 5;   // number of primary modes
    nSmode = 7;   // number of secondary modes
    NSmode = new int[nPmode];
    NSmode_int = new int[nPmode];
    NSmode[0] = 1;
    NSmode[1] = 1;
    NSmode[2] = 2;
    NSmode[3] = 2;
    NSmode[4] = 1;
    NSmode_int[0] = 1;
    NSmode_int[1] = 2;
    NSmode_int[2] = 4;
    NSmode_int[3] = 6;
    NSmode_int[4] = 7;

    ModeTag = new string[nPmode];
    ModeTag[0] = "Parameter";
    ModeTag[1] = "Flux Point";
    ModeTag[2] = "Source TS";
    ModeTag[3] = "Source TS in Each Bin";
    ModeTag[4] = "Flux UL";

    MinAlgo = new string[nPmode];
    ConfLevel  = new double[nPmode];
    NPar_total = new int[nPmode];
    ErrOpt = new bool[nPmode];
    for (int ii=0;ii<nPmode;ii++){
        if (ii==4){
            MinAlgo[ii] = "MINOS";
            ConfLevel[ii] = 1.35;
        }
        else{
            MinAlgo[ii] = "MIGRAD";
            ConfLevel[ii] = 0.5;
            //MinAlgo[ii] = "MINOS";
            //ConfLevel[ii] = 1.35;
        }
        NPar_total[ii] = npar_total;

        if (ii==1)
            ErrOpt[ii] = 0;
        else
            ErrOpt[ii] = 1;
    }

    NBinUsed  = new int*[nSmode];
    ParStatus = new int*[nSmode];
    for (int ii=0;ii<nSmode;ii++){
        NBinUsed[ii]  = new int[3];
        ParStatus[ii] = new int[npar_total+1];
        for (int jj=0;jj<3;jj++)
            NBinUsed[ii][jj] = 0;

        for (int jj=0;jj<npar_total+1;jj++)
            ParStatus[ii][jj] = 0;
    }

    int npar = 0;
    // Fitting
    // NBinUsed
    if (cf.UseWCDA)
        NBinUsed[0][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[0][1] += cf.KNEbinUsed;
    NBinUsed[0][2] = -1;

    // ParStatus
    // Srcs
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        if (Template->Srcs[isrc].Ra[3])
            ParStatus[0][npar] = 1;
        if (Template->Srcs[isrc].Dec[3])
            ParStatus[0][npar+1] = 1; 
        // SED
        for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++){
            if (Template->Srcs[isrc].SEDPar[ipar][3])
                ParStatus[0][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++){
            if (Template->Srcs[isrc].MorPar[ipar][3])
                ParStatus[0][npar+ipar] = 1;
        }
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        if (Template->Srcs_NumCon[isrc].Ra[3])
            ParStatus[0][npar] = 1;
        if (Template->Srcs_NumCon[isrc].Dec[3])
            ParStatus[0][npar+1] = 1; 
        // SED
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            if (Template->Srcs_NumCon[isrc].SEDPar[ipar][3])
                ParStatus[0][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++){
            if (Template->Srcs_NumCon[isrc].MorPar[ipar][3])
                ParStatus[0][npar+ipar] = 1;
        }
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    // Src_Temp && DGEs
    for (int isrc=0;isrc<Template->NTemp;isrc++){
        if (isrc<Template->NSrc_Temp){
            // SED
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                if (Template->Srcs_Temp[isrc].SEDPar[ipar][3])
                    ParStatus[0][npar+ipar] = 1;
            }
            npar += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++){
                if (Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][3])
                    ParStatus[0][npar+ipar] = 1;
            }
            npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }   
    ParStatus[0][npar_total] = -1;

    // Flux Point
    // NBinUsed
    if (cf.UseWCDA)
        NBinUsed[1][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[1][1] += cf.KNEbinUsed;
    NBinUsed[1][2] = 1;

    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[1][npar] = 1;
        ParStatus[1][npar+1] = 1;
        // SED
        for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[1][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[1][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[1][npar] = 1;
        ParStatus[1][npar+1] = 1;
        // SED
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[1][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[1][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    // Src_Temp && DGEs
    for (int isrc=0;isrc<Template->NTemp;isrc++){
        if (isrc<Template->NSrc_Temp){
            // SED
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[1][npar+ipar] = 1;
            }
            npar += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[1][npar+ipar] = 1;
            }
            npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }
    ParStatus[1][npar_total] = -1; 

    // source TS, Smode 0
    if (cf.UseWCDA)
        NBinUsed[2][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[2][1] += cf.KNEbinUsed;
    NBinUsed[2][2] = -1;
    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[2][npar] = 1;
        ParStatus[2][npar+1] = 1;
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[2][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[2][npar] = 1;
        ParStatus[2][npar+1] = 1;
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[2][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    ParStatus[2][npar_total] = 1; 

    // source TS, Smode 1
    if (cf.UseWCDA)
        NBinUsed[3][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[3][1] += cf.KNEbinUsed;
    NBinUsed[3][2] = -1;
    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[3][npar] = 1;
        ParStatus[3][npar+1] = 1;
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[3][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[3][npar] = 1;
        ParStatus[3][npar+1] = 1;
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[3][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    ParStatus[3][npar_total] = 1; 


    // source TS in each bin, Smode 0
    if (cf.UseWCDA)
        NBinUsed[4][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[4][1] += cf.KNEbinUsed;
    NBinUsed[4][2] = 1;
    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[4][npar] = 1;
        ParStatus[4][npar+1] = 1;
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[4][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[4][npar] = 1;
        ParStatus[4][npar+1] = 1;
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[4][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    ParStatus[4][npar_total] = 1; 

    // source TS in each bin, Smode 1
    if (cf.UseWCDA)
        NBinUsed[5][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[5][1] += cf.KNEbinUsed;
    NBinUsed[5][2] = 1;
    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[5][npar] = 1;
        ParStatus[5][npar+1] = 1;
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[5][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[5][npar] = 1;
        ParStatus[5][npar+1] = 1;
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[5][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    ParStatus[5][npar_total] = 1; 

    // Uplimit
    if (cf.UseWCDA)
        NBinUsed[6][1] += cf.NnhitUsed;
    if (cf.UseKM2A)
        NBinUsed[6][1] += cf.KNEbinUsed;
    NBinUsed[6][2] = 1;
    // ParStatus
    // Srcs
    npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        // Position
        ParStatus[6][npar] = 1;
        ParStatus[6][npar+1] = 1;
        // SED
        for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[6][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParStatus[6][npar+ipar] = 1;
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        // Position
        ParStatus[6][npar] = 1;
        ParStatus[6][npar+1] = 1;
        // SED
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[6][npar+2+ipar] = 1;
        }
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParStatus[6][npar+ipar] = 1;
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    // Src_Temp && DGEs
    for (int isrc=0;isrc<Template->NTemp;isrc++){
        if (isrc<Template->NSrc_Temp){
            // SED
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[6][npar+ipar] = 1;
            }
            npar += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[6][npar+ipar] = 1;
            }
            npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }
    ParStatus[6][npar_total] = -1; 

}

void Src_FittingMode::SetParStatus(int ipmode, int ismode, int icomp, Src_Template* Template, int binloop){

    if (icomp==-1) return;

    int iSmode = ismode+NSmode_int[ipmode-1];

    for (int ii=0;ii<npar_total;ii++)
        ParStatus[iSmode][ii] = 0;

    int npar = 0;
    // Srcs
    for (int isrc=0;isrc<Template->NSrc;isrc++){

        if (ismode==1 && isrc==icomp) continue;

        // Position
        if (isrc!=icomp){
            ParStatus[iSmode][npar] = 1;
            ParStatus[iSmode][npar+1] = 1;
        }
        else{
            //if (binloop==1){
            ParStatus[iSmode][npar] = 1;
            ParStatus[iSmode][npar+1] = 1;
            //}
            //else{
            //    ParStatus[iSmode][npar] = 0;
            //    ParStatus[iSmode][npar+1] = 0;
            //}
        }
        // SED
        //if (binloop==1){
        for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[iSmode][npar+2+ipar] = 1;
        }
        //}
        npar += 2+Template->Srcs[isrc].nSEDpar;
        // Morphology
        if (isrc!=icomp){
            for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
                ParStatus[iSmode][npar+ipar] = 1;
        }
        else{
            //if (binloop==1){
            for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
                ParStatus[iSmode][npar+ipar] = 1;
            //}
            //else{
            //    for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            //        ParStatus[iSmode][npar+ipar] = 0;
            //}
        }
        npar += Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){

        if (ismode==1 && isrc==(icomp-Template->NSrc)) continue;

        // Position
        if (isrc!=(icomp-Template->NSrc)){
            ParStatus[iSmode][npar] = 1;
            ParStatus[iSmode][npar+1] = 1;
        }
        else{
            //if (binloop==1){
            ParStatus[iSmode][npar] = 1;
            ParStatus[iSmode][npar+1] = 1;
            //}
            //else{
            //    ParStatus[iSmode][npar] = 0;
            //    ParStatus[iSmode][npar+1] = 0;
            //}
        }
        // SED
        //if (binloop==1){
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            if (ipar>=1)
                ParStatus[iSmode][npar+2+ipar] = 1;
        }
        //}
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        // Morphology
        if (isrc!=(icomp-Template->NSrc)){
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
                ParStatus[iSmode][npar+ipar] = 1;
        }
        else{
            //if (binloop==1){    
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
                ParStatus[iSmode][npar+ipar] = 1;
            //}
            //else{
            //    for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            //        ParStatus[iSmode][npar+ipar] = 0;
            //}
        }
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }   
    // Src_Temp && DGEs
    //if (binloop==1){
    for (int isrc=0;isrc<Template->NTemp;isrc++){

        if (ismode==1 && isrc==(icomp-Template->NSrc-Template->NSrc_NumCon)) continue;

        if (isrc<Template->NSrc_Temp){
            // SED
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[iSmode][npar+ipar] = 1;
            }
            npar += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++){
                if (ipar>=1)
                    ParStatus[iSmode][npar+ipar] = 1;
            }
            npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }
    //}

}

void Src_FittingMode::SetNPar_total(int ipmode, int ismode, int icomp, Src_Template* Template){

    if (icomp==-1 || ismode==0) return;
    // Srcs
    NPar_total[ipmode] = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        if (isrc!=icomp)
            NPar_total[ipmode] += 2+Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar;
    }   
    // Srcs_NumCon
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        if (isrc!=icomp-Template->NSrc)
            NPar_total[ipmode] += 2+Template->Srcs_NumCon[isrc].nSEDpar + Template->Srcs_NumCon[isrc].nMorpar;
    }   
    // Src_Temp && DGEs
    for (int isrc=0;isrc<Template->NTemp;isrc++){
        if (isrc<Template->NSrc_Temp){
            if (isrc!=icomp-(Template->NSrc+Template->NSrc_NumCon))
                NPar_total[ipmode] += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            if (isrc!=icomp-(Template->NSrc+Template->NSrc_NumCon))
                NPar_total[ipmode] += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }

}

// Fitting results
void Src_FittingMode::InitRes(Src_Template* Template){

    TS_WCDA = 0;
    TS_KM2A = 0;
    TS_Total = 0;
    logL_total = 0;
    FITSTATUS = 0;

    ParVal = new double[npar_total];
    ParErr = new double[npar_total];
    for (int ii=0;ii<npar_total;ii++){
        ParVal[ii] = 0;
        ParErr[ii] = 0;
    }

    int npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        ParVal[npar] = Template->Srcs[isrc].Ra[0];
        ParVal[npar+1] = Template->Srcs[isrc].Dec[0];
        for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
            ParVal[npar+ipar+2] = Template->Srcs[isrc].SEDPar[ipar][0];
        npar += 2+Template->Srcs[isrc].nSEDpar; 
        for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
            ParVal[npar+ipar] = Template->Srcs[isrc].MorPar[ipar][0];
        npar += Template->Srcs[isrc].nMorpar;
    }
    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
        ParVal[npar] = Template->Srcs_NumCon[isrc].Ra[0];
        ParVal[npar+1] = Template->Srcs_NumCon[isrc].Dec[0];
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++)
            ParVal[npar+2+ipar] = Template->Srcs_NumCon[isrc].SEDPar[ipar][0];
        npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
        for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
            ParVal[npar+ipar] = Template->Srcs_NumCon[isrc].MorPar[ipar][0];
        npar += Template->Srcs_NumCon[isrc].nMorpar;
    }
    for (int isrc=0;isrc<Template->NTemp;isrc++){
        if (isrc<Template->NSrc_Temp){
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++)
                ParVal[npar+ipar] = Template->Srcs_Temp[isrc].SEDPar[ipar][0];
            npar += Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++)
                ParVal[npar+ipar] = Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][0];
            npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
        }
    }

    TS_UL[0] = 0;
    TS_UL[1] = 0;
    TS_UL[2] = 4;

    Emedian = new double*[Template->NComp];
    Flux    = new double*[Template->NComp];
    FluxErr = new double*[Template->NComp];
    FluxUL  = new double*[Template->NComp];
    FNorm    = new double*[Template->NComp];
    FNormErr = new double*[Template->NComp];
    FNormUL  = new double*[Template->NComp];
    TS_Src  = new double*[Template->NComp];
    FitStatus_Src = new int*[Template->NComp];
    TS_Bin  = new double*[Template->NComp];
    FitStatus_Bin = new int*[Template->NComp];
    SEDChi2 = new double[Template->NComp*2];

    FitStatus_UL = new int[NBinUsed[4][1]];
    for (int ibin=0;ibin<NBinUsed[4][1];ibin++)
        FitStatus_UL[ibin] = 0;

    for (int ii=0;ii<Template->NComp;ii++){
        Emedian[ii] = new double[NBinUsed[0][1]*2];
        Flux[ii]    = new double[NBinUsed[0][1]*2];
        FluxErr[ii] = new double[NBinUsed[0][1]*2];
        FluxUL[ii]  = new double[NBinUsed[0][1]*2];
        FitStatus_Bin[ii] = new int[NBinUsed[0][1]*2];
        for (int jj=0;jj<NBinUsed[0][1]*2;jj++){
            Flux[ii][jj] = 0;
            FluxErr[ii][jj] = 0;
            FluxUL[ii][jj] = 0;
            Emedian[ii][jj] = 0;
            FitStatus_Bin[ii][jj] = 0;
        }

        SEDChi2[ii] = 0;
        SEDChi2[ii+Template->NComp] = 0;

        FNorm[ii]    = new double[NBinUsed[0][1]];
        FNormErr[ii] = new double[NBinUsed[0][1]];
        FNormUL[ii]  = new double[NBinUsed[0][1]];
        for (int jj=0;jj<NBinUsed[0][1];jj++){
            FNorm[ii][jj] = 0;
            FNormErr[ii][jj] = 0;
            FNormUL[ii][jj] = 0;
        }

        FitStatus_Src[ii] = new int[2];
        for (int jj=0;jj<2;jj++)
            FitStatus_Src[ii][jj] = 0;

        TS_Src[ii] = new double[4];
        for (int jj=0;jj<4;jj++)
            TS_Src[ii][jj] = 0;

        TS_Bin[ii] = new double[4*NBinUsed[0][1]];
        for (int jj=0;jj<NBinUsed[0][1]*4;jj++)
            TS_Bin[ii][jj] = 0;
    }

}

void Src_FittingMode::CalFlux(int ipmode, Src_Template* Template){


    if (cf.UseWCDA){
        for (int icomp=0;icomp<Template->NComp;icomp++)
            for (int ibin=0;ibin<cf.NnhitUsed;ibin++)
                Emedian[icomp][ibin+NBinUsed[1][1]] = Emedian[icomp][ibin];
    }

    int npar = 0;
    for (int icomp=0;icomp<Template->NComp;icomp++){

        TF1 *fSED;
        double f0_scale = 1.0;
        if (icomp>=0 && icomp<Template->NSrc){
            fSED = new TF1("fSED", Template->Srcs[icomp].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs[icomp].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs[icomp].nSEDpar+Template->Srcs[icomp].nMorpar; 
        }
        else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon)){
            fSED = new TF1("fSED", Template->Srcs_NumCon[icomp-Template->NSrc].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar+Template->Srcs_NumCon[icomp-Template->NSrc].nMorpar;
        }
        else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total){
            fSED = new TF1("fSED", Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;
        }
        else{
            f0_scale = Template->DGEs[icomp-Template->NSrc_total].Omega_total_model*Template->DGEs[icomp-Template->NSrc_total].Eta/Template->DGEs[icomp-Template->NSrc_total].Omega_total;
            fSED = new TF1("fSED", Template->DGEs[icomp-Template->NSrc_total].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->DGEs[icomp-Template->NSrc_total].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->DGEs[icomp-Template->NSrc_total].nSEDpar;
        }

        if (ipmode==1){
            for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                fSED->SetParameter(0, FNorm[icomp][ibin]*f0_scale);
                Flux[icomp][ibin] = fSED->Eval(Emedian[icomp][ibin])*Emedian[icomp][ibin]*Emedian[icomp][ibin];
                if (cf.UseKM2A)
                    Flux[icomp][ibin+NBinUsed[1][1]] = fSED->Eval(Emedian[icomp][ibin+NBinUsed[1][1]])*Emedian[icomp][ibin+NBinUsed[1][1]]*Emedian[icomp][ibin+NBinUsed[1][1]];
                fSED->SetParameter(0, FNormErr[icomp][ibin]*f0_scale);
                FluxErr[icomp][ibin] = fSED->Eval(Emedian[icomp][ibin])*Emedian[icomp][ibin]*Emedian[icomp][ibin];
                if (cf.UseKM2A)
                    FluxErr[icomp][ibin+NBinUsed[1][1]] = fSED->Eval(Emedian[icomp][ibin+NBinUsed[1][1]])*Emedian[icomp][ibin+NBinUsed[1][1]]*Emedian[icomp][ibin+NBinUsed[1][1]];
            }
        }

        if (ipmode==4){
            for (int ibin=0;ibin<NBinUsed[ipmode][1];ibin++){
                fSED->SetParameter(0, FNormUL[icomp][ibin]*f0_scale);
                FluxUL[icomp][ibin] = fSED->Eval(Emedian[icomp][ibin])*Emedian[icomp][ibin]*Emedian[icomp][ibin];
                if (cf.UseKM2A)
                    FluxUL[icomp][ibin+NBinUsed[ipmode][1]] = fSED->Eval(Emedian[icomp][ibin+NBinUsed[ipmode][1]])*Emedian[icomp][ibin+NBinUsed[ipmode][1]]*Emedian[icomp][ibin+NBinUsed[ipmode][1]];
            }
        }

    }

}

void Src_FittingMode::CalTS(int ipmode, Src_Template* Template){

    for (int icomp=0;icomp<Template->NComp;icomp++){

        if (ipmode==2){
            TS_Src[icomp][0] = TS_Src[icomp][0]-TS_Src[icomp][2];
            TS_Src[icomp][1] = TS_Src[icomp][1]-TS_Src[icomp][3];
        }

        if (ipmode==3){
            for (int ibin=0;ibin<NBinUsed[ipmode][1];ibin++){
                TS_Bin[icomp][ibin] = TS_Bin[icomp][ibin] - TS_Bin[icomp][ibin+2*NBinUsed[ipmode][1]];
                TS_Bin[icomp][ibin+NBinUsed[ipmode][1]] = TS_Bin[icomp][ibin+NBinUsed[ipmode][1]] - TS_Bin[icomp][ibin+3*NBinUsed[ipmode][1]];
            }
        }

    }

}

void Src_FittingMode::CalSEDChi2(Src_Template* Template){

    int npar = 0;
    for (int icomp=0;icomp<Template->NComp;icomp++){

        TF1 *fSED;
        double f0_scale = 1.0;
        if (icomp>=0 && icomp<Template->NSrc){
            fSED = new TF1("fSED", Template->Srcs[icomp].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs[icomp].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs[icomp].nSEDpar+Template->Srcs[icomp].nMorpar;
        }
        else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon)){
            fSED = new TF1("fSED", Template->Srcs_NumCon[icomp-Template->NSrc].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar+Template->Srcs_NumCon[icomp-Template->NSrc].nMorpar;
        }
        else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total){
            fSED = new TF1("fSED", Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;
        }
        else{
            f0_scale = Template->DGEs[icomp-Template->NSrc_total].Omega_total_model*Template->DGEs[icomp-Template->NSrc_total].Eta/Template->DGEs[icomp-Template->NSrc_total].Omega_total;
            fSED = new TF1("fSED", Template->DGEs[icomp-Template->NSrc_total].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->DGEs[icomp-Template->NSrc_total].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->DGEs[icomp-Template->NSrc_total].nSEDpar;
        }

        for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
            bool efftag = 0;
            if (cf.FitOpt[3]==0){
                if ((Flux[icomp][ibin]-FluxErr[icomp][ibin])<=0)
                    efftag = 1;
            }
            else{
                double TS = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                if (TS<=TS_UL[2])
                    efftag = 1;
            }
            if (!efftag){
                double flux = 0;
                if (!cf.UseKM2A){
                    flux = fSED->Eval(Emedian[icomp][ibin])*f0_scale*Emedian[icomp][ibin]*Emedian[icomp][ibin];
                    SEDChi2[icomp] += pow(flux-Flux[icomp][ibin], 2)/pow(FluxErr[icomp][ibin], 2);
                }
                else{
                    flux = fSED->Eval(Emedian[icomp][ibin+NBinUsed[1][1]])*f0_scale*Emedian[icomp][ibin+NBinUsed[1][1]]*Emedian[icomp][ibin+NBinUsed[1][1]];
                    SEDChi2[icomp] += pow(flux-Flux[icomp][ibin+NBinUsed[1][1]], 2)/pow(FluxErr[icomp][ibin+NBinUsed[1][1]], 2);
                }
                SEDChi2[icomp+Template->NComp]++;
            }
        }

    }

}

void Src_FittingMode::PrintRes(Src_Template* Template, int neffbins){

    cout<<" ===================================================== "<<endl;
    cout<<" |----------------- Fitting Results -----------------| "<<endl;
    cout<<" ===================================================== "<<endl;

    int npar = 0, imodel = -1;
    // ParVal and ParErr
    if (cf.FitOpt[0]){

        cout<<" |---------------- ParVal and ParErr ----------------| "<<endl;
        cout<<" STATUS : ";
        if (FITSTATUS==3)
            cout<<"OK"<<endl;
        else
            cout<<"NotOK"<<endl;
        cout<<Form("%-15s", " ParName")<<"   Value    "<<"   Error   "<<"   Limits"<<endl;
        double ra, dec;
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            // Position
            cout<<Form("%-15s", Form(" %s_F0_Unit ", Template->Srcs[isrc].Srcname.data()))<<"   "<<Form("%sTeV^-1cm^-2s^-1", Template->Srcs[isrc].F0_order.data())<<endl;
            if (!cf.CorOpt){
                cout<<Form("%-15s", Form(" %s_ra ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar])<<"   "<<Form("%8.5lf", ParErr[npar])<<Form("   %6.2lf", Template->Srcs[isrc].Ra[1])<<Form("   %6.2lf", Template->Srcs[isrc].Ra[2])<<endl;
                cout<<Form("%-15s", Form(" %s_dec ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar+1])<<"   "<<Form("%8.5lf", ParErr[npar+1])<<Form("   %6.2lf", Template->Srcs[isrc].Dec[1])<<Form("   %6.2lf", Template->Srcs[isrc].Dec[2])<<endl;
            }
            else{
                cout<<Form("%-15s", Form(" %s_l ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar])<<"   "<<Form("%8.5lf", ParErr[npar])<<Form("   %6.2lf", Template->Srcs[isrc].Ra[1])<<Form("   %6.2lf", Template->Srcs[isrc].Ra[2])<<endl;
                cout<<Form("%-15s", Form(" %s_b ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar+1])<<"   "<<Form("%8.5lf", ParErr[npar+1])<<Form("   %6.2lf", Template->Srcs[isrc].Dec[1])<<Form("   %6.2lf", Template->Srcs[isrc].Dec[2])<<endl;
                g2e(ParVal[npar], ParVal[npar+1], &ra, &dec);
                cout<<Form("%-15s", Form(" %s_ra ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", ra)<<endl;
                cout<<Form("%-15s", Form(" %s_dec ", Template->Srcs[isrc].Srcname.data()))<<Form("   %9.5lf", dec)<<endl;
            }
            // SED
            imodel = Template->Model->SEDMap[Template->Srcs[isrc].SEDtype]-1;
            for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++){
                cout<<Form("%-15s", Form(" %s_%s ", Template->Srcs[isrc].Srcname.data(), Template->Model->SEDParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+2+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+2+ipar])<<Form("   %6.2lf", Template->Srcs[isrc].SEDPar[ipar][1])<<Form("   %6.2lf", Template->Srcs[isrc].SEDPar[ipar][2]);
                if (ipar>=1 && Template->Srcs[isrc].LinkPars){
                    cout<<"   linked to Src"<<Template->Srcs[isrc].TargetSrcID;
                }
                if ((ParVal[npar+2+ipar]-0.1*ParErr[npar+2+ipar])<Template->Srcs[isrc].SEDPar[ipar][1] || (ParVal[npar+2+ipar]+0.1*ParErr[npar+2+ipar])>Template->Srcs[isrc].SEDPar[ipar][2])
                    cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                else
                    cout<<endl;
            }

            npar += 2+Template->Srcs[isrc].nSEDpar;

            // Morphology
            imodel = Template->Model->MorMap[Template->Srcs[isrc].Mortype]-1;
            for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++){
                cout<<Form("%-15s", Form(" %s_%s ", Template->Srcs[isrc].Srcname.data(), Template->Model->MorParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+ipar])<<Form("   %6.2lf", Template->Srcs[isrc].MorPar[ipar][1])<<Form("   %6.2lf", Template->Srcs[isrc].MorPar[ipar][2]);
                if ((ParVal[npar+ipar]-0.1*ParErr[npar+ipar])<Template->Srcs[isrc].MorPar[ipar][1] || (ParVal[npar+ipar]+0.1*ParErr[npar+ipar])>Template->Srcs[isrc].MorPar[ipar][2])
                    cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                else
                    cout<<endl;
            }
            npar += Template->Srcs[isrc].nMorpar;

        }  

        // NumSrcs
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            cout<<Form("%-15s", Form(" %s_F0_Unit ", Template->Srcs_NumCon[isrc].Srcname.data()))<<"   "<<Form("%sTeV^-1cm^-2s^-1", Template->Srcs_NumCon[isrc].F0_order.data())<<endl;

            // Position
            if (!cf.CorOpt){
                cout<<Form("%-15s", Form(" %s_ra ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar])<<"   "<<Form("%8.5lf", ParErr[npar])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Ra[1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Ra[2])<<endl;
                cout<<Form("%-15s", Form(" %s_dec ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar+1])<<"   "<<Form("%8.5lf", ParErr[npar+1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Dec[1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Dec[2])<<endl;
            }
            else{
                cout<<Form("%-15s", Form(" %s_l ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar])<<"   "<<Form("%8.5lf", ParErr[npar])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Ra[1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Ra[2])<<endl;
                cout<<Form("%-15s", Form(" %s_b ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", ParVal[npar+1])<<"   "<<Form("%8.5lf", ParErr[npar+1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Dec[1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].Dec[2])<<endl;
                g2e(ParVal[npar], ParVal[npar+1], &ra, &dec);
                cout<<Form("%-15s", Form(" %s_ra ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", ra)<<endl;
                cout<<Form("%-15s", Form(" %s_dec ", Template->Srcs_NumCon[isrc].Srcname.data()))<<Form("   %9.5lf", dec)<<endl;
            }
            // SED
            imodel = Template->Model->SEDMap[Template->Srcs_NumCon[isrc].SEDtype]-1;
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
                cout<<Form("%-15s", Form(" %s_%s ", Template->Srcs_NumCon[isrc].Srcname.data(), Template->Model->SEDParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+2+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+2+ipar])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].SEDPar[ipar][1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].SEDPar[ipar][2]);
                if (ipar>=1 && Template->Srcs_NumCon[isrc].LinkPars){
                    cout<<"   linked to Src"<<Template->Srcs_NumCon[isrc].TargetSrcID;
                }
                if ((ParVal[npar+2+ipar]-0.1*ParErr[npar+2+ipar])<Template->Srcs_NumCon[isrc].SEDPar[ipar][1] || (ParVal[npar+2+ipar]+0.1*ParErr[npar+2+ipar])>Template->Srcs_NumCon[isrc].SEDPar[ipar][2])
                    cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                else
                    cout<<endl;
            }
            npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;

            // Morphology
            imodel = Template->Model->MorMap[Template->Srcs_NumCon[isrc].Mortype]-1;
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++){
                cout<<Form("%-15s", Form(" %s_%s ", Template->Srcs_NumCon[isrc].Srcname.data(), Template->Model->MorParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+ipar])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].MorPar[ipar][1])<<Form("   %6.2lf", Template->Srcs_NumCon[isrc].MorPar[ipar][2]);
                if ((ParVal[npar+ipar]-0.1*ParErr[npar+ipar])<Template->Srcs_NumCon[isrc].MorPar[ipar][1] || (ParVal[npar+ipar]+0.1*ParErr[npar+ipar])>Template->Srcs_NumCon[isrc].MorPar[ipar][2])
                    cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                else
                    cout<<endl;
            }
            npar +=  Template->Srcs_NumCon[isrc].nMorpar;

        }  

        // Src_Temp && DGEs
        for (int isrc=0;isrc<Template->NTemp;isrc++){

            if (isrc<Template->NSrc_Temp){
                cout<<Form("%-15s", Form(" %s_F0_Unit ", Template->Srcs_Temp[isrc].Srcname.data()))<<"   "<<Form("%sTeV^-1cm^-2s^-1sr^-1", Template->Srcs_Temp[isrc].F0_order.data())<<endl;
                // SED
                imodel = Template->Model->SEDMap[Template->Srcs_Temp[isrc].SEDtype]-1;
                for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                    cout<<Form("%-15s", Form(" %s_%s ", Template->Srcs_Temp[isrc].Srcname.data(), Template->Model->SEDParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+ipar])<<Form("   %6.2lf", Template->Srcs_Temp[isrc].SEDPar[ipar][1])<<Form("   %6.2lf", Template->Srcs_Temp[isrc].SEDPar[ipar][2]);
                    if (ipar>=1 && Template->Srcs_Temp[isrc].LinkPars){
                        cout<<"   linked to Src"<<Template->Srcs_Temp[isrc].TargetSrcID;
                    }
                    if ((ParVal[npar+ipar]-0.1*ParErr[npar+ipar])<Template->Srcs_Temp[isrc].SEDPar[ipar][1] || (ParVal[npar+ipar]+0.1*ParErr[npar+ipar])>Template->Srcs_Temp[isrc].SEDPar[ipar][2])
                        cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                    else
                        cout<<endl;
                }
                npar += Template->Srcs_Temp[isrc].nSEDpar;
            }
            else{
                // SED
                cout<<Form("%-15s", Form(" %s_F0_Unit ", Template->DGEs[isrc-Template->NSrc_Temp].Srcname.data()))<<"   "<<Form("%sTeV^-1cm^-2s^-1sr^-1", Template->DGEs[isrc-Template->NSrc_Temp].F0_order.data())<<"   "<<"( Omega = "<<Form("%.6lf )", Template->DGEs[isrc-Template->NSrc_Temp].Omega_total)<<endl;
                imodel = Template->Model->SEDMap[Template->DGEs[isrc-Template->NSrc_Temp].SEDtype]-1;
                for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++){
                    if (ipar!=0)
                        cout<<Form("%-15s", Form(" %s_%s ", Template->DGEs[isrc-Template->NSrc_Temp].Srcname.data(), Template->Model->SEDParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+ipar])<<"   "<<Form("%8.5lf", ParErr[npar+ipar])<<Form("   %6.2lf", Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][1])<<Form("   %6.2lf", Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][2]);
                    else{
                        double factor_temp = Template->DGEs[isrc-Template->NSrc_Temp].Omega_total_model*Template->DGEs[isrc-Template->NSrc_Temp].Eta/Template->DGEs[isrc-Template->NSrc_Temp].Omega_total; 
                        cout<<Form("%-15s", Form(" %s_%s ", Template->DGEs[isrc-Template->NSrc_Temp].Srcname.data(), Template->Model->SEDParname[imodel][ipar].data()))<<Form("   %9.5lf", ParVal[npar+ipar]*factor_temp)<<"   "<<Form("%8.5lf", ParErr[npar+ipar]*factor_temp)<<Form("   %6.2lf", Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][1]*factor_temp)<<Form("   %6.2lf", Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][2]*factor_temp);
                    }

                    if ((ParVal[npar+ipar]-0.1*ParErr[npar+ipar])<Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][1] || (ParVal[npar+ipar]+0.1*ParErr[npar+ipar])>Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][2])
                        cout<<"   \033[31;1mAt Limits\033[0m"<<endl;
                    else
                        cout<<endl;
                }
                npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
            }

        }
        if (cf.UseWCDA)
            cout<<" TS_WCDA = "<<Form("%9.2lf", TS_WCDA)<<", ";
        if (cf.UseKM2A)
            cout<<" TS_KM2A = "<<Form("%9.2lf", TS_KM2A)<<", ";
        cout<<" TS      = "<<Form("%9.2lf", TS_Total)<<",  ";
        cout<<" BIC     = "<<Form("%.3lf", Template->Npar_free*log((cf.UseWCDA*cf.NnhitUsed+cf.UseKM2A*cf.KNEbinUsed)*neffbins)-logL_total)<<",  ";
        cout<<" AIC     = "<<Form("%.3lf", 2*Template->Npar_free-logL_total)<<endl;

        cout<<" ===================================================== "<<endl;
    }

    if (cf.FitOpt[2]){
        cout<<" |----------------- TS of Component -----------------| "<<endl;
        string srcname;
        for (int icomp=0;icomp<Template->NComp;icomp++){

            if (icomp>=0 && icomp<Template->NSrc)
                srcname  = Template->Srcs[icomp].Srcname;
            else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon))
                srcname  = Template->Srcs_NumCon[icomp-Template->NSrc].Srcname;
            else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total)
                srcname  = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].Srcname;
            else
                srcname  = Template->DGEs[icomp-Template->NSrc_total].Srcname;
            cout<<Form(" >>>> %-9s : ", srcname.data());
            if (cf.UseWCDA)
                cout<<" TS_W = "<<Form("%9.2lf", TS_Src[icomp][0])<<", ";
            if (cf.UseKM2A)
                cout<<" TS_K = "<<Form("%9.2lf", TS_Src[icomp][1])<<", ";
            cout<<"TS = "<<Form("%9.2lf", TS_Src[icomp][0]+TS_Src[icomp][1])<<", STATUS : ";
            if (Template->NComp>1){
                if (FitStatus_Src[icomp][0]==3 && FitStatus_Src[icomp][1]==3)
                    cout<<"OK"<<endl;
                else{
                    if (FitStatus_Src[icomp][0]!=3 && FitStatus_Src[icomp][1]==3)
                        cout<<"Step1 NotOK"<<endl;
                    else if (FitStatus_Src[icomp][0]==3 && FitStatus_Src[icomp][1]!=3)
                        cout<<"Step2 NotOK"<<endl;
                    else
                        cout<<"NotOK"<<endl;
                }
            }
            else{
                if (FitStatus_Src[icomp][0]==3)
                    cout<<"OK"<<endl;
                else
                    cout<<"NotOK"<<endl;
            }
        }
        cout<<" ===================================================== "<<endl;
    }

    if (cf.FitOpt[3]){
        cout<<" |----------- TS for each bin of Component ----------| "<<endl;
        string srcname;
        double F0_min, F0_max;
        for (int icomp=0;icomp<Template->NComp;icomp++){

            if (icomp>=0 && icomp<Template->NSrc){
                srcname  = Template->Srcs[icomp].Srcname;
                F0_min   = Template->Srcs[icomp].SEDPar[0][1];
                F0_max   = Template->Srcs[icomp].SEDPar[0][2];
            }
            else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon)){
                srcname  = Template->Srcs_NumCon[icomp-Template->NSrc].Srcname;
                F0_min   = Template->Srcs_NumCon[icomp-Template->NSrc].SEDPar[0][1];
                F0_max   = Template->Srcs_NumCon[icomp-Template->NSrc].SEDPar[0][2];
            }
            else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total){
                srcname  = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].Srcname;
                F0_min   = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDPar[0][1];
                F0_max   = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDPar[0][2];
            }
            else{
                srcname  = Template->DGEs[icomp-Template->NSrc_total].Srcname;
                F0_min   = Template->DGEs[icomp-Template->NSrc_total].SEDPar[0][1];
                F0_max   = Template->DGEs[icomp-Template->NSrc_total].SEDPar[0][2];
            }

            cout<<Form(" >>>> %-9s : ", srcname.data())<<endl;

            cout<<"  STATUS : ";
            for (int ibin=0;ibin<NBinUsed[3][1];ibin++){
                if (Template->NComp>1){
                    if (FitStatus_Bin[icomp][ibin]==3 && FitStatus_Bin[icomp][ibin+NBinUsed[3][1]]==3){
                        if (cf.FitOpt[1]){
                            if ((FNorm[icomp][ibin]+0.1*FNormErr[icomp][ibin])<F0_max && (FNorm[icomp][ibin]-0.1*FNormErr[icomp][ibin])>F0_min)
                                cout<<Form("%10s", "OK  ");
                            else
                                cout<<Form("%10s", "F0Limit  ");
                        }
                        else
                            cout<<Form("%10s", "OK  ");
                    }
                    else
                        cout<<Form("%10s", "NotOK  ");
                }
                else{
                    if (FitStatus_Bin[icomp][ibin]==3)
                        cout<<Form("%10s", "OK  ");
                    else
                        cout<<Form("%10s", "NotOK  ");
                }
            }
            cout<<endl;
            if (cf.UseWCDA){
                cout<<"    TS_W : ";
                for (int ibin=0;ibin<NBinUsed[3][1];ibin++)
                    cout<<Form("%8.2lf", TS_Bin[icomp][ibin])<<", ";
                cout<<endl;
            }
            if (cf.UseKM2A){
                cout<<"    TS_K : ";
                for (int ibin=0;ibin<NBinUsed[3][1];ibin++)
                    cout<<Form("%8.2lf", TS_Bin[icomp][ibin+NBinUsed[3][1]])<<", ";
                cout<<endl;
            }
            cout<<"    TS   : ";
            for (int ibin=0;ibin<NBinUsed[3][1];ibin++)
                cout<<Form("%8.2lf", TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[3][1]])<<", ";
            cout<<endl;
        }
        cout<<" ===================================================== "<<endl;
    }


    if (cf.FitOpt[1]){

        cout<<" |-------------------- Flux Point -------------------| "<<endl;
        cout<<" |--- Unit : Energy, TeV; Flux/Ferr, TeVcm^-2s^-1 ---|"<<endl;

        string srcname;
        string f0_order;
        double F0_min, F0_max;
        int abstag = 0;
        for (int icomp=0;icomp<Template->NComp;icomp++){

            abstag = 0;

            if (icomp>=0 && icomp<Template->NSrc){
                srcname  = Template->Srcs[icomp].Srcname;
                f0_order = Template->Srcs[icomp].F0_order;
                F0_min   = Template->Srcs[icomp].SEDPar[0][1];
                F0_max   = Template->Srcs[icomp].SEDPar[0][2];
                abstag   = Template->Srcs[icomp].GGAbsFlag; 
            }
            else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon)){
                srcname  = Template->Srcs_NumCon[icomp-Template->NSrc].Srcname;
                f0_order = Template->Srcs_NumCon[icomp-Template->NSrc].F0_order;
                F0_min   = Template->Srcs_NumCon[icomp-Template->NSrc].SEDPar[0][1];
                F0_max   = Template->Srcs_NumCon[icomp-Template->NSrc].SEDPar[0][2];
            }
            else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total){
                srcname  = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].Srcname;
                f0_order = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].F0_order;
                F0_min   = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDPar[0][1];
                F0_max   = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDPar[0][2];
            }
            else{
                srcname  = Template->DGEs[icomp-Template->NSrc_total].Srcname;
                f0_order = Template->DGEs[icomp-Template->NSrc_total].F0_order;
                F0_min   = Template->DGEs[icomp-Template->NSrc_total].SEDPar[0][1];
                F0_max   = Template->DGEs[icomp-Template->NSrc_total].SEDPar[0][2];
            }

            TF1 *ftemp = new TF1("ftemp", Form("x/%s", f0_order.data()), 0, 1);

            ofstream out(Form("%s/%s/SED_Mor/%s_SED.txt", cf.WorkDir.data(), cf.Outdir.data(), srcname.data()), ios::out);
            ofstream out_obs;
            if (abstag){
                out_obs.open(Form("%s/%s/SED_Mor/%s_SED_obs.txt", cf.WorkDir.data(), cf.Outdir.data(), srcname.data()), ios::out);
            }
            cout<<Form(" >>>> %-9s : ", srcname.data())<<"Order of flux : "<<f0_order<<endl;
            if (!cf.FitOpt[4]){

                out<<"energy flux ferrL ferrU WCDAtag"<<endl;
                if (abstag)
                    out_obs<<"energy flux ferrL ferrU WCDAtag"<<endl;

                cout<<"    STATUS  : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    if (FitStatus_Bin[0][ibin]==3){
                        if ((FNorm[icomp][ibin]+0.1*FNormErr[icomp][ibin])<F0_max && (FNorm[icomp][ibin]-0.1*FNormErr[icomp][ibin])>F0_min)
                            cout<<Form("%11s", "OK  ");
                        else
                            cout<<Form("%11s", "F0Limit  ");
                    }
                    else
                        cout<<Form("%11s", "NotOK  ");
                }
                cout<<endl;

                cout<<"    Energy  : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                    cout<<Form("%9.3lf, ", Emedian[icomp][ibin]);
                cout<<endl;
                cout<<"    Flux    : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                    cout<<Form("%9.3lf, ", ftemp->Eval(Flux[icomp][ibin]));
                cout<<endl;
                cout<<"    Ferr    : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                    cout<<Form("%9.3lf, ", ftemp->Eval(FluxErr[icomp][ibin]));
                cout<<endl;
                if (cf.UseKM2A){
                    cout<<"    Energy1 : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                        cout<<Form("%9.3lf, ", Emedian[icomp][ibin+NBinUsed[1][1]]);
                    cout<<endl;
                    cout<<"    Flux1   : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                        cout<<Form("%9.3lf, ", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]]));
                    cout<<endl;
                    cout<<"    Ferr1   : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                        cout<<Form("%9.3lf, ", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]));
                    cout<<endl;
                }

                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    if (ibin<cf.NnhitUsed){
                        out<<Form("%.3lf", Emedian[icomp][ibin])<<" "
                            <<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin]))<<" "
                            <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin]))<<" "
                            <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin]))<<" "<<1<<endl;
                    }
                    else{
                        out<<Form("%.3lf", Emedian[icomp][ibin+NBinUsed[1][1]])<<" "
                            <<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]]))<<" "
                            <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]))<<" "
                            <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]))<<" "<<0<<endl;
                    }
                }
                out.close();

                if (abstag){
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                        if (ibin<cf.NnhitUsed){
                            out_obs<<Form("%.3lf", Emedian[icomp][ibin])<<" ";
                            double ebl_abs = exp(-Template->Srcs[icomp].gg_ebl->Eval(Emedian[icomp][ibin]));
                            out_obs<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin])*ebl_abs)<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin])*ebl_abs)<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin])*ebl_abs)<<" "<<1<<endl;
                        }
                        else{
                            out_obs<<Form("%.3lf", Emedian[icomp][ibin+NBinUsed[1][1]])<<" ";
                            double ebl_abs = exp(-Template->Srcs[icomp].gg_ebl->Eval(Emedian[icomp][ibin+NBinUsed[1][1]]));
                            out_obs<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "<<0<<endl;
                        }
                    }
                    out_obs.close();
                }


            }
            else{

                out<<"energy flux ferrL ferrU TS WCDAtag"<<endl;
                if (abstag)
                    out_obs<<"energy flux ferrL ferrU TS WCDAtag"<<endl;

                cout<<"    STATUS  : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                    if (ts_bin>=TS_UL[2]){
                        if (FitStatus_Bin[0][ibin]==3){
                            if ((FNorm[icomp][ibin]+0.1*FNormErr[icomp][ibin])<F0_max && (FNorm[icomp][ibin]-0.1*FNormErr[icomp][ibin])>F0_min)
                                cout<<Form("%11s", "OK  ");
                            else
                                cout<<Form("%11s", "F0Limit  ");
                        }
                        else
                            cout<<Form("%11s", "NotOK  ");
                    }
                    else{
                        if (FitStatus_UL[ibin]==3){
                            //if ((FNorm[icomp][ibin]+FNormErr[icomp][ibin])<F0_max)
                            if (FNormUL[icomp][ibin]>F0_max*0.95 || (FNormUL[icomp][ibin]-FNorm[icomp][ibin])<1.e-5)
                                cout<<Form("%11s", "F0Limit  ");
                            else
                                cout<<Form("%11s", "OK  ");
                        }
                        else
                            cout<<Form("%11s", "NotOK  ");
                    }
                }
                cout<<endl;

                cout<<"    Energy  : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                    cout<<Form("%9.3lf, ", Emedian[icomp][ibin]);
                cout<<endl;
                cout<<"    Flux    : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                    if (ts_bin>=TS_UL[2])
                        cout<<Form("%9.3lf, ", ftemp->Eval(Flux[icomp][ibin]));
                    else
                        cout<<Form("%9.3lf, ", ftemp->Eval(FluxUL[icomp][ibin]));
                }
                cout<<endl;
                cout<<"    Ferr    : ";
                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                    if (ts_bin>=TS_UL[2])
                        cout<<Form("%9.3lf, ", ftemp->Eval(FluxErr[icomp][ibin]));
                    else 
                        cout<<Form("%9.3lf, ", 0.0);
                }
                cout<<endl;
                if (cf.UseKM2A){
                    cout<<"    Energy1 : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++)
                        cout<<Form("%9.3lf, ", Emedian[icomp][ibin+NBinUsed[1][1]]);
                    cout<<endl;
                    cout<<"    Flux1   : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                        double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                        if (ts_bin>=TS_UL[2])
                            cout<<Form("%9.3lf, ", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]]));
                        else
                            cout<<Form("%9.3lf, ", ftemp->Eval(FluxUL[icomp][ibin+NBinUsed[1][1]]));
                    }
                    cout<<endl;
                    cout<<"    Ferr1   : ";
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                        double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                        if (ts_bin>=TS_UL[2])
                            cout<<Form("%9.3lf, ", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]));
                        else 
                            cout<<Form("%9.3lf, ", 0.0);
                    }
                    cout<<endl;
                }

                for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                    double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                    if (ibin<cf.NnhitUsed){
                        out<<Form("%.3lf", Emedian[icomp][ibin])<<" ";
                        if (ts_bin>=TS_UL[2]){
                            out<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin]))<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin]))<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin]))<<" "
                                <<Form("%.2lf", ts_bin)<<" "<<1<<endl;
                        }
                        else{
                            out<<Form("%.3lf", ftemp->Eval(FluxUL[icomp][ibin]))<<" "
                                <<Form("%.3lf", 0.0)<<" "<<Form("%.3lf", 0.0)<<" "
                                <<Form("%.2lf", ts_bin)<<" "<<1<<endl;
                        }
                    }
                    else{
                        out<<Form("%.3lf", Emedian[icomp][ibin+NBinUsed[1][1]])<<" ";
                        if (ts_bin>=TS_UL[2]){
                            out<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]]))<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]))<<" "
                                <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]]))<<" "
                                <<Form("%.2lf", ts_bin)<<" "<<0<<endl;
                        }
                        else{
                            out<<Form("%.3lf", ftemp->Eval(FluxUL[icomp][ibin+NBinUsed[1][1]]))<<" "
                                <<Form("%.3lf", 0.0)<<" "<<Form("%.3lf", 0.0)<<" "
                                <<Form("%.2lf", ts_bin)<<" "<<0<<endl;
                        }
                    }
                }
                out.close();

                if (abstag){
                    for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
                        double ts_bin = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                        if (ibin<cf.NnhitUsed){
                            out_obs<<Form("%.3lf", Emedian[icomp][ibin])<<" ";
                            double ebl_abs = exp(-Template->Srcs[icomp].gg_ebl->Eval(Emedian[icomp][ibin]));
                            if (ts_bin>=TS_UL[2]){
                                out_obs<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin])*ebl_abs)<<" "
                                    <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin])*ebl_abs)<<" "
                                    <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin])*ebl_abs)<<" "
                                    <<Form("%.2lf", ts_bin)<<" "<<1<<endl;
                            }
                            else{
                                out_obs<<Form("%.3lf", ftemp->Eval(FluxUL[icomp][ibin])*ebl_abs)<<" "
                                    <<Form("%.3lf", 0.0)<<" "<<Form("%.3lf", 0.0)<<" "
                                    <<Form("%.2lf", ts_bin)<<" "<<1<<endl;
                            }
                        }
                        else{
                            out_obs<<Form("%.3lf", Emedian[icomp][ibin+NBinUsed[1][1]])<<" ";
                            double ebl_abs = exp(-Template->Srcs[icomp].gg_ebl->Eval(Emedian[icomp][ibin+NBinUsed[1][1]]));
                            if (ts_bin>=TS_UL[2]){
                                out_obs<<Form("%.3lf", ftemp->Eval(Flux[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                    <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                    <<Form("%.3lf", ftemp->Eval(FluxErr[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                    <<Form("%.2lf", ts_bin)<<" "<<0<<endl;
                            }
                            else{
                                out_obs<<Form("%.3lf", ftemp->Eval(FluxUL[icomp][ibin+NBinUsed[1][1]])*ebl_abs)<<" "
                                    <<Form("%.3lf", 0.0)<<" "<<Form("%.3lf", 0.0)<<" "
                                    <<Form("%.2lf", ts_bin)<<" "<<0<<endl;
                            }
                        }
                    }
                    out_obs.close();
                }


            }

            cout<<"    Chi2/NDF of SED = "<<Form("%.3lf/%.lf", SEDChi2[icomp], SEDChi2[icomp+Template->NComp])<<endl;

        }
        cout<<" ===================================================== "<<endl;
    }
}

void Src_FittingMode::DrawSED(Src_Template* Template){

    TCanvas *cc = new TCanvas("cc", "cc", 1000, 700);
    cc->SetLogx();
    cc->SetLogy();
    cc->SetLeftMargin(0.13);
    cc->SetRightMargin(0.07);
    TGraphErrors *gg = new TGraphErrors();
    gg->SetMarkerStyle(20);
    gg->SetMarkerSize(2.0);
    gg->SetMarkerColor(kCyan+1);
    gg->SetLineColor(kCyan+1);
    gg->SetLineWidth(2);

    int npar = 0;
    for (int icomp=0;icomp<Template->NComp;icomp++){

        string f0_order, srcname;
        TF1 *fSED;
        double f0_scale = 1.0;
        if (icomp>=0 && icomp<Template->NSrc){
            fSED = new TF1("fSED", Form("x*x*%s", Template->Srcs[icomp].SEDFormula.data()), 0.01, 1000);
            f0_order = Template->Srcs[icomp].F0_order;
            srcname  = Template->Srcs[icomp].Srcname;
            for (int ipar=0;ipar<Template->Srcs[icomp].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs[icomp].nSEDpar+Template->Srcs[icomp].nMorpar;
        }
        else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon)){
            fSED = new TF1("fSED", Form("x*x*%s", Template->Srcs_NumCon[icomp-Template->NSrc].SEDFormula.data()), 0.01, 1000);
            f0_order = Template->Srcs_NumCon[icomp-Template->NSrc].F0_order;
            srcname  = Template->Srcs_NumCon[icomp-Template->NSrc].Srcname;
            for (int ipar=0;ipar<Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar+2]);
            npar += 2+Template->Srcs_NumCon[icomp-Template->NSrc].nSEDpar+Template->Srcs_NumCon[icomp-Template->NSrc].nMorpar;
        }
        else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total){
            fSED = new TF1("fSED", Form("x*x*%s", Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].SEDFormula.data()), 0.01, 1000);
            f0_order = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].F0_order;
            srcname  = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].Srcname;
            for (int ipar=0;ipar<Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].nSEDpar;
        }
        else{
            f0_scale = Template->DGEs[icomp-Template->NSrc_total].Omega_total_model*Template->DGEs[icomp-Template->NSrc_total].Eta/Template->DGEs[icomp-Template->NSrc_total].Omega_total;
            fSED = new TF1("fSED", Form("x*x*%lf*%s", f0_scale, Template->DGEs[icomp-Template->NSrc_total].SEDFormula.data()), 0.01, 1000);
            f0_order = Template->DGEs[icomp-Template->NSrc_total].F0_order;
            srcname  = Template->DGEs[icomp-Template->NSrc_total].Srcname;
            for (int ipar=0;ipar<Template->DGEs[icomp-Template->NSrc_total].nSEDpar;ipar++)
                fSED->SetParameter(ipar, ParVal[npar+ipar]);
            npar += Template->DGEs[icomp-Template->NSrc_total].nSEDpar;
        }

        /*double *ewcda, *fwcda, *ferrwcda;
        double *ekm2a, *fkm2a, *ferrkm2a;
        if (cf.UseWCDA){
            ewcda = new double[cf.NnhitUsed];
            fwcda = new double[cf.NnhitUsed];
            ferrwcda = new double[cf.NnhitUsed];
            for (int ii=0;ii<cf.NnhitUsed;ii++){
                ewcda[ii] = 0;
                fwcda[ii] = 0;
                ferrwcda[ii] = 0;
            }
        }
        if (cf.UseKM2A){
            ekm2a = new double[cf.KNEbinUsed];
            fkm2a = new double[cf.KNEbinUsed];
            ferrkm2a = new double[cf.KNEbinUsed];
            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                ekm2a[ii] = 0;
                fkm2a[ii] = 0;
                ferrkm2a[ii] = 0;
            }
        }*/

        double emin = 10000., emax = 0;
        double fmin = 10000., fmax = 0;

        TF1 *ftemp = new TF1("ftemp", Form("x/%s", f0_order.data()), 0, 1);
        TF1 *ftemp1 = new TF1("ftemp1", Form("x*%s", f0_order.data()), 0, 1);
        for (int ibin=0;ibin<NBinUsed[1][1];ibin++){
            if (cf.FitOpt[3]==0){
                if (ftemp->Eval(Flux[icomp][ibin])>0.01){
                    if (Emedian[icomp][ibin]<emin) emin = Emedian[icomp][ibin];
                    if (Emedian[icomp][ibin]>emax) emax = Emedian[icomp][ibin];
                    if (ftemp->Eval(Flux[icomp][ibin])<fmin) fmin = ftemp->Eval(Flux[icomp][ibin]);
                    if (ftemp->Eval(Flux[icomp][ibin])>fmax) fmax = ftemp->Eval(Flux[icomp][ibin]);
                }
            }
            else{
                double TS = TS_Bin[icomp][ibin]+TS_Bin[icomp][ibin+NBinUsed[1][1]];
                if (TS>=TS_UL[2]){
                    if (Emedian[icomp][ibin]<emin) emin = Emedian[icomp][ibin];
                    if (Emedian[icomp][ibin]>emax) emax = Emedian[icomp][ibin];
                    if (ftemp->Eval(Flux[icomp][ibin])<fmin) fmin = ftemp->Eval(Flux[icomp][ibin]);
                    if (ftemp->Eval(Flux[icomp][ibin])>fmax) fmax = ftemp->Eval(Flux[icomp][ibin]);
                }
                else{
                    Flux[icomp][ibin] = FluxUL[icomp][ibin];
                    FluxErr[icomp][ibin] = 0;
                    if (ftemp->Eval(Flux[icomp][ibin])>0.01){
                        if (Emedian[icomp][ibin]<emin) emin = Emedian[icomp][ibin];
                        if (Emedian[icomp][ibin]>emax) emax = Emedian[icomp][ibin];
                        if (ftemp->Eval(Flux[icomp][ibin])<fmin) fmin = ftemp->Eval(Flux[icomp][ibin]);
                        if (ftemp->Eval(Flux[icomp][ibin])>fmax) fmax = ftemp->Eval(Flux[icomp][ibin]);
                    }
                }
            }

            gg->SetPoint(ibin, Emedian[icomp][ibin], Flux[icomp][ibin]);
            gg->SetPointError(ibin, 0, FluxErr[icomp][ibin]);
        }

        gg->SetTitle(Form("SED %s;E [ TeV ];Flux E^{2} [ TeVcm^{-2}s^{-1} ]", srcname.data()));
        gg->GetXaxis()->SetLimits(emin*0.3, emax*3);
        gg->GetYaxis()->SetRangeUser(ftemp1->Eval(fmin)*0.1, ftemp1->Eval(fmax)*5);
        cc->cd();
        gg->Draw("AP");
        fSED->SetRange(emin*0.5, emax*2);
        fSED->SetLineColor(kGray+1);
        fSED->SetLineWidth(2);
        fSED->Draw("lsame");
        TLatex *lx = new TLatex(emin, ftemp1->Eval(fmin), Form("Chi2/NDF = %.3lf/%.lf", SEDChi2[icomp], SEDChi2[icomp+Template->NComp]));
        lx->SetTextColor(kGray);
        lx->Draw();
        cc->SaveAs(Form("%s/%s/SED_Mor/SED_%s_temp.png", cf.WorkDir.data(), cf.Outdir.data(), srcname.data()));

    }

}


void Src_FittingMode::MkOutdir(){

    if (access(cf.Outdir.data(), R_OK|W_OK)!=0){
        string command = "mkdir -p ";
        command += cf.WorkDir+"/"+cf.Outdir;
        system(command.data());
    }
    string dirtemp = cf.Outdir+"/Check";
    if (access(dirtemp.data(), R_OK|W_OK)!=0){
        string command = "mkdir -p ";
        command += cf.WorkDir+"/"+dirtemp;
        system(command.data());
    }
    dirtemp = cf.Outdir+"/SED_Mor";
    if (access(dirtemp.data(), R_OK|W_OK)!=0){
        string command = "mkdir -p ";
        command += cf.WorkDir+"/"+dirtemp;
        system(command.data());
    }

}

void Src_FittingMode::DrawSigMap(Src_Template* Template,  vector<long int> cellid, double **Wnon, double **Wnbkg, double **Knon, double **Knbkg, double **Wnmodel_convo, double **Knmodel_convo, double **Wpsf, double **Kpsf){

    int Neffbins = cellid.size();
    double Xmin = 360, Xmax = 0, Ymin = 90, Ymax = -90;
    map<long int, int> cellid_reverse;
    for (int ii=0;ii<Neffbins;ii++){
        double x0 = X[0]+(cellid[ii]/nbinsY)*wbinX;
        double y0 = Y[0]+(cellid[ii]%nbinsY)*wbinY;
        if (x0<Xmin)
            Xmin = x0;
        if ((x0+0.1)>Xmax)
            Xmax = x0+0.1;
        if (y0<Ymin)
            Ymin = y0;
        if ((y0+0.1)>Ymax)
            Ymax = y0+0.1;
        cellid_reverse.insert(pair<long int, int>(cellid[ii], ii+1));
    }
    int nxbins = (Xmax-Xmin+wbinX/10)/wbinX;
    int nybins = (Ymax-Ymin+wbinY/10)/wbinY;

    string outfile = cf.WorkDir+"/"+cf.Outdir+"/Check/"+"DataSigMap.root";
    TFile *fout = TFile::Open(outfile.data(), "recreate");
    fout->cd();
    // Data Sigmap of each bin
    string *bintag = new string[NBinUsed[0][1]];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        bintag[ii] = "";
        if (cf.UseWCDA){
            if (ii<cf.NnhitUsed)
                bintag[ii] = Form("%d<=nhit<%d", cf.Nhit[ii+cf.NhitUsed[0]], cf.Nhit[ii+1+cf.NhitUsed[0]]);
            else
                bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
        }
        else
            bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
    }

    TH2D *hDataSig[NBinUsed[0][1]];
    for (int ii=0;ii<NBinUsed[0][1];ii++)
        hDataSig[ii] = new TH2D(Form("hDataSig_%d", ii), Form("DataSig        %s", bintag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
    cout<<" *** Cal Data Sigmap of each bin : "<<endl;
    double ra, dec;
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        cout<<" Bin"<<ii<<" ... "<<endl;
        for (int jj=0;jj<Neffbins;jj++){
            double x0 = X[0]+(cellid[jj]/nbinsY+0.5)*wbinX;
            double y0 = Y[0]+(cellid[jj]%nbinsY+0.5)*wbinY;
            int ibinX = (x0-Xmin)/wbinX;
            int ibinY = (y0-Ymin)/wbinY;
            if (!cf.CorOpt)
                dec = y0;
            else
                g2e(x0, y0, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;
            double psf39, psf68;
            if (cf.UseWCDA){
                if (ii<cf.NnhitUsed){
                    psf39 = Wpsf[ii][idecbin*2];
                    psf68 = Wpsf[ii][idecbin*2+1];
                }
                else{
                    psf39 = Kpsf[ii-cf.NnhitUsed][idecbin*2];
                    psf68 = Kpsf[ii-cf.NnhitUsed][idecbin*2+1];
                }
            }
            else{
                psf39 = Kpsf[ii][idecbin*2];
                psf68 = Kpsf[ii][idecbin*2+1];
            }

            int xbins0 = (x0-1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int xbins1 = (x0+1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int ybins0 = max((int)((y0-1.5*psf68-Y[0])/wbinY+0.5), 0); 
            int ybins1 = min((int)((y0+1.5*psf68-Y[0])/wbinY+0.5), nbinsY);

            double sum_on   = 0;
            double sum_on2  = 0;
            double sum_bkg  = 0;
            double sum_bkg2 = 0;

            for (int mm=xbins0;mm<=xbins1;mm++){
                int mm_temp = mm;
                if (mm<0) mm_temp = mm + nbinsX;
                if (mm>=nbinsX) mm_temp = mm - nbinsX;
                double x1 = X[0] + (mm_temp+0.5)*wbinX;

                for (int nn=ybins0;nn<=ybins1;nn++){

                    int ipixel = cellid_reverse[mm_temp*nbinsY+nn];
                    if (!ipixel) continue;
                    ipixel = ipixel-1;
                    double y1  = Y[0] + (nn+0.5)*wbinY;
                    double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                    if (space<psf68){
                        double w = exp(-(space*space)/(2.0*psf39*psf39))/(2*TMath::Pi()*psf39*psf39);
                        if (cf.UseWCDA){
                            if (ii<cf.NnhitUsed){
                                sum_on  += Wnon[ii][ipixel]*w;
                                sum_on2 += Wnon[ii][ipixel]*w*w;
                                sum_bkg += Wnbkg[ii][ipixel]*w;
                                sum_bkg2 += Wnbkg[ii][ipixel]*w*w;
                            }
                            else{
                                sum_on  += Knon[ii-cf.NnhitUsed][ipixel]*w;
                                sum_on2 += Knon[ii-cf.NnhitUsed][ipixel]*w*w;
                                sum_bkg += Knbkg[ii-cf.NnhitUsed][ipixel]*w;
                                sum_bkg2 += Knbkg[ii-cf.NnhitUsed][ipixel]*w*w;
                            }
                        }
                        else{
                            sum_on  += Knon[ii][ipixel]*w;
                            sum_on2 += Knon[ii][ipixel]*w*w;
                            sum_bkg += Knbkg[ii][ipixel]*w;
                            sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                        }
                    }
                }
            }

            double scale = (sum_on+sum_bkg)/(sum_on2+sum_bkg2);
            double sum_on_0  = sum_on*scale;
            double sum_bkg_0 = sum_bkg*scale;
            double lamda = sum_bkg_0-sum_on_0*(1-log(sum_on_0/sum_bkg_0));
            double sig = 0;
            if (sum_on_0>=sum_bkg_0)
                sig = sqrt(2)*sqrt(lamda);
            else
                sig = -sqrt(2)*sqrt(lamda);
            if (!isnan(sig) && !isinf(sig))
                hDataSig[ii]->SetBinContent(ibinX+1, ibinY+1, sig);
        }
    }
    // Residual Sigmap of each bin
    TH2D *hResSig[NBinUsed[0][1]];
    TH1D *hResSig1D[NBinUsed[0][1]];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        hResSig[ii] = new TH2D(Form("hResSig_%d", ii), Form("ResidualSig    %s", bintag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
        hResSig1D[ii] = new TH1D(Form("hResSig1D_%d", ii), Form("ResidualSig1D  %s", bintag[ii].data()), 100, -10, 10);
    }
    cout<<" *** Cal Residual Sigmap of each bin : "<<endl;
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        cout<<" Bin"<<ii<<" ... "<<endl;
        for (int jj=0;jj<Neffbins;jj++){
            double x0 = X[0]+(cellid[jj]/nbinsY+0.5)*wbinX;
            double y0 = Y[0]+(cellid[jj]%nbinsY+0.5)*wbinY;
            int ibinX = (x0-Xmin)/wbinX;
            int ibinY = (y0-Ymin)/wbinY;
            if (!cf.CorOpt)
                dec = y0;
            else
                g2e(x0, y0, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;
            double psf39, psf68;
            if (cf.UseWCDA){
                if (ii<cf.NnhitUsed){
                    psf39 = Wpsf[ii][idecbin*2];
                    psf68 = Wpsf[ii][idecbin*2+1];
                }
                else{
                    psf39 = Kpsf[ii-cf.NnhitUsed][idecbin*2];
                    psf68 = Kpsf[ii-cf.NnhitUsed][idecbin*2+1];
                }
            }
            else{
                psf39 = Kpsf[ii][idecbin*2];
                psf68 = Kpsf[ii][idecbin*2+1];
            }

            int xbins0 = (x0-1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int xbins1 = (x0+1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int ybins0 = max((int)((y0-1.5*psf68-Y[0])/wbinY+0.5), 0); 
            int ybins1 = min((int)((y0+1.5*psf68-Y[0])/wbinY+0.5), nbinsY);

            double sum_on   = 0;
            double sum_on2  = 0;
            double sum_bkg  = 0;
            double sum_bkg2 = 0;

            for (int mm=xbins0;mm<=xbins1;mm++){
                int mm_temp = mm;
                if (mm<0) mm_temp = mm + nbinsX;
                if (mm>=nbinsX) mm_temp = mm - nbinsX;
                double x1 = X[0] + (mm_temp+0.5)*wbinX;

                for (int nn=ybins0;nn<=ybins1;nn++){

                    int ipixel = cellid_reverse[mm_temp*nbinsY+nn];
                    if (!ipixel) continue;
                    ipixel = ipixel-1;
                    double y1  = Y[0] + (nn+0.5)*wbinY;
                    double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                    if (space<psf68){
                        double w = exp(-(space*space)/(2.0*psf39*psf39))/(2*TMath::Pi()*psf39*psf39);
                        if (cf.UseWCDA){
                            if (ii<cf.NnhitUsed){
                                sum_on  += Wnon[ii][ipixel]*w;
                                sum_on2 += Wnon[ii][ipixel]*w*w;
                                sum_bkg += Wnbkg[ii][ipixel]*w;
                                sum_bkg2 += Wnbkg[ii][ipixel]*w*w;
                                for (int icomp=0;icomp<Template->NComp;icomp++){
                                    sum_bkg  += Wnmodel_convo[icomp][ii*Neffbins+ipixel]*w;
                                    sum_bkg2 += Wnmodel_convo[icomp][ii*Neffbins+ipixel]*w*w;
                                }
                            }
                            else{
                                sum_on  += Knon[ii-cf.NnhitUsed][ipixel]*w;
                                sum_on2 += Knon[ii-cf.NnhitUsed][ipixel]*w*w;
                                sum_bkg += Knbkg[ii-cf.NnhitUsed][ipixel]*w;
                                sum_bkg2 += Knbkg[ii-cf.NnhitUsed][ipixel]*w*w;
                                for (int icomp=0;icomp<Template->NComp;icomp++){
                                    sum_bkg  += Knmodel_convo[icomp][(ii-cf.NnhitUsed)*Neffbins+ipixel]*w;
                                    sum_bkg2 += Knmodel_convo[icomp][(ii-cf.NnhitUsed)*Neffbins+ipixel]*w*w;
                                }
                            }
                        }
                        else{
                            sum_on  += Knon[ii][ipixel]*w;
                            sum_on2 += Knon[ii][ipixel]*w*w;
                            sum_bkg += Knbkg[ii][ipixel]*w;
                            sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                            for (int icomp=0;icomp<Template->NComp;icomp++){
                                sum_bkg  += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w;
                                sum_bkg2 += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w*w;
                            }
                        }
                    }
                }
            }

            double scale = (sum_on+sum_bkg)/(sum_on2+sum_bkg2);
            double sum_on_0  = sum_on*scale;
            double sum_bkg_0 = sum_bkg*scale;
            double lamda = sum_bkg_0-sum_on_0*(1-log(sum_on_0/sum_bkg_0));
            double sig = 0;
            if (sum_on_0>=sum_bkg_0)
                sig = sqrt(2)*sqrt(lamda);
            else
                sig = -sqrt(2)*sqrt(lamda);
            if (!isnan(sig) && !isinf(sig)){
                hResSig[ii]->SetBinContent(ibinX+1, ibinY+1, sig);
                hResSig1D[ii]->Fill(sig);
            }
        }
    }

    double LWR = 1., winLe = 800;
    int NWinX;
    if (cf.UseWCDA) NWinX = 3;
    if (cf.UseKM2A) NWinX = 5;
    int NWinY;
    if (NBinUsed[0][1]%NWinX==0)
        NWinY = NBinUsed[0][1]/NWinX;
    else
        NWinY = NBinUsed[0][1]/NWinX+1;
    int NWinTotal = NWinX*NWinY;
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(1);
    gStyle->SetPalette(kRainBow);

    TCanvas *cc = new TCanvas("cc", "cc", NWinX*winLe, NWinY*winLe/LWR);
    cc->Divide(NWinX, NWinY);
    for (int ii=0;ii<NBinUsed[0][1];ii++){

        cc->cd(ii+1);
        hDataSig[ii]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }

    }
    cc->SaveAs(Form("%s/%s/Check/DataSig.png", cf.WorkDir.data(), cf.Outdir.data()));
    cc->SaveAs(Form("%s/%s/Check/DataSig.pdf", cf.WorkDir.data(), cf.Outdir.data()));
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        cc->cd(ii+1);
        hResSig[ii]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
    }
    cc->SaveAs(Form("%s/%s/Check/DataResiSig.png", cf.WorkDir.data(), cf.Outdir.data()));
    cc->SaveAs(Form("%s/%s/Check/DataResiSig.pdf", cf.WorkDir.data(), cf.Outdir.data()));
    for (int ii=0;ii<NBinUsed[0][1];ii++)
        hResSig1D[ii]->Fit("gaus", "Q");
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        cc->cd(ii+1);
        cc->GetPad(ii+1)->SetLogy();
        hResSig1D[ii]->Draw();
    }
    cc->SaveAs(Form("%s/%s/Check/DataResiSig1D.png", cf.WorkDir.data(), cf.Outdir.data()));
    cc->SaveAs(Form("%s/%s/Check/DataResiSig1D.pdf", cf.WorkDir.data(), cf.Outdir.data()));

    for (int ii=0;ii<NBinUsed[0][1];ii++){
        hDataSig[ii]->Write();
        hResSig[ii]->Write();
        hResSig1D[ii]->Write();
    }
    fout->Close();

    outfile = cf.WorkDir+"/"+cf.Outdir+"/Check/"+"CompSigMap.root";
    fout = TFile::Open(outfile.data(), "recreate");
    fout->cd();
    // Data SigMap : nhit>=200, 25TeV<E<100TeV, E>100TeV
    const int NDet = 3;
    string maptag[NDet] = {"nhit>=200", "25TeV<E<100TeV", "E>100TeV"};
    string mapfigtag[NDet] = {"nhit_ge200", "E_25_100", "E_ge100"};
    TH2D *hDataSigS[NDet];
    for (int ii=0;ii<NDet;ii++)
        hDataSigS[ii] = new TH2D(Form("hDataSigS_%d", ii), Form("DataSig        %s", maptag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
    
    cout<<" *** Cal Data Sigmap : nhit>=200, 25TeV<E<100TeV, E>100TeV "<<endl;
    for (int iDet=0;iDet<NDet;iDet++){
        if (iDet==0 && !cf.UseWCDA) continue;
        if (iDet>=1 && !cf.UseKM2A) continue;
        for (int jj=0;jj<Neffbins;jj++){
            double x0 = X[0]+(cellid[jj]/nbinsY+0.5)*wbinX;
            double y0 = Y[0]+(cellid[jj]%nbinsY+0.5)*wbinY;
            int ibinX = (x0-Xmin)/wbinX;
            int ibinY = (y0-Ymin)/wbinY;
            if (!cf.CorOpt)
                dec = y0;
            else
                g2e(x0, y0, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

            double psf39 = 0, psf68 = 0;
            if (iDet==0){
                for (int ii=0;ii<cf.NnhitUsed;ii++)
                    if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                        psf39 = Wpsf[ii][idecbin*2];
                        psf68 = Wpsf[ii][idecbin*2+1];
                        break;
                    }
            }
            if (iDet==1){
                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0)){
                        psf39 = Kpsf[ii][idecbin*2];
                        psf68 = Kpsf[ii][idecbin*2+1];
                        break;
                    }
            }
            if (iDet==2){
                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                        psf39 = Kpsf[ii][idecbin*2];
                        psf68 = Kpsf[ii][idecbin*2+1];
                        break;
                    }
            }

            int xbins0 = (x0-1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int xbins1 = (x0+1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int ybins0 = max((int)((y0-1.5*psf68-Y[0])/wbinY+0.5), 0); 
            int ybins1 = min((int)((y0+1.5*psf68-Y[0])/wbinY+0.5), nbinsY);

            double sum_on   = 0;
            double sum_on2  = 0;
            double sum_bkg  = 0;
            double sum_bkg2 = 0;

            for (int mm=xbins0;mm<=xbins1;mm++){
                int mm_temp = mm;
                if (mm<0) mm_temp = mm + nbinsX;
                if (mm>=nbinsX) mm_temp = mm - nbinsX;
                double x1 = X[0] + (mm_temp+0.5)*wbinX;

                for (int nn=ybins0;nn<=ybins1;nn++){

                    int ipixel = cellid_reverse[mm_temp*nbinsY+nn];
                    if (!ipixel) continue;
                    ipixel = ipixel-1;
                    double y1  = Y[0] + (nn+0.5)*wbinY;
                    double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                    if (space<psf68){
                        double w = exp(-(space*space)/(2.0*psf39*psf39))/(2*TMath::Pi()*psf39*psf39);
                        if (iDet==0){
                            for (int ii=0;ii<cf.NnhitUsed;ii++)
                                if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                                    sum_on  += Wnon[ii][ipixel]*w;
                                    sum_on2 += Wnon[ii][ipixel]*w*w;
                                    sum_bkg += Wnbkg[ii][ipixel]*w;
                                    sum_bkg2 += Wnbkg[ii][ipixel]*w*w;
                                }
                        }
                        if (iDet==1){
                            for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0) && (cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep)<=log10(100.1)){
                                    sum_on  += Knon[ii][ipixel]*w;
                                    sum_on2 += Knon[ii][ipixel]*w*w;
                                    sum_bkg += Knbkg[ii][ipixel]*w;
                                    sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                }
                        }
                        if (iDet==2){
                            for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                                    sum_on  += Knon[ii][ipixel]*w;
                                    sum_on2 += Knon[ii][ipixel]*w*w;
                                    sum_bkg += Knbkg[ii][ipixel]*w;
                                    sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                }
                        }
                    }
                }
            }

            double scale = (sum_on+sum_bkg)/(sum_on2+sum_bkg2);
            double sum_on_0  = sum_on*scale;
            double sum_bkg_0 = sum_bkg*scale;
            double lamda = sum_bkg_0-sum_on_0*(1-log(sum_on_0/sum_bkg_0));
            double sig = 0;
            if (sum_on_0>=sum_bkg_0)
                sig = sqrt(2)*sqrt(lamda);
            else
                sig = -sqrt(2)*sqrt(lamda);
            if (!isnan(sig) && !isinf(sig))
                hDataSigS[iDet]->SetBinContent(ibinX+1, ibinY+1, sig);
        }
    }

    TCanvas *cc2 = new TCanvas("cc2", "cc2", 1500, 1500);
    if (cf.UseWCDA){
        cc2->cd();
        hDataSigS[0]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }

        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));
        hDataSigS[0]->Write();
    }
    if (cf.UseKM2A){
        cc2->cd();
        hDataSigS[1]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));

        hDataSigS[2]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));

        hDataSigS[1]->Write();
        hDataSigS[2]->Write();
    }

    // Residual Sigmap : nhit>=200, 25TeV<E<100TeV, E>100TeV
    TH2D *hResSigS[NDet];
    TH1D *hResSig1DS[NDet];
    for (int ii=0;ii<NDet;ii++){
        hResSigS[ii] = new TH2D(Form("hResSigS_%d", ii), Form("ResidualSig    %s", maptag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
        hResSig1DS[ii] = new TH1D(Form("hResSig1DS_%d", ii), Form("ResidualSig1D  %s", maptag[ii].data()), 100, -10, 10);
    }
    
    cout<<" *** Cal Residual Sigmap : nhit>=200, 25TeV<E<100TeV, E>100TeV "<<endl;
    for (int iDet=0;iDet<NDet;iDet++){
        if (iDet==0 && !cf.UseWCDA) continue;
        if (iDet>=1 && !cf.UseKM2A) continue;
        for (int jj=0;jj<Neffbins;jj++){
            double x0 = X[0]+(cellid[jj]/nbinsY+0.5)*wbinX;
            double y0 = Y[0]+(cellid[jj]%nbinsY+0.5)*wbinY;
            int ibinX = (x0-Xmin)/wbinX;
            int ibinY = (y0-Ymin)/wbinY;
            if (!cf.CorOpt)
                dec = y0;
            else
                g2e(x0, y0, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

            double psf39 = 0, psf68 = 0;
            if (iDet==0){
                for (int ii=0;ii<cf.NnhitUsed;ii++)
                    if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                        psf39 = Wpsf[ii][idecbin*2];
                        psf68 = Wpsf[ii][idecbin*2+1];
                        break;
                    }
            }
            if (iDet==1){
                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0)){
                        psf39 = Kpsf[ii][idecbin*2];
                        psf68 = Kpsf[ii][idecbin*2+1];
                        break;
                    }
            }
            if (iDet==2){
                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                        psf39 = Kpsf[ii][idecbin*2];
                        psf68 = Kpsf[ii][idecbin*2+1];
                        break;
                    }
            }

            int xbins0 = (x0-1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int xbins1 = (x0+1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
            int ybins0 = max((int)((y0-1.5*psf68-Y[0])/wbinY+0.5), 0); 
            int ybins1 = min((int)((y0+1.5*psf68-Y[0])/wbinY+0.5), nbinsY);

            double sum_on   = 0;
            double sum_on2  = 0;
            double sum_bkg  = 0;
            double sum_bkg2 = 0;

            for (int mm=xbins0;mm<=xbins1;mm++){
                int mm_temp = mm;
                if (mm<0) mm_temp = mm + nbinsX;
                if (mm>=nbinsX) mm_temp = mm - nbinsX;
                double x1 = X[0] + (mm_temp+0.5)*wbinX;

                for (int nn=ybins0;nn<=ybins1;nn++){

                    int ipixel = cellid_reverse[mm_temp*nbinsY+nn];
                    if (!ipixel) continue;
                    ipixel = ipixel-1;
                    double y1  = Y[0] + (nn+0.5)*wbinY;
                    double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                    if (space<psf68){
                        double w = exp(-(space*space)/(2.0*psf39*psf39))/(2*TMath::Pi()*psf39*psf39);
                        if (iDet==0){
                            for (int ii=0;ii<cf.NnhitUsed;ii++)
                                if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                                    sum_on  += Wnon[ii][ipixel]*w;
                                    sum_on2 += Wnon[ii][ipixel]*w*w;
                                    sum_bkg += Wnbkg[ii][ipixel]*w;
                                    sum_bkg2 += Wnbkg[ii][ipixel]*w*w;
                                    for (int icomp=0;icomp<Template->NComp;icomp++){
                                        sum_bkg  += Wnmodel_convo[icomp][ii*Neffbins+ipixel]*w;
                                        sum_bkg2 += Wnmodel_convo[icomp][ii*Neffbins+ipixel]*w*w;
                                    }
                                }
                        }
                        if (iDet==1){
                            for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0) && (cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep)<=log10(100.1)){
                                    sum_on  += Knon[ii][ipixel]*w;
                                    sum_on2 += Knon[ii][ipixel]*w*w;
                                    sum_bkg += Knbkg[ii][ipixel]*w;
                                    sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                    for (int icomp=0;icomp<Template->NComp;icomp++){
                                        sum_bkg  += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w;
                                        sum_bkg2 += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w*w;
                                    }
                                }
                        }
                        if (iDet==2){
                            for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                                    sum_on  += Knon[ii][ipixel]*w;
                                    sum_on2 += Knon[ii][ipixel]*w*w;
                                    sum_bkg += Knbkg[ii][ipixel]*w;
                                    sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                    for (int icomp=0;icomp<Template->NComp;icomp++){
                                        sum_bkg  += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w;
                                        sum_bkg2 += Knmodel_convo[icomp][ii*Neffbins+ipixel]*w*w;
                                    }
                                }
                        }
                    }
                }
            }

            double scale = (sum_on+sum_bkg)/(sum_on2+sum_bkg2);
            double sum_on_0  = sum_on*scale;
            double sum_bkg_0 = sum_bkg*scale;
            double lamda = sum_bkg_0-sum_on_0*(1-log(sum_on_0/sum_bkg_0));
            double sig = 0;
            if (sum_on_0>=sum_bkg_0)
                sig = sqrt(2)*sqrt(lamda);
            else
                sig = -sqrt(2)*sqrt(lamda);
            if (!isnan(sig) && !isinf(sig)){
                hResSigS[iDet]->SetBinContent(ibinX+1, ibinY+1, sig);
                hResSig1DS[iDet]->Fill(sig);
            }
        }
    }
    if (cf.UseWCDA){
        cc2->cd();
        hResSigS[0]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));

        hResSigS[0]->Write();
    }
    if (cf.UseKM2A){
        cc2->cd();
        hResSigS[1]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));

        hResSigS[2]->Draw("colz");
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            TMarker *mm = new TMarker(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            if (Template->Srcs[isrc].Mortype == "Point") continue;
            TEllipse *e1 = new TEllipse(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], Template->Srcs[isrc].MorPar[0][0]/cos(Template->Srcs[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs[isrc].MorPar[0][0]);
            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            TMarker *mm = new TMarker(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], 5);
            mm->SetMarkerSize(4);
            mm->SetMarkerColor(kBlack);
            mm->Draw();
            TEllipse *e1;
            if (Template->Srcs_NumCon[isrc].Mortype != "Ext_EGaus")
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[0][0]);
            else
                e1 = new TEllipse(Template->Srcs_NumCon[isrc].Ra[0], Template->Srcs_NumCon[isrc].Dec[0], Template->Srcs_NumCon[isrc].MorPar[0][0]/cos(Template->Srcs_NumCon[isrc].Dec[0]/180*TMath::Pi()), Template->Srcs_NumCon[isrc].MorPar[1][0], 0, 360, Template->Srcs_NumCon[isrc].MorPar[2][0]);

            e1->SetFillStyle(0);
            e1->SetLineColor(kBlack);
            e1->SetLineWidth(3);
            e1->Draw("f");
        }
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));

        hResSigS[1]->Write();
        hResSigS[2]->Write();
    }

    if (cf.UseWCDA){
        hResSig1DS[0]->Fit("gaus", "Q");
        cc2->cd();
        cc2->SetLogy();
        hResSig1DS[0]->Draw();
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[0].data()));

        hResSig1DS[0]->Write();
    }

    if (cf.UseKM2A){
        hResSig1DS[1]->Fit("gaus", "Q");
        hResSig1DS[2]->Fit("gaus", "Q");

        cc2->cd();
        cc2->SetLogy();
        hResSig1DS[1]->Draw();
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[1].data()));
        hResSig1DS[2]->Draw();
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.png", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));
        cc2->SaveAs(Form("%s/%s/Check/DataResiSig1D_%s.pdf", cf.WorkDir.data(), cf.Outdir.data(), mapfigtag[2].data()));

        hResSig1DS[1]->Write();
        hResSig1DS[2]->Write();
    }

    // Sigmap of each Component : nhit>=200, 25TeV<E<100TeV, E>100TeV
    TH2D *hCompSig[NDet*Template->NComp];
    string srcname;
    for (int ii=0;ii<NDet;ii++)
        for (int icomp=0;icomp<Template->NComp;icomp++){
            if (icomp>=0 && icomp<Template->NSrc)
                srcname  = Template->Srcs[icomp].Srcname;
            else if (icomp>=Template->NSrc && icomp<(Template->NSrc+Template->NSrc_NumCon))
                srcname  = Template->Srcs_NumCon[icomp-Template->NSrc].Srcname;
            else if (icomp>=(Template->NSrc+Template->NSrc_NumCon) && icomp<Template->NSrc_total)
                srcname  = Template->Srcs_Temp[icomp-(Template->NSrc+Template->NSrc_NumCon)].Srcname;
            else
                srcname  = Template->DGEs[icomp-Template->NSrc_total].Srcname;

            hCompSig[ii*Template->NComp+icomp] = new TH2D(Form("hCompSig_%d_%d", ii, icomp), Form("SigMap_%-15s  %s", srcname.data(), maptag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
        }
    
    cout<<" *** Cal Sigmap of each component: nhit>=200, 25TeV<E<100TeV, E>100TeV "<<endl;
    for (int iDet=0;iDet<NDet;iDet++){
        if (iDet==0 && !cf.UseWCDA) continue;
        if (iDet>=1 && !cf.UseKM2A) continue;

        for (int icomp=0;icomp<Template->NComp;icomp++){
            for (int jj=0;jj<Neffbins;jj++){
                double x0 = X[0]+(cellid[jj]/nbinsY+0.5)*wbinX;
                double y0 = Y[0]+(cellid[jj]%nbinsY+0.5)*wbinY;
                int ibinX = (x0-Xmin)/wbinX;
                int ibinY = (y0-Ymin)/wbinY;
                if (!cf.CorOpt)
                    dec = y0;
                else
                    g2e(x0, y0, &ra, &dec);

                int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

                double psf39 = 0, psf68 = 0;
                if (iDet==0){
                    for (int ii=0;ii<cf.NnhitUsed;ii++)
                        if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                            psf39 = Wpsf[ii][idecbin*2];
                            psf68 = Wpsf[ii][idecbin*2+1];
                            break;
                        }
                }
                if (iDet==1){
                    for (int ii=0;ii<cf.KNEbinUsed;ii++)
                        if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0)){
                            psf39 = Kpsf[ii][idecbin*2];
                            psf68 = Kpsf[ii][idecbin*2+1];
                            break;
                        }
                }
                if (iDet==2){
                    for (int ii=0;ii<cf.KNEbinUsed;ii++)
                        if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                            psf39 = Kpsf[ii][idecbin*2];
                            psf68 = Kpsf[ii][idecbin*2+1];
                            break;
                        }
                }

                int xbins0 = (x0-1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
                int xbins1 = (x0+1.5*psf68/cos(y0/180*TMath::Pi())-X[0])/wbinX + 0.5;
                int ybins0 = max((int)((y0-1.5*psf68-Y[0])/wbinY+0.5), 0); 
                int ybins1 = min((int)((y0+1.5*psf68-Y[0])/wbinY+0.5), nbinsY);

                double sum_on   = 0;
                double sum_on2  = 0;
                double sum_bkg  = 0;
                double sum_bkg2 = 0;

                for (int mm=xbins0;mm<=xbins1;mm++){
                    int mm_temp = mm;
                    if (mm<0) mm_temp = mm + nbinsX;
                    if (mm>=nbinsX) mm_temp = mm - nbinsX;
                    double x1 = X[0] + (mm_temp+0.5)*wbinX;

                    for (int nn=ybins0;nn<=ybins1;nn++){

                        int ipixel = cellid_reverse[mm_temp*nbinsY+nn];
                        if (!ipixel) continue;
                        ipixel = ipixel-1;
                        double y1  = Y[0] + (nn+0.5)*wbinY;
                        double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                        if (space<psf68){
                            double w = exp(-(space*space)/(2.0*psf39*psf39))/(2*TMath::Pi()*psf39*psf39);
                            if (iDet==0){
                                for (int ii=0;ii<cf.NnhitUsed;ii++)
                                    if (cf.Nhit[ii+cf.NhitUsed[0]]>=200){
                                        sum_on  += Wnon[ii][ipixel]*w;
                                        sum_on2 += Wnon[ii][ipixel]*w*w;
                                        sum_bkg += Wnbkg[ii][ipixel]*w;
                                        sum_bkg2 += Wnbkg[ii][ipixel]*w*w;
                                        for (int jcomp=0;jcomp<Template->NComp;jcomp++){
                                            if (jcomp == icomp) continue;
                                            sum_bkg  += Wnmodel_convo[jcomp][ii*Neffbins+ipixel]*w;
                                            sum_bkg2 += Wnmodel_convo[jcomp][ii*Neffbins+ipixel]*w*w;
                                        }
                                    }
                            }
                            if (iDet==1){
                                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>log10(25.0) && (cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep)<=log10(100.1)){
                                        sum_on  += Knon[ii][ipixel]*w;
                                        sum_on2 += Knon[ii][ipixel]*w*w;
                                        sum_bkg += Knbkg[ii][ipixel]*w;
                                        sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                        for (int jcomp=0;jcomp<Template->NComp;jcomp++){
                                            if (jcomp == icomp) continue;
                                            sum_bkg  += Knmodel_convo[jcomp][ii*Neffbins+ipixel]*w;
                                            sum_bkg2 += Knmodel_convo[jcomp][ii*Neffbins+ipixel]*w*w;
                                        }
                                    }
                            }
                            if (iDet==2){
                                for (int ii=0;ii<cf.KNEbinUsed;ii++)
                                    if ((cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep)>=log10(99.9)){
                                        sum_on  += Knon[ii][ipixel]*w;
                                        sum_on2 += Knon[ii][ipixel]*w*w;
                                        sum_bkg += Knbkg[ii][ipixel]*w;
                                        sum_bkg2 += Knbkg[ii][ipixel]*w*w;
                                        for (int jcomp=0;jcomp<Template->NComp;jcomp++){
                                            if (jcomp == icomp) continue;
                                            sum_bkg  += Knmodel_convo[jcomp][ii*Neffbins+ipixel]*w;
                                            sum_bkg2 += Knmodel_convo[jcomp][ii*Neffbins+ipixel]*w*w;
                                        }
                                    }
                            }
                        }
                    }
                }

                double scale = (sum_on+sum_bkg)/(sum_on2+sum_bkg2);
                double sum_on_0  = sum_on*scale;
                double sum_bkg_0 = sum_bkg*scale;
                double lamda = sum_bkg_0-sum_on_0*(1-log(sum_on_0/sum_bkg_0));
                double sig = 0;
                if (sum_on_0>=sum_bkg_0)
                    sig = sqrt(2)*sqrt(lamda);
                else
                    sig = -sqrt(2)*sqrt(lamda);
                if (!isnan(sig) && !isinf(sig))
                    hCompSig[iDet*Template->NComp+icomp]->SetBinContent(ibinX+1, ibinY+1, sig);
            }
        }
    }

    if (cf.UseWCDA){
        for (int icomp=0;icomp<Template->NComp;icomp++)
            hCompSig[icomp]->Write();
    }
    if (cf.UseKM2A){
        for (int icomp=0;icomp<Template->NComp;icomp++)
            hCompSig[Template->NComp+icomp]->Write();
        for (int icomp=0;icomp<Template->NComp;icomp++)
            hCompSig[2*Template->NComp+icomp]->Write();
    }
    fout->Close();
}

void Src_FittingMode::OutPara(Src_Template* Template, int DGEsFlag, int SrcsFlag, double epiv_global, int parstatus_global[4]){

    if (!cf.FitOpt[0]) return;
    YAML::Node Paras;
    string outfile = cf.WorkDir+"/"+cf.Outdir+"/"+cf.fOut[0];
    ofstream out(outfile.data(), ios::out);

    if (!DGEsFlag){
        Paras["DGE"]["Active"] = 0;
    }
    else{
        Paras["DGE"]["Active"] = 1;
        Paras["DGE"]["ConvoPSF"] = 1;
        for (int idge=0;idge<Template->NDGE;idge++){
            string dgeid = Form("Template%d", idge);
            Paras["DGE"][dgeid]["Name"] = Template->DGEs[idge].Srcname;
            Paras["DGE"][dgeid]["Tempfile"] = Template->DGEs[idge].TempFile;
            string outstr;
            outstr = "["+Template->DGEs[idge].HistName+"]";
            Paras["DGE"][dgeid]["TempHist"] = outstr;
            Paras["DGE"][dgeid]["Epiv"] = Template->DGEs[idge].Epiv;
            Paras["DGE"][dgeid]["SEDModel"]["type"] = Template->DGEs[idge].SEDtype;
            int imodel = Template->Model->SEDMap[Template->DGEs[idge].SEDtype]-1;
            for (int ipar=0;ipar<Template->DGEs[idge].nSEDpar;ipar++){
                outstr = "[";
                double factor_temp = 1;
                if (ipar==0)
                   factor_temp = Template->DGEs[idge].Omega_total_model*Template->DGEs[idge].Eta/Template->DGEs[idge].Omega_total;
                outstr += Form("%.5lf, ", Template->DGEs[idge].SEDPar[ipar][0]*factor_temp);
                outstr += Form("%.2lf, ", Template->DGEs[idge].SEDPar[ipar][1]*factor_temp);
                outstr += Form("%.2lf, ", Template->DGEs[idge].SEDPar[ipar][2]*factor_temp);
                outstr += Form("%d", int(Template->DGEs[idge].SEDPar[ipar][3]));
                if (ipar==0)
                    outstr += Form(", %s", Template->DGEs[idge].F0_order.data());
                outstr += Form(", %.5lf", Template->DGEs[idge].SEDPar[ipar][4]*factor_temp);
                outstr += "]";
                Paras["DGE"][dgeid]["SEDModel"][Template->Model->SEDParname[imodel][ipar]] = outstr;
            }
        }
    }

    if (!SrcsFlag){
        Paras["SRC"]["Active"] = 0;
    }
    else{
        Paras["SRC"]["UseCatalog"] = 0;
        Paras["SRC"]["Active"] = 1;

        // global
        Paras["SRC"]["Epiv"] = epiv_global;
        Paras["SRC"]["ParStatus"]["Position"] = parstatus_global[0];
        Paras["SRC"]["ParStatus"]["F0"] = parstatus_global[1];
        Paras["SRC"]["ParStatus"]["Index"] = parstatus_global[2];
        Paras["SRC"]["ParStatus"]["MorPar"] = parstatus_global[3];

        // src definition
        for (int isrc=0;isrc<Template->NSrc_total;isrc++){
            string srcid = Form("Src%d", isrc);
            for (int jsrc=0;jsrc<Template->NSrc;jsrc++){
                if (Template->Srcs[jsrc].SrcID==isrc){
                    Paras["SRC"][srcid]["Name"] = Template->Srcs[jsrc].Srcname;
                    Paras["SRC"][srcid]["Epiv"] = Template->Srcs[jsrc].Epiv;
                    if (Template->Srcs[jsrc].GGAbsFlag)
                        Paras["SRC"][srcid]["GGAbs"] = Template->Srcs[jsrc].fGGAbs;
                    if (Template->Srcs[jsrc].LinkPars)
                        Paras["SRC"][srcid]["LinkPars"]["SED"] = Template->Srcs[jsrc].TargetSrcID;
                    Paras["SRC"][srcid]["SEDModel"]["type"] = Template->Srcs[jsrc].SEDtype;
                    int imodel = Template->Model->SEDMap[Template->Srcs[jsrc].SEDtype]-1;
                    string outstr;
                    for (int ipar=0;ipar<Template->Srcs[jsrc].nSEDpar;ipar++){
                        outstr = "[";
                        outstr += Form("%.5lf, ", Template->Srcs[jsrc].SEDPar[ipar][0]);
                        outstr += Form("%.2lf, ", Template->Srcs[jsrc].SEDPar[ipar][1]);
                        outstr += Form("%.2lf, ", Template->Srcs[jsrc].SEDPar[ipar][2]);
                        outstr += Form("%d", int(Template->Srcs[jsrc].SEDPar[ipar][3]));
                        if (ipar==0)
                            outstr += Form(", %s", Template->Srcs[jsrc].F0_order.data());
                        outstr += Form(", %.5lf", Template->Srcs[jsrc].SEDPar[ipar][4]);
                        outstr += "]";
                        Paras["SRC"][srcid]["SEDModel"][Template->Model->SEDParname[imodel][ipar]] = outstr;
                    }
                    Paras["SRC"][srcid]["MorModel"]["type"] = Template->Srcs[jsrc].Mortype;
                    outstr = "[";
                    outstr += Form("%.5lf, ", Template->Srcs[jsrc].Ra[0]);
                    outstr += Form("%.2lf, ", Template->Srcs[jsrc].Ra[1]);
                    outstr += Form("%.2lf, ", Template->Srcs[jsrc].Ra[2]);
                    outstr += Form("%d, ", int(Template->Srcs[jsrc].Ra[3]));
                    outstr += Form("%.5lf", Template->Srcs[jsrc].Ra[4]);
                    outstr += "]";
                    Paras["SRC"][srcid]["MorModel"]["ra"] = outstr;
                    outstr = "[";
                    outstr += Form("%.5lf, ", Template->Srcs[jsrc].Dec[0]);
                    outstr += Form("%.2lf, ", Template->Srcs[jsrc].Dec[1]);
                    outstr += Form("%.2lf, ", Template->Srcs[jsrc].Dec[2]);
                    outstr += Form("%d, ", int(Template->Srcs[jsrc].Dec[3]));
                    outstr += Form("%.5lf", Template->Srcs[jsrc].Dec[4]);
                    outstr += "]";
                    Paras["SRC"][srcid]["MorModel"]["dec"] = outstr;
                    imodel = Template->Model->MorMap[Template->Srcs[jsrc].Mortype]-1;
                    for (int ipar=0;ipar<Template->Srcs[jsrc].nMorpar;ipar++){
                        outstr = "[";
                        outstr += Form("%.5lf, ", Template->Srcs[jsrc].MorPar[ipar][0]);
                        outstr += Form("%.2lf, ", Template->Srcs[jsrc].MorPar[ipar][1]);
                        outstr += Form("%.2lf, ", Template->Srcs[jsrc].MorPar[ipar][2]);
                        outstr += Form("%d, ", int(Template->Srcs[jsrc].MorPar[ipar][3]));
                        outstr += Form("%.5lf", Template->Srcs[jsrc].MorPar[ipar][4]);
                        outstr += "]";
                        Paras["SRC"][srcid]["MorModel"][Template->Model->MorParname[imodel][ipar]] = outstr;
                    }
                }
            }

            for (int jsrc=0;jsrc<Template->NSrc_NumCon;jsrc++){
                if (Template->Srcs_NumCon[jsrc].SrcID==isrc){
                    Paras["SRC"][srcid]["Name"] = Template->Srcs_NumCon[jsrc].Srcname;
                    Paras["SRC"][srcid]["Epiv"] = Template->Srcs_NumCon[jsrc].Epiv;
                    if (Template->Srcs_NumCon[jsrc].LinkPars)
                        Paras["SRC"][srcid]["LinkPars"]["SED"] = Template->Srcs_NumCon[jsrc].TargetSrcID;
                    Paras["SRC"][srcid]["SEDModel"]["type"] = Template->Srcs_NumCon[jsrc].SEDtype;
                    int imodel = Template->Model->SEDMap[Template->Srcs_NumCon[jsrc].SEDtype]-1;
                    string outstr;
                    for (int ipar=0;ipar<Template->Srcs_NumCon[jsrc].nSEDpar;ipar++){
                        outstr = "[";
                        outstr += Form("%.5lf, ", Template->Srcs_NumCon[jsrc].SEDPar[ipar][0]);
                        outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].SEDPar[ipar][1]);
                        outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].SEDPar[ipar][2]);
                        outstr += Form("%d", int(Template->Srcs_NumCon[jsrc].SEDPar[ipar][3]));
                        if (ipar==0)
                            outstr += Form(", %s", Template->Srcs_NumCon[jsrc].F0_order.data());
                        outstr += Form(", %.5lf", Template->Srcs_NumCon[jsrc].SEDPar[ipar][4]);
                        outstr += "]";
                        Paras["SRC"][srcid]["SEDModel"][Template->Model->SEDParname[imodel][ipar]] = outstr;
                    }
                    Paras["SRC"][srcid]["MorModel"]["type"] = Template->Srcs_NumCon[jsrc].Mortype;
                    outstr = "[";
                    outstr += Form("%.5lf, ", Template->Srcs_NumCon[jsrc].Ra[0]);
                    outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].Ra[1]);
                    outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].Ra[2]);
                    outstr += Form("%d, ", int(Template->Srcs_NumCon[jsrc].Ra[3]));
                    outstr += Form("%.5lf", Template->Srcs_NumCon[jsrc].Ra[4]);
                    outstr += "]";
                    Paras["SRC"][srcid]["MorModel"]["ra"] = outstr;
                    outstr = "[";
                    outstr += Form("%.5lf, ", Template->Srcs_NumCon[jsrc].Dec[0]);
                    outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].Dec[1]);
                    outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].Dec[2]);
                    outstr += Form("%d, ", int(Template->Srcs_NumCon[jsrc].Dec[3]));
                    outstr += Form("%.5lf", Template->Srcs_NumCon[jsrc].Dec[4]);
                    outstr += "]";
                    Paras["SRC"][srcid]["MorModel"]["dec"] = outstr;
                    imodel = Template->Model->MorMap[Template->Srcs_NumCon[jsrc].Mortype]-1;
                    for (int ipar=0;ipar<Template->Srcs_NumCon[jsrc].nMorpar;ipar++){
                        outstr = "[";
                        outstr += Form("%.5lf, ", Template->Srcs_NumCon[jsrc].MorPar[ipar][0]);
                        outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].MorPar[ipar][1]);
                        outstr += Form("%.2lf, ", Template->Srcs_NumCon[jsrc].MorPar[ipar][2]);
                        outstr += Form("%d, ", int(Template->Srcs_NumCon[jsrc].MorPar[ipar][3]));
                        outstr += Form("%.5lf", Template->Srcs_NumCon[jsrc].MorPar[ipar][4]);
                        outstr += "]";
                        Paras["SRC"][srcid]["MorModel"][Template->Model->MorParname[imodel][ipar]] = outstr;
                    }
                }
            }

            for (int jsrc=0;jsrc<Template->NSrc_Temp;jsrc++){
                if (Template->Srcs_Temp[jsrc].SrcID == isrc){
                    string outstr;
                    Paras["SRC"][srcid]["Name"] = Template->Srcs_Temp[jsrc].Srcname;
                    Paras["SRC"][srcid]["Epiv"] = Template->Srcs_Temp[jsrc].Epiv;
                    if (Template->Srcs_Temp[jsrc].LinkPars)
                        Paras["SRC"][srcid]["LinkPars"]["SED"] = Template->Srcs_Temp[jsrc].TargetSrcID;
                    Paras["SRC"][srcid]["SEDModel"]["type"] = Template->Srcs_Temp[jsrc].SEDtype;
                    int imodel = Template->Model->SEDMap[Template->Srcs_Temp[jsrc].SEDtype]-1;
                    for (int ipar=0;ipar<Template->Srcs_Temp[jsrc].nSEDpar;ipar++){
                        outstr = "[";
                        outstr += Form("%.5lf, ", Template->Srcs_Temp[jsrc].SEDPar[ipar][0]);
                        outstr += Form("%.2lf, ", Template->Srcs_Temp[jsrc].SEDPar[ipar][1]);
                        outstr += Form("%.2lf, ", Template->Srcs_Temp[jsrc].SEDPar[ipar][2]);
                        outstr += Form("%d", int(Template->Srcs_Temp[jsrc].SEDPar[ipar][3]));
                        if (ipar==0)
                            outstr += Form(", %s", Template->Srcs_Temp[jsrc].F0_order.data());
                        outstr += Form(", %.5lf", Template->Srcs_Temp[jsrc].SEDPar[ipar][4]);
                        outstr += "]";
                        Paras["SRC"][srcid]["SEDModel"][Template->Model->SEDParname[imodel][ipar]] = outstr;
                    }
                    Paras["SRC"][srcid]["MorModel"]["type"] = Template->Srcs_Temp[jsrc].Mortype;
                    Paras["SRC"][srcid]["MorModel"]["Tempfile"] = Template->Srcs_Temp[jsrc].TempFile;
                    Paras["SRC"][srcid]["MorModel"]["TempHist"][0] = Template->Srcs_Temp[jsrc].HistName;
                }
            }
        }
    }

    out<<Paras;
    out.close();

    string command = "sed -i 's/\"//g' ";
    command += outfile;
    system(command.data());

}

void Src_FittingMode::OutConvExcess(Src_Template* Template, vector<long int> cellid, double **Wnmodel_convo, double **Knmodel_convo){

    int Neffbins = cellid.size();
    double Xmin = 360, Xmax = 0, Ymin = 90, Ymax = -90;
    map<long int, int> cellid_reverse;
    for (int ii=0;ii<Neffbins;ii++){
        double x0 = X[0]+(cellid[ii]/nbinsY)*wbinX;
        double y0 = Y[0]+(cellid[ii]%nbinsY)*wbinY;
        if (x0<Xmin)
            Xmin = x0;
        if ((x0+0.1)>Xmax)
            Xmax = x0+0.1;
        if (y0<Ymin)
            Ymin = y0;
        if ((y0+0.1)>Ymax)
            Ymax = y0+0.1;
        cellid_reverse.insert(pair<long int, int>(cellid[ii], ii+1));
    }
    int nxbins = (Xmax-Xmin+wbinX/10.)/wbinX;
    int nybins = (Ymax-Ymin+wbinY/10.)/wbinY;

    string *bintag = new string[NBinUsed[0][1]];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        bintag[ii] = "";
        if (cf.UseWCDA){
            if (ii<cf.NnhitUsed)
                bintag[ii] = Form("%d<nhit<%d", cf.Nhit[ii+cf.NhitUsed[0]], cf.Nhit[ii+1+cf.NhitUsed[0]]);
            else
                bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
        }
        else
            bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
    }
    string srcname;

    string outfile = cf.WorkDir+"/"+cf.Outdir+"/"+cf.fOut[1];
    TFile *fout = TFile::Open(outfile.data(), "recreate");
    fout->cd();
    TTree *tt = new TTree();
    tt->SetName("DataUsed");
    string DetName;
    int UsedFlag;
    int nBinUsed[2];
    tt->Branch("DetName", &DetName);
    tt->Branch("UsedFlag", &UsedFlag, "UsedFlag/I");
    tt->Branch("nBinUsed", nBinUsed, "nBinUsed[2]/I");
    DetName = "WCDA";
    UsedFlag = cf.UseWCDA;
    if (cf.UseWCDA){
        nBinUsed[0] = cf.NhitUsed[0];
        nBinUsed[1] = cf.NhitUsed[1];
    }
    else{
        nBinUsed[0] = -1;
        nBinUsed[1] = -1;
    }
    tt->Fill();
    DetName = "KM2A";
    UsedFlag = cf.UseKM2A;
    if (cf.UseKM2A){
        nBinUsed[0] = cf.KEbinUsed[0];
        nBinUsed[1] = cf.KEbinUsed[1];
    }
    else{
        nBinUsed[0] = -1;
        nBinUsed[1] = -1;
    }
    tt->Fill();
    tt->Write();

    // Excess map for each bin of each component
    TH2D *hExcess[NBinUsed[0][1]*Template->NComp];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        for (int icomp=0;icomp<Template->NComp;icomp++){

            for (int jsrc=0;jsrc<Template->NSrc;jsrc++){
                if (icomp==Template->Srcs[jsrc].SrcID) 
                    srcname = Template->Srcs[jsrc].Srcname;
            }
            for (int jsrc=0;jsrc<Template->NSrc_NumCon;jsrc++){
                if (icomp==Template->Srcs_NumCon[jsrc].SrcID) 
                    srcname = Template->Srcs_NumCon[jsrc].Srcname;
            }
            for (int jsrc=0;jsrc<Template->NSrc_Temp;jsrc++){
                if (icomp==Template->Srcs_Temp[jsrc].SrcID) 
                    srcname = Template->Srcs_Temp[jsrc].Srcname;
            }
            for (int idge=0;idge<Template->NDGE;idge++){
                if (idge==icomp-Template->NSrc_total)
                    srcname = Template->DGEs[idge].Srcname;
            }


            hExcess[ii*Template->NComp+icomp] = new TH2D(Form("hExcess_%d_%d", ii, icomp), Form("%-15s %s", srcname.data(), bintag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
        }
    }

    for (int ii=0;ii<NBinUsed[0][1];ii++){
        int icomp_real = 0;
        for (int icomp=0;icomp<Template->NComp;icomp++){

            for (int jsrc=0;jsrc<Template->NSrc;jsrc++){
                if (icomp==Template->Srcs[jsrc].SrcID)
                    icomp_real = jsrc;
            }
            for (int jsrc=0;jsrc<Template->NSrc_NumCon;jsrc++){
                if (icomp==Template->Srcs_NumCon[jsrc].SrcID)
                    icomp_real = jsrc+Template->NSrc;
            }
            for (int jsrc=0;jsrc<Template->NSrc_Temp;jsrc++){
                if (icomp==Template->Srcs_Temp[jsrc].SrcID)
                    icomp_real = jsrc+Template->NSrc+Template->NSrc_NumCon;
            }
            for (int idge=0;idge<Template->NDGE;idge++){
                if (idge==icomp-Template->NSrc_total)
                    icomp_real = icomp;
            }

            for (int kk=0;kk<Neffbins;kk++){
                double x0 = X[0]+(cellid[kk]/nbinsY+0.5)*wbinX;
                double y0 = Y[0]+(cellid[kk]%nbinsY+0.5)*wbinY;
                int ibinX = (x0-Xmin)/wbinX;
                int ibinY = (y0-Ymin)/wbinY;
                
                double nexcess = 0;
                if (cf.UseWCDA){
                    if (ii<cf.NnhitUsed)
                        nexcess = Wnmodel_convo[icomp_real][ii*Neffbins+kk];
                    else
                        nexcess = Knmodel_convo[icomp_real][(ii-cf.NnhitUsed)*Neffbins+kk];
                }
                else{
                    nexcess = Knmodel_convo[icomp_real][ii*Neffbins+kk];
                }

                hExcess[ii*Template->NComp+icomp]->SetBinContent(ibinX+1, ibinY+1, nexcess);
            }
        }
    }

    for (int ii=0;ii<NBinUsed[0][1];ii++)
        for (int jj=0;jj<Template->NComp;jj++)
            hExcess[ii*Template->NComp+jj]->Write();

    fout->Close();

}

void Src_FittingMode::OutConvExcess_ai(Src_Template* Template, vector<long int> cellid, double **Wnmodel_convo, double **Knmodel_convo, double **WNbkg, double **KNbkg, int poisson_flag, UInt_t poisson_seed){

    int Neffbins = cellid.size();
    double Xmin = 360, Xmax = 0, Ymin = 90, Ymax = -90;
    map<long int, int> cellid_reverse;
    for (int ii=0;ii<Neffbins;ii++){
        double x0 = X[0]+(cellid[ii]/nbinsY)*wbinX;
        double y0 = Y[0]+(cellid[ii]%nbinsY)*wbinY;
        if (x0<Xmin)
            Xmin = x0;
        if ((x0+0.1)>Xmax)
            Xmax = x0+0.1;
        if (y0<Ymin)
            Ymin = y0;
        if ((y0+0.1)>Ymax)
            Ymax = y0+0.1;
        cellid_reverse.insert(pair<long int, int>(cellid[ii], ii+1));
    }
    int nxbins = (Xmax-Xmin+wbinX/10.)/wbinX;
    int nybins = (Ymax-Ymin+wbinY/10.)/wbinY;

    string *bintag = new string[NBinUsed[0][1]];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        bintag[ii] = "";
        if (cf.UseWCDA){
            if (ii<cf.NnhitUsed)
                bintag[ii] = Form("%d<nhit<%d", cf.Nhit[ii+cf.NhitUsed[0]], cf.Nhit[ii+1+cf.NhitUsed[0]]);
            else
                bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii-cf.NnhitUsed+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
        }
        else
            bintag[ii] = Form("%.2lfTeV<E<%.2lfTeV", pow(10, cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep), pow(10, cf.KDataErange[0]+(ii+1+cf.KEbinUsed[0])*cf.KDataErangeStep));
    }
    string srcname;

    string outfile = cf.WorkDir+"/"+cf.Outdir+"/"+cf.fOut[1];
    TFile *fout = TFile::Open(outfile.data(), "recreate");
    fout->cd();
    TTree *tt = new TTree();
    tt->SetName("DataUsed");
    string DetName;
    int UsedFlag;
    int nBinUsed[2];
    tt->Branch("DetName", &DetName);
    tt->Branch("UsedFlag", &UsedFlag, "UsedFlag/I");
    tt->Branch("nBinUsed", nBinUsed, "nBinUsed[2]/I");
    DetName = "WCDA";
    UsedFlag = cf.UseWCDA;
    if (cf.UseWCDA){
        nBinUsed[0] = cf.NhitUsed[0];
        nBinUsed[1] = cf.NhitUsed[1];
    }
    else{
        nBinUsed[0] = -1;
        nBinUsed[1] = -1;
    }
    tt->Fill();
    DetName = "KM2A";
    UsedFlag = cf.UseKM2A;
    if (cf.UseKM2A){
        nBinUsed[0] = cf.KEbinUsed[0];
        nBinUsed[1] = cf.KEbinUsed[1];
    }
    else{
        nBinUsed[0] = -1;
        nBinUsed[1] = -1;
    }
    tt->Fill();
    tt->Write();

    // Excess map for each bin of each component
    TH2D *hExcess[NBinUsed[0][1]*Template->NComp];
    TH2D *hExcess_sum[Template->NComp];
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        for (int icomp=0;icomp<Template->NComp;icomp++){

            for (int jsrc=0;jsrc<Template->NSrc;jsrc++){
                if (icomp==Template->Srcs[jsrc].SrcID) 
                    srcname = Template->Srcs[jsrc].Srcname;
            }
            for (int jsrc=0;jsrc<Template->NSrc_NumCon;jsrc++){
                if (icomp==Template->Srcs_NumCon[jsrc].SrcID) 
                    srcname = Template->Srcs_NumCon[jsrc].Srcname;
            }
            for (int jsrc=0;jsrc<Template->NSrc_Temp;jsrc++){
                if (icomp==Template->Srcs_Temp[jsrc].SrcID) 
                    srcname = Template->Srcs_Temp[jsrc].Srcname;
            }
            for (int idge=0;idge<Template->NDGE;idge++){
                if (idge==icomp-Template->NSrc_total)
                    srcname = Template->DGEs[idge].Srcname;
            }


            hExcess[ii*Template->NComp+icomp] = new TH2D(Form("Non_exp_%d_%s", ii, srcname.data()), Form("%-15s %s", srcname.data(), bintag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);

            if (ii==0)
                hExcess_sum[icomp] = new TH2D(Form("Non_exp_%s", srcname.data()), Form("%-15s %s", srcname.data(), bintag[ii].data()), nxbins, Xmin, Xmax, nybins, Ymin, Ymax);
        }
    }
   
    TRandom *rpoisson = nullptr;
    std::mt19937 rpoisson_zero(poisson_seed);
    if (poisson_seed!=0)
        rpoisson = new TRandom(poisson_seed);
    for (int ii=0;ii<NBinUsed[0][1];ii++){
        int icomp_real = 0;
        for (int icomp=0;icomp<Template->NComp;icomp++){

            for (int jsrc=0;jsrc<Template->NSrc;jsrc++){
                if (icomp==Template->Srcs[jsrc].SrcID)
                    icomp_real = jsrc;
            }
            for (int jsrc=0;jsrc<Template->NSrc_NumCon;jsrc++){
                if (icomp==Template->Srcs_NumCon[jsrc].SrcID)
                    icomp_real = jsrc+Template->NSrc;
            }
            for (int jsrc=0;jsrc<Template->NSrc_Temp;jsrc++){
                if (icomp==Template->Srcs_Temp[jsrc].SrcID)
                    icomp_real = jsrc+Template->NSrc+Template->NSrc_NumCon;
            }
            for (int idge=0;idge<Template->NDGE;idge++){
                if (idge==icomp-Template->NSrc_total)
                    icomp_real = icomp;
            }

            for (int kk=0;kk<Neffbins;kk++){
                double x0 = X[0]+(cellid[kk]/nbinsY+0.5)*wbinX;
                double y0 = Y[0]+(cellid[kk]%nbinsY+0.5)*wbinY;
                int ibinX = (x0-Xmin)/wbinX;
                int ibinY = (y0-Ymin)/wbinY;
                
                double nexcess = 0;
                if (cf.UseWCDA){
                    if (ii<cf.NnhitUsed){
                        nexcess = Wnmodel_convo[icomp_real][ii*Neffbins+kk];
                        nexcess += WNbkg[ii][kk];
                    }
                    else{
                        nexcess = Knmodel_convo[icomp_real][(ii-cf.NnhitUsed)*Neffbins+kk];
                        nexcess += KNbkg[ii-cf.NnhitUsed][kk];
                    }
                }
                else{
                    nexcess = Knmodel_convo[icomp_real][ii*Neffbins+kk];
                    nexcess += KNbkg[ii][kk];
                }

                if (!poisson_flag)
                    hExcess[ii*Template->NComp+icomp]->SetBinContent(ibinX+1, ibinY+1, nexcess);
                else{
                    double mean = nexcess>0 ? nexcess : 0;
                    if (poisson_seed==0){
                        std::poisson_distribution<int> poisson_dist(mean);
                        hExcess[ii*Template->NComp+icomp]->SetBinContent(ibinX+1, ibinY+1, poisson_dist(rpoisson_zero)*1.);
                    }
                    else
                        hExcess[ii*Template->NComp+icomp]->SetBinContent(ibinX+1, ibinY+1, rpoisson->Poisson(mean)*1.);
                }
            }
        }
    }

    for (int jj=0;jj<Template->NComp;jj++){
        for (int ii=0;ii<NBinUsed[0][1];ii++)
            hExcess_sum[jj]->Add(hExcess[ii*Template->NComp+jj], 1);
        hExcess_sum[jj]->Write();
    }

    delete rpoisson;
    fout->Close();

}

void Src_FittingMode::GeneTSJOB(char* fityaml){

    ofstream out(cf.Outdir+"/"+cf.JOBScript, ios::out);
    out<<" #!/bin/bash"<<endl;
    out<<"export EOS_MGM_URL=root://eos01.ihep.ac.cn/"<<endl;
    out<<"procid=$1"<<endl;
    out<<"segid=$[procid]"<<endl;
    out<<"WorkDir="<<cf.WorkDir<<endl;
    out<<"exeprog=Src_TSMap"<<endl;
    out<<"FitConfig="<<fityaml<<endl;
    out<<"Outdir=$WorkDir/"<<cf.Outdir<<"/TSmap"<<endl;
    out<<"[ -d $Outdir ] || mkdir -p $Outdir"<<endl;
    out<<"$WorkDir/$exeprog $WorkDir/$FitConfig $segid $Outdir/TSmap_\"$segid\".root &> $Outdir/log_\"$segid\".txt"<<endl;

}

# endif
