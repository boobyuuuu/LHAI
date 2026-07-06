# ifndef Src_Response_WCDA_h
# define Src_Response_WCDA_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "TH1D.h"
# include "TF1.h"

using namespace std;

class Src_Response_WCDA {

    public :

        Src_Response_WCDA();
        ~Src_Response_WCDA();

        bool ReadRespFile(double Ycenter);
        void Rebin();

        string Fresponse;
        double **hResp;
        double **hPSF;
        double **PSF;

};

Src_Response_WCDA::Src_Response_WCDA(){}

Src_Response_WCDA::~Src_Response_WCDA(){

    for (int ii=0;ii<cf.NnhitUsed;ii++){
        delete[] hResp[ii];
        delete[] hPSF[ii];
    }
    delete[] hResp;
    delete[] hPSF;

}

bool Src_Response_WCDA::ReadRespFile(double Ycenter){

    hResp = new double*[cf.NnhitUsed];
    hPSF  = new double*[cf.NnhitUsed];
    PSF   = new double*[cf.NnhitUsed];
    for (int ii=0;ii<cf.NnhitUsed;ii++){
        hResp[ii] = new double[cf.NEstep*cf.Nzenstep];

        for (int jj=0;jj<cf.NEstep*cf.Nzenstep;jj++)
            hResp[ii][jj] = 0;

        if (cf.PSFtype=="2Gaus"){
            hPSF[ii]  = new double[cf.NDecstep*4];
            for (int jj=0;jj<cf.NDecstep*4;jj++)
                hPSF[ii][jj] = 0;
        }
        if (cf.PSFtype=="1Gaus"){
            hPSF[ii]  = new double[cf.NDecstep*2];
            for (int jj=0;jj<cf.NDecstep*2;jj++)
                hPSF[ii][jj] = 0;
        }

        PSF[ii]  = new double[cf.NDecstep*2];
        for (int jj=0;jj<cf.NDecstep*2;jj++)
            PSF[ii][jj] = 0;
    }

    if (Ycenter<28)
        Fresponse = cf.fResponse1;
    else
        Fresponse = cf.fResponse2;

    // check and read
    TFile *ftemp = TFile::Open(Fresponse.data());
    if (!ftemp){
        cout<<"\033[31;1mError\033[0m : can not open response file : "<<Fresponse<<"! Exited."<<endl;
        return 1;
    }
    else{
        for (int ii=0;ii<cf.NnhitUsed;ii++){

            TFile *fRestemp = (TFile *) ftemp->Get(Form("Resp_%d", ii+cf.NhitUsed[0]));
            // dectection efficiency
            TH2D *htemp = (TH2D *) fRestemp->Get(Form("hresp_%d", ii+cf.NhitUsed[0]));
            if (!htemp){
                cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hresp_%d", ii+cf.NhitUsed[0])<<" in response file : "<<Fresponse<<"! Exited."<<endl;
                ftemp->Close();
                return 1;
            }
            for (int jj=0;jj<cf.NEstep;jj++)
                for (int kk=0;kk<cf.Nzenstep;kk++){
                    if (htemp->GetBinContent(jj+1, kk+1)<1.e-12)
                        hResp[ii][jj*cf.Nzenstep+kk] = 0; //htemp->GetBinContent(jj+1, kk+1);
                    else
                        hResp[ii][jj*cf.Nzenstep+kk] = htemp->GetBinContent(jj+1, kk+1);
                }

            // PSF
            for (int jj=0;jj<cf.NDecstep;jj++){
                TH1D *hpsftemp = (TH1D *) fRestemp->Get(Form("hPSF_%d_%d", ii+cf.NhitUsed[0], jj));
                if (!hpsftemp){
                    cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hPSF_%d_%d", ii+cf.NhitUsed[0], jj)<<" in response file : "<<Fresponse<<"! Exited."<<endl;
                    ftemp->Close();
                    return 1;
                }
                if (hpsftemp->GetSumOfWeights()<=0){ 
                    //cout<<"\033[31;1mWarning\033[0m : "<<Form("hPSF_%d_%d", ii+cf.NhitUsed[0], jj)<<" in response file : "<<Fresponse<<" is empty! Exited."<<endl;
                    continue;
                }
                TF1 *ftemp = (TF1 *) hpsftemp->GetFunction("fgaus");
                string formula = Form("%s*sin(x/180*TMath::Pi())", ftemp->GetExpFormula().Data());
                TF1 *fint_temp = new TF1("fint_temp", formula.data(), 0, 10);
                if (!ftemp) continue;
                if (cf.PSFtype=="1Gaus"){
                    hPSF[ii][jj*2] = ftemp->GetParameter(0);
                    hPSF[ii][jj*2+1] = ftemp->GetParameter(1);
                    fint_temp->SetParameters(hPSF[ii]+jj*2);
                }
                if (cf.PSFtype=="2Gaus"){
                    hPSF[ii][jj*4] = ftemp->GetParameter(0);
                    hPSF[ii][jj*4+1] = ftemp->GetParameter(1);
                    hPSF[ii][jj*4+2] = ftemp->GetParameter(2);
                    hPSF[ii][jj*4+3] = ftemp->GetParameter(3);
                    fint_temp->SetParameters(hPSF[ii]+jj*4);
                }
                // GetPSF
                const int num = 2;
                double prob[num] = {0.39, 0.683};
                double angle[num] = {0., 0.0};
                fint_temp->GetQuantiles(num, angle, prob);
                PSF[ii][jj*2] = angle[0];
                PSF[ii][jj*2+1] = angle[1];
            }
        }
        ftemp->Close();
    }

    return 0;

}

void Src_Response_WCDA::Rebin(){

    return;

} 

# endif
