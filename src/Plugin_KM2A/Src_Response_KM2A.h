# ifndef Src_Response_KM2A_h
# define Src_Response_KM2A_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "TH1D.h"
# include "TF1.h"

using namespace std;

class Src_Response_KM2A {

    public :

        Src_Response_KM2A();
        ~Src_Response_KM2A();

        bool ReadRespFile();

        string KFresponse;
        double **hResp;
        double **hPSF;
        double **PSF;
        double **hResp_NotFull[2];
        double **hPSF_NotFull[2];
        double **PSF_NotFull[2];

};

Src_Response_KM2A::Src_Response_KM2A(){}

Src_Response_KM2A::~Src_Response_KM2A(){

    for (int ii=0;ii<cf.KNEbinUsed;ii++){
        delete[] hResp[ii];
        delete[] hPSF[ii];
        delete[] PSF[ii];
    }
    delete[] hResp;
    delete[] hPSF;
    delete[] PSF;

    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                delete[] hResp_NotFull[idet][ii];
                delete[] hPSF_NotFull[idet][ii];
                delete[] PSF_NotFull[idet][ii];
            }
            delete[] hResp_NotFull[idet];
            delete[] hPSF_NotFull[idet];
            delete[] PSF_NotFull[idet];
        }
    }

}

bool Src_Response_KM2A::ReadRespFile(){

    double PSF_offset = 0.0;

    hResp = new double*[cf.KNEbinUsed];
    hPSF  = new double*[cf.KNEbinUsed];
    PSF   = new double*[cf.KNEbinUsed];
    for (int ii=0;ii<cf.KNEbinUsed;ii++){
        hResp[ii] = new double[cf.KNEstep*cf.KNzenstep];

        for (int jj=0;jj<cf.KNEstep*cf.KNzenstep;jj++)
            hResp[ii][jj] = 0;

        if (cf.KPSFtype=="2Gaus"){
            hPSF[ii]  = new double[cf.NDecstep*4];
            for (int jj=0;jj<cf.NDecstep*4;jj++)
                hPSF[ii][jj] = 0;
        }
        if (cf.KPSFtype=="1Gaus"){
            hPSF[ii]  = new double[cf.NDecstep*2];
            for (int jj=0;jj<cf.NDecstep*2;jj++)
                hPSF[ii][jj] = 0;
        }

        PSF[ii]  = new double[cf.NDecstep*2];
        for (int jj=0;jj<cf.NDecstep*2;jj++)
            PSF[ii][jj] = 0;

    }

    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            hResp_NotFull[idet] = new double*[cf.KNEbinUsed];
            hPSF_NotFull[idet]  = new double*[cf.KNEbinUsed];
            PSF_NotFull[idet]   = new double*[cf.KNEbinUsed];
            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                hResp_NotFull[idet][ii] = new double[cf.KNEstep*cf.KNzenstep];

                for (int jj=0;jj<cf.KNEstep*cf.KNzenstep;jj++)
                    hResp_NotFull[idet][ii][jj] = 0;

                if (cf.KPSFtype=="2Gaus"){
                    hPSF_NotFull[idet][ii]  = new double[cf.NDecstep*4];
                    for (int jj=0;jj<cf.NDecstep*4;jj++)
                        hPSF_NotFull[idet][ii][jj] = 0;
                }
                if (cf.KPSFtype=="1Gaus"){
                    hPSF_NotFull[idet][ii]  = new double[cf.NDecstep*2];
                    for (int jj=0;jj<cf.NDecstep*2;jj++)
                        hPSF_NotFull[idet][ii][jj] = 0;
                }

                PSF_NotFull[idet][ii]  = new double[cf.NDecstep*2];
                for (int jj=0;jj<cf.NDecstep*2;jj++)
                    PSF_NotFull[idet][ii][jj] = 0;

            }
        }
    }

    KFresponse = cf.fKResponse;

    // check and read
    TFile *ftemp = TFile::Open(KFresponse.data());
    if (!ftemp){
        cout<<"\033[31;1mError\033[0m : can not open response file : "<<KFresponse<<"! Exited."<<endl;
        return 1;
    }
    else{
        for (int ii=0;ii<cf.KNEbinUsed;ii++){

            TFile *fRestemp = (TFile *) ftemp->Get(Form("Resp_%d", ii+cf.KEbinUsed[0]));
            // dectection efficiency
            TH2D *htemp = (TH2D *) fRestemp->Get(Form("hresp_%d", ii+cf.KEbinUsed[0]));
            if (!htemp){
                cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hresp_%d", ii+cf.KEbinUsed[0])<<" in response file : "<<KFresponse<<"! Exited."<<endl;
                ftemp->Close();
                return 1;
            }
            for (int jj=0;jj<cf.KNEstep;jj++)
                for (int kk=0;kk<cf.KNzenstep;kk++)
                    hResp[ii][jj*cf.KNzenstep+kk] = htemp->GetBinContent(jj+1, kk+1);

            // PSF
            for (int jj=0;jj<cf.NDecstep;jj++){
                TH1D *hpsftemp = (TH1D *) fRestemp->Get(Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj));
                if (!hpsftemp){
                    cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj)<<" in response file : "<<KFresponse<<"! Exited."<<endl;
                    ftemp->Close();
                    return 1;
                }
                if (hpsftemp->GetSumOfWeights()<=0){
                    //cout<<"\033[31;1mWarning\033[0m : "<<Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj)<<" in response file : "<<KFresponse<<" is empty! Exited."<<endl;
                    continue;
                }
                TF1 *ftemp = (TF1 *) hpsftemp->GetFunction("fgaus");
                string formula = Form("%s*sin(x/180*TMath::Pi())", ftemp->GetExpFormula().Data());
                TF1 *fint_temp = new TF1("fint_temp", formula.data(), 0, 10);
                if (!ftemp) continue;
                if (cf.KPSFtype=="1Gaus"){
                    hPSF[ii][jj*2] = ftemp->GetParameter(0);
                    hPSF[ii][jj*2+1] = ftemp->GetParameter(1)+PSF_offset;
                    fint_temp->SetParameters(hPSF[ii]+jj*2);
                }
                if (cf.KPSFtype=="2Gaus"){
                    hPSF[ii][jj*4] = ftemp->GetParameter(0);
                    hPSF[ii][jj*4+1] = ftemp->GetParameter(1);
                    hPSF[ii][jj*4+2] = ftemp->GetParameter(2);
                    hPSF[ii][jj*4+3] = ftemp->GetParameter(3);
                    if (hPSF[ii][jj*4+2]<hPSF[ii][jj*4+3]){
                        hPSF[ii][jj*4+3] += PSF_offset/hPSF[ii][jj*4+2]*hPSF[ii][jj*4+3];
                        hPSF[ii][jj*4+2] += PSF_offset;
                    }
                    else{
                        hPSF[ii][jj*4+2] += PSF_offset/hPSF[ii][jj*4+3]*hPSF[ii][jj*4+2];
                        hPSF[ii][jj*4+3] += PSF_offset;
                    }
                    fint_temp->SetParameters(hPSF[ii]+jj*4);
                }
                // GetPSF
                const int num = 2;
                double prob[num] = {0.39, 0.683};
                double angle[num] = {0., 0.};
                fint_temp->GetQuantiles(num, angle, prob);
                PSF[ii][jj*2] = angle[0];
                PSF[ii][jj*2+1] = angle[1];
            }
        }

        string dettag_temp[2] = {"34", "12"};
        if (cf.UseKM2A_NotFull){
            
            for (int idet=0;idet<2;idet++){
                for (int ii=0;ii<cf.KNEbinUsed;ii++){

                    TFile *fRestemp = (TFile *) ftemp->Get(Form("Resp_%d_%s", ii+cf.KEbinUsed[0], dettag_temp[idet].data()));
                    //TFile *fRestemp = (TFile *) ftemp->Get(Form("Resp_%d", ii+cf.KEbinUsed[0]));
                    // dectection efficiency
                    TH2D *htemp = (TH2D *) fRestemp->Get(Form("hresp_%d", ii+cf.KEbinUsed[0]));
                    if (!htemp){
                        cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hresp_%d", ii+cf.KEbinUsed[0])<<" in response file : "<<KFresponse<<"! Exited."<<endl;
                        ftemp->Close();
                        return 1;
                    }
                    for (int jj=0;jj<cf.KNEstep;jj++)
                        for (int kk=0;kk<cf.KNzenstep;kk++)
                            hResp_NotFull[idet][ii][jj*cf.KNzenstep+kk] = htemp->GetBinContent(jj+1, kk+1);

                    // PSF
                    for (int jj=0;jj<cf.NDecstep;jj++){
                        TH1D *hpsftemp = (TH1D *) fRestemp->Get(Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj));
                        if (!hpsftemp){
                            cout<<"\033[31;1mError\033[0m :  there is no "<<Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj)<<" in response file : "<<KFresponse<<"! Exited."<<endl;
                            ftemp->Close();
                            return 1;
                        }
                        if (hpsftemp->GetSumOfWeights()<=0){
                            //cout<<"\033[31;1mWarning\033[0m : "<<Form("hPSF_%d_%d", ii+cf.KEbinUsed[0], jj)<<" in response file : "<<KFresponse<<" is empty! Exited."<<endl;
                            continue;
                        }
                        TF1 *ftemp = (TF1 *) hpsftemp->GetFunction("fgaus");
                        string formula = Form("%s*sin(x/180*TMath::Pi())", ftemp->GetExpFormula().Data());
                        TF1 *fint_temp = new TF1("fint_temp", formula.data(), 0, 10);
                        if (!ftemp) continue;
                        if (cf.KPSFtype=="1Gaus"){
                            hPSF_NotFull[idet][ii][jj*2] = ftemp->GetParameter(0);
                            hPSF_NotFull[idet][ii][jj*2+1] = ftemp->GetParameter(1)+PSF_offset;
                            fint_temp->SetParameters(hPSF_NotFull[idet][ii]+jj*2);
                        }
                        if (cf.KPSFtype=="2Gaus"){
                            hPSF_NotFull[idet][ii][jj*4] = ftemp->GetParameter(0);
                            hPSF_NotFull[idet][ii][jj*4+1] = ftemp->GetParameter(1);
                            hPSF_NotFull[idet][ii][jj*4+2] = ftemp->GetParameter(2);
                            hPSF_NotFull[idet][ii][jj*4+3] = ftemp->GetParameter(3);
                            if (hPSF_NotFull[idet][ii][jj*4+2]<hPSF_NotFull[idet][ii][jj*4+3]){
                                hPSF_NotFull[idet][ii][jj*4+3] += PSF_offset/hPSF_NotFull[idet][ii][jj*4+2]*hPSF_NotFull[idet][ii][jj*4+3];
                                hPSF_NotFull[idet][ii][jj*4+2] += PSF_offset;
                            }
                            else{
                                hPSF_NotFull[idet][ii][jj*4+2] += PSF_offset/hPSF_NotFull[idet][ii][jj*4+3]*hPSF_NotFull[idet][ii][jj*4+2];
                                hPSF_NotFull[idet][ii][jj*4+3] += PSF_offset;
                            }
                            fint_temp->SetParameters(hPSF_NotFull[idet][ii]+jj*4);
                        }
                        // GetPSF
                        const int num = 2;
                        double prob[num] = {0.39, 0.683};
                        double angle[num] = {0., 0.};
                        fint_temp->GetQuantiles(num, angle, prob);
                        PSF_NotFull[idet][ii][jj*2] = angle[0];
                        PSF_NotFull[idet][ii][jj*2+1] = angle[1];
                    }
                }
            }

        } 

        ftemp->Close();
    }

    return 0;

}

# endif
