# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <iostream>
# include <fstream>

# include <TFile.h>
# include <TStyle.h>
# include <TBranch.h>
# include <TTree.h>
# include <TLegend.h>
# include <TGraph.h>
# include <TLatex.h>
# include <TLine.h>
# include <TLegend.h>
# include <TH1D.h>
# include <TMath.h>
# include <TH2D.h>
# include <TF1.h>
# include <TF2.h>
# include <TMinuit.h>
# include <TCanvas.h>
# include <TGraphErrors.h>

# include "src/basic/papi.h"
# include "src/basic/astro.h"
# include "/home/lhaaso/hushicong/MyEnv/YAML_CPP/include/yaml-cpp/yaml.h"

# include "src/Src_ROI.h"
# include "src/Src_Template.h"
# include "src/Plugin_WCDA/Src_Fitting_WCDA.h"
# include "src/Plugin_KM2A/Src_Fitting_KM2A.h"
# include "src/Src_FittingMode.h"
# include "src/Src_FittingResults.h"

Src_Fitting_WCDA *WCDAFit = new Src_Fitting_WCDA();
Src_Fitting_KM2A *KM2AFit = new Src_Fitting_KM2A();
int iBinUsed0 = 0, iBinUsed1 = 0, iThisComp = 0, iThisPmode = 0;

void FCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag){

    if (cf.UseWCDA)
        WCDAFit->CalLogSig(par, npar_src, npar_numsrc, npar_dge, iBinUsed0, iBinUsed1, iThisComp, iThisPmode);
    if (cf.UseKM2A)
        KM2AFit->CalLogSig(par, npar_src, npar_numsrc, npar_dge, iBinUsed0, iBinUsed1, iThisComp, iThisPmode);

    f = 0;
    double TS = 0;
    if (cf.UseWCDA){
        f  += -WCDAFit->log_L_sig;
        TS += 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null);
        cout<<"TS_WCDA = "<<Form("%6.2lf", 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null))<<", ";
    }
    if (cf.UseKM2A){
        f  += -KM2AFit->log_L_sig;
        TS += 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null);
        cout<<"TS_KM2A = "<<Form("%6.2lf", 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null))<<", ";
    }

    cout<<"TS = "<<Form("%6.2lf", TS)<<endl;

}

void Fitting(int ipmode, int ismode, Src_FittingMode *fitmode, int icomp){

    cout<<" ****** Fitting begins ******"<<endl;
    // calculate log_L_null
    if (cf.UseWCDA)
        WCDAFit->CalLogNull(iBinUsed0, iBinUsed1);
    // calculate KM2A log_L_null
    if (cf.UseKM2A)
        KM2AFit->CalLogNull(iBinUsed0, iBinUsed1);

    cout<<" *** Fitting : npar_src = "<<npar_src<<", npar_numsrc = "<<npar_numsrc<<", npar_dge = "<<npar_dge<<", npar_total = "<<npar_total<<endl;
    cout<<" *** Fitting : npar_total of this mode = "<<fitmode->NPar_total[ipmode]<<endl;
    // minimize based TMiniut
    TMinuit *gMinuit = new TMinuit(fitmode->NPar_total[ipmode]);
    gMinuit->SetFCN(FCN);

    Double_t arglist[10];
    Int_t ierflg = 0;
    arglist[0] = fitmode->ConfLevel[ipmode];
    gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);

    int npar = 0;
    int imodel = -1;
    // Srcs
    for (int isrc=0;isrc<WCDAFit->Template->NSrc;isrc++){

        if (isrc==iThisComp) continue;

        // Position
        gMinuit->mnparm(npar  , Form("%s_X", WCDAFit->Template->Srcs[isrc].Srcname.data()), WCDAFit->Template->Srcs[isrc].Ra[0], 0.01, WCDAFit->Template->Srcs[isrc].Ra[1], WCDAFit->Template->Srcs[isrc].Ra[2], ierflg);
        if (WCDAFit->Template->Srcs[isrc].Ra[3] || fitmode->ParStatus[ismode][npar])
            gMinuit->FixParameter(npar);
        gMinuit->mnparm(npar+1, Form("%s_Y", WCDAFit->Template->Srcs[isrc].Srcname.data()), WCDAFit->Template->Srcs[isrc].Dec[0], 0.01, WCDAFit->Template->Srcs[isrc].Dec[1], WCDAFit->Template->Srcs[isrc].Dec[2], ierflg);
        if (WCDAFit->Template->Srcs[isrc].Dec[3] || fitmode->ParStatus[ismode][npar+1])
            gMinuit->FixParameter(npar+1);

        // SED
        imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs[isrc].SEDtype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs[isrc].nSEDpar;ipar++){
            gMinuit->mnparm(npar+2+ipar, Form("%s_%s", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data()), WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0], 0.001, WCDAFit->Template->Srcs[isrc].SEDPar[ipar][1], WCDAFit->Template->Srcs[isrc].SEDPar[ipar][2], ierflg);
            if (WCDAFit->Template->Srcs[isrc].SEDPar[ipar][3] || fitmode->ParStatus[ismode][npar+2+ipar])
                gMinuit->FixParameter(npar+2+ipar);
        }
        npar += 2+WCDAFit->Template->Srcs[isrc].nSEDpar;

        // Morphology
        imodel = WCDAFit->Template->Model->MorMap[WCDAFit->Template->Srcs[isrc].Mortype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs[isrc].nMorpar;ipar++){
            gMinuit->mnparm(npar+ipar, Form("%s_%s", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->MorParname[imodel][ipar].data()), WCDAFit->Template->Srcs[isrc].MorPar[ipar][0], 0.01, WCDAFit->Template->Srcs[isrc].MorPar[ipar][1], WCDAFit->Template->Srcs[isrc].MorPar[ipar][2], ierflg);
            if (WCDAFit->Template->Srcs[isrc].MorPar[ipar][3] || fitmode->ParStatus[ismode][npar+ipar])
                gMinuit->FixParameter(npar+ipar);
        }
        npar += WCDAFit->Template->Srcs[isrc].nMorpar;
    }   

    // Srcs_NumCon
    for (int isrc=0;isrc<WCDAFit->Template->NSrc_NumCon;isrc++){

        if (isrc==(iThisComp-WCDAFit->Template->NSrc)) continue;

        // Position
        gMinuit->mnparm(npar  , Form("%s_X", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data()), WCDAFit->Template->Srcs_NumCon[isrc].Ra[0], 0.01, WCDAFit->Template->Srcs_NumCon[isrc].Ra[1], WCDAFit->Template->Srcs_NumCon[isrc].Ra[2], ierflg);
        if (WCDAFit->Template->Srcs_NumCon[isrc].Ra[3] || fitmode->ParStatus[ismode][npar])
            gMinuit->FixParameter(npar);
        gMinuit->mnparm(npar+1, Form("%s_Y", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data()), WCDAFit->Template->Srcs_NumCon[isrc].Dec[0], 0.01, WCDAFit->Template->Srcs_NumCon[isrc].Dec[1], WCDAFit->Template->Srcs_NumCon[isrc].Dec[2], ierflg);
        if (WCDAFit->Template->Srcs_NumCon[isrc].Dec[3] || fitmode->ParStatus[ismode][npar+1])
            gMinuit->FixParameter(npar+1);

        // SED
        imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs_NumCon[isrc].SEDtype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            gMinuit->mnparm(npar+2+ipar, Form("%s_%s", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data()), WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0], 0.01, WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][1], WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][2], ierflg);
            if (WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][3] || fitmode->ParStatus[ismode][npar+2+ipar])
                gMinuit->FixParameter(npar+2+ipar);
        }
        npar += 2+WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;

        // Morphology
        imodel = WCDAFit->Template->Model->MorMap[WCDAFit->Template->Srcs_NumCon[isrc].Mortype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;ipar++){
            gMinuit->mnparm(npar+ipar, Form("%s_%s", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->MorParname[imodel][ipar].data()), WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][0], 0.01, WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][1], WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][2], ierflg);
            if (WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][3] || fitmode->ParStatus[ismode][npar+ipar])
                gMinuit->FixParameter(npar+ipar);
        }
        npar += WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;
    }   

    // Src_Temp && DGEs
    for (int isrc=0;isrc<WCDAFit->Template->NTemp;isrc++){

        if (isrc==(iThisComp-WCDAFit->Template->NSrc-WCDAFit->Template->NSrc_NumCon)) continue;

        // SED
        if (isrc<WCDAFit->Template->NSrc_Temp){
            imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs_Temp[isrc].SEDtype]-1;
            for (int ipar=0;ipar<WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                gMinuit->mnparm(npar+ipar, Form("%s_%s", WCDAFit->Template->Srcs_Temp[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data()), WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0], 0.01, WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][1], WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][2], ierflg);
                if (WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][3] || fitmode->ParStatus[ismode][npar+ipar])
                    gMinuit->FixParameter(npar+ipar);
            }
            npar += WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDtype]-1;
            for (int ipar=0;ipar<WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;ipar++){
                gMinuit->mnparm(npar+ipar, Form("%s_%s", WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data()), WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][0], 0.01, WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][1], WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][2], ierflg);
                if (WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][3] || fitmode->ParStatus[ismode][npar+ipar])
                    gMinuit->FixParameter(npar+ipar);
            }
            npar += WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;
        }

    }   

    arglist[0] = 10000;
    arglist[1] = 0.1;
    gMinuit->mnexcm(fitmode->MinAlgo[ipmode].data(), arglist, 2, ierflg);

    double Chi2 = gMinuit->fAmin;
    double *Par    = new double [fitmode->NPar_total[ipmode]];
    double *Parerr = new double [fitmode->NPar_total[ipmode]];
    for (int ipar=0;ipar<fitmode->NPar_total[ipmode];ipar++){
        Par[ipar] = 0;
        Parerr[ipar] = 0;
    }   
    //gMinuit->SetPrintLevel(0);
    cout<<" *** Fitting : final fitting results : "<<endl;
    cout<<"     Chi2   = "<<Chi2<<endl;
    npar = 0;
    // Srcs
    double ra, dec;
    double erL, erU, erPA, gcc;
    for (int isrc=0;isrc<WCDAFit->Template->NSrc;isrc++){

        if (isrc==iThisComp) continue;

        // Position
        gMinuit->GetParameter(npar,   Par[npar], Parerr[npar]);
        gMinuit->GetParameter(npar+1, Par[npar+1], Parerr[npar+1]);
        if (!cf.CorOpt){
            cout<<Form("%s_ra     = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", Par[npar])<<" +/- "<<Form("%.5lf", Parerr[npar])<<endl;
            cout<<Form("%s_dec    = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", Par[npar+1])<<" +/- "<<Form("%.5lf", Parerr[npar+1])<<endl;
        }
        else{
            cout<<Form("%s_l      = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", Par[npar])<<" +/- "<<Form("%.5lf", Parerr[npar])<<endl;
            cout<<Form("%s_b      = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", Par[npar+1])<<" +/- "<<Form("%.5lf", Parerr[npar+1])<<endl;
            g2e(Par[npar], Par[npar+1], &ra, &dec);
            cout<<Form("%s_ra     = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", ra)<<endl;
            cout<<Form("%s_dec    = ", WCDAFit->Template->Srcs[isrc].Srcname.data())<<Form("%.5lf", dec)<<endl;
        }
        if (ipmode==0){
            WCDAFit->Template->Srcs[isrc].Ra[0] = Par[npar];
            WCDAFit->Template->Srcs[isrc].Ra[4] = Parerr[npar];
            WCDAFit->Template->Srcs[isrc].Dec[0] = Par[npar+1];
            WCDAFit->Template->Srcs[isrc].Dec[4] = Parerr[npar+1];
            fitmode->ParVal[npar] = Par[npar];
            fitmode->ParErr[npar] = Parerr[npar];
            fitmode->ParVal[npar+1] = Par[npar+1];
            fitmode->ParErr[npar+1] = Parerr[npar+1];
        }

        // SED
        imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs[isrc].SEDtype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs[isrc].nSEDpar;ipar++){
            gMinuit->GetParameter(npar+2+ipar, Par[npar+2+ipar], Parerr[npar+2+ipar]);
            cout<<Form("%s_%s     = ", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+2+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+2+ipar])<<endl;
            // Flux UL
            if (ipmode==4 && ipar==0){
                gMinuit->mnerrs(npar+2+ipar, erU, erL, erPA, gcc);
                cout<<Form("%s_%s     = ", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+2+ipar])<<" + "<<Form("%.5lf", erU)<<" - "<<Form("%.5lf", erL)<<endl;
            }
            if (ipmode==0){
                WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0] = Par[npar+2+ipar];
                WCDAFit->Template->Srcs[isrc].SEDPar[ipar][4] = Parerr[npar+2+ipar];
                fitmode->ParVal[npar+2+ipar] = Par[npar+2+ipar];
                fitmode->ParErr[npar+2+ipar] = Parerr[npar+2+ipar];
            }
        }
        if (ipmode==1){
            fitmode->FNorm[isrc][iBinUsed0] = Par[npar+2];
            fitmode->FNormErr[isrc][iBinUsed0] = Parerr[npar+2];
        }
        if (ipmode==4){
            fitmode->FNorm[isrc][iBinUsed0] = Par[npar+2];
            fitmode->FNormErr[isrc][iBinUsed0] = Parerr[npar+2];
            fitmode->FNormUL[isrc][iBinUsed0] = Par[npar+2]+erU;
        }

        npar += 2+WCDAFit->Template->Srcs[isrc].nSEDpar;

        // Morphology
        imodel = WCDAFit->Template->Model->MorMap[WCDAFit->Template->Srcs[isrc].Mortype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs[isrc].nMorpar;ipar++){
            gMinuit->GetParameter(npar+ipar, Par[npar+ipar], Parerr[npar+ipar]);
            cout<<Form("%s_%s    = ", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->MorParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+ipar])<<endl;
            if (ipmode==0){
                WCDAFit->Template->Srcs[isrc].MorPar[ipar][0] = Par[npar+ipar];
                WCDAFit->Template->Srcs[isrc].MorPar[ipar][4] = Parerr[npar+ipar];
                fitmode->ParVal[npar+ipar] = Par[npar+ipar];
                fitmode->ParErr[npar+ipar] = Parerr[npar+ipar];
            }
        }
        npar += WCDAFit->Template->Srcs[isrc].nMorpar;

    }  

    // NumSrcs
    for (int isrc=0;isrc<WCDAFit->Template->NSrc_NumCon;isrc++){

        if (isrc==(iThisComp-WCDAFit->Template->NSrc)) continue;

        // Position
        gMinuit->GetParameter(npar,   Par[npar], Parerr[npar]);
        gMinuit->GetParameter(npar+1, Par[npar+1], Parerr[npar+1]);
        if (!cf.CorOpt){
            cout<<Form("%s_ra     = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", Par[npar])<<" +/- "<<Form("%.5lf", Parerr[npar])<<endl;
            cout<<Form("%s_dec    = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", Par[npar+1])<<" +/- "<<Form("%.5lf", Parerr[npar+1])<<endl;
        }
        else{
            cout<<Form("%s_l      = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", Par[npar])<<" +/- "<<Form("%.5lf", Parerr[npar])<<endl;
            cout<<Form("%s_b      = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", Par[npar+1])<<" +/- "<<Form("%.5lf", Parerr[npar+1])<<endl;
            g2e(Par[npar], Par[npar+1], &ra, &dec);
            cout<<Form("%s_ra     = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", ra)<<endl;
            cout<<Form("%s_dec    = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data())<<Form("%.5lf", dec)<<endl;
        }
        if (ipmode==0){
            WCDAFit->Template->Srcs_NumCon[isrc].Ra[0] = Par[npar];
            WCDAFit->Template->Srcs_NumCon[isrc].Ra[4] = Parerr[npar];
            WCDAFit->Template->Srcs_NumCon[isrc].Dec[0] = Par[npar+1];
            WCDAFit->Template->Srcs_NumCon[isrc].Dec[4] = Parerr[npar+1];
            fitmode->ParVal[npar] = Par[npar];
            fitmode->ParErr[npar] = Parerr[npar];
            fitmode->ParVal[npar+1] = Par[npar+1];
            fitmode->ParErr[npar+1] = Parerr[npar+1];
        }

        // SED
        imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs_NumCon[isrc].SEDtype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
            gMinuit->GetParameter(npar+2+ipar, Par[npar+2+ipar], Parerr[npar+2+ipar]);
            cout<<Form("%s_%s     = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+2+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+2+ipar])<<endl;
            // Flux UL
            if (ipmode==4 && ipar==0){
                gMinuit->mnerrs(npar+2+ipar, erU, erL, erPA, gcc);
                cout<<Form("%s_%s     = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+2+ipar])<<" + "<<Form("%.5lf", erU)<<" - "<<Form("%.5lf", erL)<<endl;
            }
            if (ipmode==0){
                WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = Par[npar+2+ipar];
                WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][4] = Parerr[npar+2+ipar];
                fitmode->ParVal[npar+2+ipar] = Par[npar+2+ipar];
                fitmode->ParErr[npar+2+ipar] = Parerr[npar+2+ipar];
            }
        }
        if (ipmode==1){
            fitmode->FNorm[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Par[npar+2];
            fitmode->FNormErr[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Parerr[npar+2];
        }
        if (ipmode==4){
            fitmode->FNorm[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Par[npar+2];
            fitmode->FNormErr[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Parerr[npar+2];
            fitmode->FNormUL[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Par[npar+2]+erU;
        }
        npar += 2+WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;

        // Morphology
        imodel = WCDAFit->Template->Model->MorMap[WCDAFit->Template->Srcs_NumCon[isrc].Mortype]-1;
        for (int ipar=0;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;ipar++){
            gMinuit->GetParameter(npar+ipar, Par[npar+ipar], Parerr[npar+ipar]);
            cout<<Form("%s_%s    = ", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->MorParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+ipar])<<endl;
            if (ipmode==0){
                WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][0] = Par[npar+ipar];
                WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][4] = Parerr[npar+ipar];
                fitmode->ParVal[npar+ipar] = Par[npar+ipar];
                fitmode->ParErr[npar+ipar] = Parerr[npar+ipar];
            }
        }
        npar += WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;

    }  

    // Src_Temp && DGEs
    for (int isrc=0;isrc<WCDAFit->Template->NTemp;isrc++){

        if (isrc==(iThisComp-WCDAFit->Template->NSrc-WCDAFit->Template->NSrc_NumCon)) continue;

        if (isrc<WCDAFit->Template->NSrc_Temp){
            // SED
            imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->Srcs_Temp[isrc].SEDtype]-1;
            for (int ipar=0;ipar<WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                gMinuit->GetParameter(npar+ipar, Par[npar+ipar], Parerr[npar+ipar]);
                cout<<Form("%s_%s    = ", WCDAFit->Template->Srcs_Temp[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+ipar])<<endl;
                // Flux UL
                if (ipmode==4 && ipar==0){
                    gMinuit->mnerrs(npar+ipar, erU, erL, erPA, gcc);
                    cout<<Form("%s_%s     = ", WCDAFit->Template->Srcs_Temp[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" + "<<Form("%.5lf", erU)<<" - "<<Form("%.5lf", erL)<<endl;
                }

                if (ipmode==0){
                    WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0] = Par[npar+ipar];
                    WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][4] = Parerr[npar+ipar];
                    fitmode->ParVal[npar+ipar] = Par[npar+ipar];
                    fitmode->ParErr[npar+ipar] = Parerr[npar+ipar];
                }
            }
            if (ipmode==1){
                fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
            }
            if (ipmode==4){
                fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
                fitmode->FNormUL[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar]+erU;
            }
            npar += WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            // SED
            imodel = WCDAFit->Template->Model->SEDMap[WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDtype]-1;
            for (int ipar=0;ipar<WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;ipar++){
                gMinuit->GetParameter(npar+ipar, Par[npar+ipar], Parerr[npar+ipar]);
                cout<<Form("%s_%s    = ", WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" +/- "<<Form("%.5lf", Parerr[npar+ipar])<<endl;
                // Flux UL
                if (ipmode==4 && ipar==0){
                    gMinuit->mnerrs(npar+ipar, erU, erL, erPA, gcc);
                    cout<<Form("%s_%s     = ", WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data())<<Form("%.5lf", Par[npar+ipar])<<" + "<<Form("%.5lf", erU)<<" - "<<Form("%.5lf", erL)<<endl;
                }

                if (ipmode==0){
                    WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][0] = Par[npar+ipar];
                    WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][4] = Parerr[npar+ipar];
                    fitmode->ParVal[npar+ipar] = Par[npar+ipar];
                    fitmode->ParErr[npar+ipar] = Parerr[npar+ipar];
                }
            }
            if (ipmode==1){
                fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
            }
            if (ipmode==4){
                fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
                fitmode->FNormUL[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar]+erU;
            }
            npar += WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;
        }

    }   

    // Get CORRELATION Contour
    /*int ke1 = 2, ke2 = 4, nptu = 30, ierrf;
    double xptu[30],  yptu[30];
    gMinuit->mncont(ke1, ke2, nptu, xptu, yptu, ierrf);
    for (int ii=0;ii<30;ii++)
        cout<<xptu[ii]<<" "<<yptu[ii]<<endl;*/

    // Get Fitting Status and TS
    double fmin, fedm, errdef;
    int npari, nparx, istat;
    gMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

    double TS_total = 0, TS_total_W = 0, TS_total_K = 0;
    if (cf.UseWCDA){
        TS_total_W = 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null);
        cout<<"TS_total_W = "<<Form("%.2lf", TS_total_W)<<endl;
        TS_total += TS_total_W;
    }
    if (cf.UseKM2A){
        TS_total_K = 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null);
        cout<<"TS_total_K = "<<Form("%.2lf", TS_total_K)<<endl;
        TS_total += TS_total_K;
    }
    cout<<"TS_total = "<<Form("%.2lf", TS_total)<<endl;

    if (ipmode==0){
        fitmode->TS_WCDA = TS_total_W;
        fitmode->TS_KM2A = TS_total_K;
        fitmode->TS_Total = TS_total;
        fitmode->FITSTATUS = istat;
    }
    if (ipmode==1){
        for (int ii=0;ii<WCDAFit->Template->NComp;ii++)
            fitmode->FitStatus_Bin[ii][iBinUsed0] = istat;
    }
    if (ipmode==2){
        int iSmode = ismode-fitmode->NSmode_int[ipmode-1];
        if (iSmode==0){
            for (int ii=0;ii<WCDAFit->Template->NComp;ii++){
                fitmode->TS_Src[ii][0+iSmode*2] = TS_total_W;
                fitmode->TS_Src[ii][1+iSmode*2] = TS_total_K;
                fitmode->FitStatus_Src[ii][iSmode] = istat;
            }
        }
        else{
            fitmode->TS_Src[icomp][0+iSmode*2] = TS_total_W;
            fitmode->TS_Src[icomp][1+iSmode*2] = TS_total_K;
            fitmode->FitStatus_Src[icomp][iSmode] = istat;
        }
    }
    if (ipmode==3){
        int iSmode = ismode-fitmode->NSmode_int[ipmode-1];
        if (iSmode==0){
            for (int ii=0;ii<WCDAFit->Template->NComp;ii++){
                fitmode->TS_Bin[ii][iBinUsed0+iSmode*2*fitmode->NBinUsed[ismode][1]] = TS_total_W;
                fitmode->TS_Bin[ii][iBinUsed0+fitmode->NBinUsed[ismode][1]+iSmode*2*fitmode->NBinUsed[ismode][1]] = TS_total_K;
                fitmode->FitStatus_Bin[ii][iBinUsed0] = istat;
            }
        }
        else{
            fitmode->TS_Bin[icomp][iBinUsed0+iSmode*2*fitmode->NBinUsed[ismode][1]] = TS_total_W;
            fitmode->TS_Bin[icomp][iBinUsed0+fitmode->NBinUsed[ismode][1]+iSmode*2*fitmode->NBinUsed[ismode][1]] = TS_total_K;
            fitmode->FitStatus_Bin[icomp][iBinUsed0+fitmode->NBinUsed[ismode][1]] = istat;
        }
    }
    if (ipmode==4){
        fitmode->FitStatus_UL[iBinUsed0] = istat;
        cout<<"STATUS = "<<fitmode->FitStatus_UL[iBinUsed0]<<endl;
    }

    delete[] Par;
    delete[] Parerr;

} 

int main(int argc, char *argv[]){

    if (argc!=4){
        cout<<" \033[31;1mError\033[0m : too few parameters"<<endl;
        cout<<argv[0]<<"\n  [ configfile : Fit.yaml ]\n  [ ROIID ]\n  [ OutFile ]"<<endl;
        return -1; 
    }

    // read configfile
    bool cfflag = cf.Readin(argv[1], 1); 
    if (cfflag){
        cerr<<" \033[31;1mError\033[0m : bad config file! Returned."<<endl;
        return -1; 
    }

    int SrcID = cf.TSmap_SrcID;
    cout<<" *** main : SrcID = "<<SrcID<<endl;
    int ROIID = 0;
    sscanf(argv[2], "%d", &ROIID);
    cout<<" *** main : ROIID = "<<ROIID<<endl;

    // Init Template
    Src_Template *Template = new Src_Template();
    if (Template->Init())
        return -1;
    
    Src_Template *Template_TS = new Src_Template();
    if (Template_TS->Init())
        return -1;

    // Get par of skymap
    string mapfile_temp;
    if (cf.UseWCDA)
        mapfile_temp = cf.fMap;
    else
        mapfile_temp = cf.fKMap;

    TFile *fmap = TFile::Open(mapfile_temp.data());
    TH2D *hon_geo;
    if (!cf.CorOpt)
        hon_geo = (TH2D *) fmap->Get("hon_0");
    else
        hon_geo = (TH2D *) fmap->Get("hon_gal_0");
    nbinsX = hon_geo->GetNbinsX();
    nbinsY = hon_geo->GetNbinsY();
    wbinX  = hon_geo->GetXaxis()->GetBinWidth(1);
    wbinY  = hon_geo->GetYaxis()->GetBinWidth(1);
    X[0] = hon_geo->GetXaxis()->GetBinLowEdge(1);
    X[1] = hon_geo->GetXaxis()->GetBinLowEdge(nbinsX)+wbinX;
    Y[0] = hon_geo->GetYaxis()->GetBinLowEdge(1);
    Y[1] = hon_geo->GetYaxis()->GetBinLowEdge(nbinsY)+wbinY;
    cout<<" *** main : information of input event map : "<<std::endl;
    cout<<"  X : "<<X[0]<<" - "<<X[1]<<", nbins = "<<nbinsX<<", bin width = "<<wbinX<<std::endl;
    cout<<"  Y : "<<Y[0]<<" - "<<Y[1]<<", nbins = "<<nbinsY<<", bin width = "<<wbinY<<std::endl;

    // Set ROI
    cout<<" *** main : Set ROI and read skymap ... "<<endl;
    Src_ROI *ROI = new Src_ROI();
    bool roiflag = ROI->Init_Arbitrary();
    if (roiflag!=0){
        cerr<<"\033[31;1mError\033[0m : Something wrong with reading ROIfile! Returned."<<endl;
        return -1;
    }

    int NnhitUsed_bk;
    int NhitUsed_bk[2];
    int UseWCDA_bk = 0;
    int UseKM2A_bk = 0;
    if (cf.UseWCDA){
        NnhitUsed_bk = cf.NnhitUsed;
        NhitUsed_bk[0] = cf.NhitUsed[0];
        NhitUsed_bk[1] = cf.NhitUsed[1];
        UseWCDA_bk = cf.UseWCDA;
        cf.NhitUsed[0] = cf.TSmap_WCDA[1];
        cf.NhitUsed[1] = cf.TSmap_WCDA[2];
        cf.NnhitUsed = cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1;
        cf.UseWCDA = cf.TSmap_WCDA[0];
    }
    int KNEbinUsed_bk;
    int KEbinUsed_bk[2];
    if (cf.UseKM2A){
        KNEbinUsed_bk = cf.KNEbinUsed;
        KEbinUsed_bk[0] = cf.KEbinUsed[0];
        KEbinUsed_bk[1] = cf.KEbinUsed[1];
        UseKM2A_bk = cf.UseKM2A;
        cf.KEbinUsed[0] = cf.TSmap_KM2A[1];
        cf.KEbinUsed[1] = cf.TSmap_KM2A[2];
        cf.KNEbinUsed = cf.TSmap_KM2A[2]-cf.TSmap_KM2A[1]+1;
        cf.UseKM2A = cf.TSmap_KM2A[0];
    }

    // Read Mapfile
    bool dataflag = 0;
    if (cf.UseWCDA){
        dataflag = WCDAFit->WCDAData->GetContentinROI(ROI->Cellid);
        if (dataflag!=0){
            cerr<<"\033[31;1mError\033[0m : Something wrong with reading WCDA mapfile! Returned."<<endl;
            return -1;
        }
    }
    if (cf.UseKM2A){
        dataflag = KM2AFit->KM2AData->GetContentinROI(ROI->Cellid);
        if (dataflag!=0){
            cerr<<"\033[31;1mError\033[0m : Something wrong with reading KM2A mapfile! Returned."<<endl;
            return -1;
        }
    }

    // Read Response file
    cout<<" *** main : Read Response file ... "<<endl;
    bool respflag = 0;
    if (cf.UseWCDA){
        respflag = WCDAFit->WCDAResp->ReadRespFile(ROI->Ycenter);
        if (respflag!=0){
            cerr<<"\033[31;1mError\033[0m : Fail to read WCDA reponsefile! Returned."<<endl;
            return -1;
        }
    }
    if (cf.UseKM2A){
        respflag = KM2AFit->KM2AResp->ReadRespFile();
        if (respflag!=0){
            cerr<<"\033[31;1mError\033[0m : Fail to read KM2A reponsefile! Returned."<<endl;
            return -1;
        }
    }

    // Set source model
    double epiv;
    string name, sedtype, tempfile, histname, f0_order, mortype, fggabs;
    vector<vector<double> > sedpar;
    vector<double> sed;
    vector<vector<double> > morpar;
    vector<double> mor;

    bool tempflag = 0, ggabsflag = 0;

    if (!(cf.TSmap_SrcID==-1 && cf.SubtractOpt==1)){
        cout<<" *** main : Set source model ... "<<endl;

        string fparresu = cf.WorkDir+"/"+cf.Outdir+"/"+cf.fOut[0]; 
        YAML::Node Compcf = YAML::LoadFile(fparresu.data());
        int SrcsFlag = Compcf["SRC"]["Active"].as<int>();
        int DGEsFlag = Compcf["DGE"]["Active"].as<int>();

        if (!SrcsFlag && !DGEsFlag){
            cerr<<" \033[31;1mError\033[0m : Srcs and DGEs are both not active! Returned."<<endl;
            return -1;
        }

        // Srcs
        if (SrcsFlag){
            Src_Src src;
            int srcid = 0;
            double epiv_global = Compcf["SRC"]["Epiv"].as<double>();
            int parstatus_global[4];
            parstatus_global[0] = Compcf["SRC"]["ParStatus"]["Position"].as<int>();
            parstatus_global[1] = Compcf["SRC"]["ParStatus"]["F0"].as<int>();
            parstatus_global[2] = Compcf["SRC"]["ParStatus"]["Index"].as<int>();
            parstatus_global[3] = Compcf["SRC"]["ParStatus"]["MorPar"].as<int>();
            double ra[5], dec[5];

            int iiter = 0;
            for (YAML::const_iterator it = Compcf["SRC"].begin(); it!=Compcf["SRC"].end();++it){
                if (iiter>=4){

                    srcid = iiter-4;
                    name = it->second["Name"].as<string>();
                    epiv = it->second["Epiv"].as<double>();
                    sedtype = it->second["SEDModel"]["type"].as<string>();

                    int jiter = 0;
                    for (YAML::const_iterator itt = it->second["SEDModel"].begin(); itt!=it->second["SEDModel"].end();++itt){
                        if (jiter>=1){
                            sed.clear();
                            if (itt->first.as<string>() == "F0"){
                                f0_order = itt->second[4].as<string>();
                                for (int ii=0;ii<4;ii++)
                                    if (ii!=3)
                                        sed.push_back(itt->second[ii].as<double>());
                                    else
                                        sed.push_back(itt->second[ii].as<double>()+parstatus_global[1]);
                            }
                            else{
                                for (int ii=0;ii<4;ii++)
                                    if (ii!=3)
                                        sed.push_back(itt->second[ii].as<double>());
                                    else
                                        sed.push_back(itt->second[ii].as<double>()+parstatus_global[2]);
                            }

                            sed.push_back(0.);
                            if (sed[0]<sed[1] || sed[0]>sed[2]){
                                cout<<"\033[31;1mError\033[0m : Init value of sedpar of SRC"<<iiter-4<<" not within the limit! Returned."<<endl;
                                return -1;
                            }
                            sedpar.push_back(sed);
                        }
                        jiter ++;
                    }

                    int imodel = Template->Model->SEDMap[sedtype]-1;
                    if ((jiter-1)!=Template->Model->SEDNpar[imodel]){
                        cout<<"\033[31;1mError\033[0m : Spectrum model \""<<sedtype.data()<<"\" has "<<Template->Model->SEDNpar[imodel]<<" parameters, but "<<jiter-1<<" are given! Returned."<<endl;
                        return -1;
                    }  

                    // Gamma gamma absorption
                    //YAML::Node absnode = it;
                    ggabsflag = 0;
                    if (it->second["GGAbs"].IsDefined()){
                        fggabs = it->second["GGAbs"].as<string>();
                        if (fggabs!="none")
                            ggabsflag = 1;
                    }


                    mortype = it->second["MorModel"]["type"].as<string>();
                    if (mortype=="Ext_Temp"){
                        tempfile = it->second["MorModel"]["Tempfile"].as<string>();
                        src.Init(name, mortype, sedtype, tempfile, sedpar);
                        histname = it->second["MorModel"]["TempHist"][0].as<string>();
                        tempflag = src.GetTempROI(ROI->Cellid, ROI->Cellid_model, histname);
                        if(tempflag!=0){
                            cerr<<"\033[31;1mError\033[0m : Something wrong with reading SRC"<<iiter-4<<" tempfile "<<tempfile<<"! Returned."<<endl;
                            return -1;
                        }
                    }
                    else{
                        for (int ii=0;ii<4;ii++){
                            if (ii!=3)
                                ra[ii] = it->second["MorModel"]["ra"][ii].as<double>();
                            else
                                ra[ii] = it->second["MorModel"]["ra"][ii].as<double>()+parstatus_global[0];
                        }
                        ra[4] = 0.;
                        if (ra[0]<ra[1] || ra[0]>ra[2]){
                            cout<<"\033[31;1mError\033[0m : Init value of RA of SRC"<<iiter-4<<" not within the limit! Returned."<<endl;
                            return -1;
                        }
                        for (int ii=0;ii<4;ii++){
                            if (ii!=3)
                                dec[ii] = it->second["MorModel"]["dec"][ii].as<double>();
                            else
                                dec[ii] = it->second["MorModel"]["dec"][ii].as<double>()+parstatus_global[0];
                        }
                        dec[4] = 0.;
                        if (dec[0]<dec[1] || dec[0]>dec[2]){
                            cout<<"\033[31;1mError\033[0m : Init value of DEC of SRC"<<iiter-4<<" not within the limit! Returned."<<endl;
                            return -1;
                        }

                        if (mortype!="Point"){
                            jiter = 0;
                            for (YAML::const_iterator itt = it->second["MorModel"].begin(); itt!=it->second["MorModel"].end();++itt){
                                if (jiter>=3){
                                    mor.clear();
                                    for (int ii=0;ii<4;ii++){
                                        if (ii!=3)
                                            mor.push_back(itt->second[ii].as<double>());
                                        else
                                            mor.push_back(itt->second[ii].as<double>()+parstatus_global[3]);
                                    }
                                    mor.push_back(0.);
                                    if (mor[0]<mor[1] || mor[0]>mor[2]){
                                        cout<<"\033[31;1mError\033[0m : Init value of morpar of SRC"<<iiter-4<<" not within the limit! Returned."<<endl;
                                        return -1;
                                    }
                                    morpar.push_back(mor);
                                }
                                jiter ++;
                            }
                        }
                        src.Init(name, ra, dec, mortype, sedtype, morpar, sedpar);
                    }

                    if (epiv_global<0)
                        src.SetBasicPar(epiv, f0_order);
                    else
                        src.SetBasicPar(epiv_global, f0_order);
                    src.SetSrcID(srcid);
                    src.SetfGGAbs(ggabsflag, fggabs, name);

                    Template->AddSource(src);

                    sedpar.clear();
                    morpar.clear();
                    src.Clear();
                }
                iiter++;
            }
        }

        // DGEs
        if (DGEsFlag){
            Src_DGE dge;

            int iiter = 0;
            for (YAML::const_iterator it = Compcf["DGE"].begin(); it!=Compcf["DGE"].end();++it){
                if (iiter>=2){
                    name = it->second["Name"].as<string>();
                    epiv = it->second["Epiv"].as<double>();
                    sedtype  = it->second["SEDModel"]["type"].as<string>();
                    tempfile = it->second["Tempfile"].as<string>();
                    histname = it->second["TempHist"][0].as<string>();

                    int jiter = 0;
                    for (YAML::const_iterator itt = it->second["SEDModel"].begin(); itt!=it->second["SEDModel"].end();++itt){
                        if (jiter>=1){
                            sed.clear();
                            if (itt->first.as<string>() == "F0")
                                f0_order = itt->second[4].as<string>();
                            for (int ii=0;ii<4;ii++)
                                sed.push_back(itt->second[ii].as<double>());
                            sed.push_back(0.);
                            if (sed[0]<sed[1] || sed[0]>sed[2]){
                                cout<<"\033[31;1mError\033[0m : Init value of sedpar of DGE"<<iiter-2<<" not within the limit! Returned."<<endl;
                                return -1;
                            }
                            sedpar.push_back(sed);
                        }
                        jiter ++;
                    }

                    dge.Init(name, sedtype, sedpar, tempfile);
                    dge.SetBasicPar(epiv, f0_order);
                    tempflag = dge.GetTempROI(ROI->Cellid, ROI->Cellid_model, histname);
                    if(tempflag!=0){
                        cerr<<"\033[31;1mError\033[0m : Something wrong with reading DGE"<<iiter-2<<" tempfile "<<tempfile<<"! Returned."<<endl;
                        return -1;
                    }
                    Template->AddDGE(dge);

                    sedpar.clear();
                    dge.Clear();
                }
                iiter++;
            }
        }

        if (SrcID>=Template->NComp){
            cerr<<"\033[31;1mError\033[0m : SrcID in Fit.ymal is larger than total number of components! Returned"<<endl;
            return -1;
        }
    }

    // Add excess to background map
    TFile *fExcess = new TFile();
    if (!(cf.TSmap_SrcID==-1 && cf.SubtractOpt==1)){
        string sExcess = cf.WorkDir+"/"+cf.Outdir+"/"+cf.fOut[1];
        cout<<sExcess.data()<<endl;
        fExcess = TFile::Open(sExcess.data());
        TH2D *hExcess_temp;
        int Nbins_total = 0;
        if (cf.TSmap_WCDA[0])
            Nbins_total += (cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1);
        if (cf.TSmap_KM2A[0])
            Nbins_total += (cf.TSmap_KM2A[2]-cf.TSmap_KM2A[1]+1);

        TH2D *hExcess[Nbins_total];
        for (int ii=0;ii<Nbins_total;ii++){

            if (UseWCDA_bk){
                if (cf.TSmap_WCDA[0]){
                    if (ii<(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1))
                        hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_0", ii+cf.TSmap_WCDA[1]-NhitUsed_bk[0]));
                    else
                        hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_0", ii-(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1)+cf.TSmap_KM2A[1]-KEbinUsed_bk[0]+NnhitUsed_bk));
                }
                else{
                    hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_0", ii+cf.TSmap_KM2A[1]-KEbinUsed_bk[0]+NnhitUsed_bk));
                }
            }
            else{
                hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_0", ii+cf.TSmap_KM2A[1]-KEbinUsed_bk[0]));
            }
            hExcess[ii] = (TH2D *) hExcess_temp->Clone(Form("hExcess_%d", ii));
            hExcess[ii]->Reset();

            for (int jj=0;jj<Template->NComp;jj++){

                if (SrcID!=-1){
                    if (!cf.SubtractOpt){
                        if (jj==SrcID) 
                            continue;
                    }
                    else{
                        if (jj!=SrcID)
                            continue;
                    }
                }
                else{
                    if (cf.SubtractOpt)
                        continue;
                }

                if (UseWCDA_bk){
                    if (cf.TSmap_WCDA[0]){
                        if (ii<(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1))
                            hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_%d", ii+cf.TSmap_WCDA[1]-NhitUsed_bk[0], jj));
                        else
                            hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_%d", ii-(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1)+cf.TSmap_KM2A[1]-KEbinUsed_bk[0]+NnhitUsed_bk, jj));
                    }
                    else{
                        hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_%d", ii+cf.TSmap_KM2A[1]-KEbinUsed_bk[0]+NnhitUsed_bk, jj));
                    }
                }
                else{
                    hExcess_temp = (TH2D *) fExcess->Get(Form("hExcess_%d_%d", ii+cf.TSmap_KM2A[1]-KEbinUsed_bk[0], jj));
                }

                hExcess[ii]->Add(hExcess_temp, 1);
            }
        }
    

        //double Xroi0 = hExcess[0]->GetXaxis()->GetBinLowEdge(1);
        //double Yroi0 = hExcess[0]->GetYaxis()->GetBinLowEdge(1);
        for (int ii=0;ii<ROI->Neffbins;ii++){
            int xid = ROI->Cellid[ii]/nbinsY;
            int yid = ROI->Cellid[ii]%nbinsY;
            double xx = X[0]+(xid+0.5)*wbinX;
            double yy = Y[0]+(yid+0.5)*wbinY;
            int xxid = (xx-ROI->Xroi[0])/wbinX;
            int yyid = (yy-ROI->Yroi[0])/wbinY;
            for (int jj=0;jj<Nbins_total;jj++){
                if (cf.TSmap_WCDA[0]){
                    if (jj<(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1))
                        WCDAFit->WCDAData->Nbkg[jj][ii] += hExcess[jj]->GetBinContent(xxid+1, yyid+1);
                    else
                        KM2AFit->KM2AData->Nbkg[jj-(cf.TSmap_WCDA[2]-cf.TSmap_WCDA[1]+1)][ii] += hExcess[jj]->GetBinContent(xxid+1, yyid+1);
                }
                else
                    KM2AFit->KM2AData->Nbkg[jj][ii] += hExcess[jj]->GetBinContent(xxid+1, yyid+1);
            }
        }
    }

    // Init fake source
    Src_Src src;
    double ra_TS[5] = {0, 0, 0, 1, 0};
    double dec_TS[5] = {0, 0, 0, 1, 0};
    name = "TS_fake";
    mortype = "Point";
    ggabsflag = 0;
    if (SrcID!=-1 && cf.SubtractOpt==0){
        if (SrcID < Template->NSrc_total){
            for (int ii=0;ii<Template->NSrc;ii++){
                if (Template->Srcs[ii].SrcID==SrcID){
                    f0_order= Template->Srcs[ii].F0_order;
                    epiv = Template->Srcs[ii].Epiv;
                    sedtype = Template->Srcs[ii].SEDtype;
                    for (int ipar=0;ipar<Template->Srcs[ii].nSEDpar;ipar++){
                        sed.clear();
                        sed = Template->Srcs[ii].SEDPar[ipar];
                        if (ipar==0){
                            sed[0] = 0.005;
                            sed[1] = -500.;
                        }
                        else{
                            sed[3] = 1;
                        }
                        sedpar.push_back(sed);
                    }

                    if (Template->Srcs[ii].GGAbsFlag){
                        ggabsflag = Template->Srcs[ii].GGAbsFlag;
                        fggabs = Template->Srcs[ii].fGGAbs;
                    }
                }
            } 

            for (int ii=0;ii<Template->NSrc_NumCon;ii++){
                if (Template->Srcs_NumCon[ii].SrcID==SrcID){
                    f0_order= Template->Srcs_NumCon[ii].F0_order;
                    epiv = Template->Srcs_NumCon[ii].Epiv;
                    sedtype = Template->Srcs_NumCon[ii].SEDtype;
                    for (int ipar=0;ipar<Template->Srcs_NumCon[ii].nSEDpar;ipar++){
                        sed.clear();
                        sed = Template->Srcs_NumCon[ii].SEDPar[ipar];
                        if (ipar==0){
                            sed[0] = 0.005;
                            sed[1] = -500.;
                        }
                        else{
                            sed[3] = 1;
                        }
                        sedpar.push_back(sed);
                    }
                    if (Template->Srcs_NumCon[ii].GGAbsFlag){
                        ggabsflag = Template->Srcs_NumCon[ii].GGAbsFlag;
                        fggabs = Template->Srcs_NumCon[ii].fGGAbs;
                    }
                }
            }

            for (int ii=0;ii<Template->NSrc_Temp;ii++){
                if (Template->Srcs_Temp[ii].SrcID==SrcID){
                    f0_order= Template->Srcs_Temp[ii].F0_order;
                    epiv = Template->Srcs_Temp[ii].Epiv;
                    sedtype = Template->Srcs_Temp[ii].SEDtype;
                    for (int ipar=0;ipar<Template->Srcs_Temp[ii].nSEDpar;ipar++){
                        sed.clear();
                        sed = Template->Srcs_Temp[ii].SEDPar[ipar];
                        if (ipar==0){
                            sed[0] = 0.005;
                            sed[1] = -500.;
                        }
                        else{
                            sed[3] = 1;
                        }
                        sedpar.push_back(sed);
                    }
                    if (Template->Srcs_Temp[ii].GGAbsFlag){
                        ggabsflag = Template->Srcs_Temp[ii].GGAbsFlag;
                        fggabs = Template->Srcs_Temp[ii].fGGAbs;
                    }
                }
            }

        }
        else{
            for (int ii=0;ii<Template->NDGE;ii++){
                if (ii==(SrcID-Template->NSrc_total)){
                    f0_order= Template->DGEs[ii].F0_order;
                    epiv = Template->DGEs[ii].Epiv;
                    sedtype = Template->DGEs[ii].SEDtype;
                    for (int ipar=0;ipar<Template->DGEs[ii].nSEDpar;ipar++){
                        sed.clear();
                        sed = Template->DGEs[ii].SEDPar[ipar];
                        if (ipar==0){
                            sed[0] = 0.005;
                            sed[1] = -500.;
                        }
                        else{
                            sed[3] = 1;
                        }
                        sedpar.push_back(sed);
                    }
                }
            }
        }
    }
    else{
        if (cf.UseWCDA && !cf.UseKM2A){
            f0_order = "1.e-13";
            epiv = 3.0;
            sedtype = "PL";
            sed.clear();
            sed.push_back(0.005);
            sed.push_back(-500);
            sed.push_back(500);
            sed.push_back(0.);
            sed.push_back(0.);
            sedpar.push_back(sed);
            sed.clear();
            sed.push_back(2.6);
            sed.push_back(2.0);
            sed.push_back(3.0);
            sed.push_back(1);
            sed.push_back(0.);
            sedpar.push_back(sed);
        }
        else if (!cf.UseWCDA && cf.UseKM2A){
            f0_order = "1.e-16";
            epiv = 50.0;
            sedtype = "PL";
            sed.clear();
            sed.push_back(0.005);
            sed.push_back(-500);
            sed.push_back(500);
            sed.push_back(0.);
            sed.push_back(0.);
            sedpar.push_back(sed);
            sed.clear();
            sed.push_back(3.2);
            sed.push_back(2.0);
            sed.push_back(4.0);
            sed.push_back(1);
            sed.push_back(0.);
            sedpar.push_back(sed);
        }
        else{
            f0_order = "1.e-14";
            epiv = 10.0;
            sedtype = "LP";
            sed.clear();
            sed.push_back(0.005);
            sed.push_back(-500);
            sed.push_back(500);
            sed.push_back(0.);
            sed.push_back(0.);
            sedpar.push_back(sed);
            sed.clear();
            sed.push_back(2.87440);
            sed.push_back(2.0);
            sed.push_back(4.0);
            sed.push_back(1);
            sed.push_back(0.);
            sedpar.push_back(sed);
            sed.clear();
            sed.push_back(0.09436);
            sed.push_back(0.0);
            sed.push_back(1.0);
            sed.push_back(1);
            sed.push_back(0.);
            sedpar.push_back(sed);
        }
    }

    sed.clear();
    src.SetBasicPar(epiv, f0_order);
    src.SetSrcID(0);
    src.Init(name, ra_TS, dec_TS, mortype, sedtype, morpar, sedpar);
    src.SetfGGAbs(ggabsflag, fggabs, name);

    Template_TS->AddSource(src);
    npar_src = Template_TS->Srcs[0].nSEDpar+Template_TS->Srcs[0].nMorpar+2;
    npar_total = npar_src+npar_numsrc+npar_dge; 
    cout<<" *** Set source model : npar_src = "<<npar_src<<", npar_numsrc = "<<npar_numsrc<<", npar_dge = "<<npar_dge<<", npar_total = "<<npar_total<<endl; 

    cout<<" *** main : Init fitting mode ... "<<endl;
    Src_FittingMode *FitMode = new Src_FittingMode();
    FitMode->Init(Template_TS);
    FitMode->InitRes(Template_TS);

    cout<<" *** main : Fitting ... "<<endl;
    bool initflag = 0;
    WCDAFit->SetROI(ROI);
    WCDAFit->SetTemplate(Template_TS);
    if (cf.UseWCDA){
        WCDAFit->SetBasicPar(4.e10);
        initflag = WCDAFit->Init();
        if (initflag!=0){
            cerr<<"\033[31;1mError\033[0m : Initialization of WCDAFit failed! Returned."<<endl;
            return -1;
        }
    }
    if (cf.UseKM2A){
        KM2AFit->SetROI(ROI);
        KM2AFit->SetTemplate(Template_TS);
        KM2AFit->SetBasicPar(1.e10*TMath::Pi());
        initflag = KM2AFit->Init();
        if (initflag!=0){
            cerr<<"\033[31;1mError\033[0m : Initialization of KM2AFit failed! Returned."<<endl;
            return -1;
        }
    }

    // Fitting
    iBinUsed0 = FitMode->NBinUsed[0][0];
    iBinUsed1 = FitMode->NBinUsed[0][1];
    iThisComp = -1;
    /*double Radius_TS = 5*0.5;
    if (Template->Srcs[SrcID].Mortype == "Ext_Temp")
        Radius_TS = 100000.;
    else if (Template->Srcs[SrcID].Mortype == "Point")
        Radius_TS = 5*0.5;
    else
        Radius_TS = 5*sqrt(0.5*0.5+pow(Template->Srcs[SrcID].MorPar[0][0], 2));*/

    TH2D *hSig = new TH2D("hSig", "hSig", ROI->nbinsX_roi, ROI->Xroi[0], ROI->Xroi[1], ROI->nbinsY_roi, ROI->Yroi[0], ROI->Yroi[1]);
    TH1D *hSig1D = new TH1D("hSig1D", "hSig1D", 70, -7, 7);
    hSig->SetTitle("");
    hSig->Reset();
    int Nbineff = 0;
    for (int ii=0;ii<ROI->Neffbins;ii++){

        int xid = ROI->Cellid[ii]/nbinsY;
        int yid = ROI->Cellid[ii]%nbinsY;
        double xx = X[0]+(xid+0.5)*wbinX;
        double yy = Y[0]+(yid+0.5)*wbinY;
        //double space = distance(90-yy, xx, 90-Template->Srcs[SrcID].Dec[0], Template->Srcs[SrcID].Ra[0]);
        //if (space>Radius_TS) continue;

        Nbineff++;
        Niter = 0; 
        if (!(Nbineff>ROIID*100 && Nbineff<=(ROIID+1)*100)) continue;

        Template_TS->Srcs[0].Ra[0] = xx-0.01;
        Template_TS->Srcs[0].Ra[1] = xx-4;
        Template_TS->Srcs[0].Ra[2] = xx+4;
        Template_TS->Srcs[0].Dec[0] = yy-0.01;
        Template_TS->Srcs[0].Dec[1] = yy-4;
        Template_TS->Srcs[0].Dec[2] = yy+4;
        WCDAFit->Template->Srcs[0].SEDPar[0][0] = 0.005;

        // Fitting
        if (cf.UseWCDA)
            WCDAFit->GetTobs();
        if (cf.UseKM2A)
            KM2AFit->GetTobs();

        Fitting(0, 0, FitMode, -1);

        double sig = 0;
        double TS_total = 0;
        if (cf.UseWCDA)
            TS_total += 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null);
        if (cf.UseKM2A)
            TS_total += 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null);

        if (isnan(WCDAFit->Template->Srcs[0].SEDPar[0][0])){
            WCDAFit->Template->Srcs[0].SEDPar[0][0] = 0.1;
            continue;
        }
        if (TS_total<=0) continue;
        if (WCDAFit->Template->Srcs[0].SEDPar[0][0]>=0)
            sig = sqrt(TS_total);
        else
            sig = -sqrt(TS_total);

        int xxid = (xx-ROI->Xroi[0])/wbinX;
        int yyid = (yy-ROI->Yroi[0])/wbinY;
        hSig->SetBinContent(xxid+1, yyid+1, sig);
        hSig1D->Fill(sig);

    }

    TFile *fout = TFile::Open(argv[3], "recreate");
    fout->cd();
    hSig->Write();
    hSig1D->Write();
    fout->Close();

    fExcess->Close();
    fmap->Close();
    return 0;
}
