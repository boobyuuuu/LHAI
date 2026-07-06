# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <iostream>
# include <fstream>
# include <unistd.h>

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
        //f  += WCDAFit->log_L_null;
        TS += 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null);
        cout<<"TS_WCDA = "<<Form("%6.2lf", 2*(WCDAFit->log_L_sig-WCDAFit->log_L_null))<<", ";
    }
    if (cf.UseKM2A){
        f  += -KM2AFit->log_L_sig;
        //f  += KM2AFit->log_L_null;
        TS += 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null);
        cout<<"TS_KM2A = "<<Form("%6.2lf", 2*(KM2AFit->log_L_sig-KM2AFit->log_L_null))<<", ";
    }

    cout<<"TS = "<<Form("%6.2lf", TS)<<endl;

    Niter++;

    /*for (int ii=0;ii<npar_total;ii++)
        cout<<Form("%12.7lf  ", par[ii]);
    cout<<endl;*/

    // memory monitor
    /*FILE* fp = fopen("/proc/self/status", "r");
    char line[128];
    while (fgets(line, 128, fp)!=NULL){
        if (strncmp(line, "VmRSS:", 6)==0){
            printf("%d kB\n", atoi(line+6));
            break;
        }
    }
    fclose(fp);*/

}

void Fitting(int ipmode, int ismode, Src_FittingMode *fitmode, int icomp){

    // Reset Fast Iteration
    Niter = 0;
    for (int isrc=0;isrc<WCDAFit->Template->NSrc;isrc++)
        WCDAFit->Template->Srcs[isrc].ConvoFlag = 1;
    for (int isrc=0;isrc<WCDAFit->Template->NSrc_NumCon;isrc++)
        WCDAFit->Template->Srcs_NumCon[isrc].ConvoFlag = 1;
    for (int isrc=0;isrc<WCDAFit->Template->NTemp;isrc++){
        if (isrc<WCDAFit->Template->NSrc_Temp)
            WCDAFit->Template->Srcs_Temp[isrc].ConvoFlag = 1;
        else
            WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].ConvoFlag = 1;
    }
    int npar = 0;
    for (int isrc=0;isrc<WCDAFit->Template->NSrc;isrc++){
        for (int ipar=0;ipar<WCDAFit->Template->Srcs[isrc].nSEDpar;ipar++)
            WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0] = fitmode->ParVal[npar+2+ipar];
        npar += 2+WCDAFit->Template->Srcs[isrc].nSEDpar+WCDAFit->Template->Srcs[isrc].nMorpar;
    }
    for (int isrc=0;isrc<WCDAFit->Template->NSrc_NumCon;isrc++){
        for (int ipar=0;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;ipar++)
            WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = fitmode->ParVal[npar+2+ipar];
        npar += 2+WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar+WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;
    }
    for (int isrc=0;isrc<WCDAFit->Template->NTemp;isrc++){
        if (isrc<WCDAFit->Template->NSrc_Temp){
            for (int ipar=0;ipar<WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;ipar++)
                WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0] = fitmode->ParVal[npar+ipar];
            npar += WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;
        }
        else{
            for (int ipar=0;ipar<WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;ipar++)
                WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].SEDPar[ipar][0] = fitmode->ParVal[npar+ipar];
            npar += WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;
        }
    }

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

    npar = 0;
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
            gMinuit->mnparm(npar+2+ipar, Form("%s_%s", WCDAFit->Template->Srcs[isrc].Srcname.data(), WCDAFit->Template->Model->SEDParname[imodel][ipar].data()), WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0], 0.01, WCDAFit->Template->Srcs[isrc].SEDPar[ipar][1], WCDAFit->Template->Srcs[isrc].SEDPar[ipar][2], ierflg);
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
        gMinuit->mnparm(npar  , Form("%s_X", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data()), WCDAFit->Template->Srcs_NumCon[isrc].Ra[0], 0.1, WCDAFit->Template->Srcs_NumCon[isrc].Ra[1], WCDAFit->Template->Srcs_NumCon[isrc].Ra[2], ierflg);
        if (WCDAFit->Template->Srcs_NumCon[isrc].Ra[3] || fitmode->ParStatus[ismode][npar])
            gMinuit->FixParameter(npar);
        gMinuit->mnparm(npar+1, Form("%s_Y", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data()), WCDAFit->Template->Srcs_NumCon[isrc].Dec[0], 0.1, WCDAFit->Template->Srcs_NumCon[isrc].Dec[1], WCDAFit->Template->Srcs_NumCon[isrc].Dec[2], ierflg);
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
            gMinuit->mnparm(npar+ipar, Form("%s_%s", WCDAFit->Template->Srcs_NumCon[isrc].Srcname.data(), WCDAFit->Template->Model->MorParname[imodel][ipar].data()), WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][0], 0.1, WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][1], WCDAFit->Template->Srcs_NumCon[isrc].MorPar[ipar][2], ierflg);
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

    arglist[0] = 20000;
    arglist[1] = 0.1;
    gMinuit->mnexcm(fitmode->MinAlgo[ipmode].data(), arglist, 2, ierflg);
    if (iThisPmode==1)
        gMinuit->mnexcm("HESSE", arglist, 2, ierflg);

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
            double ts_bin = fitmode->TS_Bin[isrc][iBinUsed0]+fitmode->TS_Bin[isrc][iBinUsed0+fitmode->NBinUsed[1][1]];
            if (ts_bin<fitmode->TS_UL[2]){
                fitmode->FNorm[isrc][iBinUsed0] = Par[npar+2];
                fitmode->FNormErr[isrc][iBinUsed0] = Parerr[npar+2];
                fitmode->FNormUL[isrc][iBinUsed0] = Par[npar+2]+erU;
            }
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
            double ts_bin = fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc][iBinUsed0]+fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc][iBinUsed0+fitmode->NBinUsed[1][1]];
            if (ts_bin<fitmode->TS_UL[2]){
                fitmode->FNorm[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Par[npar+2];
                fitmode->FNormErr[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Parerr[npar+2];
                fitmode->FNormUL[isrc+WCDAFit->Template->NSrc][iBinUsed0] = Par[npar+2]+erU;
            }
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
                double ts_bin = fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0]+fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0+fitmode->NBinUsed[1][1]];
                if (ts_bin<fitmode->TS_UL[2]){
                    fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                    fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
                    fitmode->FNormUL[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar]+erU;
                }
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
                double ts_bin = fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0]+fitmode->TS_Bin[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0+fitmode->NBinUsed[1][1]];
                if (ts_bin<fitmode->TS_UL[2]){
                    fitmode->FNorm[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar];
                    fitmode->FNormErr[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Parerr[npar];
                    fitmode->FNormUL[isrc+WCDAFit->Template->NSrc+WCDAFit->Template->NSrc_NumCon][iBinUsed0] = Par[npar]+erU;
                }
            }
            npar += WCDAFit->Template->DGEs[isrc-WCDAFit->Template->NSrc_Temp].nSEDpar;
        }

    }   

    // link parameters
    if (ipmode==0){
        int ipar_temp = 0;
        // Srcs
        for (int isrc=0;isrc<WCDAFit->Template->NSrc;isrc++){
            if (WCDAFit->Template->Srcs[isrc].LinkPars){
                int targetsrcid = WCDAFit->Template->Srcs[isrc].TargetSrcID_Class;
                int targetsrclass = WCDAFit->Template->Srcs[isrc].TargetSrcClass;
                for (int ipar=1;ipar<WCDAFit->Template->Srcs[isrc].nSEDpar;ipar++){
                    if (targetsrclass==0){
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][4];
                    }
                    else if (targetsrclass==1){
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][4];
                    }
                    else{
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][4];
                    }
                    fitmode->ParVal[ipar_temp+2+ipar] = WCDAFit->Template->Srcs[isrc].SEDPar[ipar][0];
                    fitmode->ParErr[ipar_temp+2+ipar] = WCDAFit->Template->Srcs[isrc].SEDPar[ipar][4];
                }
            }
            ipar_temp += 2+WCDAFit->Template->Srcs[isrc].nSEDpar+WCDAFit->Template->Srcs[isrc].nMorpar;
        }
        // NumSrcs
        for (int isrc=0;isrc<WCDAFit->Template->NSrc_NumCon;isrc++){
            if (WCDAFit->Template->Srcs_NumCon[isrc].LinkPars){
                int targetsrcid = WCDAFit->Template->Srcs_NumCon[isrc].TargetSrcID_Class;
                int targetsrclass = WCDAFit->Template->Srcs_NumCon[isrc].TargetSrcClass;
                for (int ipar=1;ipar<WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar;ipar++){
                    if (targetsrclass==0){
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][4];
                    }
                    else if (targetsrclass==1){
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][4];
                    }
                    else{
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][4];
                    }
                    fitmode->ParVal[ipar_temp+2+ipar] = WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][0];
                    fitmode->ParErr[ipar_temp+2+ipar] = WCDAFit->Template->Srcs_NumCon[isrc].SEDPar[ipar][4];
                }
            }
            ipar_temp += 2+WCDAFit->Template->Srcs_NumCon[isrc].nSEDpar+WCDAFit->Template->Srcs_NumCon[isrc].nMorpar;
        }
        // Src_Temp && DGEs
        for (int isrc=0;isrc<WCDAFit->Template->NSrc_Temp;isrc++){
            if (WCDAFit->Template->Srcs_Temp[isrc].LinkPars){
                int targetsrcid = WCDAFit->Template->Srcs_Temp[isrc].TargetSrcID_Class;
                int targetsrclass = WCDAFit->Template->Srcs_Temp[isrc].TargetSrcClass;
                for (int ipar=1;ipar<WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;ipar++){
                    if (targetsrclass==0){
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs[targetsrcid].SEDPar[ipar][4];
                    }
                    else if (targetsrclass==1){
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_NumCon[targetsrcid].SEDPar[ipar][4];
                    }
                    else{
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][0];
                        WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][4] = WCDAFit->Template->Srcs_Temp[targetsrcid].SEDPar[ipar][4];
                    }
                    fitmode->ParVal[ipar_temp+ipar] = WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][0];
                    fitmode->ParErr[ipar_temp+ipar] = WCDAFit->Template->Srcs_Temp[isrc].SEDPar[ipar][4];
                }
            }
            ipar_temp += WCDAFit->Template->Srcs_Temp[isrc].nSEDpar;
        }
    }

    // Get CORRELATION Contour
    /*int ke1 = 7, ke2 = 52, nptu = 30, ierrf;
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
        if (cf.UseWCDA)
            fitmode->logL_total += 2*WCDAFit->log_L_sig;
        if (cf.UseKM2A)
            fitmode->logL_total += 2*KM2AFit->log_L_sig; 
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

    if (argc!=3){
        cout<<" \033[31;1mError\033[0m : too few parameters"<<endl;
        cout<<argv[0]<<"\n  [ configfile : Fit.yaml ]\n  [ configfile : ParInit.yaml ]"<<endl;
        return -1; 
    }

    // read configfile
    bool cfflag = cf.Readin(argv[1], 1); 
    if (cfflag){
        cerr<<" \033[31;1mError\033[0m : bad config file! Returned."<<endl;
        return -1; 
    }

    int FitFlag = 0;
    for (int ii=0;ii<5;ii++)
        FitFlag += cf.FitOpt[ii];

    // Init Template
    Src_Template *Template = new Src_Template();
    if (Template->Init())
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

    /*TFile *fROI = TFile::Open("test_ROI.root", "recreate");
    TH2D *hROI = (TH2D *) hon_geo->Clone("hROI");
    TH2D *hROI_model = (TH2D *) hon_geo->Clone("hROI_model");
    hROI->Reset();
    hROI_model->Reset();
    for (int ii=0;ii<ROI->Neffbins;ii++){
        int xid = ROI->Cellid[ii]/nbinsY;
        int yid = ROI->Cellid[ii]%nbinsY;
        hROI->SetBinContent(xid+1, yid+1, 1);
    }
    for (int ii=0;ii<ROI->Neffbins_model;ii++){
        int xid = ROI->Cellid_model[ii]/nbinsY;
        int yid = ROI->Cellid_model[ii]%nbinsY;
        hROI_model->SetBinContent(xid+1, yid+1, 1);
    }
    fROI->cd();
    hROI->Write();
    hROI_model->Write();
    fROI->Close();
    return 0;*/

    if (FitFlag>0){

        // fine-tune initial F0 for better convergence
        double f0_finetune = 1.0;
        if (cf.FitOpt[0])
            f0_finetune = 1.0;

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
        cout<<" *** main : Set source model ... "<<endl;

        YAML::Node Compcf = YAML::LoadFile(argv[2]);
        int SrcsFlag = Compcf["SRC"]["Active"].as<int>();
        int DGEsFlag = Compcf["DGE"]["Active"].as<int>();

        if (!SrcsFlag && !DGEsFlag){
            cerr<<" \033[31;1mError\033[0m : Srcs and DGEs are both not active! Returned."<<endl;
            return -1;
        }
        if (DGEsFlag){
            if (abs(ROI->Ycenter_gb)>15){
                cerr<<" \033[31;1mError\033[0m : Activate DGEs but center of ROI is at large galactic latitude (abs(gb)>15 degree)! Returned."<<endl;
                cerr<<" \033[32;1mDebug\033[0m : Deactivate DGEs."<<endl;
                return -1;
            }
        }


        // Read ParInit
        double epiv;
        string name, sedtype, tempfile, histname, f0_order, fggabs;
        vector<vector<double> > sedpar;
        vector<double> sed;
        bool tempflag = 0, ggabsflag = 0;
        int linkflag = 0, targertsrc = -1;
        double epiv_global;
        int parstatus_global[4];
        int nparsrc_free = 0, npardge_free = 0;
        // Srcs
        if (SrcsFlag){
            Src_Src src;
            int srcid = 0;
            epiv_global = Compcf["SRC"]["Epiv"].as<double>();
            parstatus_global[0] = Compcf["SRC"]["ParStatus"]["Position"].as<int>();
            parstatus_global[1] = Compcf["SRC"]["ParStatus"]["F0"].as<int>();
            parstatus_global[2] = Compcf["SRC"]["ParStatus"]["Index"].as<int>();
            parstatus_global[3] = Compcf["SRC"]["ParStatus"]["MorPar"].as<int>();
            double ra[5], dec[5];
            string mortype;
            vector<vector<double> > morpar;
            vector<double> mor;

            int iiter = 0;
            for (YAML::const_iterator it = Compcf["SRC"].begin(); it!=Compcf["SRC"].end();++it){
                if (iiter>=4){

                    srcid = iiter-4;
                    name = it->second["Name"].as<string>();
                    epiv = it->second["Epiv"].as<double>();
                    sedtype = it->second["SEDModel"]["type"].as<string>();

                    linkflag = 0, targertsrc = -1;
                    if (it->second["LinkPars"].IsDefined()){
                        if (it->second["LinkPars"]["SED"].IsDefined())
                            targertsrc = it->second["LinkPars"]["SED"].as<int>();
                        if (targertsrc>=0)
                            linkflag = 1;
                    }

                    int jiter = 0;
                    for (YAML::const_iterator itt = it->second["SEDModel"].begin(); itt!=it->second["SEDModel"].end();++itt){
                        if (jiter>=1){
                            sed.clear();
                            if (itt->first.as<string>() == "F0"){
                                f0_order = itt->second[4].as<string>();
                                if ((itt->second[3].as<double>()+parstatus_global[1])==0.)
                                    nparsrc_free++;
                                for (int ii=0;ii<4;ii++)
                                    if (ii!=3){
                                        if ((itt->second[3].as<double>()+parstatus_global[1])!=0.)
                                            sed.push_back(itt->second[ii].as<double>());
                                        else{
                                            if (ii==0)
                                                sed.push_back(itt->second[ii].as<double>()*f0_finetune);
                                            else
                                                sed.push_back(itt->second[ii].as<double>());
                                        }
                                    }
                                    else
                                        sed.push_back(itt->second[ii].as<double>()+parstatus_global[1]);
                            }
                            else{
                                if ((itt->second[3].as<double>()+parstatus_global[2])==0 && linkflag!=1)
                                    nparsrc_free++;
                                for (int ii=0;ii<4;ii++)
                                    if (ii!=3)
                                        sed.push_back(itt->second[ii].as<double>());
                                    else{
                                        if (linkflag!=1)
                                            sed.push_back(itt->second[ii].as<double>()+parstatus_global[2]);
                                        else
                                            sed.push_back(1);
                                    }
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
                        if ((it->second["MorModel"]["ra"][3].as<double>()+parstatus_global[0])==0)
                            nparsrc_free++;
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
                        if ((it->second["MorModel"]["dec"][3].as<double>()+parstatus_global[0])==0)
                            nparsrc_free++;
                        if (dec[0]<dec[1] || dec[0]>dec[2]){
                            cout<<"\033[31;1mError\033[0m : Init value of DEC of SRC"<<iiter-4<<" not within the limit! Returned."<<endl;
                            return -1;
                        }

                        if (mortype!="Point"){
                            jiter = 0;
                            for (YAML::const_iterator itt = it->second["MorModel"].begin(); itt!=it->second["MorModel"].end();++itt){
                                if (jiter>=3){
                                    mor.clear();
                                    if ((itt->second[3].as<double>()+parstatus_global[3])==0)
                                        nparsrc_free++;
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

                    //if (ggabsflag)
                    src.SetfGGAbs(ggabsflag, fggabs, name);
                    if (linkflag==1)
                        src.SetLinkPars(targertsrc);

                    Template->AddSource(src);
                    if ((src.Mortype=="Ext_gaus" || src.Mortype=="Point" || src.Mortype == "Ext_gaus_E"))
                        npar_src += src.SEDPar.size()+src.MorPar.size()+2;
                    else if (src.Mortype=="Ext_Temp")
                        npar_dge += src.SEDPar.size();
                    else
                        npar_numsrc += src.SEDPar.size()+src.MorPar.size()+2;

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
                            if (itt->second[3].as<double>()==0)
                                npardge_free++;
                            if (itt->first.as<string>() == "F0")
                                f0_order = itt->second[4].as<string>();
                            for (int ii=0;ii<4;ii++){
                                if (itt->second[3].as<double>()!=0.)
                                    sed.push_back(itt->second[ii].as<double>());
                                else{
                                    if (ii==0)
                                        sed.push_back(itt->second[ii].as<double>()*f0_finetune);
                                    else
                                        sed.push_back(itt->second[ii].as<double>());
                                }
                            }
                            sed.push_back(0.);
                            if (sed[0]<sed[1] || sed[0]>sed[2]){
                                cout<<"\033[31;1mError\033[0m : Init value of sedpar of DGE"<<iiter-2<<" not within the limit! Returned."<<endl;
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

                    dge.Init(name, sedtype, sedpar, tempfile);
                    dge.SetBasicPar(epiv, f0_order);
                    tempflag = dge.GetTempROI(ROI->Cellid, ROI->Cellid_model, histname);
                    if(tempflag!=0){
                        cerr<<"\033[31;1mError\033[0m : Something wrong with reading DGE"<<iiter-2<<" tempfile "<<tempfile<<"! Returned."<<endl;
                        return -1;
                    }
                    Template->AddDGE(dge);
                    npar_dge += dge.SEDPar.size();

                    sedpar.clear();
                    dge.Clear();
                }
                iiter++;
            }
        }
        npar_total += npar_src+npar_dge+npar_numsrc;
        cout<<" *** Set source model : npar_src = "<<npar_src<<", npar_numsrc = "<<npar_numsrc<<", npar_dge = "<<npar_dge<<", npar_total = "<<npar_total<<", nparSrc_free = "<<nparsrc_free<<", nparDGE_free = "<<npardge_free<<endl; 
        Template->SetNparFree(nparsrc_free, npardge_free);

        cout<<" *** main : Init fitting mode ... "<<endl;
        Src_FittingMode *FitMode = new Src_FittingMode();
        FitMode->Init(Template);
        FitMode->InitRes(Template);

        cout<<" *** main : Fitting ... "<<endl;
        bool initflag = 0;
        WCDAFit->SetROI(ROI);
        WCDAFit->SetTemplate(Template);
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
            KM2AFit->SetTemplate(Template);
            KM2AFit->SetBasicPar(1.e10*TMath::Pi());
            initflag = KM2AFit->Init();
            if (initflag!=0){
                cerr<<"\033[31;1mError\033[0m : Initialization of KM2AFit failed! Returned."<<endl;
                return -1;
            }
        }

        // Fitting
        if (cf.UseWCDA)
            WCDAFit->GetTobs();
        if (cf.UseKM2A)
            KM2AFit->GetTobs();

        /*TFile *fROI = TFile::Open("test_ROI.root", "recreate");
        TH2D *hROI = (TH2D *) hon_geo->Clone("hROI");
        TH2D *hROI_model = (TH2D *) hon_geo->Clone("hROI_model");
        TH2D *hLtime = (TH2D *) hon_geo->Clone("hLtime");
        hROI->Reset();
        hROI_model->Reset();
        for (int ii=0;ii<ROI->Neffbins;ii++){
            int xid = ROI->Cellid[ii]/nbinsY;
            int yid = ROI->Cellid[ii]%nbinsY;
            hROI->SetBinContent(xid+1, yid+1, 1);
        }
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            int xid = ROI->Cellid_model[ii]/nbinsY;
            int yid = ROI->Cellid_model[ii]%nbinsY;
            hROI_model->SetBinContent(xid+1, yid+1, 1);
            hLtime->SetBinContent(xid+1, yid+1,  WCDAFit->WCDAData->Tobs[ii]);
        }
        fROI->cd();
        hROI->Write();
        hROI_model->Write();
        hLtime->Write();
        fROI->Close();
        return 0;*/

        // Loop fitting mode
        for (int ipmode=0;ipmode<FitMode->nPmode;ipmode++){
            if (cf.FitOpt[ipmode]){

                iThisPmode = ipmode;

                cout<<" *** main : Fitting "<<FitMode->ModeTag[ipmode]<<" ... "<<endl;

                for (int ismode=0;ismode<FitMode->NSmode[ipmode];ismode++){

                    if (Template->NComp==1 && ismode==1) continue;

                    int iSmode = ismode;
                    if (ipmode>0)
                        iSmode += FitMode->NSmode_int[ipmode-1];

                    if (FitMode->NBinUsed[iSmode][2]==-1){
                        iBinUsed0 = FitMode->NBinUsed[iSmode][0];
                        iBinUsed1 = FitMode->NBinUsed[iSmode][1];

                        if (FitMode->ParStatus[iSmode][npar_total]==-1){
                            iThisComp = -1;
                            Fitting(ipmode, iSmode, FitMode, -1);
                        }
                        else if (FitMode->ParStatus[iSmode][npar_total]==1){
                            for (int icomp=0;icomp<Template->NComp;icomp++){
                                FitMode->SetParStatus(ipmode, ismode, icomp, Template, FitMode->NBinUsed[iSmode][2]);
                                FitMode->SetNPar_total(ipmode, ismode, icomp, Template);
                                if (ismode==0){
                                    iThisComp = -1;
                                    if (icomp==0)
                                        Fitting(ipmode, iSmode, FitMode, icomp);
                                }
                                else{
                                    iThisComp = icomp;
                                    Fitting(ipmode, iSmode, FitMode, icomp);
                                }
                            }
                        }
                        else{
                            cerr<<" \033[31;1mError\033[0m : ParStatus[npar_total] of imode "<<iSmode<<" is wrong! Returned."<<endl;
                            return -1;
                        }

                    }
                    else if (FitMode->NBinUsed[iSmode][2]==1){
                        for (int ibin=FitMode->NBinUsed[iSmode][0];ibin<FitMode->NBinUsed[iSmode][1];ibin++){
                            iBinUsed0 = ibin;
                            iBinUsed1 = ibin+1;

                            if (FitMode->ParStatus[iSmode][npar_total]==-1){
                                if (ipmode==4){
                                    bool binloopflag = 0;
                                    for (int icomp=0;icomp<Template->NComp;icomp++){
                                        if ((FitMode->TS_Bin[icomp][ibin]+FitMode->TS_Bin[icomp][ibin+FitMode->NBinUsed[ipmode][1]])<FitMode->TS_UL[2]){
                                            binloopflag = 1;
                                            break;
                                        }
                                    }
                                    if (!binloopflag) continue;
                                }
                                iThisComp = -1;
                                Fitting(ipmode, iSmode, FitMode, -1);
                            }
                            else if (FitMode->ParStatus[iSmode][npar_total]==1){
                                for (int icomp=0;icomp<Template->NComp;icomp++){
                                    FitMode->SetParStatus(ipmode, ismode, icomp, Template, FitMode->NBinUsed[iSmode][2]);
                                    FitMode->SetNPar_total(ipmode, ismode, icomp, Template);
                                    if (ismode==0){
                                        iThisComp = -1;
                                        if (icomp==0)
                                            Fitting(ipmode, iSmode, FitMode, -1);
                                    }
                                    else{
                                        iThisComp = icomp;
                                        Fitting(ipmode, iSmode, FitMode, icomp);
                                    }
                                }
                            }
                            else{
                                cerr<<" \033[31;1mError\033[0m : ParStatus[npar_total] of imode "<<iSmode<<" is wrong! Returned."<<endl;
                                return -1;
                            }
                        }
                    }
                    else{
                        cerr<<" \033[31;1mError\033[0m : NBinUsed[2] of imode "<<iSmode<<" is wrong! Returned."<<endl;
                        return -1;
                    }

                }
                // Cal TS_Src
                if (ipmode==2)
                    FitMode->CalTS(2, Template);
                // Cal TS_Bin
                if (ipmode==3)
                    FitMode->CalTS(3, Template);
                // Output optimal value of parameter to xxx.yaml 
                if (ipmode==0 && cf.fOut[0]!="none"){
                    FitMode->MkOutdir();
                    FitMode->OutPara(Template, DGEsFlag, SrcsFlag, epiv_global, parstatus_global);
                }
                // Draw residual significance map and 1D dis of convolutional excess
                if (ipmode==0 && cf.OutDrawOpt){
                    FitMode->MkOutdir();
                    FitMode->DrawSigMap(Template, ROI->Cellid, WCDAFit->WCDAData->Non, WCDAFit->WCDAData->Nbkg, KM2AFit->KM2AData->Non, KM2AFit->KM2AData->Nbkg, WCDAFit->Nmodel_convo, KM2AFit->Nmodel_convo, WCDAFit->WCDAResp->PSF, KM2AFit->KM2AResp->PSF);
                }

                // Output convolutional Excess map
                if (ipmode==0 && cf.fOut[1]!="none"){
                    FitMode->MkOutdir();
                    FitMode->OutConvExcess(Template, ROI->Cellid, WCDAFit->Nmodel_convo, KM2AFit->Nmodel_convo);
                }

                /*if (ipmode==0 && (cf.FitOpt[1] || cf.FitOpt[4])){
                    if (cf.UseWCDA)
                        WCDAFit->CalEmedian(FitMode->Emedian, ROI->Ycenter);
                    if (cf.UseKM2A){
                        KM2AFit->CalEmedian(0, FitMode->Emedian);
                        KM2AFit->CalEmedian(1, FitMode->Emedian);
                    }
                }*/
            }

            if (ipmode==0 && (cf.FitOpt[1] || cf.FitOpt[4])){
                if (cf.UseWCDA)
                    WCDAFit->CalEmedian(FitMode->Emedian, ROI->Ycenter);
                if (cf.UseKM2A){
                    KM2AFit->CalEmedian(0, FitMode->Emedian);
                    KM2AFit->CalEmedian(1, FitMode->Emedian);
                }
            }

        }

        // Cal median energy && flux point
        if (cf.FitOpt[1] || cf.FitOpt[4]){
            /*if (cf.UseWCDA)
                WCDAFit->CalEmedian(FitMode->Emedian, ROI->Ycenter);
            if (cf.UseKM2A){
                KM2AFit->CalEmedian(0, FitMode->Emedian);
                KM2AFit->CalEmedian(1, FitMode->Emedian);
            }*/
            FitMode->MkOutdir();
            if (cf.FitOpt[1])
                FitMode->CalFlux(1, Template);
            if (cf.FitOpt[4])
                FitMode->CalFlux(4, Template);
        }

        if (cf.FitOpt[1])
            FitMode->CalSEDChi2(Template);

        FitMode->PrintRes(Template, ROI->Neffbins);

        if (cf.FitOpt[1] || cf.FitOpt[4])
            FitMode->DrawSED(Template);

    }
    fmap->Close();
    cout<<" *** main : number of iteration is "<<Niter<<endl;    

    // TSmap
    if (cf.FitTSmap){
        string command = Form("mkdir -p %s/TSmap", cf.Outdir.data());
        system(command.data());
        ofstream out(cf.Outdir+"/TSmap/"+cf.JOBScript, ios::out);
        out<<"#!/bin/bash"<<endl;
        out<<"export EOS_MGM_URL=root://eos01.ihep.ac.cn/"<<endl;
        out<<"procid=$1"<<endl;
        out<<"segid=$[procid]"<<endl;
        out<<"WorkDir="<<cf.WorkDir<<endl;
        out<<"exeprog=Src_TSMap"<<endl;
        out<<"FitConfig="<<argv[1]<<endl;
        out<<"Outdir=$WorkDir/"<<cf.Outdir<<"/TSmap"<<endl;
        out<<"[ -d $Outdir ] || mkdir -p $Outdir"<<endl;
        out<<"$WorkDir/$exeprog $WorkDir/$FitConfig $segid $Outdir/TSmap_\"$segid\".root &> $Outdir/log_\"$segid\".txt"<<endl;
        command = Form("chmod 755 %s/TSmap/%s", cf.Outdir.data(), cf.JOBScript.data());
        system(command.data());

        int Njob = ROI->Neffbins/100+1;
        command = Form("hep_sub -g lhaaso -prio 99 %s/TSmap/%s -argu \"\%{ProcId}\" -n %d", cf.Outdir.data(), cf.JOBScript.data(), Njob);
        system(command.data());
    }

    return 0;
}
