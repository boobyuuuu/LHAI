# ifndef Src_Fitting_KM2A_h
# define Src_Fitting_KM2A_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "TMath.h"
# include "Src_Response_KM2A.h"
# include "Src_Data_KM2A.h"

using namespace std;

class Src_Fitting_KM2A {

    public :

        Src_Fitting_KM2A();
        ~Src_Fitting_KM2A();

        void SetBasicPar(double s0);
        void SetROI(Src_ROI *roi);
        void SetTemplate(Src_Template *temp);
        bool Init();
        void GetDisZen_Temp();
        void GeneZenDis(double ra, double dec, TH1D *hzen, int detconf);
        void GetTobs();
        void Convolute(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_num, double *par_dge);
        void Convolute_NotFull(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double **Nexcess_exp, double *par_num, double *par_dge);
        void Convolute_NumSrc(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_dge);
        void Convolute_NumSrc_NotFull(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double *par_src, double *par_dge);
        void Convolute_DGE(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_num);
        void Convolute_DGE_NotFull(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double *par_src, double *par_num);
        void CalLogNull(int ibinUsed0, int ibinUsed1);
        void CalLogSig(double *par, int nPar_src,  int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp, int ithispmode);

        void CalLogNull_1D(int ibinUsed0, int ibinUsed1);
        void CalLogSig_1D(double *par, int nPar_src,  int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp);

        Src_Response_KM2A *KM2AResp;
        Src_Data_KM2A     *KM2AData;
        Src_ROI           *ROI;
        Src_Template      *Template;

        double Mindec;
        double Maxdec;
        int Ndecstep;
        double S0;
        double *Tobs;
        double *Tobs_NotFull[2];
        double **DisZen;
        double **DisZen_NotFull[2];
        double **DisZen_Temp;
        double **DisZen_Temp_NotFull[2];
        double **Effzen;
        double **Effzen_NotFull[2];
        double *Omega_model;
        double *Omega;
        double Omega_total;
        double Omega_total_model;
        double *Eta_ROI;
        double *NTemp_total_model;
        vector<vector <double> > PSF;
        vector<vector <long int> > PSF_id;
        double **Nmodel_convo;
        double log_L_null;
        double log_L_sig;
        double *log_L_const;

        // Get flux point
        void CalEmedian(int imode, double **Energy);
        // Tools
        void CalExposure(int ibinUsed0, int ibinUsed1,  int ipixel, double **exposure);

};

Src_Fitting_KM2A::Src_Fitting_KM2A(){

    KM2AResp = new Src_Response_KM2A();
    KM2AData = new Src_Data_KM2A();

}

Src_Fitting_KM2A::~Src_Fitting_KM2A(){

    for (int ii=0;ii<Template->NComp;ii++)
        delete[] Nmodel_convo[ii];
    delete[] Nmodel_convo;

    for (int ii=0;ii<Ndecstep;ii++)
        delete[] DisZen_Temp;
    delete[] DisZen_Temp;

    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            for (int ii=0;ii<Ndecstep;ii++)
                delete[] DisZen_Temp_NotFull[idet][ii];
            delete[] DisZen_Temp_NotFull[idet];
        }
    }

    for (int ii=0;ii<Template->NSrc;ii++)
        delete[] DisZen;
    delete[] DisZen;

    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            for (int ii=0;ii<Template->NSrc;ii++)
                delete[] DisZen_NotFull[idet][ii];
            delete[] DisZen_NotFull[idet];
        }
    }

    delete[] Omega_model;
    delete[] Omega;
    delete[] Eta_ROI;
    delete[] NTemp_total_model; 

}

void Src_Fitting_KM2A::SetBasicPar(double s0){

    S0 = s0;

}

void Src_Fitting_KM2A::SetROI(Src_ROI *roi){ ROI = roi; }

void Src_Fitting_KM2A::SetTemplate(Src_Template *temp){ Template = temp; }

bool Src_Fitting_KM2A::Init(){

    log_L_null = 0;
    log_L_sig = 0;
    log_L_const = new double[cf.KNEbinUsed];
    for (int ii=0;ii<cf.KNEbinUsed;ii++)
        log_L_const[ii] = 0;

    cout<<" Fitting KM2A Init: Initializing array... "<<endl;
    Nmodel_convo = new double*[Template->NComp];
    DisZen = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NComp;ii++){
        Nmodel_convo[ii] = new double[cf.KNEbinUsed*ROI->Neffbins];
        for (int jj=0;jj<ROI->Neffbins*cf.KNEbinUsed;jj++)
            Nmodel_convo[ii][jj] = 0;
    }
    for (int ii=0;ii<Template->NSrc;ii++){
        DisZen[ii] = new double[cf.KNzenstep];
        for (int jj=0;jj<cf.KNzenstep;jj++)
            DisZen[ii][jj] = 0;
    }
    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            DisZen_NotFull[idet] = new double*[Template->NSrc];
            for (int ii=0;ii<Template->NSrc;ii++){
                DisZen_NotFull[idet][ii] = new double[cf.KNzenstep];
                for (int jj=0;jj<cf.KNzenstep;jj++)
                    DisZen_NotFull[idet][ii][jj] = 0;
            }
        }
    }

    if ((Template->NTemp+Template->NSrc_NumCon)>0){

        // Get dis. of zen
        cout<<" Fitting Init: Get dis. of Zenith angle ... "<<endl;
        Maxdec = -90;
        Mindec = 90;
        double ra, dec;
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            if (!cf.CorOpt)
                dec = yy;
            else
                g2e(xx, yy, &ra, &dec);

            if (dec>=Maxdec) Maxdec = dec;
            if (dec<=Mindec) Mindec = dec;
        }
        Maxdec = floor(Maxdec)+1;
        Mindec = floor(Mindec);
        if (Mindec<cf.Decrange[0] || Maxdec<cf.Decrange[0]){
            std::cerr<<" \033[31;1mError\033[0m : Fitting Init: Mindec<cf.Decrange[0] || Maxdec<cf.Decrange[0]! Returned."<<endl;
            return -1;
        }
        if (Mindec>cf.Decrange[1] || Maxdec>cf.Decrange[1]){
            std::cerr<<" \033[31;1mError\033[0m : Fitting Init: Mindec>cf.Decrange[1] || Maxdec>cf.Decrange[1]! Returned."<<endl;
            return -1;
        }
        Ndecstep = (Maxdec-Mindec)/cf.DECstep;
        cout<<" Mindec = "<<Mindec<<", Maxdec = "<<Maxdec<<", Ndecstep = "<<Ndecstep<<endl;
        DisZen_Temp = new double* [Ndecstep];
        for (int ii=0;ii<Ndecstep;ii++){
            DisZen_Temp[ii] = new double[cf.KNzenstep];
            for (int jj=0;jj<cf.KNzenstep;jj++)
                DisZen_Temp[ii][jj] = 0;
        }
        if (cf.UseKM2A_NotFull){
            for (int idet=0;idet<2;idet++){
                DisZen_Temp_NotFull[idet] = new double* [Ndecstep];
                for (int ii=0;ii<Ndecstep;ii++){
                    DisZen_Temp_NotFull[idet][ii] = new double[cf.KNzenstep];
                    for (int jj=0;jj<cf.KNzenstep;jj++)
                        DisZen_Temp_NotFull[idet][ii][jj] = 0;
                }
            }
        }
        GetDisZen_Temp();

        Effzen = new double*[cf.KNEbinUsed];
        for (int ii=0;ii<cf.KNEbinUsed;ii++){
            Effzen[ii] = new double[cf.KNEstep*Ndecstep];
            for (int jj=0;jj<cf.KNEstep;jj++){
                for (int kk=0;kk<Ndecstep;kk++){
                    Effzen[ii][jj*Ndecstep+kk] = 0;
                    for (int izen=0;izen<cf.KNzenstep;izen++){
                        double zen = (izen+0.5)*cf.KZenstep;
                        Effzen[ii][jj*Ndecstep+kk] += cos(zen*papi::degrad)*KM2AResp->hResp[ii][jj*cf.KNzenstep+izen]*DisZen_Temp[kk][izen];
                    }
                }
            }
        }
        if (cf.UseKM2A_NotFull){
            for (int idet=0;idet<2;idet++){
                Effzen_NotFull[idet] = new double*[cf.KNEbinUsed];
                for (int ii=0;ii<cf.KNEbinUsed;ii++){
                    Effzen_NotFull[idet][ii] = new double[cf.KNEstep*Ndecstep];
                    for (int jj=0;jj<cf.KNEstep;jj++){
                        for (int kk=0;kk<Ndecstep;kk++){
                            Effzen_NotFull[idet][ii][jj*Ndecstep+kk] = 0;
                            for (int izen=0;izen<cf.KNzenstep;izen++){
                                double zen = (izen+0.5)*cf.KZenstep;
                                Effzen_NotFull[idet][ii][jj*Ndecstep+kk] += cos(zen*papi::degrad)*KM2AResp->hResp_NotFull[idet][ii][jj*cf.KNzenstep+izen]*DisZen_Temp_NotFull[idet][kk][izen];
                            }
                        }
                    }
                }
            }
        }

        // Get PSF(i, j), here i is ith cell
        cout<<" Fitting Init: Calculate PSF(i, j) ... "<<endl;
        cout<<"   PSF(i, j) Norm : ["<<flush;
        double p1, sigma1, sigma2;
        double **psf_total = new double*[ROI->Neffbins_model];
        for (int ii=0;ii<ROI->Neffbins_model;ii++){

            if (ii%(ROI->Neffbins_model/100)==0){
                if (ii/(ROI->Neffbins_model/100)%10==0)
                    cout<<ii/(ROI->Neffbins_model/100)<<"%"<<flush;
                else
                    cout<<"="<<flush;
            }

            psf_total[ii] = new double[cf.KNEbinUsed];
            double x1 = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double y1 = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double ibinx0 = (x1-2/cos(y1*papi::degrad)-X[0])/wbinX; 
            double ibinx1 = (x1+2/cos(y1*papi::degrad)-X[0])/wbinX;
            double ibiny0 = (y1-2-Y[0])/wbinY;
            double ibiny1 = (y1+2-Y[0])/wbinY;

            if (!cf.CorOpt)
                dec = y1;
            else
                g2e(x1, y1, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

            for (int jj=0;jj<cf.KNEbinUsed;jj++){
                psf_total[ii][jj] = 0;

                if (cf.KPSFtype == "1Gaus")
                    sigma1 = KM2AResp->hPSF[jj][idecbin*2+1];
                if (cf.KPSFtype == "2Gaus"){
                    p1 = KM2AResp->hPSF[jj][idecbin*4+1];
                    sigma1 = KM2AResp->hPSF[jj][idecbin*4+2];
                    sigma2 = KM2AResp->hPSF[jj][idecbin*4+3];
                }

                for (int mm=ibinx0;mm<=ibinx1;mm++){
                    double x2 = X[0]+(mm+0.5)*wbinX;
                    for (int nn=ibiny0;nn<=ibiny1;nn++){
                        double y2 = Y[0]+(nn+0.5)*wbinY;
                        double space1 = distance(90-y1, x1, 90-(y2-0.001), x2-0.001);
                        double omega = (cos((Y[1]-y2-0.5*wbinY)*papi::degrad)-cos((Y[1]-y2+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;
                        if (space1<2.0){
                            if (cf.KPSFtype == "1Gaus")
                                psf_total[ii][jj] += 1/(2*3.141592654)/(sigma1*sigma1+pow(cf.KPSFOffset, 2))*exp(-space1*space1/2/(sigma1*sigma1+pow(cf.KPSFOffset, 2)))*omega;
                            if (cf.KPSFtype == "2Gaus")
                                psf_total[ii][jj] += (p1/(2*3.141592654)/(sigma1*sigma1+pow(cf.KPSFOffset, 2))*exp(-space1*space1/2/(sigma1*sigma1+pow(cf.KPSFOffset, 2)))+(1-p1)/(2*3.141592654)/(sigma2*sigma2+pow(cf.KPSFOffset, 2))*exp(-space1*space1/2/(sigma2*sigma2+pow(cf.KPSFOffset, 2))))*omega;
                        }
                    }
                }
            }
        }
        cout<<"]"<<endl;

        vector<double> psf_temp;
        vector<long int> psfid_temp;
        cout<<"   PSF(i, j) : ["<<flush;
        for (int ii=0;ii<ROI->Neffbins;ii++){

            if (ii%(ROI->Neffbins/100)==0){
                if (ii/(ROI->Neffbins/100)%10==0)
                    cout<<ii/(ROI->Neffbins/100)<<"%"<<flush;
                else
                    cout<<"="<<flush;
            }

            psf_temp.clear();
            psfid_temp.clear();
            double x0 = X[0]+((ROI->Cellid[ii]/nbinsY)+0.5)*wbinX;
            double y0 = Y[0]+((ROI->Cellid[ii]%nbinsY)+0.5)*wbinY;
            double omega = (cos((Y[1]-y0-0.5*wbinY)*papi::degrad)-cos((Y[1]-y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

            for (int jj=0;jj<ROI->Neffbins_model;jj++){
                double x1 = X[0]+((ROI->Cellid_model[jj]/nbinsY)+0.5)*wbinX;
                double y1 = Y[0]+((ROI->Cellid_model[jj]%nbinsY)+0.5)*wbinY;
                double space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                if (space<2.0){

                    if (!cf.CorOpt)
                        dec = y1;
                    else
                        g2e(x1, y1, &ra, &dec);

                    int idecbin = (dec-cf.Decrange[0])/cf.Decstep;
                    psfid_temp.push_back(jj);
                    for (int inhit=0;inhit<cf.KNEbinUsed;inhit++){
                        double psf_ij;
                        if (cf.KPSFtype == "1Gaus"){
                            sigma1 = KM2AResp->hPSF[inhit][idecbin*2+1];
                            psf_ij = 1/(2*3.141592654)/(sigma1*sigma1+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+pow(cf.KPSFOffset, 2)))*omega;
                        }
                        if (cf.KPSFtype == "2Gaus"){
                            p1 = KM2AResp->hPSF[inhit][idecbin*4+1];
                            sigma1 = KM2AResp->hPSF[inhit][idecbin*4+2];
                            sigma2 = KM2AResp->hPSF[inhit][idecbin*4+3];
                            psf_ij = (p1/(2*3.141592654)/(sigma1*sigma1+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+pow(cf.KPSFOffset, 2)))+(1-p1)/(2*3.141592654)/(sigma2*sigma2+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma2*sigma2+pow(cf.KPSFOffset, 2))))*omega;
                        }
                        psf_temp.push_back(psf_ij/psf_total[jj][inhit]);
                    }
                }
            }

            PSF.push_back(psf_temp);
            PSF_id.push_back(psfid_temp);

        }
        cout<<"]"<<endl;
        psf_temp.clear();
        psf_temp.shrink_to_fit();
        psfid_temp.clear();
        psfid_temp.shrink_to_fit();

        for (int ii=0;ii<ROI->Neffbins_model;ii++)
            delete[] psf_total[ii];
        delete[] psf_total;


        cout<<" Fitting Init: calculate solid angle of each cell ... "<<endl;
        Omega_model = new double[ROI->Neffbins_model];
        Omega_total_model = 0;
        for (int jj=0;jj<ROI->Neffbins_model;jj++){
            double y0 = Y[1]-(Y[0]+((ROI->Cellid_model[jj]%nbinsY)+0.5)*wbinY);
            Omega_model[jj] = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;
            Omega_total_model += Omega_model[jj];
        }
        Omega = new double[ROI->Neffbins];
        Omega_total = 0;
        for (int jj=0;jj<ROI->Neffbins;jj++){
            double y0 = Y[1]-(Y[0]+((ROI->Cellid[jj]%nbinsY)+0.5)*wbinY);
            Omega[jj] = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;
            Omega_total += Omega[jj];
        }

        if (Template->NTemp>0){
            Eta_ROI = new double[Template->NTemp];
            NTemp_total_model = new double[Template->NTemp];
            for (int ii=0;ii<Template->NTemp;ii++){
                Eta_ROI[ii] = 0;
                NTemp_total_model[ii] = 0;

                double NTemp_total = 0;
                for (int jj=0;jj<ROI->Neffbins_model;jj++){

                    if (ii<Template->NSrc_Temp)
                        NTemp_total_model[ii] += Template->Srcs_Temp[ii].NTemp_model[jj]*Omega_model[jj];
                    else
                        NTemp_total_model[ii] += Template->DGEs[ii-Template->NSrc_Temp].NTemp_model[jj]*Omega_model[jj];
                }

                for (int jj=0;jj<ROI->Neffbins;jj++){

                    if (ii<Template->NSrc_Temp)
                        NTemp_total += Template->Srcs_Temp[ii].NTemp[jj]*Omega[jj];
                    else
                        NTemp_total += Template->DGEs[ii-Template->NSrc_Temp].NTemp[jj]*Omega[jj];

                }

                Eta_ROI[ii] = NTemp_total/NTemp_total_model[ii];
                cout<<"Omega_total = "<<Omega_total<<", Omega_total_model = "<<Omega_total_model<<", Eta = "<<Eta_ROI[ii]<<endl;

            }
        }
    }

    return 0;

}

void Src_Fitting_KM2A::GetTobs(){

    Tobs = new double[Template->NSrc];
    for (int isrc=0;isrc<Template->NSrc;isrc++)
        Tobs[isrc] = 0;
    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++){
            Tobs_NotFull[idet] = new double[Template->NSrc];
            for (int isrc=0;isrc<Template->NSrc;isrc++)
                Tobs_NotFull[idet][isrc] = 0;
        }
    }

    double ra, dec;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        double tobs = 0;
        if (!cf.CorOpt){
            ra  = Template->Srcs[isrc].Ra[0];
            dec = Template->Srcs[isrc].Dec[0];
        }
        else
            g2e(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], &ra, &dec);

        KM2AData->GetTobs(ra, dec, DisZen[isrc], tobs);
        Tobs[isrc] = tobs;
        if (cf.UseKM2A_NotFull){
            for (int idet=0;idet<2;idet++){
                double tobs = 0;
                KM2AData->GetTobs_NotFull(ra, dec, DisZen_NotFull[idet][isrc], tobs, idet);
                Tobs_NotFull[idet][isrc] = tobs;
            }
        }                                                                                                             
    }

    if ((Template->NTemp+Template->NSrc_NumCon)>0){
        KM2AData->GetTobsMap(ROI->Cellid_model);
        if (cf.UseKM2A_NotFull){
            for (int idet=0;idet<2;idet++)
                KM2AData->GetTobsMap_NotFull(ROI->Cellid_model, idet);
        }
    }

}

void Src_Fitting_KM2A::Convolute(Double_t *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_num, double *par_dge){

    if (isnan(par[2]))
        par[2] = 0.01;

    double **nexcess_exp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed];
        for (int jj=0;jj<cf.KNEbinUsed;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double **nexcess_temp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_temp[ii] = new double[cf.KNEbinUsed];
        for (int jj=0;jj<cf.KNEbinUsed;jj++)
            nexcess_temp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;

    // SED X detection efficiency
    int npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){

        if (isrc==ithisComp) continue;
        if (Template->Srcs[isrc].ConvoFlag == 0) {
            npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
            continue; 
        }

        TF1 *fSED = new TF1("fSED", Template->Srcs[isrc].SEDFormula.data(), 0.01, 1000);
        fSED->SetParameter(0, par[2+npar]);
        if (!Template->Srcs[isrc].LinkPars){
            for (int ipar=1;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[2+ipar+npar]);
        }
        else{
            int targetsrcid = Template->Srcs[isrc].TargetSrcID_Class;
            int targetsrclass = Template->Srcs[isrc].TargetSrcClass;
            int ipar_temp = 0;
            if (targetsrclass == 0){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[2+ipar_temp+ipar]);
            }
            else if (targetsrclass == 1){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_num[2+ipar_temp+ipar]);
            }
            else{
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_dge[ipar_temp+ipar]);
            }
        }

        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1);
        }

        // Calculate Nexcess_exp of each Source
        for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
            for (int iE=0;iE<cf.KNEstep;iE++){
                double e0, e1;
                e0 = TMath::Power(10, cf.KErange[0]+iE*cf.KEstep);
                e1 = TMath::Power(10, cf.KErange[0]+(iE+1)*cf.KEstep);

                double tau = 0;
                if (Template->Srcs[isrc].GGAbsFlag){
                    if ((e0+e1)/2<Template->Srcs[isrc].ebl_Emin)
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                    else if ((e0+e1)/2>Template->Srcs[isrc].ebl_Emin && (e0+e1)/2<Template->Srcs[isrc].ebl_Emax)
                        tau = Template->Srcs[isrc].gg_ebl->Eval((e0+e1)/2);
                    else
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                }

                for (int izen=0;izen<cf.KNzenstep;izen++){
                    double zen = (izen+0.5)*cf.KZenstep;
                    if (KM2AResp->hResp[inhit][iE*cf.KNzenstep+izen]<=0) continue;
                    if (DisZen[isrc][izen]<=0) continue;
                    nexcess_exp[isrc][inhit] += Flux[iE]*cos(zen*TMath::DegToRad())*KM2AResp->hResp[inhit][iE*cf.KNzenstep+izen]*DisZen[isrc][izen]*exp(-tau);
                }
            }
            nexcess_exp[isrc][inhit] = nexcess_exp[isrc][inhit]*S0*Tobs[isrc];
        }

        npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;

        delete fSED;

    }

    if (cf.UseKM2A_NotFull){
        Convolute_NotFull(par, ibinUsed0, ibinUsed1, ithisComp, 0, nexcess_exp, par_num, par_dge);
        Convolute_NotFull(par, ibinUsed0, ibinUsed1, ithisComp, 1, nexcess_exp, par_num, par_dge);
    }

    double ra, dec;
    double p1, sigma1, sigma2;
    // Morphology model X PSF
    for (int ii=0;ii<ROI->Neffbins_model;ii++){

        double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
        double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
        double omega = (cos((Y[1]-yy-0.5*wbinY)*papi::degrad)-cos((Y[1]-yy+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

        npar = 0;
        for (int isrc=0;isrc<Template->NSrc;isrc++){

            if (isrc==ithisComp) continue;
            if (Template->Srcs[isrc].ConvoFlag == 0) {
                npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
                continue;
            }

            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (!cf.CorOpt)
                dec = par[npar+1];
            else
                g2e(par[npar], par[npar+1], &ra, &dec);
            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;;

            npar = npar+2+Template->Srcs[isrc].nSEDpar;
            double sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus")
                sigma3 = par[npar];
            if (Template->Srcs[isrc].Mortype == "Point")
                sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus_E") 
                sigma3 = par[npar+1];

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (cf.KPSFtype == "1Gaus"){
                    sigma1 = KM2AResp->hPSF[inhit][idecbin*2+1];
                    nexcess_temp[isrc][inhit] += 1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                }
                if (cf.KPSFtype == "2Gaus"){
                    p1 = KM2AResp->hPSF[inhit][idecbin*4+1];
                    sigma1 = KM2AResp->hPSF[inhit][idecbin*4+2];
                    sigma2 = KM2AResp->hPSF[inhit][idecbin*4+3];
                    nexcess_temp[isrc][inhit] += p1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                    nexcess_temp[isrc][inhit] += (1-p1)/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                }
            }
            npar += Template->Srcs[isrc].nMorpar;
        }

    }

    for (int ii=0;ii<ROI->Neffbins;ii++){

        double xx = X[0]+((ROI->Cellid[ii]/nbinsY)+0.5)*wbinX;
        double yy = Y[0]+((ROI->Cellid[ii]%nbinsY)+0.5)*wbinY;
        double omega = (cos((Y[1]-yy-0.5*wbinY)*papi::degrad)-cos((Y[1]-yy+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

        npar = 0;
        for (int isrc=0;isrc<Template->NSrc;isrc++){

            if (isrc==ithisComp) continue;
            if (Template->Srcs[isrc].ConvoFlag == 0) {
                npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
                continue;
            }

            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (!cf.CorOpt)
                dec = par[npar+1];
            else
                g2e(par[npar], par[npar+1], &ra, &dec);
            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;;

            npar = npar+2+Template->Srcs[isrc].nSEDpar;
            double sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus")
                sigma3 = par[npar];
            if (Template->Srcs[isrc].Mortype == "Point")
                sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus_E") 
                sigma3 = par[npar+1];

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (cf.KPSFtype == "1Gaus"){
                    sigma1 = KM2AResp->hPSF[inhit][idecbin*2+1];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] = 1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                }
                if (cf.KPSFtype == "2Gaus"){
                    p1 = KM2AResp->hPSF[inhit][idecbin*4+1];
                    sigma1 = KM2AResp->hPSF[inhit][idecbin*4+2];
                    sigma2 = KM2AResp->hPSF[inhit][idecbin*4+3];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] = p1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] += (1-p1)/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                }
            }

            npar += Template->Srcs[isrc].nMorpar;
        }

    }

    delete[] Flux;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        delete[] nexcess_exp[isrc];
        delete[] nexcess_temp[isrc];
    }
    delete[] nexcess_exp;
    delete[] nexcess_temp;

}

void Src_Fitting_KM2A::Convolute_NotFull(Double_t *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double **Nexcess_exp, double *par_num, double *par_dge){

    if (isnan(par[2]))
        par[2] = 0.01;

    double **nexcess_exp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed];
        for (int jj=0;jj<cf.KNEbinUsed;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double **nexcess_temp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_temp[ii] = new double[cf.KNEbinUsed];
        for (int jj=0;jj<cf.KNEbinUsed;jj++)
            nexcess_temp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;

    // SED X detection efficiency
    int npar = 0;
    for (int isrc=0;isrc<Template->NSrc;isrc++){

        if (isrc==ithisComp) continue;
        if (Template->Srcs[isrc].ConvoFlag == 0) {
            npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
            continue; 
        }

        TF1 *fSED = new TF1("fSED", Template->Srcs[isrc].SEDFormula.data(), 0.01, 1000);
        fSED->SetParameter(0, par[2+npar]);
        if (!Template->Srcs[isrc].LinkPars){
            for (int ipar=1;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[2+ipar+npar]);
        }
        else{
            int targetsrcid = Template->Srcs[isrc].TargetSrcID_Class;
            int targetsrclass = Template->Srcs[isrc].TargetSrcClass;
            int ipar_temp = 0;
            if (targetsrclass == 0){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[2+ipar_temp+ipar]);
            }
            else if (targetsrclass == 1){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_num[2+ipar_temp+ipar]);
            }
            else{
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_dge[ipar_temp+ipar]);
            }
        }


        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1);
        }

        // Calculate Nexcess_exp of each Source
        for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
            for (int iE=0;iE<cf.KNEstep;iE++){
                double e0, e1;
                e0 = TMath::Power(10, cf.KErange[0]+iE*cf.KEstep);
                e1 = TMath::Power(10, cf.KErange[0]+(iE+1)*cf.KEstep);

                double tau = 0;
                if (Template->Srcs[isrc].GGAbsFlag){
                    if ((e0+e1)/2<Template->Srcs[isrc].ebl_Emin)
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                    else if ((e0+e1)/2>Template->Srcs[isrc].ebl_Emin && (e0+e1)/2<Template->Srcs[isrc].ebl_Emax)
                        tau = Template->Srcs[isrc].gg_ebl->Eval((e0+e1)/2);
                    else
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                }

                for (int izen=0;izen<cf.KNzenstep;izen++){
                    double zen = (izen+0.5)*cf.KZenstep;
                    if (KM2AResp->hResp_NotFull[detconf][inhit][iE*cf.KNzenstep+izen]<=0) continue;
                    if (DisZen_NotFull[detconf][isrc][izen]<=0) continue;
                    nexcess_exp[isrc][inhit] += Flux[iE]*cos(zen*TMath::DegToRad())*KM2AResp->hResp_NotFull[detconf][inhit][iE*cf.KNzenstep+izen]*DisZen_NotFull[detconf][isrc][izen]*exp(-tau);
                }
            }
            nexcess_exp[isrc][inhit] = nexcess_exp[isrc][inhit]*S0*Tobs_NotFull[detconf][isrc];

            Nexcess_exp[isrc][inhit] += nexcess_exp[isrc][inhit];

        }

        npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;

        delete fSED;

    }

    //double ra, dec;
    //double p1, sigma1, sigma2;
    // Morphology model X PSF
    /*for (int ii=0;ii<ROI->Neffbins_model;ii++){

        double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
        double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
        double omega = (cos((Y[1]-yy-0.5*wbinY)*papi::degrad)-cos((Y[1]-yy+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

        npar = 0;
        for (int isrc=0;isrc<Template->NSrc;isrc++){

            if (isrc==ithisComp) continue;
            if (Template->Srcs[isrc].ConvoFlag == 0) {
                npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
                continue;
            }

            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (!cf.CorOpt)
                dec = par[npar+1];
            else
                g2e(par[npar], par[npar+1], &ra, &dec);
            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;;

            npar = npar+2+Template->Srcs[isrc].nSEDpar;
            double sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus")
                sigma3 = par[npar];
            if (Template->Srcs[isrc].Mortype == "Point")
                sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus_E") 
                sigma3 = par[npar+1];

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (cf.KPSFtype == "1Gaus"){
                    sigma1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*2+1];
                    nexcess_temp[isrc][inhit] += 1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                }
                if (cf.KPSFtype == "2Gaus"){
                    p1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+1];
                    sigma1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+2];
                    sigma2 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+3];
                    nexcess_temp[isrc][inhit] += p1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                    nexcess_temp[isrc][inhit] += (1-p1)/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega;
                }
            }
            npar += Template->Srcs[isrc].nMorpar;
        }

    }

    for (int ii=0;ii<ROI->Neffbins;ii++){

        double xx = X[0]+((ROI->Cellid[ii]/nbinsY)+0.5)*wbinX;
        double yy = Y[0]+((ROI->Cellid[ii]%nbinsY)+0.5)*wbinY;
        double omega = (cos((Y[1]-yy-0.5*wbinY)*papi::degrad)-cos((Y[1]-yy+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

        npar = 0;
        for (int isrc=0;isrc<Template->NSrc;isrc++){

            if (isrc==ithisComp) continue;
            if (Template->Srcs[isrc].ConvoFlag == 0) {
                npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;
                continue;
            }

            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (!cf.CorOpt)
                dec = par[npar+1];
            else
                g2e(par[npar], par[npar+1], &ra, &dec);
            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;;

            npar = npar+2+Template->Srcs[isrc].nSEDpar;
            double sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus")
                sigma3 = par[npar];
            if (Template->Srcs[isrc].Mortype == "Point")
                sigma3 = 0;
            if (Template->Srcs[isrc].Mortype == "Ext_gaus_E") 
                sigma3 = par[npar+1];

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (cf.KPSFtype == "1Gaus"){
                    sigma1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*2+1];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] += 1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                }
                if (cf.KPSFtype == "2Gaus"){
                    p1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+1];
                    sigma1 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+2];
                    sigma2 = KM2AResp->hPSF_NotFull[detconf][inhit][idecbin*4+3];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] += p1/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] += (1-p1)/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2))*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3+pow(cf.KPSFOffset, 2)))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                }
            }

            npar += Template->Srcs[isrc].nMorpar;
        }

    }*/

    delete[] Flux;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        delete[] nexcess_exp[isrc];
        delete[] nexcess_temp[isrc];
    }
    delete[] nexcess_exp;
    delete[] nexcess_temp;

}

void Src_Fitting_KM2A::Convolute_NumSrc(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_dge){

    double **nexcess_exp = new double*[Template->NSrc_NumCon];
    for (int ii=0;ii<Template->NSrc_NumCon;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.KNEbinUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;

    int npar = 0;
    for (int idge=0;idge<Template->NSrc_NumCon;idge++){

        if (idge==(ithisComp-Template->NSrc)) continue;
        if (Template->Srcs_NumCon[idge].ConvoFlag==0) {
            npar += Template->Srcs_NumCon[idge].nMorpar + Template->Srcs_NumCon[idge].nSEDpar + 2;
            continue;
        }

        // SED X detection efficiency
        TF1 *fSED = new TF1("fSED", Template->Srcs_NumCon[idge].SEDFormula.data(), 0.01, 1000);
        fSED->SetParameter(0, par[npar+2]);       
        if (!Template->Srcs_NumCon[idge].LinkPars){
            for (int ipar=1;ipar<Template->Srcs_NumCon[idge].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[ipar+npar+2]);
        }
        else{
            int targetsrcid = Template->Srcs_NumCon[idge].TargetSrcID_Class;
            int targetsrclass = Template->Srcs_NumCon[idge].TargetSrcClass;
            int ipar_temp = 0;
            if (targetsrclass == 0){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_src[2+ipar_temp+ipar]);
            }
            else if (targetsrclass == 1){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[2+ipar_temp+ipar]);
            }
            else{
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_dge[ipar_temp+ipar]);
            }
        }

        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1);
        }

        double ra, dec;
        for (int isrc=0;isrc<ROI->Neffbins_model;isrc++){
            // Calculate Nexcess_exp of each cell

            double xx = X[0]+((ROI->Cellid_model[isrc]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[isrc]%nbinsY)+0.5)*wbinY;

            if (!cf.CorOpt)
                dec = yy;
            else
                g2e(xx, yy, &ra, &dec);
            if (dec>=Maxdec) continue;

            int idecbin = (dec-Mindec)/cf.DECstep;
            //cout<<"idecbin = "<<idecbin<<endl;

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                for (int iE=0;iE<cf.KNEstep;iE++)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen[inhit][iE*Ndecstep+idecbin]*S0*KM2AData->Tobs[isrc];
            }
        }

        int imodel = Template->Model->MorMap[Template->Srcs_NumCon[idge].Mortype]-1;
        int ndim = Template->Model->MorNDim[imodel];
        TF1 *fMor;
        TF2 *fMor2D;
        if (ndim==1) 
            fMor = new TF1("fMor", Template->Srcs_NumCon[idge].MorFormula.data(), 0., 100.);
        if (ndim==2)
            fMor2D = new TF2("fMor2D", Template->Srcs_NumCon[idge].MorFormula.data(), 0., 100., -TMath::Pi(), TMath::Pi());
        if (Template->Srcs_NumCon[idge].nMorpar==2){
            for (int ipar=0;ipar<Template->Srcs_NumCon[idge].nMorpar;ipar++){
                if (ndim==1)
                    fMor->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
                if (ndim==2)
                    fMor2D->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
            }
        }
        else{
            for (int ipar=0;ipar<Template->Srcs_NumCon[idge].nMorpar;ipar++){
                if (ndim==1)
                    fMor->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
                if (ndim==2)
                    fMor2D->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
            }
        }

        // Morphology model X PSF
        double ntotal_mor = 0;
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (ndim==1)
                ntotal_mor += fMor->Eval(space)*Omega_model[ii];
            if (ndim==2){
                double angle = position(par[npar], par[npar+1], xx, yy);
                ntotal_mor += fMor2D->Eval(space, angle/180.*TMath::Pi())*Omega_model[ii];
            }
        }
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            double angle = 0;
            if (ndim==2)
                angle = position(par[npar], par[npar+1], xx, yy);
            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (ndim==1)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+ii] = nexcess_exp[idge][inhit*ROI->Neffbins_model+ii]*fMor->Eval(space)*Omega_model[ii]/ntotal_mor;
                if (ndim==2)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+ii] = nexcess_exp[idge][inhit*ROI->Neffbins_model+ii]*fMor2D->Eval(space, angle/180.*TMath::Pi())*Omega_model[ii]/ntotal_mor;
            }
        }

        for (int ii=0;ii<cf.KNEbinUsed*ROI->Neffbins;ii++)
            Nmodel_convo[Template->NSrc+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.KNEbinUsed+jj]; 
                }
            }
        }

        npar += Template->Srcs_NumCon[idge].nMorpar + Template->Srcs_NumCon[idge].nSEDpar + 2;

        delete fSED;
        if (ndim==1) delete fMor;
        if (ndim==2) delete fMor2D;

    }

    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++)
        delete[] nexcess_exp[isrc];
    delete[] nexcess_exp;
    delete[] Flux;


}

void Src_Fitting_KM2A::Convolute_NumSrc_NotFull(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double *par_src, double *par_dge){

    double **nexcess_exp = new double*[Template->NSrc_NumCon];
    for (int ii=0;ii<Template->NSrc_NumCon;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.KNEbinUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;

    int npar = 0;
    for (int idge=0;idge<Template->NSrc_NumCon;idge++){

        if (idge==(ithisComp-Template->NSrc)) continue;
        if (Template->Srcs_NumCon[idge].ConvoFlag==0) {
            npar += Template->Srcs_NumCon[idge].nMorpar + Template->Srcs_NumCon[idge].nSEDpar + 2;
            continue;
        }

        // SED X detection efficiency
        TF1 *fSED = new TF1("fSED", Template->Srcs_NumCon[idge].SEDFormula.data(), 0.01, 1000);
        fSED->SetParameter(0, par[npar+2]);       
        if (!Template->Srcs_NumCon[idge].LinkPars){
            for (int ipar=1;ipar<Template->Srcs_NumCon[idge].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[ipar+npar+2]);
        }
        else{
            int targetsrcid = Template->Srcs_NumCon[idge].TargetSrcID_Class;
            int targetsrclass = Template->Srcs_NumCon[idge].TargetSrcClass;
            int ipar_temp = 0;
            if (targetsrclass == 0){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_src[2+ipar_temp+ipar]);
            }
            else if (targetsrclass == 1){
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[2+ipar_temp+ipar]);
            }
            else{
                for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                    ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par_dge[ipar_temp+ipar]);
            }
        }

        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1);
        }

        double ra, dec;
        for (int isrc=0;isrc<ROI->Neffbins_model;isrc++){
            // Calculate Nexcess_exp of each cell

            double xx = X[0]+((ROI->Cellid_model[isrc]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[isrc]%nbinsY)+0.5)*wbinY;

            if (!cf.CorOpt)
                dec = yy;
            else
                g2e(xx, yy, &ra, &dec);
            if (dec>=Maxdec) continue;

            int idecbin = (dec-Mindec)/cf.DECstep;
            //cout<<"idecbin = "<<idecbin<<endl;

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                for (int iE=0;iE<cf.KNEstep;iE++)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen_NotFull[detconf][inhit][iE*Ndecstep+idecbin]*S0*KM2AData->Tobs_NotFull[detconf][isrc];
            }
        }

        int imodel = Template->Model->MorMap[Template->Srcs_NumCon[idge].Mortype]-1;
        int ndim = Template->Model->MorNDim[imodel];
        TF1 *fMor;
        TF2 *fMor2D;
        if (ndim==1) 
            fMor = new TF1("fMor", Template->Srcs_NumCon[idge].MorFormula.data(), 0., 100.);
        if (ndim==2)
            fMor2D = new TF2("fMor2D", Template->Srcs_NumCon[idge].MorFormula.data(), 0., 100., -TMath::Pi(), TMath::Pi());
        if (Template->Srcs_NumCon[idge].nMorpar==2){
            for (int ipar=0;ipar<Template->Srcs_NumCon[idge].nMorpar;ipar++){
                if (ndim==1)
                    fMor->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
                if (ndim==2)
                    fMor2D->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
            }
        }
        else{
            for (int ipar=0;ipar<Template->Srcs_NumCon[idge].nMorpar;ipar++){
                if (ndim==1)
                    fMor->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
                if (ndim==2)
                    fMor2D->SetParameter(ipar, par[ipar+npar+2+Template->Srcs_NumCon[idge].nSEDpar]);
            }
        }

        // Morphology model X PSF
        double ntotal_mor = 0;
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            if (ndim==1)
                ntotal_mor += fMor->Eval(space)*Omega_model[ii];
            if (ndim==2){
                double angle = position(par[npar], par[npar+1], xx, yy);
                ntotal_mor += fMor2D->Eval(space, angle/180.*TMath::Pi())*Omega_model[ii];
            }
        }
        for (int ii=0;ii<ROI->Neffbins_model;ii++){
            double xx = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double space = distance(90-yy, xx, 90-par[npar+1], par[npar]);
            double angle = 0;
            if (ndim==2)
                angle = position(par[npar], par[npar+1], xx, yy);
            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                if (ndim==1)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+ii] = nexcess_exp[idge][inhit*ROI->Neffbins_model+ii]*fMor->Eval(space)*Omega_model[ii]/ntotal_mor;
                if (ndim==2)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+ii] = nexcess_exp[idge][inhit*ROI->Neffbins_model+ii]*fMor2D->Eval(space, angle/180.*TMath::Pi())*Omega_model[ii]/ntotal_mor;
            }
        }

        //for (int ii=0;ii<cf.KNEbinUsed*ROI->Neffbins;ii++)
        //    Nmodel_convo[Template->NSrc+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.KNEbinUsed+jj]; 
                }
            }
        }

        npar += Template->Srcs_NumCon[idge].nMorpar + Template->Srcs_NumCon[idge].nSEDpar + 2;

        delete fSED;
        if (ndim==1) delete fMor;
        if (ndim==2) delete fMor2D;

    }

    for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++)
        delete[] nexcess_exp[isrc];
    delete[] nexcess_exp;
    delete[] Flux;


}


void Src_Fitting_KM2A::Convolute_DGE(Double_t *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_num){

    double **nexcess_exp = new double*[Template->NTemp];
    for (int ii=0;ii<Template->NTemp;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.KNEbinUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;


    int npar = 0;
    for (int idge=0;idge<Template->NTemp;idge++){

        if (idge==(ithisComp-Template->NSrc-Template->NSrc_NumCon)) continue;
        if (idge<Template->NSrc_Temp){
            if (Template->Srcs_Temp[idge].ConvoFlag==0){
                npar += Template->Srcs_Temp[idge].nSEDpar;
                continue;
            }
        }
        else{
            if (Template->DGEs[idge-Template->NSrc_Temp].ConvoFlag==0){
                npar += Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;
                continue;
            }
        }


        // SED X detection efficiency
        TF1 *fSED;
        if (idge<Template->NSrc_Temp){
            fSED = new TF1("fSED", Template->Srcs_Temp[idge].SEDFormula.data(), 0.01, 1000);
            fSED->SetParameter(0, par[npar]);
            if (!Template->Srcs_Temp[idge].LinkPars){
                for (int ipar=1;ipar<Template->Srcs_Temp[idge].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[ipar+npar]);
            }
            else{
                int targetsrcid = Template->Srcs_Temp[idge].TargetSrcID_Class;
                int targetsrclass = Template->Srcs_Temp[idge].TargetSrcClass;
                int ipar_temp = 0;
                if (targetsrclass == 0){
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                    for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par_src[2+ipar_temp+ipar]);
                }
                else if (targetsrclass == 1){
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                    for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par_num[2+ipar_temp+ipar]);
                }
                else{
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                    for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par[ipar_temp+ipar]);
                }
            }

        }
        else{
            fSED = new TF1("fSED", Template->DGEs[idge-Template->NSrc_Temp].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[ipar+npar]);
        }

        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1)*Omega_total_model;
        }

        double ra, dec;
        for (int isrc=0;isrc<ROI->Neffbins_model;isrc++){
            // Calculate Nexcess_exp of each cell

            double xx = X[0]+((ROI->Cellid_model[isrc]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[isrc]%nbinsY)+0.5)*wbinY;

            if (!cf.CorOpt)
                dec = yy;
            else
                g2e(xx, yy, &ra, &dec);

            if (dec>=Maxdec) continue;
            int idecbin = (dec-Mindec)/cf.DECstep;

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                for (int iE=0;iE<cf.KNEstep;iE++){
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen[inhit][iE*Ndecstep+idecbin];
                }
                if (idge<Template->NSrc_Temp)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*KM2AData->Tobs[isrc]*Template->Srcs_Temp[idge].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
                else
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*KM2AData->Tobs[isrc]*Template->DGEs[idge-Template->NSrc_Temp].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
            }
        }

        // Morphology model X PSF
        for (int ii=0;ii<cf.KNEbinUsed*ROI->Neffbins;ii++)
            Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.KNEbinUsed+jj]; 
                }
            }
        }

        if (idge<Template->NSrc_Temp)
            npar += Template->Srcs_Temp[idge].nSEDpar;
        else
            npar += Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;

        if (fSED) delete fSED;

    } 

    for (int isrc=0;isrc<Template->NTemp;isrc++)
        delete[] nexcess_exp[isrc];
    delete[] nexcess_exp;
    delete[] Flux;

}

void Src_Fitting_KM2A::Convolute_DGE_NotFull(Double_t *par, int ibinUsed0, int ibinUsed1, int ithisComp, int detconf, double *par_src, double *par_num){

    double **nexcess_exp = new double*[Template->NTemp];
    for (int ii=0;ii<Template->NTemp;ii++){
        nexcess_exp[ii] = new double[cf.KNEbinUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.KNEbinUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.KNEstep];
    for (int iE=0;iE<cf.KNEstep;iE++)
        Flux[iE] = 0;


    int npar = 0;
    for (int idge=0;idge<Template->NTemp;idge++){

        if (idge==(ithisComp-Template->NSrc-Template->NSrc_NumCon)) continue;
        if (idge<Template->NSrc_Temp){
            if (Template->Srcs_Temp[idge].ConvoFlag==0){
                npar += Template->Srcs_Temp[idge].nSEDpar;
                continue;
            }
        }
        else{
            if (Template->DGEs[idge-Template->NSrc_Temp].ConvoFlag==0){
                npar += Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;
                continue;
            }
        }


        // SED X detection efficiency
        TF1 *fSED;
        if (idge<Template->NSrc_Temp){
            fSED = new TF1("fSED", Template->Srcs_Temp[idge].SEDFormula.data(), 0.01, 1000);
            fSED->SetParameter(0, par[npar]);
            if (!Template->Srcs_Temp[idge].LinkPars){
                for (int ipar=1;ipar<Template->Srcs_Temp[idge].nSEDpar;ipar++)
                    fSED->SetParameter(ipar, par[ipar+npar]);
            }
            else{
                int targetsrcid = Template->Srcs_Temp[idge].TargetSrcID_Class;
                int targetsrclass = Template->Srcs_Temp[idge].TargetSrcClass;
                int ipar_temp = 0;
                if (targetsrclass == 0){
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += 2+Template->Srcs[isrc_temp].nSEDpar+Template->Srcs[isrc_temp].nMorpar;

                    for (int ipar=1;ipar<Template->Srcs[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par_src[2+ipar_temp+ipar]);
                }
                else if (targetsrclass == 1){
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += 2+Template->Srcs_NumCon[isrc_temp].nSEDpar+Template->Srcs_NumCon[isrc_temp].nMorpar;

                    for (int ipar=1;ipar<Template->Srcs_NumCon[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par_num[2+ipar_temp+ipar]);
                }
                else{
                    for (int isrc_temp=0;isrc_temp<targetsrcid;isrc_temp++)
                        ipar_temp += Template->Srcs_Temp[isrc_temp].nSEDpar;

                    for (int ipar=1;ipar<Template->Srcs_Temp[targetsrcid].nSEDpar;ipar++)
                        fSED->SetParameter(ipar, par[ipar_temp+ipar]);
                }
            }

        }
        else{
            fSED = new TF1("fSED", Template->DGEs[idge-Template->NSrc_Temp].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;ipar++)
                fSED->SetParameter(ipar, par[ipar+npar]);
        }

        for (int iE=0;iE<cf.KNEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.KErange[0]+iE*cf.KEstep);
            e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep);
            Flux[iE] = fSED->Integral(e0, e1)*Omega_total_model;
        }

        double ra, dec;
        for (int isrc=0;isrc<ROI->Neffbins_model;isrc++){
            // Calculate Nexcess_exp of each cell

            double xx = X[0]+((ROI->Cellid_model[isrc]/nbinsY)+0.5)*wbinX;
            double yy = Y[0]+((ROI->Cellid_model[isrc]%nbinsY)+0.5)*wbinY;

            if (!cf.CorOpt)
                dec = yy;
            else
                g2e(xx, yy, &ra, &dec);

            if (dec>=Maxdec) continue;
            int idecbin = (dec-Mindec)/cf.DECstep;

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                for (int iE=0;iE<cf.KNEstep;iE++){
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen_NotFull[detconf][inhit][iE*Ndecstep+idecbin];
                }
                if (idge<Template->NSrc_Temp)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*KM2AData->Tobs_NotFull[detconf][isrc]*Template->Srcs_Temp[idge].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
                else
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*KM2AData->Tobs_NotFull[detconf][isrc]*Template->DGEs[idge-Template->NSrc_Temp].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
            }
        }

        // Morphology model X PSF
        //for (int ii=0;ii<cf.KNEbinUsed*ROI->Neffbins;ii++)
        //    Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.KNEbinUsed+jj]; 
                }
            }
        }

        if (idge<Template->NSrc_Temp)
            npar += Template->Srcs_Temp[idge].nSEDpar;
        else
            npar += Template->DGEs[idge-Template->NSrc_Temp].nSEDpar;

        if (fSED) delete fSED;

    } 

    for (int isrc=0;isrc<Template->NTemp;isrc++)
        delete[] nexcess_exp[isrc];
    delete[] nexcess_exp;
    delete[] Flux;

}


void Src_Fitting_KM2A::GetDisZen_Temp(){

    TH1D *hzen = new TH1D("hzen", "hzen", cf.KNzenstep, 0, cf.KZenrange[1]+5.);
    for (int ii=0;ii<Ndecstep;ii++){
        hzen->Reset();
        double dec = Mindec + (ii+0.5)*cf.DECstep;
        GeneZenDis(ROI->Xcenter, dec, hzen, -1);
        for (int jj=0;jj<cf.KNzenstep;jj++)
            DisZen_Temp[ii][jj] = hzen->GetBinContent(jj+1);
        if (cf.UseKM2A_NotFull){
            for (int idet=0;idet<2;idet++){
                hzen->Reset();
                GeneZenDis(ROI->Xcenter, dec, hzen, idet);
                for (int jj=0;jj<cf.KNzenstep;jj++)
                    DisZen_Temp_NotFull[idet][ii][jj] = hzen->GetBinContent(jj+1);
            }
        }
    }

}

void Src_Fitting_KM2A::GeneZenDis(double ra, double dec, TH1D *hzen, int detconf){

    int tsecs = KM2AData->hSide->GetNbinsX();
    double wbinside = KM2AData->hSide->GetXaxis()->GetBinWidth(1);
    double tside0 = KM2AData->hSide->GetXaxis()->GetBinLowEdge(1);

    double zen, azi;
    for (int it=0;it<tsecs;it++){
        double tside = tside0+(it+0.5)*wbinside;
        double ha = tside-ra;

        papi::eql2hcs(ha*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.KZenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.KZenrange[0]) continue;
        if (detconf==-1){
            hzen->Fill(zen*papi::raddeg, KM2AData->hSide->GetBinContent(it+1));
        }
        else{
            if (cf.UseKM2A_NotFull)
                hzen->Fill(zen*papi::raddeg, KM2AData->Ntransit_NotFull[detconf]);
        }
    }   

    hzen->Scale(1/hzen->GetSum());

}

void Src_Fitting_KM2A::CalLogNull(int ibinUsed0, int ibinUsed1){

    log_L_null = 0;
    if (cf.UseWCDA){
        if (ibinUsed0==0 && ibinUsed1>cf.KNEbinUsed)
            ibinUsed1 = cf.KNEbinUsed;
        if ((ibinUsed1-ibinUsed0)==1){
            if (ibinUsed0<cf.NnhitUsed)
                return;
            else{
                ibinUsed0 -= cf.NnhitUsed;
                ibinUsed1 -= cf.NnhitUsed;
            }
        }
    }

    cout<<" ibinUsed0 = "<<ibinUsed0<<", ibinUsed1 = "<<ibinUsed1<<endl;

    for (int ii=0;ii<cf.KNEbinUsed;ii++)
        log_L_const[ii] = 0;

    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (KM2AData->Non[inhit][ii]>=0 && KM2AData->Nbkg[inhit][ii]>0){
                if (KM2AData->Non[inhit][ii]==0)
                    log_L_null += -KM2AData->Nbkg[inhit][ii];
                else{
                    log_L_null += -KM2AData->Nbkg[inhit][ii] + KM2AData->Non[inhit][ii]*log(KM2AData->Nbkg[inhit][ii]);
                    for (int kk=1;kk<=KM2AData->Non[inhit][ii];kk++){
                        log_L_null -= log(kk);
                        log_L_const[inhit] -= log(kk);
                    }
                }   
                if (isnan(log_L_null)) {
                    cout<<" iEbin = "<<inhit<<", ibin = "<<ii<<endl;
                    cout<<" Non = "<<KM2AData->Non[inhit][ii]<<", Nbkg = "<<KM2AData->Nbkg[inhit][ii]<<endl;
                    return;
                }   
            }   
        }   
    }   

    std::cout<<" *** Fitting : KM2A log_L_null = "<<log_L_null<<std::endl;

}

void Src_Fitting_KM2A::CalLogSig(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp, int ithispmode){

    int npar = 0;
    if (cf.FastIteration==1){
        int fastiteropt = 1;
        if (cf.UseWCDA && (ithispmode==0 || ithispmode==2))
            fastiteropt = 0;

        if (Niter>=1 && fastiteropt){
            for (int isrc=0;isrc<Template->NSrc;isrc++){
                if (isrc==ithisComp) continue;
                Template->Srcs[isrc].ConvoFlag = 0;
                if (abs(Template->Srcs[isrc].Ra[0]-par[npar])>FastIter_th)  Template->Srcs[isrc].ConvoFlag = 1;
                if (abs(Template->Srcs[isrc].Dec[0]-par[npar+1])>FastIter_th) Template->Srcs[isrc].ConvoFlag = 1;
                for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
                    if (abs(Template->Srcs[isrc].SEDPar[ipar][0]-par[npar+2+ipar])>FastIter_th)
                        Template->Srcs[isrc].ConvoFlag = 1;
                npar += 2+Template->Srcs[isrc].nSEDpar;
                for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
                    if (abs(Template->Srcs[isrc].MorPar[ipar][0]-par[npar+ipar])>FastIter_th)
                        Template->Srcs[isrc].ConvoFlag = 1;
                npar += Template->Srcs[isrc].nMorpar;
            }
            for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
                if (isrc==(ithisComp-Template->NSrc)) continue;
                Template->Srcs_NumCon[isrc].ConvoFlag = 0;
                if (abs(Template->Srcs_NumCon[isrc].Ra[0]-par[npar])>FastIter_th) Template->Srcs_NumCon[isrc].ConvoFlag = 1;
                if (abs(Template->Srcs_NumCon[isrc].Dec[0]-par[npar+1])>FastIter_th) Template->Srcs_NumCon[isrc].ConvoFlag = 1;
                for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++)
                    if (abs(Template->Srcs_NumCon[isrc].SEDPar[ipar][0]-par[npar+2+ipar])>FastIter_th)
                        Template->Srcs_NumCon[isrc].ConvoFlag = 1;
                npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
                for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
                    if (abs(Template->Srcs_NumCon[isrc].MorPar[ipar][0]-par[npar+ipar])>FastIter_th)
                        Template->Srcs_NumCon[isrc].ConvoFlag = 1;
                npar += Template->Srcs_NumCon[isrc].nMorpar;
            }
            for (int isrc=0;isrc<Template->NTemp;isrc++){
                if (isrc==(ithisComp-Template->NSrc-Template->NSrc_NumCon)) continue;
                if (isrc<Template->NSrc_Temp){
                    Template->Srcs_Temp[isrc].ConvoFlag = 0;
                    for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++)
                        if (abs(Template->Srcs_Temp[isrc].SEDPar[ipar][0]-par[npar+ipar])>FastIter_th)
                            Template->Srcs_Temp[isrc].ConvoFlag = 1;
                    npar += Template->Srcs_Temp[isrc].nSEDpar;
                }
                else{
                    Template->DGEs[isrc-Template->NSrc_Temp].ConvoFlag = 0;
                    for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++)
                        if (abs(Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][0]-par[npar+ipar])>FastIter_th)
                            Template->DGEs[isrc-Template->NSrc_Temp].ConvoFlag = 1;;
                    npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
                }
            }

            // link pars -> ConvoFlag
            for (int isrc=0;isrc<Template->NSrc;isrc++){
                if (isrc==ithisComp) continue;
                if (Template->Srcs[isrc].LinkPars){
                    int targetsrcid = Template->Srcs[isrc].TargetSrcID_Class;
                    int targetsrclass = Template->Srcs[isrc].TargetSrcClass;
                    if (targetsrclass == 0){
                        Template->Srcs[isrc].ConvoFlag += Template->Srcs[targetsrcid].ConvoFlag;
                    }
                    else if (targetsrclass == 1){
                        Template->Srcs[isrc].ConvoFlag += Template->Srcs_NumCon[targetsrcid].ConvoFlag;
                    }
                    else{
                        Template->Srcs[isrc].ConvoFlag += Template->Srcs_Temp[targetsrcid].ConvoFlag;
                    }
                }
            }
            for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
                if (isrc==(ithisComp-Template->NSrc)) continue;
                if (Template->Srcs_NumCon[isrc].LinkPars){
                    int targetsrcid = Template->Srcs_NumCon[isrc].TargetSrcID_Class;
                    int targetsrclass = Template->Srcs_NumCon[isrc].TargetSrcClass;
                    if (targetsrclass == 0){
                        Template->Srcs_NumCon[isrc].ConvoFlag += Template->Srcs[targetsrcid].ConvoFlag;
                    }
                    else if (targetsrclass == 1){
                        Template->Srcs_NumCon[isrc].ConvoFlag += Template->Srcs_NumCon[targetsrcid].ConvoFlag;
                    }
                    else{
                        Template->Srcs_NumCon[isrc].ConvoFlag += Template->Srcs_Temp[targetsrcid].ConvoFlag;
                    }
                }
            }
            for (int isrc=0;isrc<Template->NSrc_Temp;isrc++){
                if (isrc==(ithisComp-Template->NSrc-Template->NSrc_NumCon)) continue;
                if (Template->Srcs_Temp[isrc].LinkPars){
                    int targetsrcid = Template->Srcs_Temp[isrc].TargetSrcID_Class;
                    int targetsrclass = Template->Srcs_Temp[isrc].TargetSrcClass;
                    if (targetsrclass == 0){
                        Template->Srcs_Temp[isrc].ConvoFlag += Template->Srcs[targetsrcid].ConvoFlag;
                    }
                    else if (targetsrclass == 1){
                        Template->Srcs_Temp[isrc].ConvoFlag += Template->Srcs_NumCon[targetsrcid].ConvoFlag;
                    }
                    else{
                        Template->Srcs_Temp[isrc].ConvoFlag += Template->Srcs_Temp[targetsrcid].ConvoFlag;
                    }
                }
            }

        }
    }


    log_L_sig = 0;
    if (cf.UseWCDA){
        if (ibinUsed0==0 && ibinUsed1>cf.KNEbinUsed)
            ibinUsed1 = cf.KNEbinUsed;
        if ((ibinUsed1-ibinUsed0)==1){
            if (ibinUsed0<cf.NnhitUsed)
                return;
            else{
                ibinUsed0 -= cf.NnhitUsed;
                ibinUsed1 -= cf.NnhitUsed;
            }
        }
    }

    if (ithisComp!=-1){
        // Srcs
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            if (isrc==ithisComp)
                nPar_src = nPar_src-(2+Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar);
        }   
        // Srcs_NumCon
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            if (isrc==(ithisComp-Template->NSrc))
                nPar_numsrc = nPar_numsrc-(2+Template->Srcs_NumCon[isrc].nSEDpar + Template->Srcs_NumCon[isrc].nMorpar);
        }   
        // Src_Temp && DGEs
        for (int isrc=0;isrc<Template->NTemp;isrc++){
            if (isrc==(ithisComp-Template->NSrc-Template->NSrc_NumCon)){
                if (isrc<Template->NSrc_Temp){
                    nPar_dge = nPar_dge-Template->Srcs_Temp[isrc].nSEDpar;
                }
                else{
                    nPar_dge = nPar_dge-Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
                }
            }
        }
    }

    double *par_src = new double[nPar_src];
    double *par_numsrc = new double[nPar_numsrc];
    double *par_dge = new double[nPar_dge];
    for (int ii=0;ii<nPar_src;ii++)
        par_src[ii] = 0;
    for (int ii=0;ii<nPar_numsrc;ii++)
        par_numsrc[ii] = 0;
    for (int ii=0;ii<nPar_dge;ii++)
        par_dge[ii] = 0;
    for (int ii=0;ii<nPar_src+nPar_numsrc+nPar_dge;ii++){
        if (ii<nPar_src)
            par_src[ii] = par[ii];
        else if (ii>=nPar_src && ii<(nPar_src+nPar_numsrc))
            par_numsrc[ii-nPar_src] = par[ii];
        else
            par_dge[ii-(nPar_src+nPar_numsrc)] = par[ii];
    }

    if (nPar_src>0){
        Convolute(par_src, ibinUsed0, ibinUsed1, ithisComp, par_numsrc, par_dge);
        /*if (cf.UseKM2A_NotFull){
            Convolute_NotFull(par_src, ibinUsed0, ibinUsed1, ithisComp, 0);
            Convolute_NotFull(par_src, ibinUsed0, ibinUsed1, ithisComp, 1);
        }*/
    }
    if (nPar_numsrc>0){
        Convolute_NumSrc(par_numsrc, ibinUsed0, ibinUsed1, ithisComp, par_src, par_dge);
        if (cf.UseKM2A_NotFull){
            Convolute_NumSrc_NotFull(par_numsrc, ibinUsed0, ibinUsed1, ithisComp, 0, par_src, par_dge);
            Convolute_NumSrc_NotFull(par_numsrc, ibinUsed0, ibinUsed1, ithisComp, 1, par_src, par_dge);
        }
    }
    if (nPar_dge>0){
        Convolute_DGE(par_dge, ibinUsed0, ibinUsed1, ithisComp, par_src, par_numsrc);
        if (cf.UseKM2A_NotFull){
            Convolute_DGE_NotFull(par_dge, ibinUsed0, ibinUsed1, ithisComp, 0, par_src, par_numsrc);
            Convolute_DGE_NotFull(par_dge, ibinUsed0, ibinUsed1, ithisComp, 1, par_src, par_numsrc);
        }
    }

    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (KM2AData->Non[inhit][ii]>=0 && KM2AData->Nbkg[inhit][ii]>0){

                double on_expect = 0;
                for (int isrc=0;isrc<Template->NComp;isrc++)
                    if (isrc!=ithisComp)
                        on_expect += Nmodel_convo[isrc][inhit*ROI->Neffbins+ii];
                on_expect += KM2AData->Nbkg[inhit][ii];

                if (KM2AData->Non[inhit][ii]==0)
                    log_L_sig += -on_expect;
                else{
                    log_L_sig += -on_expect + KM2AData->Non[inhit][ii]*log(on_expect);
                    //for (int kk=1;kk<=KM2AData->Non[inhit][ii];kk++)
                    //    log_L_sig -= log(kk);
                }

            }
        }
        log_L_sig += log_L_const[inhit];
    } 

    delete[] par_src;
    delete[] par_numsrc;
    delete[] par_dge;

    // Fast iteration
    if (cf.FastIteration==1){
        npar = 0;
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            if (isrc==ithisComp) continue;
            Template->Srcs[isrc].Ra[0] = par[npar];
            Template->Srcs[isrc].Dec[0] = par[npar+1];
            for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
                Template->Srcs[isrc].SEDPar[ipar][0] = par[npar+2+ipar];
            npar += 2+Template->Srcs[isrc].nSEDpar;
            for (int ipar=0;ipar<Template->Srcs[isrc].nMorpar;ipar++)
                Template->Srcs[isrc].MorPar[ipar][0] = par[npar+ipar];
            npar += Template->Srcs[isrc].nMorpar;
        }
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            if (isrc==(ithisComp-Template->NSrc)) continue;
            Template->Srcs_NumCon[isrc].Ra[0] = par[npar];
            Template->Srcs_NumCon[isrc].Dec[0] = par[npar+1];
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nSEDpar;ipar++)
                Template->Srcs_NumCon[isrc].SEDPar[ipar][0] = par[npar+2+ipar];
            npar += 2+Template->Srcs_NumCon[isrc].nSEDpar;
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc].nMorpar;ipar++)
                Template->Srcs_NumCon[isrc].MorPar[ipar][0] = par[npar+ipar];
            npar += Template->Srcs_NumCon[isrc].nMorpar;
        }
        for (int isrc=0;isrc<Template->NTemp;isrc++){
            if (isrc==(ithisComp-Template->NSrc-Template->NSrc_NumCon)) continue;
            if (isrc<Template->NSrc_Temp){
                for (int ipar=0;ipar<Template->Srcs_Temp[isrc].nSEDpar;ipar++)
                    Template->Srcs_Temp[isrc].SEDPar[ipar][0] = par[npar+ipar];
                npar += Template->Srcs_Temp[isrc].nSEDpar;
            }
            else{
                for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;ipar++)
                    Template->DGEs[isrc-Template->NSrc_Temp].SEDPar[ipar][0] = par[npar+ipar];
                npar += Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
            }
        }
    }

}

void Src_Fitting_KM2A::CalEmedian(int imode, double **Energy){

    /* 
        imode : 0, return 0.7, 0.9, 1.1, ...
                1, cal median energy according to position and spectrum of Components
    */

    int binbias = 0;
    if (cf.UseWCDA)
        binbias += cf.NnhitUsed;

    if (imode==0){

        cout<<" KM2A CalEmedian (middle point of energy bin) :"<<endl;
        for (int isrc=0;isrc<Template->NComp;isrc++){
            if (isrc<Template->NSrc)
                cout<<Form(" *** median energy of %s : ", Template->Srcs[isrc].Srcname.data())<<endl;
            else if (isrc>=Template->NSrc && isrc<(Template->NSrc_NumCon+Template->NSrc))
                cout<<Form(" *** median energy of %s : ", Template->Srcs_NumCon[isrc-Template->NSrc].Srcname.data())<<endl;
            else if (isrc>=(Template->NSrc_NumCon+Template->NSrc) && isrc<Template->NSrc_total)
                cout<<Form(" *** median energy of %s : ", Template->Srcs_Temp[isrc-Template->NSrc-Template->NSrc_NumCon].Srcname.data())<<endl;
            else
                cout<<Form(" *** median energy of %s : ", Template->DGEs[isrc-Template->NSrc_total].Srcname.data())<<endl;

            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                Energy[isrc][ii+binbias+imode*(binbias+cf.KNEbinUsed)] = pow(10, cf.KDataErange[0]+(ii+cf.KEbinUsed[0]+0.5)*cf.KDataErangeStep);
                cout<<Energy[isrc][ii+binbias+imode*(binbias+cf.KNEbinUsed)]<<", ";
            }
            cout<<endl;
        }

    }
    else{
        
        TH1D *hzen[Template->NComp];
        double ra, dec;
        for (int isrc=0;isrc<Template->NComp;isrc++){
            hzen[isrc] = new TH1D(Form("hzen_%d", isrc), Form("hzen_%d", isrc), 200, 0, cf.KZenrange[1]+5.);
            if (isrc<Template->NSrc){
                if (!cf.CorOpt){
                    ra  = Template->Srcs[isrc].Ra[0];
                    dec = Template->Srcs[isrc].Dec[0];
                }
                else{
                    g2e(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], &ra, &dec);
                }
                GeneZenDis(ra, dec, hzen[isrc], -1);
            }
            else if (isrc>=Template->NSrc && isrc<Template->NSrc_NumCon){
                if (!cf.CorOpt){
                    ra  = Template->Srcs_NumCon[isrc-Template->NSrc].Ra[0];
                    dec = Template->Srcs_NumCon[isrc-Template->NSrc].Dec[0];
                }
                else{
                    g2e(Template->Srcs_NumCon[isrc-Template->NSrc].Ra[0], Template->Srcs_NumCon[isrc-Template->NSrc].Dec[0], &ra, &dec);
                }
                GeneZenDis(ra, dec, hzen[isrc], -1);
            }
            else{
                GeneZenDis(ROI->Xcenter, ROI->Ycenter, hzen[isrc], -1);
            }
        }


        // Gene zen dis of simu data
        TH1D *hzen_ref = new TH1D("hzen_ref", "hzen_ref", 200, 0, cf.KZenrange[1]+5.);
        for (int ii=0;ii<200;ii++){
            double theta0 = ii*(cf.Zenrange[1]+5.)/200*papi::degrad;
            double theta1 = (ii+1)*(cf.Zenrange[1]+5.)/200*papi::degrad;
            hzen_ref->SetBinContent(ii+1, cos(theta0)-cos(theta1));
        }
        hzen_ref->Scale(1./hzen_ref->GetSumOfWeights());

        // SED
        TF1 *fSED_ref = new TF1("fSED_ref", "2.83*1.e-14*pow(x/10, -2.)", 0.01, 1000);
        TF1 *fSED[Template->NComp];
        // Srcs
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            fSED[isrc] = new TF1(Form("hSED_%d", isrc), Template->Srcs[isrc].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs[isrc].nSEDpar;ipar++)
                fSED[isrc]->SetParameter(ipar, Template->Srcs[isrc].SEDPar[ipar][0]);
        }
        // NumSrcs
        for (int isrc=Template->NSrc;isrc<(Template->NSrc_NumCon+Template->NSrc);isrc++){
            fSED[isrc] = new TF1(Form("hSED_%d", isrc), Template->Srcs_NumCon[isrc-Template->NSrc].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_NumCon[isrc-Template->NSrc].nSEDpar;ipar++)
                fSED[isrc]->SetParameter(ipar, Template->Srcs_NumCon[isrc-Template->NSrc].SEDPar[ipar][0]);
        }
        // Srcs_Temp
        for (int isrc=(Template->NSrc_NumCon+Template->NSrc);isrc<Template->NSrc_total;isrc++){
            fSED[isrc] = new TF1(Form("hSED_%d", isrc), Template->Srcs_Temp[isrc-(Template->NSrc_NumCon+Template->NSrc)].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->Srcs_Temp[isrc-(Template->NSrc_NumCon+Template->NSrc)].nSEDpar;ipar++)
                fSED[isrc]->SetParameter(ipar, Template->Srcs_Temp[isrc-(Template->NSrc_NumCon+Template->NSrc)].SEDPar[ipar][0]);
        }
        // DGEs
        for (int isrc=Template->NSrc_total;isrc<Template->NSrc_total+Template->NDGE;isrc++){
            fSED[isrc] = new TF1(Form("hSED_%d", isrc), Template->DGEs[isrc-Template->NSrc_total].SEDFormula.data(), 0.01, 1000);
            for (int ipar=0;ipar<Template->DGEs[isrc-Template->NSrc_total].nSEDpar;ipar++)
                fSED[isrc]->SetParameter(ipar, Template->DGEs[isrc-Template->NSrc_total].SEDPar[ipar][0]);
        }


        TH1D *henergy[cf.KNEbinUsed*Template->NComp];
        for (int ii=0;ii<cf.KNEbinUsed;ii++)
            for (int jj=0;jj<Template->NComp;jj++)
                henergy[ii*Template->NComp+jj] = new TH1D(Form("henergy_%d_src%d", ii, jj), Form("henergy_%d_src%d", ii, jj), 100, cf.KErange[0], cf.KErange[1]);

        TFile *fSimu = TFile::Open(cf.fKSimu.data());
        TTree *tSimu = (TTree *) fSimu->Get("km2aevents");
        // Declaration of leaf types
        Double_t        E;
        Double_t        Rec_E;
        Double_t        Theta;
        // List of branches
        TBranch        *b_E;   //!
        TBranch        *b_Rec_E;   //!
        TBranch        *b_Theta;   //! 

        tSimu->SetBranchAddress("E", &E, &b_E);
        tSimu->SetBranchAddress("Rec_E", &Rec_E, &b_Rec_E);
        tSimu->SetBranchAddress("Theta", &Theta, &b_Theta);
        Long64_t nentries = tSimu->GetEntriesFast();
        cout<<" KM2A CalEmedian (Cal Emedian according to SED and position of Components) : "<<endl;
        cout<<"   Event Loop: ["<<flush;
        for (Long64_t jentry=0; jentry<nentries;jentry++) {

            if (jentry%(nentries/100)==0){
                if (jentry/(nentries/100)%10==0)
                    cout<<jentry/(nentries/100)<<"%"<<flush;
                else
                    cout<<"="<<flush;
            }

            tSimu->GetEntry(jentry);
            Theta *= papi::raddeg;
            if (Theta>=cf.KZenrange[1]+5.) continue;

            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                if (Rec_E>=(cf.KDataErange[0]+(ii+cf.KEbinUsed[0])*cf.KDataErangeStep) && Rec_E<(cf.KDataErange[0]+(ii+cf.KEbinUsed[0]+1)*cf.KDataErangeStep)){
                    for (int isrc=0;isrc<Template->NComp;isrc++){
                        double weight_sed = fSED[isrc]->Eval(E/1000)/fSED_ref->Eval(E/1000);
                        int izenbin = Theta/((cf.KZenrange[1]+5.)/200);
                        double weight_zen = hzen[isrc]->GetBinContent(izenbin+1)/hzen_ref->GetBinContent(izenbin+1);

                        if (isnan(weight_zen) || isnan(weight_sed)){
                            cout<<weight_sed<<" "<<weight_zen<<" "<<Theta<<endl;
                            continue;
                        }

                        double tau = 0;
                        if (isrc<Template->NSrc && Template->Srcs[isrc].GGAbsFlag){
                            if (E/1000<Template->Srcs[isrc].ebl_Emin)
                                tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                            else if (E/1000>Template->Srcs[isrc].ebl_Emin && E/1000<Template->Srcs[isrc].ebl_Emax)
                                tau = Template->Srcs[isrc].gg_ebl->Eval(E/1000);
                            else
                                tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                        }

                        henergy[ii*Template->NComp+isrc]->Fill(log10(E)-3, weight_sed*weight_zen*exp(-tau));
                    }
                    break;
                }
            }
        }
        cout<<"]"<<endl;

        // median energy
        for (int isrc=0;isrc<Template->NComp;isrc++){
            if (isrc<Template->NSrc)
                cout<<Form(" *** median energy of %s : ", Template->Srcs[isrc].Srcname.data())<<endl;
            else if (isrc>=Template->NSrc && isrc<(Template->NSrc_NumCon+Template->NSrc))
                cout<<Form(" *** median energy of %s : ", Template->Srcs_NumCon[isrc-Template->NSrc].Srcname.data())<<endl;
            else if (isrc>=(Template->NSrc_NumCon+Template->NSrc) && isrc<Template->NSrc_total)
                cout<<Form(" *** median energy of %s : ", Template->Srcs_Temp[isrc-(Template->NSrc_NumCon+Template->NSrc)].Srcname.data())<<endl;
            else
                cout<<Form(" *** median energy of %s : ", Template->DGEs[isrc-Template->NSrc_total].Srcname.data())<<endl;

            for (int ii=0;ii<cf.KNEbinUsed;ii++){
                double Ntotal = henergy[ii*Template->NComp+isrc]->GetSumOfWeights();
                double Nevent = 0;
                double emedian = 0;
                for (int iE=0;iE<100;iE++){
                    Nevent += henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1);
                    if (Nevent/Ntotal>=0.5){
                        Nevent -= henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1);
                        emedian = henergy[ii*Template->NComp+isrc]->GetBinLowEdge(iE+1)+(Ntotal*0.5-Nevent)/henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1)*henergy[ii*Template->NComp+isrc]->GetBinWidth(iE+1);
                        break;
                    }   
                }   
                cout<<pow(10, emedian)<<", ";
                Energy[isrc][ii+binbias+imode*(binbias+cf.KNEbinUsed)] = pow(10, emedian);
            }   
            cout<<endl;
        }

        fSimu->Close();

    }

}


void Src_Fitting_KM2A::CalExposure(int ibinUsed0, int ibinUsed1, int ipixel, double **exposure){ 

    for (int jj=0;jj<cf.KNEbinUsed;jj++)
        for (int kk=0;kk<cf.KNEstep;kk++)
            exposure[jj][kk] = 0;

    for (int iE=0;iE<cf.KNEstep;iE++){
        double e0, e1;
        e0 = pow(10, cf.KErange[0]+iE*cf.KEstep)/1000;
        e1 = pow(10, cf.KErange[0]+(iE+1)*cf.KEstep)/1000;
    }

    double ra, dec;
    // Calculate exposure
    double xx = X[0]+((ROI->Cellid_model[ipixel]/nbinsY)+0.5)*wbinX;
    double yy = Y[0]+((ROI->Cellid_model[ipixel]%nbinsY)+0.5)*wbinY;
    if (!cf.CorOpt)
        dec = yy;
    else
        g2e(xx, yy, &ra, &dec);
    if (dec>=Maxdec) return;
    int idecbin = (dec-Mindec)/cf.DECstep;
    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
        for (int iE=0;iE<cf.KNEstep;iE++)
            exposure[inhit][iE] = Effzen[inhit][iE*Ndecstep+idecbin]*S0*KM2AData->Tobs[ipixel]/10000;
    }
}

void Src_Fitting_KM2A::CalLogNull_1D(int ibinUsed0, int ibinUsed1){

    log_L_null = 0;
    if (cf.UseWCDA){
        if (ibinUsed0==0 && ibinUsed1>cf.KNEbinUsed)
            ibinUsed1 = cf.KNEbinUsed;
        if ((ibinUsed1-ibinUsed0)==1){
            if (ibinUsed0<cf.NnhitUsed)
                return;
            else{
                ibinUsed0 -= cf.NnhitUsed;
                ibinUsed1 -= cf.NnhitUsed;
            }
        }
    }

    cout<<" ibinUsed0 = "<<ibinUsed0<<", ibinUsed1 = "<<ibinUsed1<<endl;

    for (int ii=0;ii<cf.KNEbinUsed;ii++)
        log_L_const[ii] = 0;
    
    double *Non_total = new double[cf.KNEbinUsed];
    double *Nbkg_total = new double[cf.KNEbinUsed];
    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){

        Non_total[inhit] = 0;
        Nbkg_total[inhit] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (KM2AData->Non[inhit][ii]>=0 && KM2AData->Nbkg[inhit][ii]>0){
                Non_total[inhit] += KM2AData->Non[inhit][ii];
                Nbkg_total[inhit] += KM2AData->Nbkg[inhit][ii];
            }  
        }   

        log_L_null += -Nbkg_total[inhit]+Non_total[inhit]*log(Nbkg_total[inhit]);
        for (int kk=1;kk<=int(Non_total[inhit]);kk++){
            log_L_null -= log(kk);
            log_L_const[inhit] -= log(kk);
        }

    }   

    delete[] Non_total;
    delete[] Nbkg_total;

    std::cout<<" *** Fitting : KM2A log_L_null = "<<log_L_null<<std::endl;

}

void Src_Fitting_KM2A::CalLogSig_1D(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp){

    log_L_sig = 0;
    if (cf.UseWCDA){
        if (ibinUsed0==0 && ibinUsed1>cf.KNEbinUsed)
            ibinUsed1 = cf.KNEbinUsed;
        if ((ibinUsed1-ibinUsed0)==1){
            if (ibinUsed0<cf.NnhitUsed)
                return;
            else{
                ibinUsed0 -= cf.NnhitUsed;
                ibinUsed1 -= cf.NnhitUsed;
            }
        }
    }

    if (ithisComp!=-1){
        // Srcs
        for (int isrc=0;isrc<Template->NSrc;isrc++){
            if (isrc==ithisComp)
                nPar_src = nPar_src-(2+Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar);
        }   
        // Srcs_NumCon
        for (int isrc=0;isrc<Template->NSrc_NumCon;isrc++){
            if (isrc==(ithisComp-Template->NSrc))
                nPar_numsrc = nPar_numsrc-(2+Template->Srcs_NumCon[isrc].nSEDpar + Template->Srcs_NumCon[isrc].nMorpar);
        }   
        // Src_Temp && DGEs
        for (int isrc=0;isrc<Template->NTemp;isrc++){
            if (isrc==(ithisComp-Template->NSrc-Template->NSrc_NumCon)){
                if (isrc<Template->NSrc_Temp){
                    nPar_dge = nPar_dge-Template->Srcs_Temp[isrc].nSEDpar;
                }
                else{
                    nPar_dge = nPar_dge-Template->DGEs[isrc-Template->NSrc_Temp].nSEDpar;
                }
            }
        }
    }

    double *par_src = new double[nPar_src];
    double *par_numsrc = new double[nPar_numsrc];
    double *par_dge = new double[nPar_dge];
    for (int ii=0;ii<nPar_src;ii++)
        par_src[ii] = 0;
    for (int ii=0;ii<nPar_numsrc;ii++)
        par_numsrc[ii] = 0;
    for (int ii=0;ii<nPar_dge;ii++)
        par_dge[ii] = 0;
    for (int ii=0;ii<nPar_src+nPar_numsrc+nPar_dge;ii++){
        if (ii<nPar_src)
            par_src[ii] = par[ii];
        else if (ii>=nPar_src && ii<(nPar_src+nPar_numsrc))
            par_numsrc[ii-nPar_src] = par[ii];
        else
            par_dge[ii-(nPar_src+nPar_numsrc)] = par[ii];
    }

    if (nPar_src>0)
        Convolute(par_src, ibinUsed0, ibinUsed1, ithisComp, par_numsrc, par_dge);
    if (nPar_numsrc>0)
        Convolute_NumSrc(par_numsrc, ibinUsed0, ibinUsed1, ithisComp, par_src, par_dge);
    if (nPar_dge>0)
        Convolute_DGE(par_dge, ibinUsed0, ibinUsed1, ithisComp, par_src, par_numsrc);

    double *Non_total = new double[cf.KNEbinUsed];
    double *Nbkg_total = new double[cf.KNEbinUsed];
    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){

        Non_total[inhit] = 0; 
        Nbkg_total[inhit] = 0; 

        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (KM2AData->Non[inhit][ii]>=0 && KM2AData->Nbkg[inhit][ii]>0){
                Non_total[inhit] += KM2AData->Non[inhit][ii];
                Nbkg_total[inhit] += KM2AData->Nbkg[inhit][ii];
            }
        }

        for (int isrc=0;isrc<Template->NComp;isrc++)
            if (isrc!=ithisComp){
                for (int ii=0;ii<ROI->Neffbins;ii++){
                    Nbkg_total[inhit] += Nmodel_convo[isrc][inhit*ROI->Neffbins+ii];
                }
            }

        log_L_sig += -Nbkg_total[inhit]+Non_total[inhit]*log(Nbkg_total[inhit]);
        log_L_sig += log_L_const[inhit];
    } 

    delete[] Non_total;
    delete[] Nbkg_total;
    delete[] par_src;
    delete[] par_numsrc;
    delete[] par_dge;

}


# endif
