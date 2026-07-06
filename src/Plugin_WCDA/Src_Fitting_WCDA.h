# ifndef Src_Fitting_WCDA_h
# define Src_Fitting_WCDA_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "TMath.h"
# include "Src_Response_WCDA.h"
# include "Src_Data_WCDA.h"

using namespace std;

class Src_Fitting_WCDA {

    public :

        Src_Fitting_WCDA();
        ~Src_Fitting_WCDA();

        void SetBasicPar(double s0);
        void SetROI(Src_ROI *roi);
        void SetTemplate(Src_Template *temp);
        bool Init();
        void GetDisZen_Temp();
        void GetTobs();
        void Convolute(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_num, double *par_dge);
        void Convolute_NumSrc(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_dge);
        void Convolute_DGE(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_num);
        void CalLogNull(int ibinUsed0, int ibinUsed1);
        void CalLogSig(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp, int ithispmode);

        void CalLogNull_1D(int ibinUsed0, int ibinUsed1);
        void CalLogSig_1D(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp);

        Src_Response_WCDA *WCDAResp;
        Src_Data_WCDA     *WCDAData;
        Src_ROI           *ROI;
        Src_Template      *Template;

        double Mindec;
        double Maxdec;
        int Ndecstep;
        double S0;
        double *Tobs;
        double **DisZen;
        double **DisZen_Temp;
        double **Effzen;
        double *Omega_model;
        double *Omega;
        double Omega_total;
        double Omega_total_model;
        double *Eta_ROI;
        double *NTemp_total_model;
        vector<vector <float> > PSF;
        vector<vector <long int> > PSF_id;
        double **Nmodel_convo;
        double log_L_null;
        double log_L_sig;
        double *log_L_const;

        // Get flux point
        void CalEmedian(double **Emedian, double Ycenter);
        void CalEmedian_Mk(double **Emedian, double Ycenter);
        void CalEmedian_Cod(double **Emedian, double Ycenter);
        void CalEmedian2(double **Emedian);

        // Tools
        void CalExposure(int ibinUsed0, int ibinUsed1,  int ipixel, double **exposure);
};

Src_Fitting_WCDA::Src_Fitting_WCDA(){

    WCDAResp = new Src_Response_WCDA();
    WCDAData = new Src_Data_WCDA();

}

Src_Fitting_WCDA::~Src_Fitting_WCDA(){

    for (int ii=0;ii<Template->NComp;ii++)
        delete[] Nmodel_convo[ii];
    delete[] Nmodel_convo;

    for (int ii=0;ii<Ndecstep;ii++)
        delete[] DisZen_Temp;
    delete[] DisZen_Temp;

    for (int ii=0;ii<Template->NSrc;ii++)
        delete[] DisZen;
    delete[] DisZen;

    delete[] Omega_model;
    delete[] Omega;
    delete[] Eta_ROI;
    delete[] NTemp_total_model;

}

void Src_Fitting_WCDA::SetBasicPar(double s0){

    S0 = s0;

}

void Src_Fitting_WCDA::SetROI(Src_ROI *roi){ ROI = roi; }

void Src_Fitting_WCDA::SetTemplate(Src_Template *temp){ Template = temp; };

bool Src_Fitting_WCDA::Init(){

    log_L_null = 0;
    log_L_sig = 0;
    log_L_const = new double[cf.NnhitUsed];
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        log_L_const[ii] = 0;

    cout<<" Fitting WCDA Init: Initializing array... "<<endl;
    Nmodel_convo = new double*[Template->NComp];
    DisZen = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NComp;ii++){
        Nmodel_convo[ii] = new double[cf.NnhitUsed*ROI->Neffbins];
        for (int jj=0;jj<ROI->Neffbins*cf.NnhitUsed;jj++)
            Nmodel_convo[ii][jj] = 0;
    }
    for (int ii=0;ii<Template->NSrc;ii++){
        DisZen[ii] = new double[cf.Nzenstep];
        for (int jj=0;jj<cf.Nzenstep;jj++)
            DisZen[ii][jj] = 0;
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
            DisZen_Temp[ii] = new double[cf.Nzenstep];
            for (int jj=0;jj<cf.Nzenstep;jj++)
                DisZen_Temp[ii][jj] = 0;
        }
        GetDisZen_Temp();

        Effzen = new double*[cf.NnhitUsed];
        for (int ii=0;ii<cf.NnhitUsed;ii++){
            Effzen[ii] = new double[cf.NEstep*Ndecstep];
            for (int jj=0;jj<cf.NEstep;jj++){
                for (int kk=0;kk<Ndecstep;kk++){
                    Effzen[ii][jj*Ndecstep+kk] = 0;
                    for (int izen=0;izen<cf.Nzenstep;izen++){
                        double zen = (izen+0.5)*cf.Zenstep;;
                        Effzen[ii][jj*Ndecstep+kk] += cos(zen*papi::degrad)*WCDAResp->hResp[ii][jj*cf.Nzenstep+izen]*DisZen_Temp[kk][izen];
                    }
                }
            }
        }

        // Get PSF(i, j), here i is ith cell
        cout<<" Fitting Init: Calculate PSF(i, j) ... "<<endl;
        cout<<"   PSF(i, j) Norm : ["<<flush;
        float **psf_total = new float*[ROI->Neffbins_model];
        for (int ii=0;ii<ROI->Neffbins_model;ii++){

            if (ii%(ROI->Neffbins_model/100)==0){
                if (ii/(ROI->Neffbins_model/100)%10==0)
                    cout<<ii/(ROI->Neffbins_model/100)<<"%"<<flush;
                else
                    cout<<"="<<flush;
            }

            psf_total[ii] = new float[cf.NnhitUsed];
            double x1 = X[0]+((ROI->Cellid_model[ii]/nbinsY)+0.5)*wbinX;
            double y1 = Y[0]+((ROI->Cellid_model[ii]%nbinsY)+0.5)*wbinY;
            double ibinx0 = (x1-2.2/cos(y1*papi::degrad)-X[0])/wbinX; 
            double ibinx1 = (x1+2.2/cos(y1*papi::degrad)-X[0])/wbinX;
            double ibiny0 = (y1-2.2-Y[0])/wbinY;
            double ibiny1 = (y1+2.2-Y[0])/wbinY;

            if (!cf.CorOpt)
                dec = y1;
            else
                g2e(x1, y1, &ra, &dec);

            int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

            for (int jj=0;jj<cf.NnhitUsed;jj++){
                psf_total[ii][jj] = 0;

                float p1 = WCDAResp->hPSF[jj][idecbin*4+1];
                float sigma1 = WCDAResp->hPSF[jj][idecbin*4+2];
                float sigma2 = WCDAResp->hPSF[jj][idecbin*4+3];

                for (int mm=ibinx0;mm<=ibinx1;mm++){
                    double x2 = X[0]+(mm+0.5)*wbinX;
                    for (int nn=ibiny0;nn<=ibiny1;nn++){
                        double y2 = Y[0]+(nn+0.5)*wbinY;
                        float space1 = distance(90-y1, x1, 90-(y2-0.001), x2-0.001);
                        float omega = (cos((Y[1]-y2-0.5*wbinY)*papi::degrad)-cos((Y[1]-y2+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;
                        if (space1<2.0)
                            psf_total[ii][jj] += (p1/(2*3.141592654)/sigma1/sigma1*exp(-space1*space1/2/sigma1/sigma1)+(1-p1)/(2*3.141592654)/sigma2/sigma2*exp(-space1*space1/2/sigma2/sigma2))*omega;
                    }
                }
            }
        }
        cout<<"]"<<endl;

        vector<float> psf_temp;
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
            float omega = (cos((Y[1]-y0-0.5*wbinY)*papi::degrad)-cos((Y[1]-y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad;

            double xmin = x0-2.2/cos(y0*papi::degrad);
            double xmax = x0+2.2/cos(y0*papi::degrad);
            double ymin = y0-2.2;
            double ymax = y0+2.2;

            for (int jj=0;jj<ROI->Neffbins_model;jj++){
                double x1 = X[0]+((ROI->Cellid_model[jj]/nbinsY)+0.5)*wbinX;
                double y1 = Y[0]+((ROI->Cellid_model[jj]%nbinsY)+0.5)*wbinY;

                if (x1<xmin || x1>xmax) continue;
                if (y1<ymin || y1>ymax) continue;

                float space = distance(90-y0, x0, 90-(y1-0.001), x1-0.001);
                if (space<2.0){

                    if (!cf.CorOpt)
                        dec = y1;
                    else
                        g2e(x1, y1, &ra, &dec);

                    int idecbin = (dec-cf.Decrange[0])/cf.Decstep;
                    psfid_temp.push_back(jj);
                    for (int inhit=0;inhit<cf.NnhitUsed;inhit++){
                        float p1 = WCDAResp->hPSF[inhit][idecbin*4+1];
                        float sigma1 = WCDAResp->hPSF[inhit][idecbin*4+2];
                        float sigma2 = WCDAResp->hPSF[inhit][idecbin*4+3];
                        float psf_ij = (p1/(2*3.141592654)/sigma1/sigma1*exp(-space*space/2/sigma1/sigma1)+(1-p1)/(2*3.141592654)/sigma2/sigma2*exp(-space*space/2/sigma2/sigma2))*omega;
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

void Src_Fitting_WCDA::GetTobs(){

    Tobs = new double[Template->NSrc];
    double ra, dec;
    for (int isrc=0;isrc<Template->NSrc;isrc++){
        Tobs[isrc] = 0;
        double tobs = 0;
        if (!cf.CorOpt){
            ra  = Template->Srcs[isrc].Ra[0];
            dec = Template->Srcs[isrc].Dec[0];
        }
        else
            g2e(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], &ra, &dec);

        WCDAData->GetTobs(ra, dec, DisZen[isrc], tobs);
        Tobs[isrc] = tobs;
    }

    /*TFile *fzen_temp = TFile::Open("/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/vtemp/config/SunGamma/track_sun_202103-202407.root");
    TH1D *hzen_temp = (TH1D *) fzen_temp->Get("h1");
    for (int izen=50;izen<70;izen++)
        hzen_temp->SetBinContent(izen+1, 0);
    //hzen_temp->Scale(0.01);
    Tobs[1] = hzen_temp->GetSumOfWeights();
    hzen_temp->Scale(1./Tobs[1]);
    for (int izen=0;izen<cf.Nzenstep;izen++){
        DisZen[1][izen] = 0;
        DisZen[1][izen] = hzen_temp->GetBinContent(izen+1);
    }
    fzen_temp->Close();*/

    if ((Template->NTemp+Template->NSrc_NumCon)>0)
        WCDAData->GetTobsMap(ROI->Cellid_model);

}

void Src_Fitting_WCDA::Convolute(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_num, double *par_dge){

    if (isnan(par[2]))
        par[2] = 0.01;

    double **nexcess_exp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_exp[ii] = new double[cf.NnhitUsed];
        for (int jj=0;jj<cf.NnhitUsed;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double **nexcess_temp = new double*[Template->NSrc];
    for (int ii=0;ii<Template->NSrc;ii++){
        nexcess_temp[ii] = new double[cf.NnhitUsed];
        for (int jj=0;jj<cf.NnhitUsed;jj++)
            nexcess_temp[ii][jj] = 0;
    }
    double *Flux = new double[cf.NEstep];
    for (int iE=0;iE<cf.NEstep;iE++)
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
        for (int iE=0;iE<cf.NEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.Erange[0]+iE*cf.Estep)/1000;
            e1 = pow(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;
            Flux[iE] = fSED->Integral(e0, e1);
        }

        // Calculate Nexcess_exp of each Source
        for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
            for (int iE=0;iE<cf.NEstep;iE++){
                double e0, e1;
                e0 = TMath::Power(10, cf.Erange[0]+iE*cf.Estep)/1000;
                e1 = TMath::Power(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;

                double tau = 0;
                if (Template->Srcs[isrc].GGAbsFlag){
                    if ((e0+e1)/2<Template->Srcs[isrc].ebl_Emin)
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                    else if ((e0+e1)/2>Template->Srcs[isrc].ebl_Emin && (e0+e1)/2<Template->Srcs[isrc].ebl_Emax)
                        tau = Template->Srcs[isrc].gg_ebl->Eval((e0+e1)/2);
                    else
                        tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                }

                for (int izen=0;izen<cf.Nzenstep;izen++){
                    double zen = (izen+0.5)*cf.Zenstep;
                    if (WCDAResp->hResp[inhit][iE*cf.Nzenstep+izen]<=0) continue;
                    if (DisZen[isrc][izen]<=0) continue;
                    nexcess_exp[isrc][inhit] += Flux[iE]*cos(zen*TMath::DegToRad())*WCDAResp->hResp[inhit][iE*cf.Nzenstep+izen]*DisZen[isrc][izen]*exp(-tau);
                }
            }
            nexcess_exp[isrc][inhit] = nexcess_exp[isrc][inhit]*S0*Tobs[isrc];

        }

        npar += Template->Srcs[isrc].nSEDpar + Template->Srcs[isrc].nMorpar + 2;

        delete fSED;
    }


    double ra, dec;
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
            if (cf.PSFtype == "2Gaus"){

                if (!cf.CorOpt)
                    dec = par[npar+1];
                else
                    g2e(par[npar], par[npar+1], &ra, &dec);
                int idecbin = (dec-cf.Decrange[0])/cf.Decstep;

                npar = npar+2+Template->Srcs[isrc].nSEDpar;
                double sigma3 = 0;
                if (Template->Srcs[isrc].Mortype == "Ext_gaus")
                    sigma3 = par[npar];
                if (Template->Srcs[isrc].Mortype == "Point")
                    sigma3 = 0;
                if (Template->Srcs[isrc].Mortype == "Ext_gaus_E")
                    sigma3 = par[npar];

                for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                    double p1 = WCDAResp->hPSF[inhit][idecbin*4+1];
                    double sigma1 = WCDAResp->hPSF[inhit][idecbin*4+2];
                    double sigma2 = WCDAResp->hPSF[inhit][idecbin*4+3];
                    nexcess_temp[isrc][inhit] += p1/(sigma1*sigma1+sigma3*sigma3)*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3))*omega;
                    nexcess_temp[isrc][inhit] += (1-p1)/(sigma2*sigma2+sigma3*sigma3)*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3))*omega;
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
            if (cf.PSFtype == "2Gaus"){

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
                    sigma3 = par[npar];

                for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                    double p1 = WCDAResp->hPSF[inhit][idecbin*4+1];
                    double sigma1 = WCDAResp->hPSF[inhit][idecbin*4+2];
                    double sigma2 = WCDAResp->hPSF[inhit][idecbin*4+3];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] = p1/(sigma1*sigma1+sigma3*sigma3)*exp(-space*space/2/(sigma1*sigma1+sigma3*sigma3))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit];
                    Nmodel_convo[isrc][inhit*ROI->Neffbins+ii] += (1-p1)/(sigma2*sigma2+sigma3*sigma3)*exp(-space*space/2/(sigma2*sigma2+sigma3*sigma3))*omega/nexcess_temp[isrc][inhit]*nexcess_exp[isrc][inhit]; 
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

void Src_Fitting_WCDA::Convolute_NumSrc(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_dge){ 

    double **nexcess_exp = new double*[Template->NSrc_NumCon];
    for (int ii=0;ii<Template->NSrc_NumCon;ii++){
        nexcess_exp[ii] = new double[cf.NnhitUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.NnhitUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.NEstep];
    for (int iE=0;iE<cf.NEstep;iE++)
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

        for (int iE=0;iE<cf.NEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.Erange[0]+iE*cf.Estep)/1000;
            e1 = pow(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;
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

            for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
                for (int iE=0;iE<cf.NEstep;iE++)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen[inhit][iE*Ndecstep+idecbin]*S0*WCDAData->Tobs[isrc];
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

        for (int ii=0;ii<cf.NnhitUsed*ROI->Neffbins;ii++)
            Nmodel_convo[Template->NSrc+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.NnhitUsed+jj]; 
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

void Src_Fitting_WCDA::Convolute_DGE(double *par, int ibinUsed0, int ibinUsed1, int ithisComp, double *par_src, double *par_num){

    // SED X detection efficiency

    double **nexcess_exp = new double*[Template->NTemp];
    for (int ii=0;ii<Template->NTemp;ii++){
        nexcess_exp[ii] = new double[cf.NnhitUsed*ROI->Neffbins_model];
        for (int jj=0;jj<cf.NnhitUsed*ROI->Neffbins_model;jj++)
            nexcess_exp[ii][jj] = 0;
    }
    double *Flux = new double[cf.NEstep];
    for (int iE=0;iE<cf.NEstep;iE++)
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

        for (int iE=0;iE<cf.NEstep;iE++){
            double e0, e1;
            e0 = pow(10, cf.Erange[0]+iE*cf.Estep)/1000;
            e1 = pow(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;
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
                for (int iE=0;iE<cf.NEstep;iE++)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] += Flux[iE]*Effzen[inhit][iE*Ndecstep+idecbin];

                if (idge<Template->NSrc_Temp)
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*WCDAData->Tobs[isrc]*Template->Srcs_Temp[idge].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
                else
                    nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc] = nexcess_exp[idge][inhit*ROI->Neffbins_model+isrc]*S0*WCDAData->Tobs[isrc]*Template->DGEs[idge-Template->NSrc_Temp].NTemp_model[isrc]*Omega_model[isrc]/NTemp_total_model[idge];
            }
        }

        // Morphology model X PSF
        for (int ii=0;ii<cf.NnhitUsed*ROI->Neffbins;ii++)
            Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][ii] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            for (int jj=ibinUsed0;jj<ibinUsed1;jj++){
                for (int kk=0;kk<PSF_id[ii].size();kk++){
                    Nmodel_convo[Template->NSrc+Template->NSrc_NumCon+idge][jj*ROI->Neffbins+ii] += nexcess_exp[idge][jj*ROI->Neffbins_model+PSF_id[ii][kk]]*PSF[ii][kk*cf.NnhitUsed+jj]; 
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

void Src_Fitting_WCDA::GetDisZen_Temp(){

    TH1D *hzen = new TH1D("hzen", "hzen", cf.Nzenstep, 0, cf.Zenrange[1]+5.);
    for (int ii=0;ii<Ndecstep;ii++){
        hzen->Reset();
        double dec = Mindec + (ii+0.5)*cf.DECstep;
        WCDAData->GeneZenDis(ROI->Xcenter, dec, hzen);
        for (int jj=0;jj<cf.Nzenstep;jj++)
            DisZen_Temp[ii][jj] = hzen->GetBinContent(jj+1);
    }

}

void Src_Fitting_WCDA::CalLogNull(int ibinUsed0, int ibinUsed1){

    log_L_null = 0;
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        log_L_const[ii] = 0;

    if (ibinUsed0>=cf.NnhitUsed)
        return;
    if (ibinUsed1>cf.NnhitUsed)
        ibinUsed1 = cf.NnhitUsed;

    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){
        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (WCDAData->Non[inhit][ii]>=0 && WCDAData->Nbkg[inhit][ii]>0){
                if (WCDAData->Non[inhit][ii]==0)
                    log_L_null += -WCDAData->Nbkg[inhit][ii];
                else{
                    log_L_null += -WCDAData->Nbkg[inhit][ii] + WCDAData->Non[inhit][ii]*log(WCDAData->Nbkg[inhit][ii]);
                    for (int kk=1;kk<=WCDAData->Non[inhit][ii];kk++){
                        log_L_null -= log(kk);
                        log_L_const[inhit] -= log(kk);
                    }
                }   
                if (isnan(log_L_null)) {
                    cout<<" inhit = "<<inhit<<", ibin = "<<ii<<endl;
                    cout<<" Non = "<<WCDAData->Non[inhit][ii]<<", Nbkg = "<<WCDAData->Nbkg[inhit][ii]<<endl;
                    return;
                }   
            }   
        }   
    }
     
    std::cout<<" *** Fitting : WCDA log_L_null = "<<log_L_null<<std::endl;

}

void Src_Fitting_WCDA::CalLogSig(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp, int ithispmode){

    // Fast Convo
    int npar = 0;
    if (cf.FastIteration==1){
        if (Niter>=1){
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

    // Convolution
    log_L_sig = 0;

    if (ibinUsed0>=cf.NnhitUsed)
        return;
    if (ibinUsed1>cf.NnhitUsed)
        ibinUsed1 = cf.NnhitUsed;

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


    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){

        /*cout<<"ihit"<<inhit<<", ";
        for (int isrc=0;isrc<Template->NComp;isrc++){
            double Nexcess = 0;
            for (int ii=0;ii<ROI->Neffbins;ii++){
                if (WCDAData->Non[inhit][ii]>=0 && WCDAData->Nbkg[inhit][ii]>0)
                    Nexcess += Nmodel_convo[isrc][inhit*ROI->Neffbins+ii];
            }
            cout<<"Nexcess"<<isrc<<" = "<<Nexcess<<", ";
        }
        cout<<endl;*/

        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (WCDAData->Non[inhit][ii]>=0 && WCDAData->Nbkg[inhit][ii]>0){

                double on_expect = 0;
                for (int isrc=0;isrc<Template->NComp;isrc++)
                    if (isrc!=ithisComp)
                        on_expect += Nmodel_convo[isrc][inhit*ROI->Neffbins+ii];
                on_expect += WCDAData->Nbkg[inhit][ii];

                if (WCDAData->Non[inhit][ii]==0)
                    log_L_sig += -on_expect;
                else{
                    log_L_sig += -on_expect + WCDAData->Non[inhit][ii]*log(on_expect);
                    //for (int kk=1;kk<=WCDAData->Non[inhit][ii];kk++)
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
        int fastiteropt = 1;
        if (cf.UseKM2A && (ithispmode==0 || ithispmode==2))
            fastiteropt = 0;

        if (fastiteropt){
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

}

void Src_Fitting_WCDA::CalEmedian(double **Emedian, double Ycenter){

    if (cf.Version == "Mk")
        CalEmedian_Mk(Emedian, Ycenter);
    else 
        CalEmedian_Cod(Emedian, Ycenter);

}

void Src_Fitting_WCDA::CalEmedian_Mk(double **Emedian, double Ycenter){

    TH1D *hzen[Template->NComp];
    double ra, dec;
    for (int isrc=0;isrc<Template->NComp;isrc++){
        hzen[isrc] = new TH1D(Form("hzen_%d", isrc), Form("hzen_%d", isrc), 200, 0, cf.Zenrange[1]+5.);
        if (isrc<Template->NSrc){
            if (!cf.CorOpt){
                ra  = Template->Srcs[isrc].Ra[0];
                dec = Template->Srcs[isrc].Dec[0];
            }
            else{
                g2e(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], &ra, &dec);
            }
            WCDAData->GeneZenDis(ra, dec, hzen[isrc]);

            /*if (isrc==1){
                TFile *fzen_temp = TFile::Open("/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/vtemp/config/SunGamma/track_sun_202103-202407.root");
                TH1D *hzen_temp = (TH1D *) fzen_temp->Get("h1");
                for (int izen=50;izen<70;izen++)
                    hzen_temp->SetBinContent(izen+1, 0);
                hzen_temp->Scale(1./hzen_temp->GetSumOfWeights());
                for (int izen=0;izen<cf.Nzenstep;izen++){
                    hzen[isrc]->SetBinContent(izen+1, 0);
                    hzen[isrc]->SetBinContent(izen+1, hzen_temp->GetBinContent(izen+1));
                }
                fzen_temp->Close();
            }*/

        }
        else if (isrc>=Template->NSrc && isrc<Template->NSrc_NumCon){
            if (!cf.CorOpt){
                ra  = Template->Srcs_NumCon[isrc-Template->NSrc].Ra[0];
                dec = Template->Srcs_NumCon[isrc-Template->NSrc].Dec[0];
            }
            else{
                g2e(Template->Srcs[isrc-Template->NSrc].Ra[0], Template->Srcs[isrc-Template->NSrc].Dec[0], &ra, &dec);
            }
            WCDAData->GeneZenDis(ra, dec, hzen[isrc]);
        }
        else{
            WCDAData->GeneZenDis(ROI->Xcenter, ROI->Ycenter, hzen[isrc]);
        }
    }

    // Gene zen dis of simu data
    TH1D *hzen_ref = new TH1D("hzen_ref", "hzen_ref", 200, 0, cf.Zenrange[1]+5.);
    for (int ii=0;ii<200;ii++){
        double theta0 = ii*(cf.Zenrange[1]+5.)/200*papi::degrad;
        double theta1 = (ii+1)*(cf.Zenrange[1]+5.)/200*papi::degrad;
        hzen_ref->SetBinContent(ii+1, cos(theta0)-cos(theta1));
    }
    hzen_ref->Scale(1./hzen_ref->GetSumOfWeights());

    // SED
    TF1 *fSED_Crab = new TF1("fSED_Crab", "2.83*1.e-11*pow(x, -2.62)", 0.01, 1000);
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

    TH1D *henergy[cf.NnhitUsed*Template->NComp];
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        for (int jj=0;jj<Template->NComp;jj++)
            henergy[ii*Template->NComp+jj] = new TH1D(Form("henergy_%d_src%d", ii, jj), Form("henergy_%d_src%d", ii, jj), 100, cf.Erange[0], cf.Erange[1]);

    string fsimu_temp;
    if (Ycenter<28)
        fsimu_temp = cf.fSimu1.data();
    else
        fsimu_temp = cf.fSimu2.data();

    TFile *fSimu = TFile::Open(fsimu_temp.data());
    TTree *tSimu = (TTree *) fSimu->Get("wcdaevents");

    // Declaration of leaf types
    Double_t        mctheta;
    Double_t        energy;
    Int_t           Nhit;
    Float_t         eweit;
    // List of branches
    TBranch        *b_mctheta;   //!
    TBranch        *b_energy;   //!
    TBranch        *b_Nhit;   //!
    TBranch        *b_eweit;   //!
    tSimu->SetBranchAddress("mctheta", &mctheta, &b_mctheta);
    tSimu->SetBranchAddress("energy", &energy, &b_energy);
    tSimu->SetBranchAddress("Nhit", &Nhit, &b_Nhit);
    tSimu->SetBranchAddress("eweit", &eweit, &b_eweit);
    Long64_t nentries = tSimu->GetEntriesFast();
    cout<<" WCDA CalEmedian : "<<endl;
    cout<<"   Event Loop : ["<<flush;

    for (Long64_t jentry=0; jentry<nentries;jentry++) {

        if (jentry%(nentries/100)==0)
            if (jentry/(nentries/100)%10==0)
                cout<<jentry/(nentries/100)<<"%"<<flush;
            else
                cout<<"="<<flush;

        tSimu->GetEntry(jentry);
        mctheta *= papi::raddeg;
        if (mctheta>=cf.Zenrange[1]+5.) continue;
        if (mctheta<cf.Zenrange[0]) continue;
        
        for (int ii=0;ii<cf.NnhitUsed;ii++){
            if (Nhit>=cf.Nhit[ii+cf.NhitUsed[0]] && Nhit<cf.Nhit[ii+cf.NhitUsed[0]+1]){
                //double space = distance(mctheta, mcphi, theta, phi);
                //if (space<=cf.PSF_simu[ii]){
                for (int isrc=0;isrc<Template->NComp;isrc++){
                    double weight_sed = fSED[isrc]->Eval(energy/1000)/fSED_Crab->Eval(energy/1000);
                    int izenbin = mctheta/((cf.Zenrange[1]+5)/200);
                    double weight_zen = hzen[isrc]->GetBinContent(izenbin+1)/hzen_ref->GetBinContent(izenbin+1);

                    if (isnan(weight_zen) || isnan(weight_sed) || isinf(weight_zen) || isinf(weight_sed)){
                        cout<<weight_sed<<" "<<weight_zen<<" "<<mctheta<<endl;
                        continue;
                    }

                    double tau = 0;
                    if (isrc<Template->NSrc && Template->Srcs[isrc].GGAbsFlag){
                        if (energy/1000<Template->Srcs[isrc].ebl_Emin)
                            tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                        else if (energy/1000>Template->Srcs[isrc].ebl_Emin && energy/1000<Template->Srcs[isrc].ebl_Emax)
                            tau = Template->Srcs[isrc].gg_ebl->Eval(energy/1000);
                        else
                            tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                    }

                    henergy[ii*Template->NComp+isrc]->Fill(log10(energy), eweit*weight_sed*weight_zen*exp(-tau));
                }
                //}
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

        for (int ii=0;ii<cf.NnhitUsed;ii++){
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
            cout<<pow(10, emedian-3)<<", ";
            Emedian[isrc][ii] = pow(10, emedian-3);
        }   
        cout<<endl;
    }

}


void Src_Fitting_WCDA::CalEmedian_Cod(double **Emedian, double Ycenter){

    TH1D *hzen[Template->NComp];
    double ra, dec;
    for (int isrc=0;isrc<Template->NComp;isrc++){
        hzen[isrc] = new TH1D(Form("hzen_%d", isrc), Form("hzen_%d", isrc), 200, 0, cf.Zenrange[1]+5.);
        if (isrc<Template->NSrc){
            if (!cf.CorOpt){
                ra  = Template->Srcs[isrc].Ra[0];
                dec = Template->Srcs[isrc].Dec[0];
            }
            else{
                g2e(Template->Srcs[isrc].Ra[0], Template->Srcs[isrc].Dec[0], &ra, &dec);
            }
            WCDAData->GeneZenDis(ra, dec, hzen[isrc]);
        }
        else if (isrc>=Template->NSrc && isrc<Template->NSrc_NumCon){
            if (!cf.CorOpt){
                ra  = Template->Srcs_NumCon[isrc-Template->NSrc].Ra[0];
                dec = Template->Srcs_NumCon[isrc-Template->NSrc].Dec[0];
            }
            else{
                g2e(Template->Srcs[isrc-Template->NSrc].Ra[0], Template->Srcs[isrc-Template->NSrc].Dec[0], &ra, &dec);
            }
            WCDAData->GeneZenDis(ra, dec, hzen[isrc]);
        }
        else{
            WCDAData->GeneZenDis(ROI->Xcenter, ROI->Ycenter, hzen[isrc]);
        }
    }

    // Gene zen dis of simu data
    TH1D *hzen_ref = new TH1D("hzen_ref", "hzen_ref", 200, 0, cf.Zenrange[1]+5.);
    /*for (int ii=0;ii<200;ii++){
        double theta0 = ii*(cf.Zenrange[1]+5.)/200*papi::degrad;
        double theta1 = (ii+1)*(cf.Zenrange[1]+5.)/200*papi::degrad;
        hzen_ref->SetBinContent(ii+1, cos(theta0)-cos(theta1));
    }
    hzen_ref->Scale(1./hzen_ref->GetSumOfWeights());*/
    WCDAData->GeneZenDis_MJD(83.63, 22.02, hzen_ref);

    // SED
    TF1 *fSED_Crab = new TF1("fSED_Crab", "2.83*1.e-11*pow(x, -2.62)", 0.01, 1000);
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

    TH1D *henergy[cf.NnhitUsed*Template->NComp];
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        for (int jj=0;jj<Template->NComp;jj++)
            henergy[ii*Template->NComp+jj] = new TH1D(Form("henergy_%d_src%d", ii, jj), Form("henergy_%d_src%d", ii, jj), 100, cf.Erange[0], cf.Erange[1]);

    string fsimu_temp;
    if (Ycenter<28)
        fsimu_temp = cf.fSimu1.data();
    else
        fsimu_temp = cf.fSimu2.data();
    TFile *fSimu = TFile::Open(fsimu_temp.data());
    TTree *tSimu = (TTree *) fSimu->Get("wcdaevents");

    // Declaration of leaf types
    Float_t         mctheta;
    Float_t         energy;
    Int_t           Nhit;
    Float_t         eweit;
    Double_t        wperiod;
    Int_t           bkgflag;
    // List of branches
    TBranch        *b_mctheta;   //!
    TBranch        *b_energy;   //!
    TBranch        *b_Nhit;   //!
    TBranch        *b_eweit;   //!
    TBranch        *b_wperiod;   //!
    TBranch        *b_bkgflag;   //!
    tSimu->SetBranchAddress("mctheta", &mctheta, &b_mctheta);
    tSimu->SetBranchAddress("energy", &energy, &b_energy);
    tSimu->SetBranchAddress("Nhit", &Nhit, &b_Nhit);
    tSimu->SetBranchAddress("eweit", &eweit, &b_eweit);
    tSimu->SetBranchAddress("wperiod", &wperiod, &b_wperiod);
    tSimu->SetBranchAddress("bkgflag", &bkgflag, &b_bkgflag);

    Long64_t nentries = tSimu->GetEntriesFast();
    cout<<" WCDA CalEmedian : "<<endl;
    cout<<"   Event Loop : ["<<flush;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {

        if (jentry%(nentries/100)==0){
            if (jentry/(nentries/100)%10==0)
                cout<<jentry/(nentries/100)<<"%"<<flush;
            else
                cout<<"="<<flush;
        }

        tSimu->GetEntry(jentry);
        mctheta *= papi::raddeg;
        if (mctheta>=cf.Zenrange[1]+5.) continue;
        if (mctheta<cf.Zenrange[0]) continue;

        for (int ii=0;ii<cf.NnhitUsed;ii++){
            if (Nhit>=cf.Nhit[ii+cf.NhitUsed[0]] && Nhit<cf.Nhit[ii+cf.NhitUsed[0]+1]){
                //double space = distance(mctheta, mcphi, theta, phi);
                //if (space<=cf.PSF_simu[ii]){
                for (int isrc=0;isrc<Template->NComp;isrc++){
                    double weight_sed = fSED[isrc]->Eval(energy/1000)/fSED_Crab->Eval(energy/1000);
                    int izenbin = mctheta/((cf.Zenrange[1]+5)/200);
                    double weight_zen = hzen[isrc]->GetBinContent(izenbin+1)/hzen_ref->GetBinContent(izenbin+1);

                    if (isnan(weight_zen) || isnan(weight_sed) || isinf(weight_zen) || isinf(weight_sed)){
                        //cout<<weight_sed<<" "<<weight_zen<<" "<<mctheta<<endl;
                        continue;
                    }

                    double tau = 0;
                    if (isrc<Template->NSrc && Template->Srcs[isrc].GGAbsFlag){
                        if (energy/1000<Template->Srcs[isrc].ebl_Emin)
                            tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emin);
                        else if (energy/1000>Template->Srcs[isrc].ebl_Emin && energy/1000<Template->Srcs[isrc].ebl_Emax)
                            tau = Template->Srcs[isrc].gg_ebl->Eval(energy/1000);
                        else
                            tau = Template->Srcs[isrc].gg_ebl->Eval(Template->Srcs[isrc].ebl_Emax);
                    }

                    if (bkgflag)
                        henergy[ii*Template->NComp+isrc]->Fill(log10(energy), eweit*weight_sed*weight_zen*wperiod*exp(-tau));
                    else
                        henergy[ii*Template->NComp+isrc]->Fill(log10(energy), -eweit*weight_sed*weight_zen*wperiod*exp(-tau));
                }
                //}
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

        for (int ii=0;ii<cf.NnhitUsed;ii++){
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

            // abnormal case process
            if (emedian<1.5){
                int maxbin = henergy[ii*Template->NComp+isrc]->GetMaximumBin();
                int ebin0 = max(maxbin-1.5/henergy[ii*Template->NComp+isrc]->GetBinWidth(1), 1.);
                int ebin1 = min(maxbin+1.5/henergy[ii*Template->NComp+isrc]->GetBinWidth(1), henergy[ii*Template->NComp+isrc]->GetNbinsX()*1.);
                Ntotal = henergy[ii*Template->NComp+isrc]->Integral(ebin0, ebin1);
                Nevent = 0;
                emedian = 0;
                for (int iE=ebin0;iE<=ebin1;iE++){
                    Nevent += henergy[ii*Template->NComp+isrc]->GetBinContent(iE);
                    if (Nevent/Ntotal>=0.5){
                        Nevent -= henergy[ii*Template->NComp+isrc]->GetBinContent(iE);
                        emedian = henergy[ii*Template->NComp+isrc]->GetBinLowEdge(iE)+(Ntotal*0.5-Nevent)/henergy[ii*Template->NComp+isrc]->GetBinContent(iE)*henergy[ii*Template->NComp+isrc]->GetBinWidth(iE);
                        break;
                    }   
                }  
            } 

            cout<<pow(10, emedian-3)<<", ";
            Emedian[isrc][ii] = pow(10, emedian-3);
        }   
        cout<<endl;
    }

    fSimu->Close();

}

void Src_Fitting_WCDA::CalEmedian2(double **Emedian){

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

    TH1D *henergy[cf.NnhitUsed*Template->NComp];
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        for (int jj=0;jj<Template->NComp;jj++)
            henergy[ii*Template->NComp+jj] = new TH1D(Form("henergy_%d_SRC%d", ii, jj), Form("henergy_%d_SRC%d", ii, jj), cf.NEstep, cf.Erange[0], cf.Erange[1]);

    // Calculate Nexcess_exp of each Source
    for (int isrc=0;isrc<Template->NComp;isrc++){
        
        TTree *tt_ebl = new TTree();
        TGraph *gg_ebl = new TGraph();
        if (Template->Srcs[isrc].GGAbsFlag){
            tt_ebl->ReadFile(Template->Srcs[isrc].fGGAbs.data());
            double ebl_E, ebl_Tau;
            tt_ebl->SetBranchAddress("E", &ebl_E);
            tt_ebl->SetBranchAddress("Tau", &ebl_Tau);
            int Npoint = tt_ebl->GetEntries();
            for (int ii=0;ii<Npoint;ii++){
                tt_ebl->GetEntry(ii);
                gg_ebl->SetPoint(ii, ebl_E, ebl_Tau);
            }
        }


        for (int inhit=0;inhit<cf.NnhitUsed;inhit++){
            for (int iE=0;iE<cf.NEstep;iE++){
                double e0, e1;
                e0 = TMath::Power(10, cf.Erange[0]+iE*cf.Estep)/1000;
                e1 = TMath::Power(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;

                double tau = 0;
                if (Template->Srcs[isrc].GGAbsFlag){
                    if ((e0+e1)/2<0.01)
                        tau = gg_ebl->Eval(0.01);
                    else if ((e0+e1)/2>0.01 && (e0+e1)/2<100)
                        tau = gg_ebl->Eval((e0+e1)/2);
                    else
                        tau = gg_ebl->Eval(100);
                }

                double excess_temp = 0;
                for (int izen=0;izen<cf.Nzenstep;izen++){
                    double zen = (izen+0.5)*cf.Zenstep;
                    if (WCDAResp->hResp[inhit][iE*cf.Nzenstep+izen]<=0) continue;
                    if (DisZen[isrc][izen]<=0) continue;
                    excess_temp += fSED[isrc]->Integral(e0, e1)*cos(zen*TMath::DegToRad())*WCDAResp->hResp[inhit][iE*cf.Nzenstep+izen]*DisZen[isrc][izen]; 
                }

                henergy[inhit*Template->NComp+isrc]->SetBinContent(iE+1, excess_temp*S0*Tobs[isrc]*exp(-tau));
            }
        }
    }

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

        for (int ii=0;ii<cf.NnhitUsed;ii++){
            double Ntotal = henergy[ii*Template->NComp+isrc]->GetSumOfWeights();
            double Nevent = 0;
            double emedian = 0;
            for (int iE=0;iE<cf.NEstep;iE++){
                Nevent += henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1);
                if (Nevent/Ntotal>=0.5){
                    Nevent -= henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1);
                    emedian = henergy[ii*Template->NComp+isrc]->GetBinLowEdge(iE+1)+(Ntotal*0.5-Nevent)/henergy[ii*Template->NComp+isrc]->GetBinContent(iE+1)*henergy[ii*Template->NComp+isrc]->GetBinWidth(iE+1);
                    break;
                }   
            }   
            cout<<pow(10, emedian-3)<<", ";
            Emedian[isrc][ii] = pow(10, emedian-3);
        }   
        cout<<endl;
    }

}

void Src_Fitting_WCDA::CalExposure(int ibinUsed0, int ibinUsed1, int ipixel, double **exposure){ 

    for (int jj=0;jj<cf.NnhitUsed;jj++)
        for (int kk=0;kk<cf.NEstep;kk++)
            exposure[jj][kk] = 0;

    for (int iE=0;iE<cf.NEstep;iE++){
        double e0, e1;
        e0 = pow(10, cf.Erange[0]+iE*cf.Estep)/1000;
        e1 = pow(10, cf.Erange[0]+(iE+1)*cf.Estep)/1000;
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
        for (int iE=0;iE<cf.NEstep;iE++)
            exposure[inhit][iE] = Effzen[inhit][iE*Ndecstep+idecbin]*S0*WCDAData->Tobs[ipixel]/10000;
    }
}

void Src_Fitting_WCDA::CalLogNull_1D(int ibinUsed0, int ibinUsed1){

    log_L_null = 0;
    for (int ii=0;ii<cf.NnhitUsed;ii++)
        log_L_const[ii] = 0;

    if (ibinUsed0>=cf.NnhitUsed)
        return;
    if (ibinUsed1>cf.NnhitUsed)
        ibinUsed1 = cf.NnhitUsed;

    int *Non_total = new int[cf.NnhitUsed];
    double *Nbkg_total = new double[cf.NnhitUsed];
    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){

        Non_total[inhit] = 0;
        Nbkg_total[inhit] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (WCDAData->Non[inhit][ii]>=0 && WCDAData->Nbkg[inhit][ii]>0){
                Non_total[inhit] += WCDAData->Non[inhit][ii];
                Nbkg_total[inhit] += WCDAData->Nbkg[inhit][ii];
            }
        }

        log_L_null += -Nbkg_total[inhit]+Non_total[inhit]*log(Nbkg_total[inhit]);
        for (int kk=1;kk<=Non_total[inhit];kk++){
            log_L_null -= log(kk);
            log_L_const[inhit] -= log(kk);
        }

    }

    delete[] Non_total;
    delete[] Nbkg_total;

    std::cout<<" *** Fitting : WCDA log_L_null = "<<log_L_null<<std::endl;

}

void Src_Fitting_WCDA::CalLogSig_1D(double *par, int nPar_src, int nPar_numsrc, int nPar_dge, int ibinUsed0, int ibinUsed1, int ithisComp){

    log_L_sig = 0;

    if (ibinUsed0>=cf.NnhitUsed)
        return;
    if (ibinUsed1>cf.NnhitUsed)
        ibinUsed1 = cf.NnhitUsed;

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

    int *Non_total = new int[cf.NnhitUsed];
    double *Nbkg_total = new double[cf.NnhitUsed];
    for (int inhit=ibinUsed0;inhit<ibinUsed1;inhit++){

        Non_total[inhit] = 0;
        Nbkg_total[inhit] = 0;

        for (int ii=0;ii<ROI->Neffbins;ii++){
            if (WCDAData->Non[inhit][ii]>=0 && WCDAData->Nbkg[inhit][ii]>0){
                Non_total[inhit] += WCDAData->Non[inhit][ii];
                Nbkg_total[inhit] += WCDAData->Nbkg[inhit][ii];
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

    delete[] par_src;
    delete[] par_numsrc;
    delete[] par_dge;
    delete[] Non_total;
    delete[] Nbkg_total;

}

# endif
