# ifndef Src_Data_WCDA_h
# define Src_Data_WCDA_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"

using namespace std;

class Src_Data_WCDA {

    public :

        Src_Data_WCDA();
        ~Src_Data_WCDA();

        bool GetContentinROI(vector<long int> cellid);
        void Rebin();
        void GetTobs(double ra, double dec, double *diszen, double &tobs);
        void GeneZenDis(double ra, double dec, TH1D *hzen);
        void GeneZenDis_MJD(double ra, double dec, TH1D *hzen);
        void GetTobsMap(vector<long int> cellid_model);

        // WCDA
        double **Non;
        double **Nbkg;
        float *Tobs; 
        TH1D *hSide;
        TH2D *hFOV;

};

Src_Data_WCDA::Src_Data_WCDA(){

    hSide = new TH1D("hside", "hside", 86164, 0, 360);
    hFOV  = new TH2D("hfOV", "hfOV", 1800, -90, 90, 1800, -90, 90);

}

Src_Data_WCDA::~Src_Data_WCDA(){

    for (int ii=0;ii<cf.NnhitUsed;ii++){
        delete[] Non[ii];
        delete[] Nbkg[ii];
    }
    delete[] Non;
    delete[] Nbkg;
    delete[] Tobs;

}

bool Src_Data_WCDA::GetContentinROI(vector<long int> cellid){

    // Init
    Non  = new double*[cf.NnhitUsed];
    Nbkg = new double*[cf.NnhitUsed];
    int Neffbins = cellid.size();
    for (int ii=0;ii<cf.NnhitUsed;ii++){
        Non[ii]  = new double[Neffbins];
        Nbkg[ii] = new double[Neffbins];
        for (int jj=0;jj<Neffbins;jj++){
            Non[ii][jj]  = 0;
            Nbkg[ii][jj] = 0;
        } 
    }

    TH2D *hon  = new TH2D();
    TH2D *hbkg = new TH2D();

    TFile *fmap = TFile::Open(cf.fMap.data());
    if (!fmap){
        cout<<"\033[31;1mError\033[0m : can not open Map file : "<<cf.fMap<<"! Exited."<<endl;
        return -1;
    }
    else{
        TH1D *hside_temp = (TH1D *) fmap->Get("hSide");
        TH2D *hfov_temp = (TH2D *) fmap->Get("hFOV");
        if (!hside_temp){ //|| !hFOV){
            cout<<"\033[31;1mError\033[0m : No hSide or hFOV in "<<cf.fMap<<"! Exited."<<endl;
            return -1;
        }

        int nbins_temp = hside_temp->GetNbinsX();
        for (int ibin=0;ibin<nbins_temp;ibin++)
            hSide->SetBinContent(ibin+1, hside_temp->GetBinContent(ibin+1)/10);
        int nbinsx_temp = hfov_temp->GetNbinsX();
        int nbinsy_temp = hfov_temp->GetNbinsY();
        for (int ibinx=0;ibinx<nbinsx_temp;ibinx++){
            for (int ibiny=0;ibiny<nbinsy_temp;ibiny++)
                hFOV->SetBinContent(ibinx+1, ibiny+1, hfov_temp->GetBinContent(ibinx+1, ibiny+1));
        }
        
        for (int ii=0;ii<cf.NnhitUsed;ii++){
            if (!cf.CorOpt){
                hon  = (TH2D *) fmap->Get(Form("hon_%d", ii+cf.NhitUsed[0]));
                hbkg = (TH2D *) fmap->Get(Form("hbkg_%d", ii+cf.NhitUsed[0]));
            }
            else{
                hon  = (TH2D *) fmap->Get(Form("hon_gal_%d", ii+cf.NhitUsed[0]));
                hbkg = (TH2D *) fmap->Get(Form("hbkg_gal_%d", ii+cf.NhitUsed[0]));
            }
            for (int jj=0;jj<Neffbins;jj++){
                int xid = cellid[jj]/nbinsY;
                int yid = cellid[jj]%nbinsY;
                Non[ii][jj]  = hon->GetBinContent(xid+1, yid+1);
                Nbkg[ii][jj] = hbkg->GetBinContent(xid+1, yid+1);
            }
        }
    }

    fmap->Close();

    return 0;

}

void Src_Data_WCDA::GetTobs(double ra, double dec, double *diszen, double &tobs){

    int tsecs = hSide->GetNbinsX();
    double wbinside = hSide->GetXaxis()->GetBinWidth(1);
    double tside0 = hSide->GetXaxis()->GetBinLowEdge(1);

    double zen = 0, azi = 0;
    //TH1D *hzen = new TH1D("hzen_x", "hzen_x", cf.Nzenstep, 0, cf.Zenrange[1]+5.);

    for (int it=0;it<tsecs;it++){
        double tside = tside0+(it+0.5)*wbinside;
        double ha = tside-ra;

        papi::eql2hcs(ha*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.Zenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.Zenrange[0]) continue;
        tobs += hSide->GetBinContent(it+1);

        int izen = (int) (zen*papi::raddeg/cf.Zenstep);
        diszen[izen] += hSide->GetBinContent(it+1);
        //hzen->Fill(zen*papi::raddeg, hSide->GetBinContent(it+1));
    }   
    cout<<Form(" WCDA Tobs (ra=%.2lf, dec=%.2lf, %.lf<zen<%.lf) = %.2lf seconds", ra, dec, cf.Zenrange[0], cf.Zenrange[1]+5., tobs)<<endl;

    //hzen->Scale(1./hzen->GetSumOfWeights());
    for (int jj=0;jj<cf.Nzenstep;jj++)
        diszen[jj] = diszen[jj]/tobs; //hzen->GetBinContent(jj+1);

}

void Src_Data_WCDA::GeneZenDis(double ra, double dec, TH1D *hzen){

    int tsecs = hSide->GetNbinsX();
    double wbinside = hSide->GetXaxis()->GetBinWidth(1);
    double tside0 = hSide->GetXaxis()->GetBinLowEdge(1);

    double zen = 0, azi = 0;

    for (int it=0;it<tsecs;it++){
        double tside = tside0+(it+0.5)*wbinside;
        double ha = tside-ra;

        papi::eql2hcs(ha*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.Zenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.Zenrange[0]) continue;
        hzen->Fill(zen*papi::raddeg, hSide->GetBinContent(it+1));
    }   

    hzen->Scale(1./hzen->GetSumOfWeights());

}

void Src_Data_WCDA::GeneZenDis_MJD(double ra, double dec, TH1D *hzen){

    double mjd0 = 59000, mjd1 = 59001;
    double mjd_step = 1./86400.;
    int nstep = (mjd1-mjd0)/mjd_step-236;

    double zen = 0, azi = 0;
    for (int it=0;it<nstep;it++){
        double mjd = mjd0+(it+0.5)*mjd_step;

        papi::eqm2hcs(mjd, 0, ra*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.Zenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.Zenrange[0]) continue;
        hzen->Fill(zen*papi::raddeg, hSide->GetBinContent(it+1));
    }   

    hzen->Scale(1./hzen->GetSumOfWeights());

}

void Src_Data_WCDA::GetTobsMap(vector<long int> cellid_model){

    /*if (cf.ROIfile!="none"){
        TFile *fROI = TFile::Open(cf.ROIfile.data());
        TH2D *hLtime = (TH2D *) fROI->Get("hLtime");
        
        int Neffbins_model = cellid_model.size();
        Tobs = new float[Neffbins_model];
        for (int ii=0;ii<Neffbins_model;ii++){
            Tobs[ii] = 0;
            int xid = cellid_model[ii]/nbinsY;
            int yid = cellid_model[ii]%nbinsY;
            Tobs[ii] = hLtime->GetBinContent(xid+1, yid+1);
        }

        fROI->Close();
    }
    else{*/
        int tsecs = hSide->GetNbinsX();
        double wbinside = hSide->GetXaxis()->GetBinWidth(1);
        double tside0 = hSide->GetXaxis()->GetBinLowEdge(1);
        double HA0 = hFOV->GetXaxis()->GetBinLowEdge(1);

        double *Tside = new double[tsecs];
        for (int it=0;it<tsecs;it++)
            Tside[it] = tside0+(it+0.5)*wbinside;;

        int Neffbins_model = cellid_model.size();
        Tobs = new float[Neffbins_model];
        for (int ii=0;ii<Neffbins_model;ii++)
            Tobs[ii] = 0;

        double ra, dec;
        int yyid;
        for (int jj=0;jj<Neffbins_model;jj++){
            if (jj%(Neffbins_model/10)==0)
                cout<<" Cal Tobs of WCDA, Cell loop : "<<jj/(Neffbins_model/10)*10<<" % ... "<<endl;

            int xid = cellid_model[jj]/nbinsY;
            int yid = cellid_model[jj]%nbinsY;
            double xx = X[0] + (xid+0.5)*wbinX;
            double yy = Y[0] + (yid+0.5)*wbinY;
            if (!cf.CorOpt){
                ra  = xx;
                dec = yy;
                yyid = yid;
            }
            else{
                g2e(xx, yy, &ra, &dec);
                yyid = (dec-Y[0])/wbinX;
            }

            // Get livetime of each sky bin
            double ltime = 0;
            for (int it=0;it<tsecs;it++){
                double ha = Tside[it]-ra;
                //papi::eql2hcs(ha*papi::degrad, dec1*papi::degrad, zen, azi);
                //if (zen*papi::raddeg>50) continue;
                if (ha>180) ha -= 360;
                if (ha<-180) ha += 360;
                int ihabin = (ha-HA0)/wbinX;
                if (hFOV->GetBinContent(ihabin+1, yyid+1)<=0) continue;
                ltime += hSide->GetBinContent(it+1);
            }
            Tobs[jj] = ltime;
        }

        delete[] Tside;
    //}

}

void Src_Data_WCDA::Rebin(){

    return;

}

# endif
