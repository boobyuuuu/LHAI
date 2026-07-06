# ifndef Src_Data_KM2A_h
# define Src_Data_KM2A_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"

using namespace std;

class Src_Data_KM2A {

    public :

        Src_Data_KM2A();
        ~Src_Data_KM2A();

        bool GetContentinROI(vector<long int> cellid);
        void Rebin();
        void GetTobs(double ra, double dec, double *diszen, double &tobs);
        void GetTobsMap(vector<long int> cellid_model);
        void GetTobs_NotFull(double ra, double dec, double *diszen, double &tobs, int detconf);
        void GetTobsMap_NotFull(vector<long int> cellid_model, int detconf);

        // KM2A
        double **Non;
        double **Nbkg;
        double Ntransit;
        double Ntransit_NotFull[2];
        float *Tobs;
        float *Tobs_NotFull[2];
        TH2D *hFOV;
        TH1D *hSide;
        const string dettag[2] = {"3/4", "1/2"};

};

Src_Data_KM2A::Src_Data_KM2A(){}

Src_Data_KM2A::~Src_Data_KM2A(){

    for (int ii=0;ii<cf.KNEbinUsed;ii++){
        delete[] Non[ii];
        delete[] Nbkg[ii];
    }
    delete[] Non;
    delete[] Nbkg;

    delete[] Tobs;
    if (cf.UseKM2A_NotFull){
        for (int idet=0;idet<2;idet++)
            delete[] Tobs_NotFull[idet];
    }

}

bool Src_Data_KM2A::GetContentinROI(vector<long int> cellid){

    // Init
    Non  = new double*[cf.KNEbinUsed];
    Nbkg = new double*[cf.KNEbinUsed];
    int Neffbins = cellid.size();
    for (int ii=0;ii<cf.KNEbinUsed;ii++){
        Non[ii]  = new double[Neffbins];
        Nbkg[ii] = new double[Neffbins];
        for (int jj=0;jj<Neffbins;jj++){
            Non[ii][jj]  = 0;
            Nbkg[ii][jj] = 0;
        } 
    }
    hFOV = new TH2D("hfOVK", "hfOVK", 1800, -90, 90, 1800, -90, 90);
    hSide = new TH1D("hsideK", "hsideK", 86164, 0, 360);

    TH2D *hon  = new TH2D();
    TH2D *hbkg = new TH2D();
    TFile *fmap = TFile::Open(cf.fKMap.data());
    if (!fmap){
        cout<<"\033[31;1mError\033[0m : can not open Map file : "<<cf.fKMap<<"! Exited."<<endl;
        return -1;
    }
    else{
        TH1D *header = (TH1D *) fmap->Get("header");
        TH2D *hfov_temp = (TH2D *) fmap->Get("hFOV");
        if (!header){
            cout<<"\033[31;1mError\033[0m : No header or hFOV in "<<cf.fKMap<<"! Exited."<<endl;
            return -1;
        }

        Ntransit = header->GetBinContent(1)*86400/(86400-236);
        if (cf.UseKM2A_NotFull){
            Ntransit_NotFull[0] = header->GetBinContent(2)*86400/(86400-236);
            Ntransit_NotFull[1] = header->GetBinContent(3)*86400/(86400-236);
        }
        for (int ii=0;ii<86164;ii++)
            hSide->SetBinContent(ii+1, Ntransit);

        int nbinsx_temp = hfov_temp->GetNbinsX();
        int nbinsy_temp = hfov_temp->GetNbinsY();
        for (int ibinx=0;ibinx<nbinsx_temp;ibinx++){
            for (int ibiny=0;ibiny<nbinsy_temp;ibiny++)
                hFOV->SetBinContent(ibinx+1, ibiny+1, hfov_temp->GetBinContent(ibinx+1, ibiny+1));
        }

        for (int ii=0;ii<cf.KNEbinUsed;ii++){
            if (!cf.CorOpt){
                hon  = (TH2D *) fmap->Get(Form("hon_%d", ii+cf.KEbinUsed[0]));
                hbkg = (TH2D *) fmap->Get(Form("hbkg_%d", ii+cf.KEbinUsed[0]));
            }
            else{
                hon  = (TH2D *) fmap->Get(Form("hon_gal_%d", ii+cf.KEbinUsed[0]));
                hbkg = (TH2D *) fmap->Get(Form("hbkg_gal_%d", ii+cf.KEbinUsed[0]));
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

void Src_Data_KM2A::GetTobs(double ra, double dec, double *diszen, double &tobs){

    int tsecs = hSide->GetNbinsX();
    double wbinside = hSide->GetXaxis()->GetBinWidth(1);
    double tside0 = hSide->GetXaxis()->GetBinLowEdge(1);

    double zen = 0, azi = 0;
    //TH1D *hzen = new TH1D("hzen_x", "hzen_x", cf.KNzenstep, 0, cf.KZenrange[1]+5.);

    for (int it=0;it<tsecs;it++){
        double tside = tside0+(it+0.5)*wbinside;
        double ha = tside-ra;

        papi::eql2hcs(ha*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.KZenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.KZenrange[0]) continue;
        tobs += hSide->GetBinContent(it+1);

        int izen = (int) (zen*papi::raddeg/cf.KZenstep);
        diszen[izen] += hSide->GetBinContent(it+1);
        //hzen->Fill(zen*papi::raddeg, hSide->GetBinContent(it+1));
    }   

    cout<<Form(" KM2A(full) Tobs (ra=%.2lf, dec=%.2lf, 0<zen<%.lf) = %.2lf seconds", ra, dec, cf.KZenrange[1]+5., tobs)<<endl;

    //hzen->Scale(1./hzen->GetSumOfWeights());
    for (int jj=0;jj<cf.KNzenstep;jj++)
        diszen[jj] = diszen[jj]/tobs; //hzen->GetBinContent(jj+1);

}

void Src_Data_KM2A::GetTobs_NotFull(double ra, double dec, double *diszen, double &tobs, int detconf){

    // detconf 0 : 3/4 array; detconf 1 : 1/2 array

    int tsecs = hSide->GetNbinsX();
    double wbinside = hSide->GetXaxis()->GetBinWidth(1);
    double tside0 = hSide->GetXaxis()->GetBinLowEdge(1);

    double zen = 0, azi = 0;
    //TH1D *hzen = new TH1D("hzen_x", "hzen_x", cf.KNzenstep, 0, cf.KZenrange[1]+5.);

    for (int it=0;it<tsecs;it++){
        double tside = tside0+(it+0.5)*wbinside;
        double ha = tside-ra;

        papi::eql2hcs(ha*papi::degrad, dec*papi::degrad, zen, azi);
        if (zen*papi::raddeg>cf.KZenrange[1]+5.) continue;
        if (zen*papi::raddeg<cf.KZenrange[0]) continue;
        tobs += Ntransit_NotFull[detconf];

        int izen = (int) (zen*papi::raddeg/cf.KZenstep);
        diszen[izen] += Ntransit_NotFull[detconf];
        //hzen->Fill(zen*papi::raddeg, Ntransit_NotFull[detconf]);
    }   

    if (detconf==0)
        cout<<Form(" KM2A(3/4)  Tobs (ra=%.2lf, dec=%.2lf, 0<zen<%.lf) = %.2lf seconds", ra, dec, cf.KZenrange[1]+5., tobs)<<endl;
    else
        cout<<Form(" KM2A(1/2)  Tobs (ra=%.2lf, dec=%.2lf, 0<zen<%.lf) = %.2lf seconds", ra, dec, cf.KZenrange[1]+5., tobs)<<endl;

    //hzen->Scale(1./hzen->GetSumOfWeights());
    for (int jj=0;jj<cf.KNzenstep;jj++)
        diszen[jj] = diszen[jj]/tobs; //= hzen->GetBinContent(jj+1);


}

void Src_Data_KM2A::GetTobsMap(vector<long int> cellid_model){

    /*if (cf.ROIfile!="none"){
      TFile *fROI = TFile::Open(cf.ROIfile.data());
      TH2D *hLtime = (TH2D *) fROI->Get("hKLtime");

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

    double HA0 = hFOV->GetXaxis()->GetBinLowEdge(1);
    double mjd0 = 59000, mjd1 = 59001, mjdstep = 1./86400;
    int Nmjd = (mjd1-mjd0)/mjdstep - 236;

    double *Tside = new double[Nmjd];
    for (int it=0;it<Nmjd;it++){
        Tside[it] = 0;
        double mjd = mjd0 + (it+0.5)*mjdstep;
        Tside[it] = papi::getlast(mjd, 0)*papi::raddeg;
    }

    int Neffbins_model = cellid_model.size();
    Tobs = new float[Neffbins_model];
    for (int ii=0;ii<Neffbins_model;ii++)
        Tobs[ii] = 0;

    double ra, dec;
    int yyid;
    for (int jj=0;jj<Neffbins_model;jj++){
        if (jj%(Neffbins_model/10)==0)
            cout<<" Cal Tobs of KM2A(full), Cell loop : "<<jj/(Neffbins_model/10)*10<<" % ... "<<endl;

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
        for (int it=0;it<Nmjd;it++){
            double ha = Tside[it]-ra;
            if (ha>180) ha -= 360;
            if (ha<-180) ha += 360;
            int ihabin = (ha-HA0)/wbinX;
            if (hFOV->GetBinContent(ihabin+1, yyid+1)<=0) continue;
            ltime ++;
        }   
        Tobs[jj] = ltime*Ntransit;
    }

    delete[] Tside; 

    //}
}

void Src_Data_KM2A::GetTobsMap_NotFull(vector<long int> cellid_model, int detconf){

    // detconf 0 : 3/4 array; detconf 1 : 1/2 array

    double HA0 = hFOV->GetXaxis()->GetBinLowEdge(1);
    double mjd0 = 59000, mjd1 = 59001, mjdstep = 1./86400;
    int Nmjd = (mjd1-mjd0)/mjdstep - 236;

    double *Tside = new double[Nmjd];
    for (int it=0;it<Nmjd;it++){
        Tside[it] = 0;
        double mjd = mjd0 + (it+0.5)*mjdstep;
        Tside[it] = papi::getlast(mjd, 0)*papi::raddeg;
    }

    int Neffbins_model = cellid_model.size();
    Tobs_NotFull[detconf] = new float[Neffbins_model];
    for (int ii=0;ii<Neffbins_model;ii++)
        Tobs_NotFull[detconf][ii] = 0;

    double ra, dec;
    int yyid;
    for (int jj=0;jj<Neffbins_model;jj++){
        if (jj%(Neffbins_model/10)==0)
            cout<<" Cal Tobs of KM2A("<<dettag[detconf]<<"),  Cell loop : "<<jj/(Neffbins_model/10)*10<<" % ... "<<endl;

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
        for (int it=0;it<Nmjd;it++){
            double ha = Tside[it]-ra;
            if (ha>180) ha -= 360;
            if (ha<-180) ha += 360;
            int ihabin = (ha-HA0)/wbinX;
            if (hFOV->GetBinContent(ihabin+1, yyid+1)<=0) continue;
            ltime ++;
        }   
        Tobs_NotFull[detconf][jj] = ltime*Ntransit_NotFull[detconf];
    }

    delete[] Tside; 

}


void Src_Data_KM2A::Rebin(){

    return;

}

# endif
