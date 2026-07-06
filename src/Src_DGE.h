# ifndef Src_DGE_h
# define Src_DGE_h

# include <iostream>
# include <string>
# include <vector>

using namespace std;

class Src_DGE {

    public :

        Src_DGE();
        Src_DGE(string name, string sedtype, vector<vector<double> > sedpar, string tempfile);
        void Init(string name, string sedtype, vector<vector<double> > sedpar, string tempfile);
        void SetBasicPar(double epiv, string f0_order);
        void SetFormula(string  sedformula);
        bool GetTempROI(vector<long int> cellid, vector<long int> cellid_model, string histname);
        void Clear();
        ~Src_DGE();

        string Srcname;
        string HistName;
        // Morphology 
        string TempFile;
        vector<double> NTemp;
        vector<double> NTemp_model;

        // SED
        string SEDtype;
        vector<vector<double> > SEDPar;
        string SEDFormula;
        int nSEDpar;
        double Epiv;
        string F0_order;
        double NTemp_total_model;
        double Omega_total_model;
        double NTemp_total;
        double Omega_total;
        double Eta;

        // Fast iteration
        bool ConvoFlag;

};

Src_DGE::Src_DGE(){ ConvoFlag = 1; }

Src_DGE::Src_DGE(string name, string sedtype, vector<vector<double> > sedpar, string tempfile){
    ConvoFlag = 1;
    Init(name, sedtype, sedpar, tempfile);
}

Src_DGE::~Src_DGE(){

    NTemp.clear();
    NTemp.shrink_to_fit();
    NTemp_model.clear();
    NTemp_model.shrink_to_fit();

}

void Src_DGE::Init(string name, string sedtype, vector<vector<double> > sedpar, string tempfile){

    Srcname = name;
    // Morphology
    TempFile = tempfile;
    // SED
    SEDtype = sedtype;
    SEDPar  = sedpar;
    nSEDpar = sedpar.size();

    cout<<" INFO : "<<"initialize DGE component "<<Form("\"%s\"", Srcname.data())
        <<" with tempfile "<<tempfile.data()
        <<" and sedtype "<<Form("\"%s\"", sedtype.data())<<endl;

}

void Src_DGE::SetBasicPar(double epiv, string f0_order){

    Epiv = epiv;
    F0_order = f0_order;

}

void Src_DGE::SetFormula(string sedformula){

    SEDFormula = sedformula;

}

bool Src_DGE::GetTempROI(vector<long int> cellid, vector<long int> cellid_model, string histname){

    HistName = histname;

    int Neffbins = cellid.size();
    int Neffbins_model = cellid_model.size();
    /*NTemp = new double[Neffbins];
      for (int jj=0;jj<Neffbins;jj++)
      NTemp[jj] = 0;
      NTemp_model = new double[Neffbins_model];
      for (int jj=0;jj<Neffbins_model;jj++)
      NTemp_model[jj] = 0;*/

    if (TempFile == "ISO"){
        for (int jj=0;jj<Neffbins;jj++){
            NTemp.push_back(1);
            int yid = cellid[jj]%nbinsY;
            double y0 = Y[1]-(Y[0]+(yid+0.5)*wbinY);
            double omega = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad; 
            Omega_total += omega;
            NTemp_total += omega;
        }
        for (int jj=0;jj<Neffbins_model;jj++){
            NTemp_model.push_back(1);
            int yid = cellid_model[jj]%nbinsY;
            double y0 = Y[1]-(Y[0]+(yid+0.5)*wbinY);
            double omega = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad; 
            Omega_total_model += omega;
            NTemp_total_model += omega;
        }
    }
    else{

        TFile *ftemp;
        TH2D *htemp;
        if (histname!="hTXT"){
            ftemp = TFile::Open(TempFile.data());
            if (!ftemp){
                cout<<"\033[31;1mError\033[0m : can not open tempfile : "<<TempFile<<"! Exited."<<endl;
                return -1;
            }
            else{
                htemp = (TH2D *) ftemp->Get(histname.data());
                //htemp->Scale(1.e-23);
                if (!htemp){
                    cout<<"\033[31;1mError\033[0m : there is no histogram \""<<histname.data()<<"\" in tempfile "<<TempFile<<"! Exited."<<endl;
                    return -1;
                }
                else{
                    if (htemp->GetSum()<=0){
                        cout<<"\033[31;1mError\033[0m : histogram \""<<histname.data()<<"\" in tempfile "<<TempFile<<" is empty! Exited."<<endl;
                        return -1;
                    }
                }
            }
        }
        else{
            double xx, yy, density;
            TTree *tt = new TTree();
            tt->ReadFile(TempFile.data(), "xx/D:yy/D:dd/D");
            tt->SetBranchAddress("xx", &xx);
            tt->SetBranchAddress("yy", &yy);
            tt->SetBranchAddress("dd", &density);
            tt->GetEntry(0);
            double xx0 = xx;
            int nbinsx = yy;
            double wbinx = density;
            tt->GetEntry(1);
            double yy0 = xx;
            int nbinsy = yy;
            double wbiny = density;
            htemp = new TH2D("htemp", "htemp", nbinsx, xx0, xx0+nbinsx*wbinx, nbinsy, yy0, yy0+nbinsy*wbiny);
            for (int ii=0;ii<nbinsx;ii++)
                for (int jj=0;jj<nbinsy;jj++){
                    tt->GetEntry(2+jj+ii*nbinsy);
                    htemp->SetBinContent(ii+1, jj+1, density);
                }
        }

        NTemp_total_model = 0;
        Omega_total_model = 0;
        double x0_temp = htemp->GetXaxis()->GetBinLowEdge(1);
        double x1_temp = htemp->GetXaxis()->GetBinLowEdge(htemp->GetNbinsX()+1);
        double y0_temp = htemp->GetYaxis()->GetBinLowEdge(1);
        double y1_temp = htemp->GetYaxis()->GetBinLowEdge(htemp->GetNbinsY()+1);
        for (int ii=0;ii<Neffbins_model;ii++){
            int xid = cellid_model[ii]/nbinsY;
            int yid = cellid_model[ii]%nbinsY;
            double yy = Y[0]+(yid+0.5)*wbinY;
            double xx = X[0]+(xid+0.5)*wbinX;

            if (xx<x0_temp || xx>x1_temp || yy<y0_temp || yy>y1_temp){
                NTemp_model.push_back(0);
                //cout<<"Warning: model pixel ["<<xid<<", "<<yid<<"] out of template histogram domain, set to 0"<<endl;
            }
            else
                NTemp_model.push_back(htemp->Interpolate(xx, yy));
            if (NTemp_model[ii]<0) {
                NTemp_model[ii] = 0;
                continue;
            }

            double y0 = Y[1]-(Y[0]+(yid+0.5)*wbinY);
            double omega = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad; 
            Omega_total_model += omega;
            //NTemp_total_model += htemp->GetBinContent(xid+1, yid+1)*omega;
            NTemp_total_model += NTemp_model[ii]*omega;
        }

        NTemp_total = 0;
        Omega_total = 0;
        for (int ii=0;ii<Neffbins;ii++){
            int xid = cellid[ii]/nbinsY;
            int yid = cellid[ii]%nbinsY;
            double yy = Y[0]+(yid+0.5)*wbinY;
            double xx = X[0]+(xid+0.5)*wbinX;

            if (xx<x0_temp || xx>x1_temp || yy<y0_temp || yy>y1_temp){
                NTemp.push_back(0);
                //cout<<"Warning: model pixel ["<<xid<<", "<<yid<<"] out of template histogram domain, set to 0"<<endl;
            }
            else
                NTemp.push_back(htemp->Interpolate(xx, yy));
            if (NTemp[ii]<0){
                NTemp[ii] = 0;
                continue;
            }

            double y0 = Y[1]-(Y[0]+(yid+0.5)*wbinY);
            double omega = (cos((y0-0.5*wbinY)*papi::degrad)-cos((y0+0.5*wbinY)*papi::degrad))*wbinX*papi::degrad; 
            Omega_total += omega;
            //NTemp_total += htemp->GetBinContent(xid+1, yid+1)*omega;
            NTemp_total += NTemp[ii]*omega;
        }

        Eta = NTemp_total/NTemp_total_model;
        SEDPar[0][0] = SEDPar[0][0]*Omega_total/Eta/Omega_total_model;
        SEDPar[0][1] = SEDPar[0][1]*Omega_total/Eta/Omega_total_model;
        SEDPar[0][2] = SEDPar[0][2]*Omega_total/Eta/Omega_total_model;

        if (histname!="hTXT")
            ftemp->Close();
    }

    return 0;

}

void Src_DGE::Clear(){

    NTemp.clear();
    NTemp_model.clear();
    SEDPar.clear();

}

# endif
