# ifndef Src_Src_h
# define Src_Src_h

# include <iostream>
# include <string>
# include <vector>

using namespace std;

class Src_Src {

    public :

        Src_Src();
        Src_Src(string name, double *ra, double *dec, string mortype, string sedtype, vector<vector<double> > morpar, vector<vector<double> > sedpar);
        Src_Src(string name, string mortype, string sedtype, string tempfile, vector<vector<double> > sedpar);
        void Init(string name, double *ra, double *dec, string mortype, string sedtype, vector<vector<double> > morpar, vector<vector<double> > sedpar);
        void Init(string name, string mortype, string sedtype, string tempfile, vector<vector<double> > sedpar);
        void SetSrcID(int srcid);
        void SetBasicPar(double epiv, string f0_order);
        void SetFormula(string morformula, string sedformula);
        bool GetTempROI(vector<long int> cellid, vector<long int> cellid_model, string histname);
        void Clear();
        ~Src_Src();

        int SrcID;
        string Srcname;
        // Position
        double Ra[5];
        double Dec[5];
        // Morphology 
        string Mortype;
        vector<vector<double> > MorPar;
        string MorFormula;
        string TempFile;
        string HistName;
        int nMorpar;
        vector<double> NTemp;
        vector<double> NTemp_model;
        // SED
        string SEDtype;
        vector<vector<double> > SEDPar;
        string SEDFormula;
        int nSEDpar;
        double Epiv;
        string F0_order;

        bool TempFlag;
        double Tobs;

        // Gamma-gamma absorption
        bool GGAbsFlag;
        string fGGAbs;
        TGraph *gg_ebl;
        double ebl_Emin, ebl_Emax;
        void SetfGGAbs(bool absflag, string fabs, string srcname);

        // Fast iteration
        bool ConvoFlag;

        // LinkPars
        bool LinkPars;
        int TargetSrcID;
        int TargetSrcClass;
        int TargetSrcID_Class;
        void SetLinkPars(int targetSrcID);
        void SetLinkPars(int targetSrcClass, int targetSrcID_Class);
};

Src_Src::Src_Src(){

    GGAbsFlag = 0;
    ConvoFlag = 1;
    LinkPars  = 0;
    TargetSrcID = -1;
    ebl_Emin = 9999999;
    ebl_Emax = 1.e-4;

}

Src_Src::Src_Src(string name, double *ra, double *dec, string mortype, string sedtype, vector<vector<double> > morpar, vector<vector<double> > sedpar){

    GGAbsFlag = 0;
    ConvoFlag = 1;
    LinkPars  = 0;
    TargetSrcID = -1;
    ebl_Emin = 9999999;
    ebl_Emax = 1.e-4;
    Init(name, ra, dec, mortype, sedtype, morpar, sedpar);

}

Src_Src::Src_Src(string name, string mortype, string sedtype, string tempfile, vector<vector<double> > sedpar){
    
    GGAbsFlag = 0;
    ConvoFlag = 1;
    LinkPars  = 0;
    TargetSrcID = -1;
    ebl_Emin = 9999999;
    ebl_Emax = 1.e-4;
    Init(name, mortype, sedtype, tempfile, sedpar);

}

Src_Src::~Src_Src(){

    if (TempFlag){
        NTemp.clear();
        NTemp.shrink_to_fit();
        NTemp_model.clear();
        NTemp_model.shrink_to_fit();
    }

}

void Src_Src::Init(string name, double *ra, double *dec, string mortype, string sedtype, vector<vector<double> > morpar, vector<vector<double> > sedpar){

    Srcname = name;
    // Position
    for (int ii=0;ii<5;ii++){
        Ra[ii]  = ra[ii];
        Dec[ii] = dec[ii];
    }
    // Morphology
    Mortype = mortype;
    MorPar  = morpar;
    nMorpar = morpar.size();
    // SED
    SEDtype = sedtype;
    SEDPar  = sedpar;
    nSEDpar = sedpar.size(); 

    TempFlag = 0;

    cout<<" INFO : "<<"initialize source "<<Form("\"%s\"", Srcname.data())
        <<" with mortype "<<Form("\"%s\"", mortype.data())
        <<" and sedtype "<<Form("\"%s\"", sedtype.data())<<endl;

}

void Src_Src::Init(string name, string mortype, string sedtype, string tempfile, vector<vector<double> > sedpar){

    Srcname = name;
    // Morphology
    Mortype  = mortype;
    TempFile = tempfile;
    nMorpar  = 0;
    // SED
    SEDtype = sedtype;
    SEDPar  = sedpar;
    nSEDpar = sedpar.size();

    TempFlag = 1;
    
    cout<<" INFO : "<<"initialize source "<<Form("\"%s\"", Srcname.data())
        <<" with tempfile "<<tempfile.data()
        <<" and sedtype "<<Form("\"%s\"", sedtype.data())<<endl;
}

void Src_Src::SetSrcID(int srcid){

    SrcID = srcid;

}

void Src_Src::SetBasicPar(double epiv, string f0_order){

    Epiv = epiv;
    F0_order = f0_order;

}

void Src_Src::SetFormula(string morformula, string sedformula){

    MorFormula = morformula;
    SEDFormula = sedformula;

}

void Src_Src::SetfGGAbs(bool absflag, string fabs, string srcname){

    GGAbsFlag = absflag;
    fGGAbs    = fabs;
    if (GGAbsFlag){
        TTree *tt_ebl = new TTree();
        gg_ebl = new TGraph();
        tt_ebl->ReadFile(fGGAbs.data());
        double ebl_E, ebl_Tau;
        tt_ebl->SetBranchAddress("E", &ebl_E);
        tt_ebl->SetBranchAddress("Tau", &ebl_Tau);
        int Npoint = tt_ebl->GetEntries();
        for (int ii=0;ii<Npoint;ii++){
            tt_ebl->GetEntry(ii);
            gg_ebl->SetPoint(ii, ebl_E, ebl_Tau);
            if (ebl_E<ebl_Emin) ebl_Emin = ebl_E;
            if (ebl_E>ebl_Emax) ebl_Emax = ebl_E;
        }
        cout<<" INFO : "<<"Set gamma-gamma absorpotion for source "<<Form("\"%s\"", srcname.data())
            <<" with depthfile "<<fabs.data()<<endl;
    }

}

bool Src_Src::GetTempROI(vector<long int> cellid, vector<long int> cellid_model, string histname){

    HistName = histname;
    int Neffbins = cellid.size();
    int Neffbins_model = cellid_model.size();
    /*NTemp = new double[Neffbins];
    for (int jj=0;jj<Neffbins;jj++)
        NTemp[jj] = 0;
    NTemp_model = new double[Neffbins_model];
    for (int jj=0;jj<Neffbins_model;jj++)
        NTemp_model[jj] = 0;*/

    TFile *ftemp = TFile::Open(TempFile.data());
    if (!ftemp){
        cout<<"\033[31;1mError\033[0m : can not open tempfile : "<<TempFile<<"! Exited."<<endl;
        return 1;
    }
    else{
        TH2D *htemp;
        if (histname!="hTXT"){
            htemp = (TH2D *) ftemp->Get(histname.data());
            //htemp->Scale(1.e-23);
            if (!htemp){
                cout<<"\033[31;1mError\033[0m : there is no histogram \""<<histname.data()<<"\" in tempfile "<<TempFile<<"! Exited."<<endl;
                return 1;
            }
            else{
                if (htemp->GetSum()<=0){
                    cout<<"\033[31;1mError\033[0m : histogram \""<<histname.data()<<"\" in tempfile "<<TempFile<<" is empty! Exited."<<endl;
                    return 1;
                }
            }
        }

        int temp_nbinX = htemp->GetNbinsX();
        int temp_nbinY = htemp->GetNbinsY();
        double temp_x[2] = {htemp->GetXaxis()->GetBinLowEdge(1), htemp->GetXaxis()->GetBinLowEdge(temp_nbinX+1)};
        double temp_y[2] = {htemp->GetYaxis()->GetBinLowEdge(1), htemp->GetYaxis()->GetBinLowEdge(temp_nbinY+1)};
        for (int ii=0;ii<Neffbins_model;ii++){
            int xid = cellid_model[ii]/nbinsY;
            int yid = cellid_model[ii]%nbinsY;
            double yy = Y[0]+(yid+0.5)*wbinY;
            double xx = X[0]+(xid+0.5)*wbinX;

            if (xx>=temp_x[1] || xx<=temp_x[0]){
                NTemp_model.push_back(0);
                continue;
            }
            if (yy>=temp_y[1] || yy<=temp_y[0]){
                NTemp_model.push_back(0);
                continue;
            }

            //NTemp_model.push_back(htemp->GetBinContent(xid+1, yid+1));
            NTemp_model.push_back(htemp->Interpolate(xx, yy));
            if (NTemp_model[ii]<=0) NTemp_model[ii] = 0;
        }
        for (int ii=0;ii<Neffbins;ii++){
            int xid = cellid[ii]/nbinsY;
            int yid = cellid[ii]%nbinsY;
            double yy = Y[0]+(yid+0.5)*wbinY;
            double xx = X[0]+(xid+0.5)*wbinX;

            if (xx>=temp_x[1] || xx<=temp_x[0]){
                NTemp.push_back(0);
                continue;
            }
            if (yy>=temp_y[1] || yy<=temp_y[0]){
                NTemp.push_back(0);
                continue;
            }

            //NTemp.push_back(htemp->GetBinContent(xid+1, yid+1));
            NTemp.push_back(htemp->Interpolate(xx, yy));
            if (NTemp[ii]<=0) NTemp[ii] = 0;
        }

    }

    ftemp->Close();

    return 0;

}

void Src_Src::SetLinkPars(int targetSrcID){

    LinkPars = 1;
    TargetSrcID = targetSrcID;

}

void Src_Src::SetLinkPars(int targetSrcClass, int targetSrcID_Class){

    TargetSrcClass = targetSrcClass;
    TargetSrcID_Class = targetSrcID_Class;

}

void Src_Src::Clear(){

    MorPar.clear();
    SEDPar.clear();
    NTemp.clear();
    NTemp_model.clear();

}; 

# endif
