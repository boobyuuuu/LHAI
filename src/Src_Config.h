# ifndef Src_Config_h
# define Src_Config_h

# include <iostream>
# include <string>
# include <fstream>
# include <vector>
# include <unistd.h>

using namespace std;

class Src_Config {

    public:
        Src_Config();
        ~Src_Config();

        string WorkDir;
        // Data and response
        int CorOpt;
        double Decrange[2];
        double Decstep;
        double DECstep;
        int NDecstep;
        // WCDA
        int UseWCDA;
        string Version;
        int Nnhit;
        vector<int> Nhit;
        int NnhitUsed;
        int NhitUsed[2];
        double Zenrange[2];
        double Zenstep;
        int Nzenstep;
        double Erange[2];
        double Estep;
        int NEstep;
        string fMap;
        string fResponse1;
        string fResponse2;
        string fSimu1;
        string fSimu2;
        // PSF
        string PSFtype;
        int RebinOpt;
        int *Rebin;

        // KM2A
        int UseKM2A;
        int UseKM2A_NotFull;
        int KNEbin;
        int KNEbinUsed;
        double KDataErange[2];
        double KDataErangeStep;
        int KEbinUsed[2];
        double KZenrange[2];
        double KZenstep;
        int KNzenstep;
        double KErange[2];
        double KEstep;
        int KNEstep;
        string fKMap;
        string fKResponse;
        string fKSimu;
        string KPSFtype;
        double KPSFOffset;

        // ROI
        string ROIfile;
        double ROI_In[6];
        int ROIExOpt;
        double ROI_Ex[5];

        // Src && DGE
        string fInitConifg;

        // Fast Iteration
        int FastIteration;
        // Fit
        int FitOpt[5];

        // TS map
        int FitTSmap;
        int TSmap_WCDA[3];
        int TSmap_KM2A[3];
        int TSmap_SrcID;
        int SubtractOpt;
        string JOBScript;

        // Output
        int OutDrawOpt;
        string Outdir;
        string fOut[2];

        bool Readin(string inputcard, bool outflag);

};

Src_Config::Src_Config(){}
Src_Config::~Src_Config(){}


bool Src_Config::Readin(string inputcard, bool outflag){

    YAML::Node FitConfig = YAML::LoadFile(inputcard.data());

    WorkDir = FitConfig["WorkDir"].as<string>();
    UseWCDA = FitConfig["DataUsed"]["WCDA"]["Active"].as<int>();
    UseKM2A = FitConfig["DataUsed"]["KM2A"]["Active"].as<int>();
    CorOpt  = FitConfig["CorOpt"].as<int>();
    if (!UseWCDA && !UseKM2A){
        cerr<<"\033[31;1mError\033[0m : WCDA not active && KM2A not active! Exited."<<endl;
        return -1;
    }

    // Data and response
    // WCDA
    string fWCDAData    = FitConfig["DataConfig"]["WCDA"].as<string>();
    fWCDAData = WorkDir+"/"+fWCDAData;
    YAML::Node WCDAData = YAML::LoadFile(fWCDAData.data());
    Decrange[0] = WCDAData["Decrange"][0].as<double>();
    Decrange[1] = WCDAData["Decrange"][1].as<double>();
    Decstep     = WCDAData["Decstep"].as<double>();
    DECstep     = WCDAData["DECstep"].as<double>();
    NDecstep    = (Decrange[1]-Decrange[0])/Decstep+0.5;
    NnhitUsed   = 0;
    if (UseWCDA){
        Nnhit      = WCDAData["Nnhit"].as<int>();
        for (int ii=0;ii<Nnhit+1;ii++)
            Nhit.push_back(WCDAData["Nhit"][ii].as<int>());
        NhitUsed[0]  = FitConfig["DataUsed"]["WCDA"]["NbinUsed"][0].as<int>();
        NhitUsed[1]  = FitConfig["DataUsed"]["WCDA"]["NbinUsed"][1].as<int>();
        if (NhitUsed[1]<NhitUsed[0]){
            cerr<<"\033[31;1mError\033[0m : WCDA: NhitUsed[1]<NhitUsed[0]! Exited."<<endl;
            return -1;
        }
        if (NhitUsed[1]>=Nnhit){
            cerr<<"\033[31;1mError\033[0m : WCDA: Bin number used in Fit.yaml larger than maximum bin number! Exited."<<endl;
            return -1;
        }
        NnhitUsed = NhitUsed[1]-NhitUsed[0]+1;
        Version      = WCDAData["Version"].as<string>();
        Zenrange[0]  = WCDAData["Zenrange"][0].as<double>();
        Zenrange[1]  = WCDAData["Zenrange"][1].as<double>();
        if (Zenrange[1]<=Zenrange[0]){
            cerr<<"\033[31;1mError\033[0m : WCDA: Zenrange[1]<=Zenrange[0]! Exited."<<endl;
            return -1;
        }
        Zenstep      = WCDAData["Zenstep"].as<double>();
        Nzenstep     = (Zenrange[1]+5)/Zenstep+0.5;
        Erange[0]    = WCDAData["Erange"][0].as<double>();
        Erange[1]    = WCDAData["Erange"][1].as<double>();
        if (Erange[1]<=Erange[0]){
            cerr<<"\033[31;1mError\033[0m : WCDA: Erange[1]<=Erange[0]! Exited."<<endl;
            return -1;
        }
        Estep        = WCDAData["Estep"].as<double>();
        NEstep       = (Erange[1]-Erange[0])/Estep+0.5;
        fMap         = WCDAData["Mapfile"].as<string>();
        fResponse1   = WCDAData["Responsefile1"].as<string>();
        fResponse2   = WCDAData["Responsefile2"].as<string>();
        fSimu1       = WCDAData["Simufile1"].as<string>();
        fSimu2       = WCDAData["Simufile2"].as<string>();
        PSFtype      = WCDAData["PSFtype"].as<string>();
        RebinOpt     = FitConfig["DataUsed"]["WCDA"]["ReBin"]["Active"].as<int>();
        if (RebinOpt){
            Rebin = new int[RebinOpt];
            for (int ii=0;ii<RebinOpt;ii++)
                Rebin[ii] = FitConfig["DataUsed"]["WCDA"]["ReBin"]["Rebin"][ii].as<int>();
        }
    }

    if (FitConfig["DataUsed"]["KM2A"]["12_and_34"].IsDefined())
        UseKM2A_NotFull = FitConfig["DataUsed"]["KM2A"]["12_and_34"].as<int>();
    else
        UseKM2A_NotFull = 0;
    KNEbinUsed = 0;
    if (UseKM2A){
        string fKM2AData    = FitConfig["DataConfig"]["KM2A"].as<string>();
        fKM2AData = WorkDir+"/"+fKM2AData;
        YAML::Node KM2AData = YAML::LoadFile(fKM2AData.data());
        KNEbin           = KM2AData["KNEbin"].as<int>();
        KDataErange[0]   = KM2AData["DataErange"][0].as<double>();
        KDataErange[1]   = KM2AData["DataErange"][1].as<double>();
        if (KDataErange[1]<=KDataErange[0]){
            cerr<<"\033[31;1mError\033[0m : KM2A: KDataErange[1]<=KDataErange[0]! Exited."<<endl;
            return -1;
        }
        KDataErangeStep  = KM2AData["DataErangeStep"].as<double>();
        double knbinused[2];
        knbinused[0]     = FitConfig["DataUsed"]["KM2A"]["NbinUsed"][0].as<double>();
        knbinused[1]     = FitConfig["DataUsed"]["KM2A"]["NbinUsed"][1].as<double>();
        if (knbinused[1]<=knbinused[0]){
            cerr<<"\033[31;1mError\033[0m : KM2A: knbinused[1]<=knbinused[0]! Exited."<<endl;
            return -1;
        }
        KEbinUsed[0]     = (knbinused[0]-KDataErange[0])/KDataErangeStep+0.5;
        KEbinUsed[1]     = (knbinused[1]-KDataErange[0])/KDataErangeStep-0.5;
        KNEbinUsed = KEbinUsed[1]-KEbinUsed[0]+1;
        KZenrange[0]     = KM2AData["Zenrange"][0].as<double>();
        KZenrange[1]     = KM2AData["Zenrange"][1].as<double>();
        if (KZenrange[1]<KZenrange[0]){
            cerr<<"\033[31;1mError\033[0m : KM2A: KZenrange[1]<KZenrange[0]! Exited."<<endl;
            return -1;
        }
        KZenstep         = KM2AData["Zenstep"].as<double>();
        KNzenstep        = (KZenrange[1]+5)/KZenstep+0.5;
        KErange[0]       = KM2AData["Erange"][0].as<double>();
        KErange[1]       = KM2AData["Erange"][1].as<double>();
        if (KErange[1]<=KErange[0]){
            cerr<<"\033[31;1mError\033[0m : KM2A: KErange[1]<=KErange[0]! Exited."<<endl;
            return -1;
        }
        KEstep           = KM2AData["Estep"].as<double>();
        KNEstep          = (KErange[1]-KErange[0])/KEstep+0.5;
        if (!UseKM2A_NotFull){
            fKMap            = KM2AData["Mapfile"].as<string>();
            fKSimu           = KM2AData["Simufile"].as<string>();
            fKResponse       = KM2AData["Responsefile"].as<string>();
        }
        else{
            fKMap            = KM2AData["Mapfile_All"].as<string>();
            fKSimu           = KM2AData["Simufile_All"].as<string>();
            fKResponse       = KM2AData["Responsefile_All"].as<string>();
        }
        KPSFtype         = KM2AData["PSFtype"].as<string>();
        KPSFOffset       = KM2AData["PSFOffset"].as<double>();
    }

    // ROI
    ROIfile = FitConfig["ROI"]["ROIfile"].as<string>();
    ifstream in;
    in.open(ROIfile.data());
    if (!in.good()){
        for (int ii=0;ii<6;ii++)
            ROI_In[ii] = FitConfig["ROI"]["Include"][ii].as<double>();

        if (ROI_In[1]==0){
            if (ROI_In[4]>ROI_In[5]){
                cerr<<"\033[31;1mError\033[0m : Model radius of ROI smaller than data radius! Returned."<<endl;
                return -1;
            }
        }
        else if (ROI_In[1]==1){
            if (ROI_In[2]>ROI_In[3]){
                cerr<<"\033[31;1mError\033[0m : Xmin of ROI bigger than Xmax! Returned."<<endl;
                return -1;
            }
            if (ROI_In[4]>ROI_In[5]){
                cerr<<"\033[31;1mError\033[0m : Ymin of ROI bigger than Ymax! Returned."<<endl;
                return -1;
            }
        }
        else{
            cerr<<"\033[31;1mError\033[0m : Wrong ROI shape (ROI_In[1]!=0 && ROI_In[1]!=1)! Returned."<<endl;
            return -1;
        }


        ROIExOpt = FitConfig["ROI"]["Exclude"]["Active"].as<int>();
        if (ROIExOpt){
            for (int jj=0;jj<5;jj++)
                ROI_Ex[jj] = FitConfig["ROI"]["Exclude"]["Region"][jj].as<double>();
        }
    }
    else
        in.close();

    // Src && DGE
    //fInitConifg = FitConfig["ParInit"].as<string>();

    // Fast Iteration
    FastIteration = 1;
    if (FitConfig["FastIteration"].IsDefined())
        FastIteration = FitConfig["FastIteration"].as<int>();

    // Fit
    int ii = 0;
    for (YAML::const_iterator it=FitConfig["Fit"].begin(); it!=FitConfig["Fit"].end();++it)
        FitOpt[ii++] = it->second.as<int>();

    /*if (FitOpt[4] && !FitOpt[3]){
        cerr<<"\033[31;1mError\033[0m : \"TS_Bin\" option must be actived if \"FluxUL\" option is actived! Exited."<<endl;
        return -1;
    }*/

    // TSmap
    FitTSmap = FitConfig["TSmap"]["Active"].as<int>();
    for (int jj=0;jj<3;jj++)
        TSmap_WCDA[jj] = FitConfig["TSmap"]["WCDA"][jj].as<int>();
    if (FitTSmap && !UseWCDA && TSmap_WCDA[0]){
        cerr<<"\033[31;1mError\033[0m : WCDA is not active in global fitting but active in TSmap fitting! Exited"<<endl;
        return -1;
    }
    if (FitTSmap && TSmap_WCDA[0] && TSmap_WCDA[1]<NhitUsed[0]){
        cerr<<"\033[31;1mError\033[0m : WCDA : Minimum bin number used in TSmap smaller than that in DataUsed! Exited"<<endl;
        return -1;
    }
    if (FitTSmap && TSmap_WCDA[0] && TSmap_WCDA[2]>NhitUsed[1]){
        cerr<<"\033[31;1mError\033[0m : WCDA : Maximum bin number used in TSmap larger than that in DataUsed! Exited"<<endl;
        return -1;
    }
    double tsmap_km2a[3];
    for (int jj=0;jj<3;jj++)
        tsmap_km2a[jj] = FitConfig["TSmap"]["KM2A"][jj].as<double>();
    TSmap_KM2A[0] = (int) tsmap_km2a[0];
    if (FitTSmap && !UseKM2A && TSmap_KM2A[0]){
        cerr<<"\033[31;1mError\033[0m : KM2A is not active in global fitting but active in TSmap fitting! Exited"<<endl;
        return -1;
    }
    if (UseKM2A){
        TSmap_KM2A[1] = (tsmap_km2a[1]-KDataErange[0])/KDataErangeStep+0.5;
        TSmap_KM2A[2] = (tsmap_km2a[2]-KDataErange[0])/KDataErangeStep-0.5;
    }
    TSmap_SrcID = FitConfig["TSmap"]["SrcID"].as<int>();
    if (TSmap_SrcID<-1){
        cerr<<"\033[31;1mError\033[0m : Invalid TSmap SrcID (input value = "<<TSmap_SrcID<<")! Exited"<<endl;
        return -1;
    }
    SubtractOpt = FitConfig["TSmap"]["Subtract"].as<int>();
    JOBScript = FitConfig["TSmap"]["JOBScript"].as<string>();

    // Output
    OutDrawOpt = FitConfig["Output"]["DrawOpt"].as<int>();
    Outdir  = FitConfig["Output"]["Outdir"].as<string>();
    fOut[0] = FitConfig["Output"]["fParResu"].as<string>();
    fOut[1] = FitConfig["Output"]["fConExcess"].as<string>();

    if (FitTSmap && fOut[0]=="none"){
        cerr<<"\033[31;1mError\033[0m : fParResu is none but FitTSmap is active! Exited"<<endl;
        return -1;
    }
    if (FitTSmap && fOut[1]=="none"){
        cerr<<"\033[31;1mError\033[0m : fConExcess is none but FitTSmap is active! Exited"<<endl;
        return -1;
    }

    if (outflag){
        cout<<" ****** Input parameters ****** "<<endl;
        cout<<"    WorkDir       : "<<WorkDir<<endl;
        cout<<"    UseWCDA       : "<<UseWCDA<<endl;
        cout<<"    UseKM2A       : "<<UseKM2A<<endl;
        cout<<"    CorOpt        : "<<CorOpt<<endl;
        cout<<"    Decrange      : "<<Decrange[0]<<" - "<<Decrange[1]<<endl;
        cout<<"    Decstep       : "<<Decstep
            <<"\n    NDecstep      : "<<NDecstep<<endl;
        if (UseWCDA){
            cout<<" ****** WCDA data and response : "<<endl;
            cout<<"    Version       : "<<Version<<endl;
            cout<<"    Nnhit         : "<<Nnhit<<endl;
            cout<<"    Nhit          : ";
            for (int ii=0;ii<Nnhit+1;ii++)
                cout<<Nhit[ii]<<", ";
            cout<<endl;
            cout<<"    NhitUsed      : "<<NhitUsed[0]<<" - "<<NhitUsed[1]<<endl;
            cout<<"    NnhitUsed     : "<<NnhitUsed<<endl; 
            cout<<"    Zenrange      : "<<Zenrange[0]<<" - "<<Zenrange[1]<<endl;
            cout<<"    Zenstep       : "<<Zenstep
                <<"\n    Nzenstep      : "<<Nzenstep<<endl;
            cout<<"    Erange        : "<<Erange[0]<<" - "<<Erange[1]<<endl;
            cout<<"    Estep         : "<<Estep
                <<"\n    NEstep        : "<<NEstep<<endl;
            cout<<"    fMap          : "<<fMap<<endl;
            cout<<"    fResponse     : "<<fResponse1<<endl;
            cout<<"    fSimu         : "<<fSimu1<<endl;
            cout<<"    PSFtype       : "<<PSFtype<<endl;
            if (RebinOpt){
                cout<<"    Rebin         : ";
                for (int ii=0;ii<RebinOpt;ii++)
                    cout<<Rebin[ii]<<", ";
                cout<<endl;
            }
        }
        if (UseKM2A){
            cout<<" ****** KM2A data and response : "<<endl;
            cout<<"    1/2&3/4 Data  : "<<UseKM2A_NotFull<<endl;
            cout<<"    KNEbin        : "<<KNEbin<<endl;
            cout<<"    KEbinUsed     : "<<KEbinUsed[0]<<" - "<<KEbinUsed[1]<<endl;
            cout<<"    KNEbinUsed    : "<<KNEbinUsed<<endl;
            cout<<"    KZenrange     : "<<KZenrange[0]<<" - "<<KZenrange[1]<<endl;
            cout<<"    KZenstep      : "<<KZenstep
                <<"\n    KNzenstep     : "<<KNzenstep<<endl;
            cout<<"    KErange       : "<<KErange[0]<<" - "<<KErange[1]<<endl;
            cout<<"    KEstep        : "<<KEstep
                <<"\n    KNEstep       : "<<KNEstep<<endl;
            cout<<"    fKMap         : "<<fKMap<<endl;
            cout<<"    fKSimu        : "<<fKSimu<<endl;
            cout<<"    fKResponse    : "<<fKResponse<<endl;
            cout<<"    KPSFtype      : "<<KPSFtype<<endl;
        }
        cout<<" ****** ROI : "<<endl;
        cout<<"    ROIfile       : "<<ROIfile<<endl;
        cout<<"    ROI_In        : ";
        for (int ii=0;ii<6;ii++)
            cout<<ROI_In[ii]<<", ";
        cout<<endl;
        if (ROIExOpt){
            cout<<"    ROI_Ex        : ";
            for (int ii=0;ii<5;ii++)
                cout<<ROI_Ex[ii]<<", ";
            cout<<endl;
        }
        //cout<<" ****** ParInit : "<<endl;
        //cout<<"    fInitConifg   : "<<fInitConifg<<endl;
        cout<<" ****** Fit : "<<endl;
        cout<<"    FastIteration : "<<FastIteration<<endl;
        cout<<"    Fitting       : "<<FitOpt[0]<<endl;
        cout<<"    FluxPoint     : "<<FitOpt[1]<<endl;
        cout<<"    TS_Src        : "<<FitOpt[2]<<endl;
        cout<<"    TS_Bin        : "<<FitOpt[3]<<endl;
        cout<<"    FluxUL        : "<<FitOpt[4]<<endl;
        cout<<" ****** TSmap : "<<endl;
        cout<<"    TSmap         : "<<FitTSmap<<endl;
        cout<<"    TSmap_WCDA    : ";
        for (int ii=0;ii<3;ii++)
            cout<<TSmap_WCDA[ii]<<", ";
        cout<<endl;
        cout<<"    TSmap_KM2A    : ";
        for (int ii=0;ii<3;ii++)
            cout<<TSmap_KM2A[ii]<<", ";
        cout<<endl;
        cout<<"    TSmap_SrcID   : "<<TSmap_SrcID<<endl;
    }
    cout<<" ****** OutPut : "<<endl;
    cout<<"    OutDrawOpt    : "<<OutDrawOpt<<endl;
    cout<<"    Outdir        : "<<Outdir<<endl;
    cout<<"    fParResu      : "<<fOut[0]<<endl;
    cout<<"    fConExcess    : "<<fOut[1]<<endl;

    return 0;
}

# endif
