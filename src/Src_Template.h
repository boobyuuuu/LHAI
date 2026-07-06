# ifndef Src_Template_h
# define Src_Template_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "Src_Src.h"
# include "Src_DGE.h"
# include "Src_Model.h"

using namespace std;

class Src_Template {

    public :

        Src_Template();
        ~Src_Template();
        bool Init();
        string ReplaceEpiv(string sin, string epiv);

        // SED and Morphology models
        Src_Model *Model;

        // Src  Template
        bool AddSource(Src_Src src);
        int NSrc_total;
        int NSrc;
        int NSrc_NumCon;
        int NSrc_2DMor;
        int NSrc_Temp;
        vector<Src_Src> Srcs;
        vector<Src_Src> Srcs_NumCon;
        vector<Src_Src> Srcs_Temp;

        // DGE Template
        bool AddDGE(Src_DGE dge);
        int NDGE;
        vector<Src_DGE> DGEs;

        int NComp;   // NSrc_total + NDGE
        int NTemp;   // NSrc_Temp  + NDGE

        int NparSrc_free;
        int NparDGE_free;
        int Npar_free;
        void SetNparFree(int nparsrc_free, int npardge_free);
};

Src_Template::Src_Template(){ 

    NSrc        = 0;
    NSrc_NumCon = 0;
    NSrc_Temp   = 0;
    NSrc_total  = 0;
    NDGE        = 0;

    NComp = 0;
    NTemp = 0;

    NparSrc_free = 0;
    NparDGE_free = 0;
    Npar_free = 0;

}

Src_Template::~Src_Template(){

    Srcs.clear();
    Srcs.shrink_to_fit();
    Srcs_NumCon.clear();
    Srcs_NumCon.shrink_to_fit();
    Srcs_Temp.clear();
    Srcs_Temp.shrink_to_fit();
    DGEs.clear();
    DGEs.shrink_to_fit();

}

bool Src_Template::Init(){

    Model = new Src_Model();
    return Model->Init();

}

string Src_Template::ReplaceEpiv(string sin, string epiv){
    
    string sout = sin;
    int pos_now = 0;
    while(sout.find("Epiv", pos_now)<=sout.length()){
        pos_now = sout.find("Epiv", 0);
        sout.replace(pos_now, 4, epiv);
    }

    return sout;

}

bool Src_Template::AddSource(Src_Src src){

    if (!Model->MorMap[src.Mortype]){
        cerr<<"\033[31;1mError\033[0m : Undefined Mortype \""<<src.Mortype<<"\"! Exited."<<endl;
        return -1;
    }
    if (!Model->SEDMap[src.SEDtype]){
        cerr<<"\033[31;1mError\033[0m : Undefined SEDtype \""<<src.SEDtype<<"\"! Exited."<<endl;
        return -1;
    }

    string epiv = Form("%.2lf", src.Epiv);
    string sedformula = ReplaceEpiv(Model->SEDFormula[Model->SEDMap[src.SEDtype]-1], epiv);
    sedformula += Form("*%s", src.F0_order.data());
    if (src.LinkPars){
        for (int ii=0;ii<NSrc;ii++){
            if (Srcs[ii].SrcID == src.TargetSrcID){
                src.SetLinkPars(0, ii);
                cout<<" INFO : \""<<src.Srcname<<"\"'s SED linked to \""<<Srcs[ii].Srcname<<"\""<<endl;
            }
        }
        for (int ii=0;ii<NSrc_NumCon;ii++){
            if (Srcs_NumCon[ii].SrcID == src.TargetSrcID){
                src.SetLinkPars(1, ii);
                cout<<" INFO : \""<<src.Srcname<<"\"'s SED linked to \""<<Srcs_NumCon[ii].Srcname<<"\""<<endl;
            }
        }
        for (int ii=0;ii<NSrc_Temp;ii++){
            if (Srcs_Temp[ii].SrcID == src.TargetSrcID){
                src.SetLinkPars(2, ii);
                cout<<" INFO : \""<<src.Srcname<<"\"'s SED linked to \""<<Srcs_Temp[ii].Srcname<<"\""<<endl;
            }
        }
    }

    if (src.Mortype=="Ext_Temp"){
        NTemp++;
        NSrc_Temp++;
        src.SetFormula(Model->MorFormula[Model->MorMap[src.Mortype]-1], sedformula);
        Srcs_Temp.push_back(src);
    }
    else if (src.Mortype=="Ext_gaus" || src.Mortype=="Point" || src.Mortype=="Ext_gaus_E"){
        NSrc++;
        src.SetFormula(Model->MorFormula[Model->MorMap[src.Mortype]-1], sedformula);
        Srcs.push_back(src);
    }
    else{
        NSrc_NumCon++;
        src.SetFormula(Model->MorFormula[Model->MorMap[src.Mortype]-1], sedformula);
        Srcs_NumCon.push_back(src);
    }

    if (Model->MorNDim[Model->MorMap[src.Mortype]-1] == 2)
        NSrc_2DMor++;

    NSrc_total ++;
    NComp ++;

    return 0;

}

bool Src_Template::AddDGE(Src_DGE dge){

    if (!Model->SEDMap[dge.SEDtype]){
        cerr<<"\033[31;1mError\033[0m : Undefined SEDtype \""<<dge.SEDtype<<"\"! Exited."<<endl;
        return -1;
    }

    string epiv = Form("%.2lf", dge.Epiv);
    string sedformula = ReplaceEpiv(Model->SEDFormula[Model->SEDMap[dge.SEDtype]-1], epiv);
    sedformula += Form("*%s", dge.F0_order.data());
    dge.SetFormula(sedformula);
    DGEs.push_back(dge);
    NDGE  ++;
    NTemp ++;
    NComp ++;

    return 0;

}


void Src_Template::SetNparFree(int nparsrc_free, int npardge_free){

    NparSrc_free = nparsrc_free;
    NparDGE_free = npardge_free;
    Npar_free = NparSrc_free+NparDGE_free;

}

# endif
