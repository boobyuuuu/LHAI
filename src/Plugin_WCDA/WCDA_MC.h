//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jul 24 22:08:31 2023 by ROOT version 6.24/08
// from TTree wcdaevents/wcdaevents
// found on file: /eos/user/h/hushicong/WCDA/1_CRS/Fullarray_simu_vs_exp/Nig/v8_new8/WCDAMC_Rec_Nig_Filter.root
//////////////////////////////////////////////////////////

#ifndef WCDA_MC_h
#define WCDA_MC_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class WCDA_MC {
    public :
        TTree          *fChain;   //!pointer to the analyzed TTree or TChain
        Int_t           fCurrent; //!current Tree number in a TChain

        // Fixed size dimensions of array or collections stored in the TTree if any.

        // Declaration of leaf types
        Float_t        mctheta;
        Float_t        mcphi;
        Float_t        energy;
        Float_t         theta;
        Float_t         phi;
        Int_t           Nhit;
        Float_t         eweit;

        // List of branches
        TBranch        *b_mctheta;   //!
        TBranch        *b_mcphi;   //!
        TBranch        *b_energy;   //!
        TBranch        *b_theta;   //!
        TBranch        *b_phi;   //!
        TBranch        *b_Nhit;   //!
        TBranch        *b_eweit;   //!

        WCDA_MC(TTree *tree=0);
        virtual ~WCDA_MC();
        virtual Int_t    Cut(Long64_t entry);
        virtual Int_t    GetEntry(Long64_t entry);
        virtual Long64_t LoadTree(Long64_t entry);
        virtual void     Init(TTree *tree);
        virtual void     Loop();
        virtual Bool_t   Notify();
        virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef WCDA_MC_cxx
WCDA_MC::WCDA_MC(TTree *tree) : fChain(0) 
{
    // if parameter tree is not specified (or zero), connect the file
    // used to generate this class and read the Tree.
    if (tree == 0) {
        TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/eos/user/h/hushicong/WCDA/1_CRS/Fullarray_simu_vs_exp/Nig/v8_new8/WCDAMC_Rec_Nig_Filter.root");
        if (!f || !f->IsOpen()) {
            f = new TFile("/eos/user/h/hushicong/WCDA/1_CRS/Fullarray_simu_vs_exp/Nig/v8_new8/WCDAMC_Rec_Nig_Filter.root");
        }
        f->GetObject("wcdaevents",tree);

    }
    Init(tree);
}

WCDA_MC::~WCDA_MC()
{
    if (!fChain) return;
    delete fChain->GetCurrentFile();
}

Int_t WCDA_MC::GetEntry(Long64_t entry)
{
    // Read contents of entry.
    if (!fChain) return 0;
    return fChain->GetEntry(entry);
}
Long64_t WCDA_MC::LoadTree(Long64_t entry)
{
    // Set the environment to read one entry
    if (!fChain) return -5;
    Long64_t centry = fChain->LoadTree(entry);
    if (centry < 0) return centry;
    if (fChain->GetTreeNumber() != fCurrent) {
        fCurrent = fChain->GetTreeNumber();
        Notify();
    }
    return centry;
}

void WCDA_MC::Init(TTree *tree)
{
    // The Init() function is called when the selector needs to initialize
    // a new tree or chain. Typically here the branch addresses and branch
    // pointers of the tree will be set.
    // It is normally not necessary to make changes to the generated
    // code, but the routine can be extended by the user if needed.
    // Init() will be called many times when running on PROOF
    // (once per file to be processed).

    // Set branch addresses and branch pointers
    if (!tree) return;
    fChain = tree;
    fCurrent = -1;
    fChain->SetMakeClass(1);

    fChain->SetBranchAddress("mctheta", &mctheta, &b_mctheta);
    fChain->SetBranchAddress("mcphi", &mcphi, &b_mcphi);
    fChain->SetBranchAddress("energy", &energy, &b_energy);
    fChain->SetBranchAddress("theta", &theta, &b_theta);
    fChain->SetBranchAddress("phi", &phi, &b_phi);
    fChain->SetBranchAddress("Nhit", &Nhit, &b_Nhit);
    fChain->SetBranchAddress("eweit", &eweit, &b_eweit);
    Notify();
}

Bool_t WCDA_MC::Notify()
{
    // The Notify() function is called when a new file is opened. This
    // can be either for a new TTree in a TChain or when when a new TTree
    // is started when using PROOF. It is normally not necessary to make changes
    // to the generated code, but the routine can be extended by the
    // user if needed. The return value is currently not used.

    return kTRUE;
}

void WCDA_MC::Show(Long64_t entry)
{
    // Print contents of entry.
    // If entry is not specified, print current entry
    if (!fChain) return;
    fChain->Show(entry);
}
Int_t WCDA_MC::Cut(Long64_t entry)
{
    // This function may be called from Loop.
    // returns  1 if entry is accepted.
    // returns -1 otherwise.
    return 1;
}
#endif // #ifdef WCDA_MC_cxx
