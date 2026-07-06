//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jul 25 11:29:52 2023 by ROOT version 6.24/08
// from TTree km2aevents/km2aevents
// found on file: /home/lhaaso/hushicong/KM2A/DATA/Phase1/KM2AMC_Rec_Filter.root
//////////////////////////////////////////////////////////

#ifndef KM2A_MC_h
#define KM2A_MC_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class KM2A_MC {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Double_t        E;
   Double_t        Theta;
   Double_t        Phi;
   Double_t        corex;
   Double_t        corey;
   Double_t        Angle;
   Double_t        Redge;
   Double_t        Rec_E;
   Double_t        Rec_theta;
   Double_t        Rec_phi;
   Double_t        Rec_x;
   Double_t        Rec_y;

   // List of branches
   TBranch        *b_E;   //!
   TBranch        *b_Theta;   //!
   TBranch        *b_Phi;   //!
   TBranch        *b_corex;   //!
   TBranch        *b_corey;   //!
   TBranch        *b_Angle;   //!
   TBranch        *b_Redge;   //!
   TBranch        *b_Rec_E;   //!
   TBranch        *b_Rec_theta;   //!
   TBranch        *b_Rec_phi;   //!
   TBranch        *b_Rec_x;   //!
   TBranch        *b_Rec_y;   //!

   KM2A_MC(TTree *tree=0);
   virtual ~KM2A_MC();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef KM2A_MC_cxx
KM2A_MC::KM2A_MC(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/home/lhaaso/hushicong/KM2A/DATA/Phase1/KM2AMC_Rec_Filter.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/home/lhaaso/hushicong/KM2A/DATA/Phase1/KM2AMC_Rec_Filter.root");
      }
      f->GetObject("km2aevents",tree);

   }
   Init(tree);
}

KM2A_MC::~KM2A_MC()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t KM2A_MC::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t KM2A_MC::LoadTree(Long64_t entry)
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

void KM2A_MC::Init(TTree *tree)
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

   fChain->SetBranchAddress("E", &E, &b_E);
   fChain->SetBranchAddress("Theta", &Theta, &b_Theta);
   fChain->SetBranchAddress("Phi", &Phi, &b_Phi);
   fChain->SetBranchAddress("corex", &corex, &b_corex);
   fChain->SetBranchAddress("corey", &corey, &b_corey);
   fChain->SetBranchAddress("Angle", &Angle, &b_Angle);
   fChain->SetBranchAddress("Redge", &Redge, &b_Redge);
   fChain->SetBranchAddress("Rec_E", &Rec_E, &b_Rec_E);
   fChain->SetBranchAddress("Rec_theta", &Rec_theta, &b_Rec_theta);
   fChain->SetBranchAddress("Rec_phi", &Rec_phi, &b_Rec_phi);
   fChain->SetBranchAddress("Rec_x", &Rec_x, &b_Rec_x);
   fChain->SetBranchAddress("Rec_y", &Rec_y, &b_Rec_y);
   Notify();
}

Bool_t KM2A_MC::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void KM2A_MC::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t KM2A_MC::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef KM2A_MC_cxx
