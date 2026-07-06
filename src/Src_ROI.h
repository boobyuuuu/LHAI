# ifndef Src_ROI_h
# define Src_ROI_h

# include <iostream>
# include <string>
# include <vector>

# include "TH2D.h"
# include "Src_Config.h"

using namespace std;

static int nbinsX = 0, nbinsY = 0;
static double X[2] = {0, 0}, Y[2] = {0, 0}; 
static double wbinX = 0, wbinY = 0;
static Src_Config cf;
static int npar_src = 0, npar_numsrc = 0, npar_dge = 0, npar_total = 0;
static int Niter = 0;
static double FastIter_th = 1.e-10;

class Src_ROI {

    public :

        Src_ROI();
        ~Src_ROI();
        bool Init_Arbitrary();

        double Xcenter;
        double Ycenter;
        double Xcenter_gl;
        double Ycenter_gb;
        double Xroi[2];
        double Yroi[2];
        int nbinsX_roi;
        int nbinsY_roi;
        int Neffbins;
        int Neffbins_model; 
        vector<long int> Cellid;
        vector<long int> Cellid_model;

};

Src_ROI::Src_ROI(){}

Src_ROI::~Src_ROI(){

    Cellid.clear();
    Cellid.shrink_to_fit();
    Cellid_model.clear();
    Cellid_model.shrink_to_fit();

}

bool Src_ROI::Init_Arbitrary(){

    Xcenter = 0;
    Ycenter = 0;
    Xcenter_gl = 0;
    Ycenter_gb = 0;
    Neffbins = 0;
    Neffbins_model = 0;

    TH2D *hROI = new TH2D("hROI", "hROI", nbinsX, X[0], X[1], nbinsY, Y[0], Y[1]);
    TH2D *hROI_model = new TH2D("hROI_model", "hROI_model", nbinsX, X[0], X[1], nbinsY, Y[0], Y[1]);
    Neffbins = 0; Neffbins_model = 0;
    if (cf.ROIfile!="none"){
        TFile *fROI = TFile::Open(cf.ROIfile.data());
        if (!fROI){
            cerr<<"\033[31;1mError\033[0m : ROIfile "<<cf.ROIfile<<" not exist! Exited."<<endl;
            return -1;
        }
        hROI = (TH2D *) fROI->Get("hROI");
        hROI_model = (TH2D *) fROI->Get("hROI_model");
        for (int ii=0;ii<nbinsX;ii++)
            for (int jj=0;jj<nbinsY;jj++){
                if (hROI->GetBinContent(ii+1, jj+1)>0){
                    Cellid.push_back(ii*nbinsY+jj);
                    Neffbins++;
                    Xcenter += hROI->GetXaxis()->GetBinCenter(ii+1);
                    Ycenter += hROI->GetYaxis()->GetBinCenter(jj+1);
                }
                if (hROI_model->GetBinContent(ii+1, jj+1)>0){
                    Cellid_model.push_back(ii*nbinsY+jj);
                    Neffbins_model++;
                }
            }

        Xcenter = Xcenter/Neffbins;
        Ycenter = Ycenter/Neffbins;

        cout<<" Center of ROI : ("<<Xcenter<<", "<<Ycenter<<")"<<endl;

        if (!cf.CorOpt){
            e2g(Xcenter, Ycenter, &Xcenter_gl, &Ycenter_gb);
        }
        else{
            Xcenter_gl = Xcenter;
            Ycenter_gb = Ycenter;
            g2e(Xcenter_gl, Ycenter_gb, &Xcenter, &Ycenter);
        }

        fROI->Close();
    }
    else{
        if (!cf.ROI_In[1]){

            double Xc0 = cf.ROI_In[2];
            double Yc0 = cf.ROI_In[3];
            double model_radius = cf.ROI_In[5];
            double data_radius  = cf.ROI_In[4];
            double radius = model_radius-data_radius;

            double Xc00 = 0, Yc00 = 0;
            if (cf.ROI_In[0]!=cf.CorOpt){
                if (!cf.CorOpt){
                    g2e(Xc0, Yc0, &Xc00, &Yc00);
                    Xcenter = Xc00;
                    Ycenter = Yc00;
                }
                else{
                    Xcenter = Xc0;
                    Ycenter = Yc0;
                    e2g(Xc0, Yc0, &Xc00, &Yc00);
                }
            }
            else{
                Xc00 = Xc0;
                Yc00 = Yc0;
                if (!cf.ROI_In[0]){
                    Xcenter = Xc00;
                    Ycenter = Yc00; 
                }
                else{
                    g2e(Xc0, Yc0, &Xcenter, &Ycenter);
                }
            }
            e2g(Xcenter, Ycenter, &Xcenter_gl, &Ycenter_gb);


            int xbin0 = (Xc00-1.5*model_radius/cos(Yc00*papi::degrad)-X[0])/wbinX;
            int xbin1 = (Xc00+1.5*model_radius/cos(Yc00*papi::degrad)-X[0])/wbinX;
            int ybin0 = (Yc00-1.5*model_radius-Y[0])/wbinY;
            int ybin1 = (Yc00+1.5*model_radius-Y[0])/wbinY;
            int nbinsx = xbin1-xbin0+1;
            int nbinsy = ybin1-ybin0+1;
            /*cout<<" *** main : ROI for fitting morphology : "<<endl;
            cout<<"  X : "<<xbin0<<" - "<<xbin1<<", nbins = "<<nbinsx<<endl;
            cout<<"  Y : "<<ybin0<<" - "<<ybin1<<", nbins = "<<nbinsy<<endl;*/

            double xx0, yy0;
            int ii_temp, mm_temp;
            for (int ii=xbin0;ii<=xbin1;ii++){

                ii_temp = ii;
                if (ii<0) ii_temp = ii+nbinsX;
                if (ii>=nbinsX) ii_temp = ii-nbinsX;

                double xx = X[0]+(ii_temp+0.5)*wbinX;
                for (int jj=ybin0;jj<=ybin1;jj++){
                    double yy = Y[0]+(jj+0.5)*wbinY;

                    if (cf.ROI_In[0]!=cf.CorOpt){
                        if (!cf.ROI_In[0])
                            g2e(xx, yy, &xx0, &yy0);
                        else
                            e2g(xx, yy, &xx0, &yy0);
                    }
                    else{
                        xx0 = xx;
                        yy0 = yy;
                    }

                    double space = distance(90-Yc0, Xc0, 90-yy0, xx0);
                    bool maskflag = 0;
                    if (cf.ROIExOpt){
                        if (!cf.ROI_Ex[0]){
                            double space1 = distance(90-cf.ROI_Ex[2], cf.ROI_Ex[1], 90-yy0, xx0);
                            if (space1<cf.ROI_Ex[3])
                                maskflag = 1;
                        }
                        else{
                            if (xx0>=cf.ROI_Ex[1] && xx0<=cf.ROI_Ex[2] && yy0>=cf.ROI_Ex[3] && yy0<=cf.ROI_Ex[4])
                                maskflag = 1;
                        }
                    }
                    // Data ROI
                    if (space<data_radius && !maskflag){
                        Neffbins++;
                        Cellid.push_back(ii_temp*nbinsY+jj);
                        hROI->SetBinContent(ii_temp+1, jj+1, 1);

                        // Model ROI
                        double ibinx0 = (xx-radius*1.2/cos(yy*papi::degrad)-X[0])/wbinX;
                        double ibinx1 = (xx+radius*1.2/cos(yy*papi::degrad)-X[0])/wbinX;
                        double ibiny0 = (yy-radius*1.2-Y[0])/wbinY;
                        double ibiny1 = (yy+radius*1.2-Y[0])/wbinY;

                        for (int mm=ibinx0;mm<=ibinx1;mm++){

                            mm_temp = mm;
                            if (mm<0) mm_temp = mm+nbinsX;
                            if (mm>=nbinsX) mm_temp = mm-nbinsX;

                            double xx1 = X[0] + (mm_temp+0.5)*wbinX;
                            for (int nn=ibiny0;nn<=ibiny1;nn++){
                                double yy1 = Y[0] + (nn+0.5)*wbinY;

                                space = distance(90-yy, xx, 90-(yy1-0.001), xx1-0.001);
                                if (space<radius && hROI_model->GetBinContent(mm_temp+1, nn+1)<=0){
                                    hROI_model->SetBinContent(mm_temp+1, nn+1, 1);
                                    Cellid_model.push_back(mm_temp*nbinsY+nn);
                                    Neffbins_model++;
                                }
                            }
                        }
                    }
                }   
            }   
            cout<<" Neffbins = "<<Neffbins<<", Neffbins_model = "<<Neffbins_model<<endl;
        }
        else{

            double xcenter = (cf.ROI_In[2]+cf.ROI_In[3])/2;
            double ycenter = (cf.ROI_In[4]+cf.ROI_In[5])/2;
            double space_1 = distance(90-ycenter, xcenter, 90-cf.ROI_In[4], cf.ROI_In[2]);
            double space_2 = distance(90-ycenter, xcenter, 90-cf.ROI_In[4], cf.ROI_In[3]);
            double space_3 = distance(90-ycenter, xcenter, 90-cf.ROI_In[5], cf.ROI_In[2]);
            double space_4 = distance(90-ycenter, xcenter, 90-cf.ROI_In[5], cf.ROI_In[3]);
            double data_radius = TMath::Max(space_1, TMath::Max(space_2, TMath::Max(space_3, space_4)));
            double model_radius = data_radius+2;
            double radius = model_radius - data_radius;

            double Xc00, Yc00;
            int ii_temp, mm_temp;
            if (cf.ROI_In[0]!=cf.CorOpt){
                if (!cf.CorOpt){
                    g2e(xcenter, ycenter, &Xc00, &Yc00);
                    Xcenter = Xc00;
                    Ycenter = Yc00;
                }
                else{
                    Xcenter = xcenter;
                    Ycenter = ycenter;
                    e2g(xcenter, ycenter, &Xc00, &Yc00);
                }
            }
            else{
                Xc00 = xcenter;
                Yc00 = ycenter;
                if (!cf.ROI_In[0]){
                    Xcenter = Xc00;
                    Ycenter = Yc00;
                }
                else{
                    g2e(xcenter, ycenter, &Xcenter, &Ycenter);
                }
            }

            int xbin0 = (Xc00-1.5*model_radius/cos(Yc00*papi::degrad)-X[0])/wbinX;
            int xbin1 = (Xc00+1.5*model_radius/cos(Yc00*papi::degrad)-X[0])/wbinX;
            int ybin0 = (Yc00-1.5*model_radius-Y[0])/wbinY;
            int ybin1 = (Yc00+1.5*model_radius-Y[0])/wbinY;
            int nbinsx = xbin1-xbin0+1;
            int nbinsy = ybin1-ybin0+1;
            /*cout<<" *** main : ROI for fitting morphology : "<<endl;
            cout<<"  X : "<<xbin0<<" - "<<xbin1<<", nbins = "<<nbinsx<<endl;
            cout<<"  Y : "<<ybin0<<" - "<<ybin1<<", nbins = "<<nbinsy<<endl;*/

            double xx0, yy0;
            for (int ii=xbin0;ii<=xbin1;ii++){

                ii_temp = ii;
                if (ii<0) ii_temp = ii+nbinsX;
                if (ii>=nbinsX) ii_temp = ii-nbinsX;

                double xx = X[0]+(ii_temp+0.5)*wbinX;
                for (int jj=ybin0;jj<=ybin1;jj++){
                    double yy = Y[0]+(jj+0.5)*wbinY;

                    if (cf.ROI_In[0]!=cf.CorOpt){
                        if (!cf.ROI_In[0])
                            g2e(xx, yy, &xx0, &yy0);
                        else
                            e2g(xx, yy, &xx0, &yy0);
                    }
                    else{
                        xx0 = xx;
                        yy0 = yy;
                    }

                    bool maskflag = 0;
                    if (cf.ROIExOpt){
                        if (!cf.ROI_Ex[0]){
                            double space1 = distance(90-cf.ROI_Ex[2], cf.ROI_Ex[1], 90-yy0, xx0);
                            if (space1<cf.ROI_Ex[3])
                                maskflag = 1;
                        }
                        else{
                            if (xx0>=cf.ROI_Ex[1] && xx0<=cf.ROI_Ex[2] && yy0>=cf.ROI_Ex[3] && yy0<=cf.ROI_Ex[4])
                                maskflag = 1;
                        }
                    }

                    // Data ROI
                    if (xx0>=cf.ROI_In[2] && xx0<=cf.ROI_In[3] && yy0>=cf.ROI_In[4] && yy0<=cf.ROI_In[5] && !maskflag){
                        Neffbins++;
                        Cellid.push_back(ii_temp*nbinsY+jj);
                        hROI->SetBinContent(ii_temp+1, jj+1, 1);

                        // Model ROI
                        double ibinx0 = (xx-radius*1.2/cos(yy*papi::degrad)-X[0])/wbinX;
                        double ibinx1 = (xx+radius*1.2/cos(yy*papi::degrad)-X[0])/wbinX;
                        double ibiny0 = (yy-radius*1.2-Y[0])/wbinY;
                        double ibiny1 = (yy+radius*1.2-Y[0])/wbinY;

                        for (int mm=ibinx0;mm<=ibinx1;mm++){

                            mm_temp = mm;
                            if (mm<0) mm_temp = mm+nbinsX;
                            if (mm>=nbinsX) mm_temp = mm-nbinsX;

                            double xx1 = X[0] + (mm_temp+0.5)*wbinX;
                            for (int nn=ibiny0;nn<=ibiny1;nn++){
                                double yy1 = Y[0] + (nn+0.5)*wbinY;

                                double space = distance(90-yy, xx, 90-(yy1-0.001), xx1-0.001);
                                if (space<radius && hROI_model->GetBinContent(mm_temp+1, nn+1)<=0){
                                    hROI_model->SetBinContent(mm_temp+1, nn+1, 1);
                                    Cellid_model.push_back(mm_temp*nbinsY+nn);
                                    Neffbins_model++;
                                }
                            }
                        }
                    }
                }   
            }   
            cout<<" Neffbins = "<<Neffbins<<", Neffbins_model = "<<Neffbins_model<<endl;
        }
    }

    Xroi[0] = 360; Xroi[1] = 0; Yroi[0] = 90; Yroi[1] = -90;
    for (int ii=0;ii<Neffbins;ii++){
        double x0 = X[0]+(Cellid[ii]/nbinsY)*wbinX;
        double y0 = Y[0]+(Cellid[ii]%nbinsY)*wbinY;
        if (x0<Xroi[0])
            Xroi[0] = x0;
        if ((x0+0.1)>Xroi[1])
            Xroi[1] = x0+0.1;
        if (y0<Yroi[0])
            Yroi[0] = y0;
        if ((y0+0.1)>Yroi[1])
            Yroi[1] = y0+0.1;
    }
    nbinsX_roi = (Xroi[1]-Xroi[0]+wbinX/10.)/wbinX;
    nbinsY_roi = (Yroi[1]-Yroi[0]+wbinY/10.)/wbinY;

    cout<<" *** main : ROI : "<<endl;
    cout<<" X range: "<<Xroi[0]<<" - "<<Xroi[1]<<endl;
    cout<<" Y range: "<<Yroi[0]<<" - "<<Yroi[1]<<endl;
    cout<<" nbinsX_roi = "<<nbinsX_roi<<", nbinsY_roi = "<<nbinsY_roi<<endl;

    return 0;

}

# endif
