void temp(){

    const int nwcda = 7;
    const int nkm2a = 10;
    double ratiow = 0.6;
    double ratiok = 0.8;
    double timew = 40/12.;
    double timek = 3+300./365;
    double energyw[nwcda] = {0.556, 0.862, 1.424, 2.389, 3.874, 7.255, 14.827};
    double energyk[nkm2a] = {28.672, 45.493, 71.768, 112.962, 179.676, 283.935, 445.481, 707.241, 1122.856, 1790.415};

    double excessw[nwcda];
    double excessk[nkm2a];
    TFile *fin = TFile::Open("ConExcess.root");
    for (int ii=0;ii<(nwcda+nkm2a);ii++){
        TH2D *htemp = (TH2D *) fin->Get(Form("hExcess_%d_0", ii));
        if (ii<nwcda)
            excessw[ii] = htemp->GetSumOfWeights()/ratiow/timew;
        else
            excessk[ii-nwcda] = htemp->GetSumOfWeights()/ratiok/timek;
    }


    gStyle->SetGridColor(kGray);
    gStyle->SetGridStyle(7);
    TGraph *gexw = new TGraph(nwcda, energyw, excessw);
    TGraph *gexk = new TGraph(nkm2a, energyk, excessk);
    TCanvas *cc = new TCanvas("cc", "cc", 1000, 700);
    cc->SetLogx();
    cc->SetLogy();
    cc->SetGridx();
    cc->SetGridy();
    gexw->GetXaxis()->SetLimits(0.3, 5000);
    gexw->GetYaxis()->SetRangeUser(0.05, 5.e5);
    gexw->SetTitle(";Energy [ TeV ];Number of photons / year w/o G/P");
    gexw->SetMarkerStyle(20);
    gexw->SetMarkerSize(2.0);
    gexw->SetLineColor(kBlack);
    gexw->SetLineWidth(3.0);
    gexw->SetMarkerColor(kGreen+1);
    gexw->Draw("APL");
    gexk->SetMarkerStyle(20);
    gexk->SetMarkerSize(2.0);
    gexk->SetLineColor(kBlack);
    gexk->SetLineWidth(3.0);
    gexk->SetMarkerColor(kBlue);
    gexk->Draw("PLsame");
    TLegend *ll = new TLegend();
    ll->AddEntry(gexw, "WCDA", "PL");
    ll->AddEntry(gexk, "KM2A", "PL");
    ll->SetFillStyle(0);
    ll->SetBorderSize(0);
    ll->Draw();

}
