double exposureYears = 2.5;
double AVOGADRO=6.022140e23;
double xmin = -1;
double xmax = 1;

double EstimateBackgroundEvents(double backgroundEfficiency, double isotopeMass, double molarMass, double halfLife)
{
  // Get the number of atoms you start with
  double nSourceAtoms=AVOGADRO * (isotopeMass*1000)/molarMass; //Avogadro is grams
  // The exponent is around 10^-20, it's too small for TMath::Exp to deal with
  // Can we just go for a Taylor expansion for 1-e^-x where x is v small?
  // 1( - e^-x) ~ x so...
  double totalDecays=nSourceAtoms * (TMath::Log(2) * exposureYears/halfLife);
  // Multiply by the efficiency and that is the amount of background events you expect to see
  double events=totalDecays * backgroundEfficiency;
  //cout<<totalDecays<<" backgrounds, of which we see "<<events<<endl;
  return events;
}
// Adapted from James Mott's LimitCalculationFunctions, thanks James!
double ExpectedLimitSigEvts(double ConfidenceLevel, TH1D* h_signal, TH1D* h_background, TH1D* h_data ) {

  double low_bound = 0.1/h_signal->Integral();
  double high_bound = 1000.0/h_signal->Integral();

  TH1D* null_hyp_signal = (TH1D*) h_signal->Clone("null_hyp_signal"); null_hyp_signal->Scale(low_bound);
  TH1D* disc_hyp_signal = (TH1D*) h_signal->Clone("disc_hyp_signal"); disc_hyp_signal->Scale(high_bound);

  TLimitDataSource* mydatasource = new TLimitDataSource(null_hyp_signal, h_background, h_data);

  //50000 is the number of MC experiments to produce
  TConfidenceLevel* myconfidence = TLimit::ComputeLimit(mydatasource, 50000);
  double low_bound_cl = myconfidence->CLs();

  delete mydatasource;

  mydatasource = new TLimitDataSource(disc_hyp_signal, h_background, h_data);

  myconfidence = TLimit::ComputeLimit(mydatasource, 50000);

  double high_bound_cl = myconfidence->CLs();

  delete mydatasource;

  double accuracy = 0.01;
  double this_cl = 0;
  double this_val = 0;

  while  (fabs(high_bound - low_bound) * h_signal->Integral() > accuracy) {
    // bisection
     this_val = low_bound+(high_bound - low_bound)/3;

    TH1D* this_signal = (TH1D*) h_signal->Clone("test_signal");
    this_signal->Scale(this_val);

    mydatasource = new TLimitDataSource(this_signal, h_background, h_data);
    myconfidence = TLimit::ComputeLimit(mydatasource, 50000);

    this_cl = myconfidence->GetExpectedCLs_b();
    if (this_cl > ConfidenceLevel) {
      low_bound = this_val;
      low_bound_cl = this_cl;
    } else {
      high_bound = this_val;
      high_bound_cl = this_cl;
    }

    delete mydatasource;
    delete this_signal;
    delete myconfidence;
  }

  delete null_hyp_signal;
  delete disc_hyp_signal;

  return h_signal->Integral() * this_val;
}

void plotPredictions()
{
  // Read in the BDT results for each category
  string bdt_0nu="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/equalWeights/test-pred-0nu-1E7.root";
  string bdt_2nu="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/equalWeights/test-pred-2nu-2E8.root";
  string bdt_bi_Foils="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/equalWeights/test-pred-bifoil-2E8.root";
  string bdt_tl_Foils="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/equalWeights/test-pred-tlfoil-2E8.root";
  string bdt_radon="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/equalWeights/test-pred-radon-2E8.root";

  TFile *fbdt_0nu=new TFile(bdt_0nu.c_str());
  TFile *fbdt_2nu=new TFile(bdt_2nu.c_str());
  TFile *fbdt_bi_Foils=new TFile(bdt_bi_Foils.c_str());
  TFile *fbdt_tl_Foils=new TFile(bdt_tl_Foils.c_str());
  TFile *fbdt_radon=new TFile(bdt_radon.c_str());

  TTree *tbdt_0nu=(TTree*)fbdt_0nu->Get("BDToutput");
  TTree *tbdt_2nu=(TTree*)fbdt_2nu->Get("BDToutput");
  TTree *tbdt_bi_Foils=(TTree*)fbdt_bi_Foils->Get("BDToutput");
  TTree *tbdt_tl_Foils=(TTree*)fbdt_tl_Foils->Get("BDToutput");
  TTree *tbdt_radon=(TTree*)fbdt_radon->Get("BDToutput");

  TH1F *h1 = new TH1F("h1", "h1", 70, -1, 1);
  TH1F *h2 = new TH1F("h2", "h2", 70, -1, 1);
  TH1F *h3 = new TH1F("h3", "h3", 70, -1, 1);
  TH1F *h4 = new TH1F("h4", "h4", 70, -1, 1);
  TH1F *h5 = new TH1F("h5", "h5", 70, -1, 1);
  h4->SetTitle("BDT scores for all events, normalised to expected number of events");
  h4->GetYaxis()->SetTitle("Arbitrary Units");
  h4->GetXaxis()->SetTitle("BDT Output");
  //h2->GetXaxis()->SetRangeUser(-0.7,1);

  gStyle->SetOptStat(0);

  h1->SetLineColor(kRed);
  h2->SetLineColor(kBlue);
  h3->SetLineColor(kOrange);
  h4->SetLineColor(kGreen);
  h5->SetLineColor(kMagenta);

  double y_0nu=0;
  double y_2nu=0;
  double y_biFoil=0;
  double y_tlFoil=0;
  double y_radon=0;

  tbdt_0nu->SetBranchAddress("y", &y_0nu);
  tbdt_2nu->SetBranchAddress("y", &y_2nu);
  tbdt_bi_Foils->SetBranchAddress("y", &y_biFoil);
  tbdt_tl_Foils->SetBranchAddress("y", &y_tlFoil);
  tbdt_radon->SetBranchAddress("y", &y_radon);

  int entries0nu=tbdt_0nu->GetEntries();
  int entries2nu=tbdt_2nu->GetEntries();
  int entriesBiFoil=tbdt_bi_Foils->GetEntries();
  int entriesTlFoil=tbdt_tl_Foils->GetEntries();
  int entriesRadon=tbdt_radon->GetEntries();

  for(int entry = 0; entry < entries0nu; entry++){
    tbdt_0nu->GetEntry(entry);
    h1->Fill(y_0nu);
  }
  for(int entry = 0; entry < entries2nu; entry++){
    tbdt_2nu->GetEntry(entry);
    h2->Fill(y_2nu);
  }
  for(int entry = 0; entry < entriesBiFoil; entry++){
    tbdt_bi_Foils->GetEntry(entry);
    h3->Fill(y_biFoil);
  }
  for(int entry = 0; entry < entriesTlFoil; entry++){
    tbdt_tl_Foils->GetEntry(entry);
    h4->Fill(y_tlFoil);
  }
  for(int entry = 0; entry < entriesRadon; entry++){
    tbdt_radon->GetEntry(entry);
    h5->Fill(y_radon);
  }
  double zeroNuEfficiency=entries0nu/1E5;
  cout<<"0nubb efficiency: "<<zeroNuEfficiency<<endl;

  double se82IsotopeMass=6.2;
  // double se82MinEnergy=2;
  // double se82MaxEnergy=3.2;
  double se82MolarMass=82;
  double se82HalfLife=10.07e19;

  // h2->Sumw2();
  // h3->Sumw2();
  // h4->Sumw2();
  // h5->Sumw2();

  // h2->Scale(1/h2->Integral());
  // h3->Scale(1/h3->Integral());
  // h4->Scale(1/h4->Integral());
  // h5->Scale(1/h5->Integral());

  TH1D *totalBkgd=(TH1D*)h2->Clone();
  totalBkgd->Add(h3);
  totalBkgd->Add(h4);
  totalBkgd->Add(h5);

  TH1D *tempSignal=(TH1D*)h1->Clone();
  // tempSignal->Draw("hist");

  TH1D *tempData=(TH1D*)totalBkgd->Clone();
  // tempData->Draw("hist SAME");

  // tempSignal->SetTitle("BDT scores for all events");
  // tempSignal->GetYaxis()->SetTitle("Arbitrary Units");
  // tempSignal->GetXaxis()->SetTitle("BDT Output");
  
  // TLegend *leg = new TLegend(0.1293878,0.5787037,0.295102,0.9074074);
  // leg->SetFillColor(0);
  // leg->AddEntry(tempSignal,"Signal 0nubb","l");
  // leg->AddEntry(tempData,"Background","l");
  // leg->Draw("same");

  //returns the number of signal events expected?
  double totalExpectedSignalEventLimit=ExpectedLimitSigEvts(0.1, tempSignal,totalBkgd, tempData);
  cout<<"Total expected signal event limit: "<<totalExpectedSignalEventLimit<<endl;
  double totalTLimitSensitivity= (zeroNuEfficiency/totalExpectedSignalEventLimit) * ((se82IsotopeMass*1000 * AVOGADRO)/se82MolarMass) * TMath::Log(2) * exposureYears;

  TH1D *scaledSignal=(TH1D*)h1->Clone();
  double h1_scale=((totalExpectedSignalEventLimit)/h1->Integral());
  scaledSignal->Sumw2();

  scaledSignal->Scale(h1_scale);

  h4->Draw("hist");
  scaledSignal->Draw("hist SAME");
  h3->Draw("hist SAME");
  h2->Draw("hist SAME");
  h5->Draw("hist SAME");

  TAxis *axis = scaledSignal->GetXaxis();
  int bmin = axis->FindBin(xmin);
  int bmax = axis->FindBin(xmax);
  double integral = scaledSignal->Integral(bmin,bmax);
  integral -= ((scaledSignal->GetBinContent(bmin)*(xmin-axis->GetBinLowEdge(bmin)))/axis->GetBinWidth(bmin));
  integral -= ((scaledSignal->GetBinContent(bmax)*(axis->GetBinUpEdge(bmax)-xmax))/axis->GetBinWidth(bmax));

  cout<<"Number of entries 0nubb: "<<integral<<endl;
  cout<<"Number of entries 2nubb: "<<h2->GetEntries()<<endl;
  cout<<"Number of entries {}^{214}Bi: "<<h3->GetEntries()<<endl;
  cout<<"Number of entries {}^{208}Tl: "<<h4->GetEntries()<<endl;
  cout<<"Number of entries Rn: "<<h5->GetEntries()<<endl;

  TLegend *leg = new TLegend(0.1293878,0.5787037,0.295102,0.9074074);
  leg->SetFillColor(0);
  leg->AddEntry(h1,"Signal 0nubb","l");
  leg->AddEntry(h2,"2nubb","l");
  leg->AddEntry(h3,"{}^{214}Bi foil","l");
  leg->AddEntry(h4,"{}^{208}Tl foil","l");
  leg->AddEntry(h5,"Radon","l");
  leg->Draw("same");

  cout<<"Sensitivity from TLimit including background isotopes: "<<totalTLimitSensitivity<<" years "<<endl;

}
