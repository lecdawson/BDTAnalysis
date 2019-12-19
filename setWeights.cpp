double exposureYears = 2.5;
double AVOGADRO=6.022140e23;

void addBranch(string filename, double weight) {
    TFile *f = new TFile(filename.c_str(), "UPDATE");

    double new_v;
    TTree *tNew = (TTree*)f->Get("Sensitivity");
    TBranch *newBranch = tNew->Branch("weights", &new_v, "weights/d");
    Long64_t nentries = tNew->GetEntries(); // read the number of entries in the tree
    for (Long64_t i = 0; i < nentries; i++) {
        new_v= weight;
        newBranch->Fill();
    }
    tNew->Write("", TObject::kOverwrite); // save only the new version of the tree
    f->Close();
    delete f;
}

double EstimateBackgroundEvents(double backgroundEfficiency, double isotopeMass, double molarMass, double halfLife)
{
  // Get the number of atoms you start with
  double nSourceAtoms=AVOGADRO * (isotopeMass*1000)/molarMass; //molar mass is in grams
  cout<<"number of source atoms: "<<nSourceAtoms<<endl;
  // The exponent is around 10^-20, it's too small for TMath::Exp to deal with
  // Can we just go for a Taylor expansion for 1-e^-x where x is v small?
  // 1( - e^-x) ~ x so...
  double totalDecays=nSourceAtoms * (TMath::Log(2) * exposureYears/halfLife);
  cout<<"number of total decays: "<<totalDecays<<endl;
  // Multiply by the efficiency and that is the amount of background events you expect to see
  double events=totalDecays * backgroundEfficiency;
  cout<<"number of 2nubb events: "<<events<<endl;
  //cout<<totalDecays<<" backgrounds, of which we see "<<events<<endl;
  return events;
}

void setWeights()
{
  //Below is just for events from the foil
  //Read in all of the sensitivity files for different generators
  string zeroNu="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/0nubb/sensitivity_0nubb_1E7_Pre_Cut.root";
  string twoNu="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/2nubb/sensitivity_2nubb_2E8_Pre_Cut.root";
  string bi_Foils="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/Bi214/sensitivity_Bi214_Foils_2E8_Pre_Cut.root";
  string tl_Foils="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/Tl208/sensitivity_Tl208_Foils_2E8_Pre_Cut.root";
  string radon="/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/Radon/sensitivity_Bi214_Wires_2E8_Pre_Cut.root";

  TFile *f0nubb=new TFile(zeroNu.c_str());
  TFile *f2nubb=new TFile(twoNu.c_str());
  TFile *fbiFoils=new TFile(bi_Foils.c_str());
  TFile *ftlFoils=new TFile(tl_Foils.c_str());
  TFile *fradon=new TFile(radon.c_str());

  TTree *t0nubb=(TTree*)f0nubb->Get("Sensitivity");
  TTree *t2nubb=(TTree*)f2nubb->Get("Sensitivity");
  TTree *tbiFoils=(TTree*)fbiFoils->Get("Sensitivity");
  TTree *ttlFoils=(TTree*)ftlFoils->Get("Sensitivity");
  TTree *tradon=(TTree*)fradon->Get("Sensitivity");

  int entries0nu=t0nubb->GetEntries();
  int entries2nu=t2nubb->GetEntries();
  int entriesBiFoil=tbiFoils->GetEntries();
  int entriesTlFoil=ttlFoils->GetEntries();
  int entriesRadon=tradon->GetEntries();

  //all of the below are using the target activities not the actual values.
  //actual values
  //Bismuth in bulk 4.94 mBq total
  //Bismuth in tracker 4.25 ± 0.48 mBq total
  //Tl208 0.545 mBq

  double activityBiBulk=62.5416E-6; //update 23/09/19 see https://nemo.lpc-caen.in2p3.fr/wiki/NEMO/SuperNEMO/Analysis/Baselines
  double activityBiTracker=2.28E-3;
  double activityBiWire=activityBiTracker*0.922;
  double activityTl=12.50832E-6; //update 23/09/19 see https://nemo.lpc-caen.in2p3.fr/wiki/NEMO/SuperNEMO/Analysis/Baselines

  double biEfficiency=entriesBiFoil/2E8;
  double tlEfficiency=entriesTlFoil/2E8;
  double radonEfficiency=entriesRadon/2E8;
  double twoNuEfficiency=entries2nu/2E8;
  double zeroNuEfficiency=entries0nu/1E7;

  cout<<"2nubb efficiency: "<<twoNuEfficiency<<endl;

  // double biEfficiency=entriesBiFoil/1E5;
  // double tlEfficiency=entriesTlFoil/1E5;
  // double radonEfficiency=entriesRadon/1E5;
  // double twoNuEfficiency=entries2nu/1E5;
  // double zeroNuEfficiency=entries0nu/1E5;

  double se82IsotopeMass=6.3;
  double se82MolarMass=82;
  double se82HalfLife=10.07e19;

  double eventsBi=activityBiBulk * biEfficiency * (exposureYears*3600*24*365.25);
  double eventsTl=activityTl * tlEfficiency * (exposureYears*3600*24*365.25);
  double eventsRn=activityBiWire * radonEfficiency * (exposureYears*3600*24*365.25);

  double events2nu=EstimateBackgroundEvents(twoNuEfficiency,se82IsotopeMass,se82MolarMass,se82HalfLife);

  cout<<"Bi Foil events: "<<eventsBi<<endl;
  cout<<"Tl foil events: "<<eventsTl<<endl;
  cout<<"Radon events: "<<eventsRn<<endl;
  cout<<"2nubb events: "<<events2nu<<endl;

  double totalBkgEvents=eventsBi + eventsTl + eventsRn + events2nu;

  double pct2nu = events2nu/totalBkgEvents;
  double pctBi = eventsBi/totalBkgEvents;
  double pctTl = eventsTl/totalBkgEvents;
  double pctRn = eventsRn/totalBkgEvents;

  double weight0nu = 1;
  double weight2nu = 1;//(pct2nu/entries2nu);
  double weightBi = 1;//(pctBi/entriesBiFoil);
  double weightTl = 1;//(pctTl/entriesTlFoil);
  double weightRn = 1;//(pctRn/entriesRadon);

  cout<<"0nu weight: "<<weight0nu<<endl;
  cout<<"2nu weight: "<<weight2nu<<endl;
  cout<<"Bi weight: "<<weightBi<<endl;
  cout<<"Tl weight: "<<weightTl<<endl;
  cout<<"Rn weight: "<<weightRn<<endl;

  // Add the weights to the trees
  // Create a new branch
  // Fill with the appropriate value for each events
  addBranch(zeroNu,weight0nu);
  addBranch(twoNu,weight2nu);
  addBranch(bi_Foils,weightBi);
  addBranch(tl_Foils,weightTl);
  addBranch(radon,weightRn);

}
