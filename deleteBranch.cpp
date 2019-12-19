void deleteBranch()
{
  TTree:TFile *input = TFile::Open("/home/vagrant/data/2nubb/sensitivity_2nubb_2E8_Pre_Cut_2.root"); 
  TTree *inputtree; 
  input->GetObject("Sensitivity",inputtree); 
  TFile *output = TFile::Open("/home/vagrant/data/2nubb/sensitivity_2nubb_2E8_Pre_Cut_2.root","RECREATE"); 
  inputtree->SetBranchStatus("bdt",0); 
  inputtree->SetBranchStatus("weights",0); 
  TTree *outputtree = inputtree->CloneTree(-1,"fast"); 
  output->Write(); 
  delete input; 
  delete output;
}