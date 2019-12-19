#include <string>
#include "TFile.h"
void removeTree(){

  std::string file_name = "/home/vagrant/data/Radon/sensitivity_Bi214_Wires_2E8_Pre_Cut_2.root";
  TFile *file=new TFile((file_name).c_str(),"update");
  std::string object_to_remove="Sensitivity;2";
  //the object can be a tree, a histogram, etc, in this case "test1" is a TTree
  //notice the ";1" which means cycle 1; to remove all cycles do ";*"
  //if your object is not at the top directory, but in a directory in the .root file, called foo
  // you do first
  //file->cd("foo");
  //then continue with the Delete command which is only applied to the current gDirectory
  gDirectory->Delete(object_to_remove.c_str());
  file->Close();
}