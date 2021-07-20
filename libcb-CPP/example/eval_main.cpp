#include "../include/libcb/evaluation.h"

using namespace std;
int main(int argc, char* argv[]) {
  if (argc != 3) {
    cerr << "incorrect number of arguments." << endl;
    exit(1);
  }
  // load prediction and label
  vector<double> prediction = load_items<double>(argv[1]);
  vector<bool> label = load_items<bool>(argv[2]);
  cout << "AUC: " << get_auc<double>(prediction, label) << endl;
  return 0;
}

// end of file
