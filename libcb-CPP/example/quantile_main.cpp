#include <fstream>
#include <iostream>
#include "../include/libcb/util.h"

using namespace std;

int main(int argc, const char* argv[]) {
  vector<double> data = load_items<double>(argv[1]);
  cout << quantile<double>(data, stod(argv[2])) << endl;
  return 0;
}
