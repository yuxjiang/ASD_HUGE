#include <string>
#include <iostream>
#include "../include/libcb/distribution.h"

using namespace std;

int main(int argc, const char* argv[]) {
  int a = stoi(argv[1]);
  int b = stoi(argv[2]);
  int c = stoi(argv[3]);
  int d = stoi(argv[4]);
  htest::Tail t = htest::BOTH;
  if (stoi(argv[5]) == 0) {
    t = htest::BOTH;
  } else if (stoi(argv[5]) < 0) {
    t = htest::LEFT;
  } else if (stoi(argv[5]) > 0) {
    t = htest::RIGHT;
  }
  cout << "P-value: " << htest::fishertest<>(a, b, c, d, t) << endl;
  return 0;
}
