//! libcb/util.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <algorithm>

#include "../include/libcb/util.h"

using namespace std;
using namespace arma;

unordered_map<string, string> parse_config(const string& cfile) {
  unordered_map<string, string> res;
  ifstream ifs(cfile, ifstream::in);
  string buf;
  while (ifs.good()) {
    getline(ifs, buf);
    if (buf.empty()) continue;
    // trim comments
    buf = buf.substr(0, buf.find_first_of('#'));
    if (buf.empty()) continue;
    // split according to =
    size_t eq_pos = buf.find('=');
    if (eq_pos == string::npos) {
      error_and_exit(__PRETTY_FUNCTION__, string("Not a valid configuration line: ") + buf);
    }
    string name  = trim(buf.substr(0, eq_pos));
    string value = trim(buf.substr(eq_pos + 1));
    res[name] = value;
  }
  ifs.close();
  return res;
}

string trim(const string& s, const string& whitespaces) {
  string res = s;
  size_t pos;
  pos = res.find_first_not_of(whitespaces);
  res = (pos == string::npos ? "" : res.substr(pos));
  if (res.empty()) return res;
  pos = res.find_last_not_of(whitespaces);
  res = res.substr(0, pos + 1);
  return res;
}

uvec randsample(index_t n, index_t k, bool replacement) {
  if (n == 0) {
    error_and_exit(__PRETTY_FUNCTION__, "Empty population.");
  }
  if (k == 0) {
    error_and_exit(__PRETTY_FUNCTION__, "Need to have at least one samples.");
  }
  uvec sampled = zeros<uvec>(k);
  // random_device rd;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  if (replacement) {
    std::uniform_int_distribution<index_t> int_distr(0, n - 1);
    std::default_random_engine generator(seed);
    for (size_t i = 0; i < k; ++i) {
      sampled(i) = int_distr(generator);
    }
  } else {
    if (k > n) {
      error_and_exit(__PRETTY_FUNCTION__,
          "Cannot have a sample size more than the population without replacement.");
    }
    sampled = uvec(shuffle(regspace<uvec>(0, n - 1))).subvec(0, k - 1);
  }
  return sampled;
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 17 Sep 2019 11:42:35 PM P
