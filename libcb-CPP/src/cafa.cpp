#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "../include/libcb/util.h"
#include "../include/libcb/cafa.h"
#include "../include/libcb/ontology.h"

using namespace std;
using namespace arma;

dmat_t cafa_load(const Ontology& ont, const string& filename) {
  dmat_t pred;
  vector<string> sequences, terms = ont.terms();
  if (terms.empty()) {
    return pred;
  }
  // collect predictions
  map<string, unordered_map<string, double>> cafa; // raw cafa prediction
  string buf;
  ifstream ifs;
  ifs.open(filename, ifstream::in);
  while (getline(ifs, buf)) {
    if (buf.empty()) continue;
    istringstream iss(buf);
    string s, t;
    double score;
    iss >> s;
    if (s.compare("AUTHOR") == 0 ||
        s.compare("MODEL") == 0 ||
        s.compare("KEYWORDS") == 0 ||
        s.compare("ACCURACY") == 0) {
      continue;
    }
    if (s.compare("END") == 0) {
      break;
    }
    iss >> t >> score;
    string validated_term = ont.get_id(t);
    if (validated_term.empty()) {
      // unknown term ID
      continue;
    }
    cafa[s][validated_term] = score;
  }
  ifs.close();
  // consolidate
  if (!cafa.empty()) {
    unordered_map<string, index_t> sindex, tindex;
    for (const auto row : cafa) {
      sindex[row.first] = sequences.size();
      sequences.push_back(row.first);
    }
    pred = dmat_t(sequences, terms);
    for (index_t i = 0; i < terms.size(); ++i) {
      tindex[terms[i]] = i;
    }
    // initialize "pred"
    for (const auto row : cafa) {
      for (const auto entry : row.second) {
        pred.data(sindex[row.first], tindex[entry.first]) = entry.second;
      }
    }

    for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
      string& term = *it;
      index_t j = tindex[term];
      set<string> parents = ont.parents(term);
      for (const auto& p : parents) {
        index_t u = tindex[p];
        for (index_t i = 0; i < pred.data.n_rows; ++i) {
          pred.data(i, u) = std::max(pred.data(i, u), pred.data(i, j));
        }
      }
    }
  }
  return pred;
}

void cafa_save(const string& filename, const mat& pred, const vector<string>& seqs,
    const vector<string>& terms, index_t max_terms) {
  if (pred.min() < 0 || pred.max() > 1) {
    error_and_exit(__PRETTY_FUNCTION__, "Predictions fall out of the range [0, 1].");
  }
  if (seqs.size() != pred.n_rows) {
    error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of sequences.");
  }
  if (terms.size() != pred.n_cols) {
    error_and_exit(__PRETTY_FUNCTION__, "Inconsistent number of terms.");
  }
  ofstream ofs;
  ofs.open(filename, ofstream::out);
  double min_score = 0;
  index_t skip_count = 0;
  int num_digits = 6;
  double scale = std::pow(10, num_digits);
  for (size_t i = 0; i < seqs.size(); ++i) {
    vec pred_per_seq = pred.row(i).t();
    pred_per_seq.replace(datum::nan, 0); // takes care of NaN
    /**
     * [deprecated] 0-1 normalization
     * pred_per_seq /= pred_per_seq.max();
     * this is probably not a good idea
     */
    uvec order = sort_index(pred_per_seq, "descend");
    pred_per_seq = round(pred_per_seq * scale) / scale;
    // if (pred_per_seq.max() < 0.01) {
    //   cerr << "scores are too small for " << seqs[i] << ", skipping" << endl;
    //   skip_count ++;
    //   continue;
    // }
    for (size_t j = 0; j < std::min(max_terms, (index_t)terms.size()); ++j) {
      double score = pred_per_seq(order(j));
      if (score <= min_score) break;
      ofs << seqs[i] << "\t" << terms[order(j)] << "\t"
        << fixed << setprecision(num_digits) << score << "\n";
    }
  }
  ofs.close();
  if (skip_count > 0) {
    cerr << "skipped " << skip_count << " out of " << seqs.size() << " sequences." << endl;
  }
}

void cafa_save(const string& filename, const dmat_t& dm, index_t max_terms) {
  cafa_save(filename, dm.data, dm.row_tag, dm.col_tag, max_terms);
}
