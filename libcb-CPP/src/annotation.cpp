//! libcb/annotation.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>

#include "../include/libcb/annotation.h"

using namespace std;

void AnnotationTable::load_parsed_gaf_with_ontology(const vector<string>& afiles,
    const Ontology& ont) {
  ifstream ifs;
  string buf;
  // insert one column for each term in the ontology
  vector<string> terms = ont.terms();
  for (const auto& term : terms) {
    add_col(term);
  }
  for (const auto& afile : afiles) {
    ifs.open(afile, ifstream::in);
    while (ifs.good()) {
      getline(ifs, buf);
      if (buf.empty()) continue;
      istringstream iss(buf);
      string seq, term;
      iss >> seq >> term;
      string id = ont.get_id(term); // map to valid id
      if (!id.empty()) {
        add_element(seq, id, gaf::positive); // add a positive annotation
      }
    }
    ifs.close();
  }
}

void Annotation::replenish() {
  vector<string> terms = m_ont.terms();
  // get an order from leaves to root(s)
  reverse(terms.begin(), terms.end());
  for (auto it = terms.begin(); it != terms.end(); ++it) {
    set<string> parents = m_ont.parents(*it);
    for (const auto& p : parents) {
      m_a.transfer_annotated_sequences(*it, p);
    }
  }
  m_replenished = true;
}

void Annotation::deplete() { // TODO
  m_replenished = false;
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Wed 25 Sep 2019 12:00:22 AM P
