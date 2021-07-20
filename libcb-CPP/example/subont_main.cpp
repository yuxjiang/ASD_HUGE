#include <iostream>
#include <fstream>
#include <string>
#include <set>

#include "../include/libcb/util.h"
#include "../include/libcb/ontology.h"

using namespace std;
using namespace arma;

int main(int argc, const char* argv[]) {
  if (argc < 5) {
    cerr << "usage:" << endl;
    cerr << "  subont [original ontology] [subsample terms] [subont term] [subont rel]" << endl;
    exit(1);
  }
  Ontology ont(argv[1]);
  fstream ifs;
  ifs.open(argv[2], ifstream::in);
  set<string> sel;
  string buf;
  while (getline(ifs, buf)) {
    cout << buf << endl;
    sel.insert(buf);
  }
  ifs.close();
  Ontology subont = ont.subontology(sel);
  cout << "subont size: " << subont.size() << endl;

  subont.save_term_as_text(argv[3]);
  subont.save_relationship_as_text(argv[4]);
  return 0;
}
