#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstdlib>

#include "../include/libcb/util.h"
#include "../include/libcb/ontology.h"

using namespace std;

void usage(void) {
    cout << "Usage: load/test ontologies." << endl;
    cout << "    ont.run [obo source] [bin file]" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        usage();
        exit(1);
    }

    arma::arma_rng::set_seed_random();

    set<string> relationships = {"undefined"};
    Ontology ont(argv[1], relationships);
    cout << "read original OBO with " << ont.size() << " terms." << endl;

    ofstream ofs;
    ofs.open(argv[2], ofstream::out);
    ont.serialize(ofs);
    ofs.close();

    cout << "serialization done." << endl;

    ifstream ifs;
    ifs.open(argv[2], ifstream::in);
    ont.deserialize(ifs);
    ifs.close();

    cout << "read saved binary with " << ont.size() << " terms." << endl;

    cout << ont.size() << endl;
    cout << "branching factor: " << ont.branching_factor() << endl;

    cout << "random sample 100 terms to form a subontology" << endl;
    auto terms = ont.terms();
    set<string> sel;
    auto indices = randsample(terms.size(), 100, false);
    for (index_t i = 0; i < 100; ++i) {
      sel.insert(terms[indices(i)]);
    }
    Ontology subont = ont.subontology(sel);
    cout << "subontology size: " << subont.size() << endl;
    cout << "subontology br: " << subont.branching_factor() << endl;

    // vector<Ontology> onts = ont.split();
    // cout << onts.size() << endl;
    // for (size_t i = 0; i < onts.size(); ++i) {
    //     set<string> rts = onts[i].roots();
    //     if (rts.size() != 1) {
    //         error_and_exit(__PRETTY_FUNCTION__, "More than one roots.");
    //     }
    //     string r = *rts.begin();
    //     cout << "Ontology [" << i + 1 << "]" << endl;
    //     cout << "    root term:        " << r << "\t" << onts[i].get_name(r) << endl;
    //     cout << "    number of terms:  " << onts[i].size() << endl;
    //     cout << "    branching factor: " << onts[i].branching_factor() << endl;
    //     cout << "    overall depth:    " << onts[i].depth() << endl;
    // }

    return 0;
}
