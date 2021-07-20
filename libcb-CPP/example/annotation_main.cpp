#include <iostream>
#include <vector>

#include "../include/libcb/util.h"
#include "../include/libcb/annotation.h"

using namespace std;

void usage(void) {
    cout << "Usage: load/test ontology annotations." << endl;
    cout << "    annotation.run [obo file] [annotation raw file] [sequence id]" << endl << endl;
    cout << "*   [annotation raw file] must be a tab-split format which has two columns:" << endl;
    cout << "    [sequence id] [term id]" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        usage();
        exit(1);
    }

    Ontology ont(argv[1]);
    Annotation ann(ont, argv[2]);
    cout << "number of annotations read: " << ann.number_of_annotations() << endl;
    ann.replenish();

    set<string> terms = ont.leaves_of(ann.annotated_terms_of(argv[3]));
    cout << "annotated terms of [" << argv[3] << "] are: " << endl;
    for (auto const& term : terms) {
        cout << term << "\t" << ont.get_name(term) << endl;
    }
    return 0;
}
