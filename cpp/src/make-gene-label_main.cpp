#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <libcb/util.h>

using namespace std;

void usage() {
    cerr
        << "NAME\n"
        << "    make-gene-label.run - attach label to gene list\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    make-gene-label.run <GENE FILE> <POSITIVE GENE FILE>\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program makes a gene label file ready for running nbp.run.\n"
        << "\n"
        << "    Both files are simply lists of Entrez IDs.\n"
        << "\n";
}

int main(int argc, char const* argv[]) {
    if (argc != 3) {
        usage();
        exit(1);
    }
    // load genes
    vector<string> entrezs = load_items<string>(argv[1]);
    vector<string> positives = load_items<string>(argv[2]);
    set<string> union_set(entrezs.begin(), entrezs.end());
    union_set.insert(positives.begin(), positives.end());
    
    // sort(entrezs.begin(), entrezs.end());
    // load positive genes
    set<string> positive_set(positives.begin(), positives.end());
    for (auto const& entrez : union_set) {
        if (CONTAINS(positive_set, entrez)) {
            cout << entrez << "\t1\n";
        } else {
            cout << entrez << "\t0\n";
        }
    }
    cerr
        << "Read " << positive_set.size() << " positive genes\n"
        << " and " << entrezs.size() << " total genes.\n";
    return 0;
}
