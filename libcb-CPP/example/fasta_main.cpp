#include <string>
#include "../include/libcb/sequence.h"

using namespace std;

void usage(void) {
    cout << "Usage: load/test fasta IO." << endl;
    cout << "    fasta.run [fasta file] [index]" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        usage();
        exit(1);
    }

    vector<Protein> sequences = fastaread<Protein>(argv[1]);
    index_t index = stoi(argv[2]);
    if (sequences.size() < index) {
        cerr << "[ERROR] index out of range." << endl;
        exit(1);
    } else {
        cout << ">" << sequences[index-1].id << endl;
        cout << sequences[index-1].data << endl;
    }
    return 0;
}
