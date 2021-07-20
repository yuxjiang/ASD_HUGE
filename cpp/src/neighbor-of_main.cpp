#include <iostream>
#include <fstream>
#include <iomanip>
#include <libcb/gene_network.h>

using namespace std;

struct Options {
    string nfile;
    string entrez;
    float cutoff = 0.5;
} opts;

void usage(void) {
    cerr
        << "NAME\n"
        << "    neighbor-of.run - neighbor of\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    nbp.run -n <NETWORK FILE> -g <ENTREZ ID> [OPTION]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program prints the neighbor of a given genes in a network.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n <NETWORK FILE>\n"
        << "        A network file in BINARY format, i.e., a serialized network.\n"
        << "\n"
        << "    -g <ENTREZ ID>\n"
        << "        An entrez ID of a gene of interest.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -cutoff <Real, non-negative> default: 0.5\n"
        << "        The network edge weight cutoff.\n"
        << "\n";
}

void parse_arguments(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    int i = 1;
    while (i < argc) {
        if (string(argv[i]).compare("-g") == 0) {
            opts.entrez = argv[++i];
        } else if (string(argv[i]).compare("-n") == 0) {
            opts.nfile = argv[++i];
            if (!has_file(opts.nfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Network file does not exist.");
            }
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Cutoff should be >= 0.");
            }
        } else {
            usage();
            error_and_exit(__PRETTY_FUNCTION__, "Unknown option.");
        }
        i += 1;
    }
    // check mandatory arguments
    if (opts.nfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No network file specified.");
    }
    if (opts.entrez.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No gene specified.");
    }
}

int main(int argc, const char* argv[]) {
    parse_arguments(argc, argv);
    // make a gene
    Gene gene(opts.entrez);
    // load the gene network
    SGraph<Gene> gn;
    ifstream ifs(opts.nfile, ifstream::in);
    gn.deserialize(ifs);
    ifs.close();
    // filter edges and rewire if necessary
    gn.filter_edges(opts.cutoff);
    // find the gene index
    vector<Gene> genes = gn.get_vertices();
    index_t index = genes.size();
    for (size_t i = 0; i < genes.size(); ++i) {
        if (genes[i] == gene) {
            index = i;
            break;
        }
    }
    if (index == genes.size()) {
        cerr << "Gene is not in this network.\n";
        exit(1);
    }
    vector<index_t> neighbors = gn.neighbor_of(index);
    for (const auto& neighbor : neighbors) {
        cout << genes[neighbor].id << "\n";
    }
    return 0;
}
