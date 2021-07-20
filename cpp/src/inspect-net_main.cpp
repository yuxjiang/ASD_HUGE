#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>

#include <libcb/util.h>
#include <libcb/gene_network.h>

using namespace std;

struct {
    string nfile  = "";
    string dfile  = "";
    string gfile  = "";
    float  cutoff = 0.5;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    inspect-net.run - inspect network properties\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    inspect-net.run -n <NETWORK FILE> [OPTION]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program returns the corresponding property of a given network.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n <NETWORK FILE>\n"
        << "        A network file in BINARY format.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -d <FILE>, default: (EMPTY)\n"
        << "        Output degree of each node.\n"
        << "    -cutoff <Real, non-negative>, default: 0.5\n"
        << "        The network edge weight cutoff.\n"
        << "    -goi <FILE>, default: (EMPTY)\n"
        << "        The gene list of interest.\n"
        << "        If given, this program returns how many of them are non-singletons.\n"
        << "\n";
}

void parse_arguments(int argc, char const* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    int i = 1;
    while (i < argc) {
        if (string(argv[i]).compare("-n") == 0) {
            opts.nfile = argv[++i];
            if (!has_file(opts.nfile)) {
                error_and_exit(__PRETTY_FUNCTION__, "Network file does not exist.");
            }
        } else if (string(argv[i]).compare("-d") == 0) {
            opts.dfile = argv[++i];
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Cutoff should be >= 0.");
            }
        } else if (string(argv[i]).compare("-goi") == 0) {
            opts.gfile = argv[++i];
        } else {
            usage();
            error_and_exit(__PRETTY_FUNCTION__, "Unknown option.");
        }
        i += 1;
    }
    if (opts.nfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No network file specified.");
    }
}

int main(int argc, char const* argv[]) {
    parse_arguments(argc, argv);
    ifstream ifs(opts.nfile, ifstream::in);
    SGraph<Gene> gn;
    gn.deserialize(ifs);
    ifs.close();
    // filter network
    gn.filter_edges(opts.cutoff);
    auto genes = gn.get_vertices();
    auto nv = genes.size();
    auto ne = gn.number_of_edges();
    // process goi if given
    unordered_set<string> goi;
    int goi_ns = 0; // non-singleton count
    if (!opts.gfile.empty()) {
        vector<string> goi_vec = load_items<string>(opts.gfile);
        goi = unordered_set<string>(goi_vec.begin(), goi_vec.end());
        for (size_t i = 0; i < nv; ++i) {
            if(CONTAINS(goi, genes[i].id) && gn.degree_of(static_cast<index_t>(i)) > 1) {
                goi_ns ++;
            }
        }
        cout << "Non-singleton nodes: " << goi_ns << " / " << goi.size() << endl;
    }
    cout
        << "Network property\n"
        << "----------------\n"
        << "Number of nodes: [" << nv << "]\n"
        << "Number of edges: [" << ne << "]\n"
        << "Number of CC:    [" << gn.number_of_connected_component() << "]\n"
        << "Sparsity:        [" << static_cast<double>(ne) / ((nv * nv - nv) >> 1) << "]\n"
        << "\n";

    if (!opts.dfile.empty()) {
        ofstream ofs(opts.dfile, iostream::out);
        for (size_t i = 0; i < nv; ++i) {
            ofs
                << genes[i].id << "\t"
                << gn.degree_of(static_cast<index_t>(i))
                << "\n";
        }
        ofs.close();
    }
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Mon 07 Sep 2020 05:08:59 PM E
