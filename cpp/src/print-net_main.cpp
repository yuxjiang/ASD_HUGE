#include <fstream>
#include <utility>
#include <libcb/gene_network.h>

#define ERRMSG(x) error_and_exit(__PRETTY_FUNCTION__, x)
#define LOGMSG(x) log_message(__PRETTY_FUNCTION__, x)

using namespace std;

struct {
    string nFile;
    double cutoff = 0.5;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    print-net.run - print a network.\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    print-net.run -n <NETWORK FILE> [OPTIONS]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program prints every edges of a BINARY network.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n <NETWORK FILE>\n"
        << "        A network file in its BINARY format.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -cutoff <Real, [0, 1]>, default: 0.5\n"
        << "        A network edge cutoff.\n"
        << "\n";
}

void parseArgument(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    string msg;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]).compare("-n") == 0) {
            opts.nFile = argv[++i];
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0 || opts.cutoff > 1) {
                usage();
                msg = string("Cutoff must be in the interval [0, 1].");
                ERRMSG(msg);
            }
        } else {
            usage();
            msg = string("Unknown option [") + string(argv[i]) + string("].");
            ERRMSG(msg);
        }
    }
    if (opts.nFile.empty()) {
        usage();
        ERRMSG(string("Missing network file."));
    }
}

int main(int argc, char const* argv[]) {
    parseArgument(argc, argv);
    ifstream ifs(opts.nFile, ofstream::in);
    GeneNetwork gn;
    gn.deserialize(ifs);
    ifs.close();
    // filter edges
    gn.filter_edges(opts.cutoff);
    vector<Gene> genes = gn.get_vertices();
    vector<GeneNetwork::weighted_edge_type> weightedEdges = gn.get_weighted_edges();
    for (const auto& e: weightedEdges) {
        index_t u, v;
        GeneNetwork::weight_type w;
        tie(u, v, w) = e;
        cout << genes[u].id << "\t"
             << genes[v].id << "\t"
             << w << "\n";
    }
    return 0;
}
