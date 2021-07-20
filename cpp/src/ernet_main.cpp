#include <iostream>
#include <string>
#include <random>
#include <chrono>

#include <libcb/util.h>
#include <libcb/gene_network.h>

#define ERRMSG(x) error_and_exit(__PRETTY_FUNCTION__, x)
#define LOGMSG(x) log_message(__PRETTY_FUNCTION__, x)

using namespace std;

struct {
    string NFile     = "";
    string nFile     = "";
    string oFile     = "";
    double p         = 0.0;
    double cutoff    = 0.5;
    unsigned long  M = 0;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    ernet.run - makes an Erdos-Renyi random network.\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    ernet.run -o <OUTPUT FILE> [-n <NODE FILE>|-N <NET FILE>] [-p <Real>|-M <Integer>] [OPTIONS]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program makes an Erdos-Renyi random network in its BINARY format.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -o <OUTPUT FILE>\n"
        << "        An output file name.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -n <NODE FILE>\n"
        << "        A file of a list of nodes in the network.\n"
        << "\n"
        << "    -N <NET FILE>\n"
        << "        A network BINARY file.\n"
        << "        If this file is given, it shadows [-n], [-p] and [-M], since the ER network will\n"
        << "        have exactly the same genes and number of edges as the given one.\n"
        << "\n"
        << "    -p <Real, [0, 1]>\n"
        << "        The probability of connecting two nodes.\n"
        << "\n"
        << "    -M <Integer, positive>\n"
        << "        The total number of edges.\n"
        << "        A non-zero value of [-M] shadows [-p]\n"
        << "\n"
        << "    -cutoff <Real, [0, 1]>, default: 0.5\n"
        << "        This cutoff takes effect if an existing network is given through [-N].\n"
        << "\n";
}

void parseArguments(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    string msg;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]).compare("-N") == 0) {
            opts.NFile = argv[++i];
        } else if (string(argv[i]).compare("-n") == 0) {
            opts.nFile = argv[++i];
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.oFile = argv[++i];
        } else if (string(argv[i]).compare("-p") == 0) {
            opts.p = stod(argv[++i]);
            if (opts.p < 0 || opts.p > 1) {
                usage();
                msg = string("Probability must be in the interval [0, 1]");
                ERRMSG(msg);
            }
        } else if (string(argv[i]).compare("-M") == 0) {
            opts.M = stoi(argv[++i]);
            if (opts.M <= 0) {
                usage();
                msg = string("-M must be positive.");
                ERRMSG(msg);
            }
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0 || opts.cutoff > 1) {
                usage();
                ERRMSG(string("[-cutff] must be within the interval [0, 1]."));
            }
        } else {
            usage();
            msg = string("Unknown option [") + string(argv[i]) + string("].");
            ERRMSG(msg);
        }
    }
    if (opts.oFile.empty()) {
        usage();
        ERRMSG(string("Missing output file."));
    }
    if (opts.NFile.empty() && opts.nFile.empty()) {
        usage();
        ERRMSG(string("Must specify either <NET FILE> or <NODE FILE>."));
    }
}

int main(int argc, const char* argv[]) {
    parseArguments(argc, argv);
    // load/make gene network
    GeneNetwork gn;
    if (!opts.NFile.empty()) {
        ifstream ifs(opts.NFile, ifstream::in);
        gn.deserialize(ifs);
        ifs.close();
        gn.filter_edges(opts.cutoff);
        opts.p = 0; // will be ignored
        opts.M = gn.number_of_edges();
        gn.remove_all_edges();
    } else {
        gn = load_gene_network_by_nodes(opts.nFile);
    }
    // create ER net
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    index_t vertexCount = gn.number_of_vertices();
    if (opts.M > 0) {
        vector<GeneNetwork::edge_type> edges;
        for (index_t r = 0; r < vertexCount; ++r) {
            for (index_t c = r + 1; c < vertexCount; ++c) {
                edges.push_back(make_pair(r, c));
            }
        }
        shuffle(edges.begin(), edges.end(), generator);
        for (size_t i = 0; i < opts.M; ++i) {
            index_t r, c;
            std::tie(r, c) = edges[i];
            gn.set_edge_weight(r, c, 1);
        }
    } else {
        uniform_real_distribution<double> distribution(0.0, 1.0);
        for (index_t r = 0; r < vertexCount; ++r) {
            for (index_t c = r + 1; c < vertexCount; ++c) {
                if (distribution(generator) < opts.p) {
                    gn.set_edge_weight(r, c, 1);
                }
            }
        }
    }
    ofstream ofs(opts.oFile, ofstream::out);
    gn.serialize(ofs);
    ofs.close();
    return 0;
}
