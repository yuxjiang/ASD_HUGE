#include <fstream>
#include <boost/program_options.hpp>
#include <libcb/util.h>
#include <libcb/gene_network.h>

#define ERRMSG(x) error_and_exit(__PRETTY_FUNCTION__, x)
#define LOGMSG(x) log_message(__PRETTY_FUNCTION__, x)

using namespace std;

struct {
    string eFile;
    string oFile;
    string nFile;
    char delim;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    make-net.run - make a network.\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    make-net.run -e <EDGE FILE> -o <OUTPUT FILE> [OPTIONS]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program makes a network in its BINARY format.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -e <EDGE FILE>\n"
        << "        A 3-column tab-splitted edge file with the format: <src> <dst> <weight>\n"
        << "\n"
        << "    -o <OUTPUT FILE>\n"
        << "        An output file name.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -n <NODE FILE>, default: (empty)\n"
        << "        A file of a list of nodes in the network.\n"
        << "        The output network will be constructed only from <EDGE FILE> if this file is not given.\n"
        << "    -d <delimiter>, default: TAB\n"
        << "        Delimiter in the edge file.\n"
        << "\n";
}

void parseArgument(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    string msg;
    opts.delim = '\t'; // default delimiter
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]).compare("-e") == 0) {
            opts.eFile = argv[++i];
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.oFile = argv[++i];
        } else if (string(argv[i]).compare("-n") == 0) {
            opts.nFile = argv[++i];
        } else if (string(argv[i]).compare("-d") == 0) {
            opts.delim = argv[++i][0];
        } else {
            usage();
            msg = string("Unknown option [") + string(argv[i]) + string("].");
            ERRMSG(msg);
        }
    }
    if (opts.eFile.empty()) {
        usage();
        ERRMSG(string("Missing edge file."));
    }
    if (opts.oFile.empty()) {
        usage();
        ERRMSG(string("Missing output file."));
    }
}

int main(int argc, const char* argv[]) {
    parseArgument(argc, argv);

#ifdef ENABLE_LOGGING
    string msg;
#endif

#ifdef ENABLE_LOGGING
    msg = "Start creating gene network from [";
    if (!opts.nFile.empty()) {
        msg += opts.nFile;
        msg += "](nodes) and [";
    } 
    msg += opts.eFile;
    msg += "](edges).";
    LOGMSG(msg);
#endif

    GeneNetwork gn;
    if (opts.nFile.empty()) {
        gn = load_gene_network_by_edges(opts.eFile, opts.delim);
    } else {
        gn = load_gene_network_by_nodes_and_edges(opts.nFile, opts.eFile, opts.delim);
    }

#ifdef ENABLE_LOGGING
    msg = "Network created with ";
    msg += to_string(gn.number_of_vertices());
    msg += " genes and ";
    msg += to_string(gn.number_of_edges());
    msg += " edges.";
    LOGMSG(msg);
#endif

    ofstream ofs(opts.oFile, ofstream::out);

#ifdef ENABLE_LOGGING
    msg = "Saving network to [";
    msg += opts.oFile;
    msg += "].";
    LOGMSG(msg);
#endif

    gn.serialize(ofs);
    ofs.close();
    return 0;
}
