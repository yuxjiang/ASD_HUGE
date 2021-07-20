#include <string>
#include <fstream>

#include <libcb/graph.h>
#include <libcb/gene_network.h>

using namespace std;

#define ERRMSG(x) error_and_exit(__PRETTY_FUNCTION__, x)
#define LOGMSG(x) log_message(__PRETTY_FUNCTION__, x)

struct {
    string iFile1;
    string iFile2;
    string oFile;
    SGraph<Gene>::merge_scheme scheme = SGraph<Gene>::minimum;
    bool removeSingleton = true;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    merge-net.run - merges two networks\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    merge-net.run -n1 <NET1 FILE> -n2 <NET2 FILE> -o <OUTPUT FILE> [OPTION]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    This program merges two gene networks.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n1 <NET1 FILE>\n"
        << "        The file name of the 1st network in its BINARY format.\n"
        << "\n"
        << "    -n2 <NET2 FILE>\n"
        << "        The file name of the 2nd network in its BINARY format.\n"
        << "\n"
        << "    -o <OUTPUT FILE>\n"
        << "        The file name of the resulting network.\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -scheme <add|min|max> default: min\n"
        << "        min - minimum:  e = min(e1, e2)\n"
        << "        max - maximum:  e = max(e1, e2)\n"
        << "        add - addition: e = e1 + e2\n"
        << "\n"
        << "    -remove-singletons <y|n> default: y\n"
        << "        Remove singletons from the resulting network or not.\n"
        << "\n";
}

void parseArguments(int argc, const char* argv[]) {
    if (argc <= 1) {
        usage();
        exit(1);
    }
    string msg;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]).compare("-n1") == 0) {
            opts.iFile1 = argv[++i];
        } else if (string(argv[i]).compare("-n2") == 0) {
            opts.iFile2 = argv[++i];
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.oFile = argv[++i];
        } else if (string(argv[i]).compare("-scheme") == 0) {
            string choiceString = argv[++i];
            if (choiceString.compare("min") == 0) {
                opts.scheme = SGraph<Gene>::minimum;
            } else if (choiceString.compare("max") == 0) {
                opts.scheme = SGraph<Gene>::maximum;
            } else if (choiceString.compare("add") == 0) {
                opts.scheme = SGraph<Gene>::addition;
            } else {
                msg = string("Unknown merging scheme: [") + choiceString + string("].");
                ERRMSG(msg);
            }
        } else if (string(argv[i]).compare("-remove-singletons") == 0) {
            string yesOrNo = argv[++i];
            if (yesOrNo.compare("Y") == 0 || yesOrNo.compare("y") == 0) {
                opts.removeSingleton = true;
            } else if (yesOrNo.compare("N") == 0 || yesOrNo.compare("n") == 0) {
                opts.removeSingleton = false;
            } else {
                msg = string("Unknown choice of [-remove-singletons]: [") + yesOrNo + string("].");
                ERRMSG(msg);
            }
        } else {
            msg = string("Unknown option: [") + string(argv[i]) + string("].");
            ERRMSG(msg);
        }
    }
    if (opts.iFile1.empty()) {
        usage();
        msg = "Missing network 1.";
        ERRMSG(msg);
    }
    if (opts.iFile2.empty()) {
        usage();
        msg = "Missing network 2.";
        ERRMSG(msg);
    }
    if (opts.oFile.empty()) {
        usage();
        msg = "Missing output file.";
        ERRMSG(msg);
    }
}

int main(int argc, const char* argv[]) {
    parseArguments(argc, argv);
    SGraph<Gene> gn1, gn2;
    // read two graphs
    ifstream ifs;
    ifs.open(opts.iFile1, istream::in);
    gn1.deserialize(ifs);
    ifs.close();
    ifs.open(opts.iFile2, istream::in);
    gn2.deserialize(ifs);
    ifs.close();

#ifdef ENABLE_LOGGING
    std::string msg("");
    msg += "Start merging two gene networks:\n  ";
    msg += std::to_string(gn1.number_of_vertices());
    msg += " genes, ";
    msg += std::to_string(gn1.number_of_edges());
    msg += " edges from [";
    msg += opts.iFile1;
    msg += "]\n  ";
    msg += std::to_string(gn2.number_of_vertices());
    msg += " genes, ";
    msg += std::to_string(gn2.number_of_edges());
    msg += " edges from [";
    msg += opts.iFile2;
    msg += "].";
    LOGMSG(msg);
#endif

    SGraph<Gene> gn3 = gn1.merge(gn2, opts.scheme);
    if (opts.removeSingleton) {
        gn3.remove_singletons();
    }

#ifdef ENABLE_LOGGING
    msg = "Finish merging.\n  ";
    msg += "After removing singletons: ";
    msg += std::to_string(gn3.number_of_vertices());
    msg += " genes, ";
    msg += std::to_string(gn3.number_of_edges());
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

    gn3.serialize(ofs);
    ofs.close();
    return 0;
}
