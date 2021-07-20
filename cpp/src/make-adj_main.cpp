#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <libcb/gene_network.h>

using namespace std;

struct {
    string nfile;
    string ofile;
    string dfile;
    float cutoff       = 0.5;
    size_t shift       = 0;
    bool inc_singleton = true;
    bool bilateral     = true;
} opts;

void usage() {
    cerr
        << "NAME\n"
        << "    make-adj.run - make adjacency matrix\n"
        << "\n"
        << "SYNOPSIS\n"
        << "    make-adj.run -n <NETWORK FILE> -o <OUTPUT FILE> -d <INDEX FILE> [OPTION]...\n"
        << "\n"
        << "DESCRIPTION\n"
        << "    Makes an plain-text (ASCII) adjacency file, and an node index file.\n"
        << "\n"
        << "MANDATORY ARGUMENTS\n"
        << "    -n <NETWORK FILE>\n"
        << "        A network file in its BINARY format.\n"
        << "\n"
        << "    -o <OUTPUT FILE>\n"
        << "        The output adjacency file.\n"
        << "\n"
        << "    -d <INDEX FILE>\n"
        << "        The node index file.\n"
        << "        <index> <node id>\n"
        << "\n"
        << "OPTIONAL ARGUMENTS\n"
        << "    -cutoff <Real, non-negative> default: 0.5\n"
        << "        Network edge cutoff, aka. tau\n"
        << "\n"
        << "    -begin <Integer, non-negative> default: 0\n"
        << "        The beginning of index.\n"
        << "\n"
        << "    -inc-singleton <Y|N> default: Y\n"
        << "        To include singleton or not.\n"
        << "\n"
        << "    -bilateral <Y|N> default: Y\n"
        << "        To output adjacency matrix in the bilateral maner. E.g., for an edge (x, y):\n"
        << "        Y - both x and y will be in the row of the other.\n"
        << "        N - only y will be in the row of x (assuming the index x < y).\n"
        << "\n";
}

void parse_argument(int argc, const char* argv[]) {
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
        } else if (string(argv[i]).compare("-o") == 0) {
            opts.ofile = argv[++i];
        } else if (string(argv[i]).compare("-d") == 0) {
            opts.dfile = argv[++i];
        } else if (string(argv[i]).compare("-begin") == 0) {
            int choice = stoi(argv[++i]);
            if (choice < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Beginning of index must be >= 0.");
            }
            opts.shift = choice;
        } else if (string(argv[i]).compare("-cutoff") == 0) {
            opts.cutoff = stod(argv[++i]);
            if (opts.cutoff < 0) {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Cutoff should be >= 0.");
            }
        } else if (string(argv[i]).compare("-inc-singleton") == 0) {
            string sel(argv[++i]);
            if (sel.compare("Y") == 0 || sel.compare("y") == 0) {
                opts.inc_singleton = true;
            } else if (sel.compare("N") == 0 || sel.compare("n") == 0) {
                opts.inc_singleton = false;
            } else {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Exclude singleton must be [Y/N].");
            }
        } else if (string(argv[i]).compare("-bilateral") == 0) {
            string bilateral(argv[++i]);
            if (bilateral.compare("Y") == 0 || bilateral.compare("y") == 0) {
                opts.bilateral = true;
            } else if (bilateral.compare("N") == 0 || bilateral.compare("n") == 0) {
                opts.bilateral = false;
            } else {
                usage();
                error_and_exit(__PRETTY_FUNCTION__, "Bilateral must be [Y/N].");
            }
        } else {
            usage();
            error_and_exit(__PRETTY_FUNCTION__, "Unknown argument.");
        }
        i += 1;
    }
    if (opts.nfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No network file specified.");
    }
    if (opts.ofile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No output file specified.");
    }
    if (opts.dfile.empty()) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "No index file specified.");
    }
}

int main(int argc, const char* argv[]) {
    parse_argument(argc, argv);
    SGraph<Gene> gn;
    ifstream ifs(opts.nfile, ifstream::in);
    gn.deserialize(ifs);
    ifs.close();
    gn.filter_edges(opts.cutoff); // filter edges
    if (!opts.inc_singleton) {
        gn.remove_singletons();
    }
    auto genes = gn.get_vertices();
    ofstream ofs;
    ofs.open(opts.dfile, ofstream::out);
    for (size_t i = 0; i < genes.size(); ++i) {
        ofs << i + opts.shift << "\t" << genes[i].id << "\n";
    }
    ofs.close();
    ofs.open(opts.ofile, ofstream::out);
    if (opts.bilateral) {
        for (size_t i = 0; i < genes.size(); ++i) {
            ofs << i + opts.shift;
            auto neighbors = gn.neighbor_of(static_cast<index_t>(i));
            for (const auto& neighbor : neighbors) {
                ofs << "\t" << neighbor + opts.shift;
            }
            ofs << "\n";
        }
    } else {
        for (size_t i = 0; i < genes.size(); ++i) {
            ofs << i + opts.shift;
            auto neighbors = gn.neighbor_of(static_cast<index_t>(i));
            for (const auto& neighbor : neighbors) {
                if (neighbor > i) {
                    ofs << "\t" << neighbor + opts.shift;
                }
            }
            ofs << "\n";
        }
    }
    ofs.close();
    return 0;
}
